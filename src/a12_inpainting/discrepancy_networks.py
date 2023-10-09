import torch, torchvision
from torch import nn
import numpy as np
from kornia.losses.focal import FocalLoss
from easydict import EasyDict
from collections import OrderedDict

from ..pipeline.transforms_pytorch import torch_onehot
from ..common.util_networks import Padder

#from ..a05_differences.networks import CorrDifference01

class Correlation(nn.Module):

	@staticmethod
	def operation(a, b):
		"""
		B x C x H x W
		"""
		return torch.sum(a * b, dim=1, keepdim=True)

	def forward(self, a, b):
		return self.operation(a, b)


class VggFeatures(nn.Module):
	LAYERS_VGG16 = [3, 8, 15, 22, 29]

	def __init__(self, vgg_mod, layers_to_extract, freeze=True):
		super().__init__()

		vgg_features = vgg_mod.features

		ends = np.array(layers_to_extract, dtype=np.int32) + 1
		starts = [0] + list(ends[:-1])

		# print(list(zip(starts, ends)))

		self.slices = nn.Sequential(*[
			nn.Sequential(*vgg_features[start:end])
			for (start, end) in zip(starts, ends)
		])

		if freeze:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, image, **_):
		results = []
		value = image
		for slice in self.slices:
			value = slice(value)
			results.append(value)

		return results


class FeatureExtractorResNet(nn.Module):
	ResNet_layers_all = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

	# ResNet_layers_extract_default = ['maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
	ResNet_layers_default = {'relu': 64, 'maxpool': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
	ResNet_layers_extract_default = list(ResNet_layers_default.keys())

	"""
	image: (1, 3, 1024, 2048)
	relu: (1, 64, 512, 1024)
	maxpool: (1, 64, 256, 512)
	layer1: (1, 256, 256, 512)
	layer2: (1, 512, 128, 256)
	layer3: (1, 1024, 64, 128)
	layer4: (1, 2048, 32, 64)
	"""

	def __init__(self, backbone = torchvision.models.resnext50_32x4d, freeze=True, layers_to_extract = None):
		super().__init__()

		# TODO cache
		if isinstance(backbone, torchvision.models.resnet.ResNet):
			pass
		else:
			backbone = backbone(pretrained=True)
				
		self.layers_to_extract = layers_to_extract or self.ResNet_layers_extract_default
		self.copy_layers_from_pretrained(backbone)
		#self.resnet = self.resnet.to('cuda')
		
		if freeze:
			for param in self.resnet.parameters():
				param.requires_grad = False

				
	def copy_layers_from_pretrained(self, backbone_net):
		layers_to_extract_remaining = set(self.layers_to_extract)

		self.layers_sequential_names = []
		self.feat_channels = [] # number of channels in extracted layers

		# extract the needed layers
		self.resnet = torch.nn.Module()
		for layer_name in self.ResNet_layers_all:
			setattr(self.resnet, layer_name, getattr(backbone_net, layer_name))

			self.layers_sequential_names.append(layer_name)
			
			
			if layer_name in layers_to_extract_remaining:
				layers_to_extract_remaining.remove(layer_name)
				self.feat_channels.append(self.ResNet_layers_default[layer_name])


				if not layers_to_extract_remaining:
					# all useful layers extracted, finish
					return
		

	def forward(self, image, **_):
		results = OrderedDict()
		value = image
		
		for layer_name in self.layers_sequential_names:
			value = getattr(self.resnet, layer_name)(value)
			
			if layer_name in self.layers_to_extract:
				results[layer_name] = value

		return results

	@classmethod
	def get(cls, name, **opts):
		backbone = {
			'resnext50_32x4d': torchvision.models.resnext50_32x4d,
			'resnext101_32x8d': torchvision.models.resnext101_32x8d,
		}[name]

		return cls(backbone=backbone, **opts)



class ZeroedVggFeatures:

	vgg16_num_channels = [
		64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
	]
	vgg16_size_ratios = [
		1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 32,
	]


	@staticmethod
	def determine_backbone_shapes(backbone_net, test_image):
		"""
		backbone_net = torchvision.models.vgg16(pretrained=True).features
		test_image = torch.zeros((1, 3, 64, 128), dtype=torch.float32)

		r = ZeroedVggFeatures.determine_backbone_shapes(backbone_net, test_image)
		"""

		net_device = next(backbone_net.parameters()).device

		img_on_device = test_image.to(net_device)
		intermediate_repr = img_on_device
		
		initial_shape = tuple(img_on_device.shape)
		shapes = []

		for i, layer in enumerate(backbone_net):
			intermediate_repr = layer(intermediate_repr)
			shapes.append(tuple(intermediate_repr.shape))
		
		# assuming BCHW
		num_channels = [s[1] for s in shapes]

		size_ratios = [initial_shape[2] // s[2] for s in shapes]

		print('num channels =', num_channels)
		print('size ratios =', size_ratios)

		return dict(
			layer_num_channels = num_channels,
			layer_size_ratios = size_ratios,
		)



	def __init__(self, layers_to_extract):
		self.layers_to_extract = layers_to_extract

	def __call__(self, img):
		b, c, h, w = img.shape
		out = []

		for lid in self.layers_to_extract:
			ratio = self.vgg16_size_ratios[lid]

			feat_sh = (b, self.vgg16_num_channels[lid], h // ratio, w // ratio)

			out.append(torch.zeros(size = feat_sh, dtype=img.dtype, device=img.device))

		return out



class FeatureExtractorForComparator(nn.Module):

	vgg_layers_to_extract = VggFeatures.LAYERS_VGG16[:4]

	def __init__(self, freeze=True, separate_gen_image=True, perspective=False, backbone_name='vgg'):
		super().__init__()

		self.separate_gen_image = separate_gen_image
		self.perspective = perspective
		self.backbone_name = backbone_name
		
		if backbone_name == 'vgg':
			self.vgg_extractor = VggFeatures(
				vgg_mod = torchvision.models.vgg16(pretrained=True),
				layers_to_extract = self.vgg_layers_to_extract,
				freeze = freeze,
			)

			self.feat_channels = [64, 128, 256, 512]

		elif backbone_name.startswith('resne'):
			self.backbone = FeatureExtractorResNet.get(
				backbone_name,
				# we choose relu instead of maxpool, because our pyramid is set to 
				# change scale 2x on every layer
				layers_to_extract = ['relu', 'layer1', 'layer2', 'layer3'],
				freeze = freeze,
			)
			self.feat_channels = self.backbone.feat_channels

		else:
			raise NotImplementedError(f'Backbone {backbone_name}')

		# perspective added as an extra feature
		if self.perspective:
			self.feat_channels = [c+1 for c in self.feat_channels]
		

	def extract(self, image):
		if self.backbone_name == 'vgg':
			return self.vgg_extractor(image)
		else:
			feats_by_name = self.backbone(image)
			return list(feats_by_name.values())


	def extract_for_single_image(self, image, perspective_scale_map=None):
		feats = self.extract(image)

		# add perspective scale map as another channel
		if self.perspective:
			if perspective_scale_map is None:
				raise ValueError(f'perspective is true but perspective_scale_map is None')
			
			for i in range(feats.__len__()):
				feats[i] = torch.cat([feats[i], perspective_scale_map[:, None]], dim=1)
				perspective_scale_map = perspective_scale_map[:, ::2, ::2]
			
		return feats

	def forward(self, image, gen_image = None, perspective_scale_map = None, **_):
		"""
		@return [stream_1_feats, stream_2_feats]
		"""

		if self.perspective:
			# scale to try prevent NANs
			perspective_scale_map = perspective_scale_map * 0.001

		feats_img = self.extract_for_single_image(image, perspective_scale_map=perspective_scale_map)

		feats_streams = [feats_img, feats_img]

		# 2nd stream is gen-image
		if self.separate_gen_image:
			if gen_image is None:
				raise ValueError(f'separate_gen_image is true but gen_image is None')

			feats_gen = self.extract_for_single_image(gen_image, perspective_scale_map=perspective_scale_map)

			feats_streams[1] = feats_gen

		return feats_streams


class ComparatorUNet(nn.Module):
	"""
	- extracts features from 2 streams - image, gen_image
	- mix features
	- combine in an up-conv pyramid
	"""

	class UpBlock(nn.Sequential):
		def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True):

			modules = [
				nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
				nn.SELU(inplace=True),
				nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
				nn.SELU(inplace=True),
			]

			if b_upsample:
				modules += [
					nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
				]

			super().__init__(*modules)

	class CatMixCorr(nn.Module):
		def __init__(self, in_ch):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch*2, in_ch, kernel_size=1)

		def forward(self, prev, feats_img, feats_rec):
			channels = [prev] if prev is not None else []

			channels += [
				Correlation.operation(feats_img, feats_rec),
				self.conv_1x1(torch.cat([feats_img, feats_rec], 1)),
			]

			# print('cat', [ch.shape[1] for ch in channels])

			return torch.cat(channels, 1)

	class CatMix(nn.Module):
		def __init__(self, in_ch):
			super().__init__()
			# +1 to compensate for the removed correlation
			self.conv_1x1 = nn.Conv2d(in_ch*2, in_ch+1, kernel_size=1)

		def forward(self, prev, feats_img, feats_rec):
			channels = [prev] if prev is not None else []

			channels += [
				self.conv_1x1(torch.cat([feats_img, feats_rec], 1)),
			]

			return torch.cat(channels, 1)

	class CatMixIgnore(nn.Module):
		def forward(self, prev, feats_img, feats_rec):
			return prev

	#def __init__(self, num_outputs=2, freeze=True, correlation=True, separate_gen_image=True, perspective=False, skips_start_from=0, backbone=None):
	def __init__(self, extractor, num_outputs=2, freeze=True, correlation=True, skips_start_from=0):
	
		super().__init__()

		self.feature_extractor = extractor

		self.construct_fusion_pyramid(
			feat_channels = self.feature_extractor.feat_channels,
			out_channels = [256, 256, 128, 64],
			correlation = correlation,
			num_outputs = num_outputs,
			skips_start_from = skips_start_from,
		)


	def construct_fusion_pyramid(self, feat_channels, out_channels, num_outputs : int, correlation : bool, skips_start_from : int = 0):
		# feat_channels = [512, 256, 128, 64]
		# out_chans = [256, 256, 128, 64]

		# the pyramid goes from top (thick features) to bottom 
		feat_channels_rev = feat_channels[::-1]
		
		prev_chans = [0] + out_channels[:-1]
		cmis = []
		decs = []

		for i, fc, oc, pc in zip(range(feat_channels_rev.__len__(), 0, -1), feat_channels_rev, out_channels, prev_chans):

			#print(i, fc)
			#print(i, fc+1+pc, oc, oc)

			if i < skips_start_from:
				cmi = self.CatMixIgnore(fc)
			else:
				cmi = self.CatMixCorr(fc) if correlation else self.CatMix(fc)
			
			#print(f'fc {fc} + 1 + pc {pc} -> oc {oc}')

			dec = self.UpBlock(fc+1+pc, oc, oc, b_upsample=(i != 1))

			cmis.append(cmi)
			decs.append(dec)

			# self.add_module('cmi_{i}'.format(i=i), cmi)
			# self.add_module('dec_{i}'.format(i=i), dec)

		self.cmis = nn.ModuleList(cmis)
		self.decs = nn.ModuleList(decs)
		self.final = nn.Conv2d(out_channels[-1], num_outputs, kernel_size=1)


	def forward(self, image, gen_image = None, perspective_scale_map=None, **_):


		if gen_image is not None and gen_image.shape != image.shape:
			gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]

		if not self.training:
			padder = Padder(image.shape, 16)
			image = padder.pad(image)
			if gen_image is not None:
				gen_image = padder.pad(gen_image)
			if perspective_scale_map is not None:
				perspective_scale_map = padder.pad(perspective_scale_map[:, None])[:, 0]

		feat_streams = self.feature_extractor(
			image = image,
			gen_image = gen_image,
			perspective_scale_map = perspective_scale_map,
		)

		str_1, str_2 = feat_streams

		# def p_stream(name, s):
		# 	print('name:')
		# 	for i, f in enumerate(s):
		# 		print(f'	{i} - {tuple(f.shape)}')

		# p_stream('str_1', str_1)
		# p_stream('str_2', str_2)


		value = None
		num_steps = self.cmis.__len__()

		for i in range(num_steps):
			i_inv = num_steps-(i+1)
			# print(f'fuse {i_inv} - {tuple(str_1[i_inv].shape)}, {tuple(str_2[i_inv].shape)}')
			value = self.decs[i](
				self.cmis[i](value, str_1[i_inv], str_2[i_inv])
			)
			# print(f' value -> {tuple(value.shape)}')

		result = self.final(value)

		# if out size is too small, upsample
		if result.shape[2] < image.shape[2]:
			result = torch.nn.functional.interpolate(
				result,
				size = image.shape[2:4],			
				mode = 'bilinear',
				align_corners = False, # to get rid of warning
			)

		if not self.training:
			result = padder.unpad(result)

		return result



class ComparatorImageToImage(nn.Module):

	vgg_layers_to_extract = VggFeatures.LAYERS_VGG16[:4]

	CatMix = ComparatorUNet.CatMix
	CatMixCorr = ComparatorUNet.CatMixCorr
	UpBlock = ComparatorUNet.UpBlock
	
	def __init__(self, num_outputs=2, freeze=True, correlation=True):
		super().__init__()

		self.vgg_extractor = VggFeatures(
			vgg_mod = torchvision.models.vgg16(pretrained=True),
			layers_to_extract = self.vgg_layers_to_extract,
			freeze = freeze,
		)


		feat_channels = [512, 256, 128, 64]
		out_chans = [256, 256, 128, 64]
		prev_chans = [0] + out_chans[:-1]
		cmis = []
		decs = []

		for i, fc, oc, pc in zip(range(feat_channels.__len__(), 0, -1), feat_channels, out_chans, prev_chans):

			#print(i, fc)
			#print(i, fc+1+pc, oc, oc)

			cmi = self.CatMixCorr(fc) if correlation else self.CatMix(fc)
			dec = self.UpBlock(fc+1+pc, oc, oc, b_upsample=(i != 1))

			cmis.append(cmi)
			decs.append(dec)

			# self.add_module('cmi_{i}'.format(i=i), cmi)
			# self.add_module('dec_{i}'.format(i=i), dec)

		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

	def extractor_for_role(self, role):
		""" role is 'image' or 'gen' """
		return self.vgg_extractor

	def extract_features(self, image, gen_image, **_):
		return dict(
			feats_img = self.extractor_for_role('image')(image),
			feats_gen = self.extractor_for_role('gen')(gen_image),
		)

	def forward(self, image, gen_image, **_):

		if gen_image.shape != image.shape:
			gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]

		if not self.training:
			padder = Padder(image.shape, 16)
			image, gen_image = (padder.pad(x) for x in (image, gen_image))

		features = self.extract_features(image=image, gen_image=gen_image)		
		vgg_feats_img = features['feats_img']
		vgg_feats_gen = features['feats_gen']

		value = None
		num_steps = self.cmis.__len__()

		for i in range(num_steps):
			i_inv = num_steps-(i+1)
			value = self.decs[i](
				self.cmis[i](value, vgg_feats_img[i_inv], vgg_feats_gen[i_inv])
			)

		result = self.final(value)

		if not self.training:
			result = padder.unpad(result)

		return result


class ComparatorImageToEmpty(ComparatorImageToImage):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.zeroed_vgg_features = ZeroedVggFeatures(
			layers_to_extract = self.vgg_layers_to_extract,
		)

	def extractor_for_role(self, role):
		return {
			'image': self.vgg_extractor,
			'gen': self.zeroed_vgg_features,
		}[role]

	def forward(self, image, **_):
		return super().forward(image=image, gen_image=image)


class ComparatorImageToSelf(ComparatorImageToImage):
	def extract_features(self, image, gen_image=None, **_):
		feats = self.extractor_for_role('image')(image)
		return dict(
			feats_img = feats,
			feats_gen = feats,
		)
		
	def forward(self, image, **_):
		return super().forward(image=image, gen_image=image)


class FocalLossIgnoreInvalid(FocalLoss):

	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'none', weights : torch.Tensor = None):
		super().__init__(alpha=alpha, gamma=gamma, reduction='none')

		self.reduction_after_ignore = reduction
		self.weights = weights

	def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

		mask_valid = (target != 255).detach()

		if self.weights is not None:
			num_classes = input.shape[1]
			num_weights = self.weights.__len__()
			assert num_weights == num_classes, f'Weights have {num_weights} elements but there are {num_classes} classes'

			loss_weight_spatial = torch.zeros_like(target, dtype=torch.float32)

			for cl, weight in enumerate(self.weights):
				loss_weight_spatial += (target == cl).float() * weight

			#print('ssh', loss_weight_spatial.shape)
			
		else:
			loss_weight_spatial = mask_valid.float()
			#print('ssh', loss_weight_spatial.shape, 'tsh', target.shape)


		result = super().forward(
			input = input,
			target = target * mask_valid,
		)

		#print(result.shape, loss_weight_spatial.shape)
		result *= loss_weight_spatial

		if self.reduction_after_ignore == 'none':
			loss = result
		elif self.reduction_after_ignore == 'mean':
			loss = torch.mean(result)
		elif self.reduction_after_ignore == 'sum':
			loss = torch.sum(result)
		else:
			raise NotImplementedError(f"Invalid reduction mode: {self.reduction_after_ignore}")

		return dict(loss = loss)


	@staticmethod
	def try_weight_map():
		import torch 
		from src.common.jupyter_show_image import show
		from src.paths import DIR_DATA
		from src.a12_inpainting.synth_obstacle_dset import SynthObstacleDset
		from src.a12_inpainting.discrepancy_experiments import Exp1205_Discrepancy_ImgVsInpaiting

		dset = SynthObstacleDset.from_disk(DIR_DATA / '1204-SynthObstacleDset-v1-Ctc' / 'train')

		frt = dset[5]
		frt.update(Exp1205_Discrepancy_ImgVsInpaiting.translate_from_inpainting_dset_to_gen_image_terms(**frt))

		def try_weight_map(target, weights):
			mask_valid = (target != 255).detach()

			if weights is not None:
				loss_weight_spatial = torch.zeros_like(target, dtype=torch.float32)

				for cl, weight in enumerate(weights):
					loss_weight_spatial += (target == cl).float() * weight
			
			else:
				loss_weight_spatial = mask_valid.float()
				
			return loss_weight_spatial


		show([
			try_weight_map(torch.from_numpy(frt.semseg_errors_label), weights=torch.Tensor([1.5, 19.0])).numpy(),
			try_weight_map(torch.from_numpy(frt.semseg_errors_label), weights=None).numpy(),	
		])



NOISE_LAYERS_DEFAULT = (
	(0.18, 1.0),
 	(0.31, 0.5),
 	(0.84, 0.2),
)

def make_hierarchical_noise(dim_bchw = (1, 1, 1024, 2048), layers = NOISE_LAYERS_DEFAULT, device : torch.device = None, zero_centered : bool = True):
	
	dim_batch = dim_bchw[0]
	dim_ch = dim_bchw[1]
	dim_spatial = dim_bchw[2:4]
	dim_spatial_ar = torch.tensor(data=dim_spatial, device=torch.device('cpu'), requires_grad=False)
	
	out = torch.zeros(dim_bchw, device=device, requires_grad=False)
	
	for relative_size, magnitude in layers:
		h, w = (dim_spatial_ar * relative_size).int()
		
		noise = torch.rand((dim_batch, dim_ch, h, w), device=device, requires_grad=False)
		
		if zero_centered:
			noise -= 0.5
		noise *= magnitude
		
		out += torch.nn.functional.interpolate(
			noise, 
			size = dim_spatial,
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)
		
	return out


class NoiseAugmentation:
	def __init__(self, layer_defs = NOISE_LAYERS_DEFAULT, magnitude_range = (0.5, 1.0)):
		self.layer_defs = layer_defs
		
		mag_min, mag_max = magnitude_range

		self.magnitude_min = mag_min
		self.magnitude_spread = mag_max - mag_min

	
	def __call__(self, image_tr):
		dim_b, dim_c, dim_h, dim_w = image_tr.shape
		
		noise = make_hierarchical_noise(
			dim_bchw = (dim_b, 1, dim_h, dim_w),
			layers = self.layer_defs,
			device = image_tr.device,
		)

		magnitude = self.magnitude_min + torch.rand(dim_b) * self.magnitude_spread


		noise *= magnitude[:, None, None, None].to(image_tr.device)

		return image_tr + noise
		

import cv2 as cv

class ModBlurInput:

	def __init__(self, cfg):
		ks = cfg['kernel_size']
		self.kernel_shape = (ks, ks)
		
	def __call__(self, image, **_):
		return dict(
			image = cv.GaussianBlur(image, self.kernel_shape, 0),
		)


def morphology_cleaning_v1(score):
	ks = 7
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ks, ks))

	sc = cv.morphologyEx(
		score,
		cv.MORPH_OPEN,
		kernel = kernel,
		iterations = 1,
	)
	sc = cv.dilate(
		sc,
		kernel = kernel,
		iterations = 2,
	)

	return sc

def morphology_cleaning_v2(score):
	ks = 7
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ks, ks))

	sc = cv.morphologyEx(
		score,
		cv.MORPH_OPEN,
		kernel = kernel,
		iterations = 1,
	)
	sc = cv.dilate(
		sc,
		kernel = kernel,
		iterations = 3,
	)

	return sc


# from pytorch_semantic_segmentation.models.psp_net import PSPNet as PSPNet_Orig

# class Seg_PSPNet(PSPNet_Orig):
# 	def __init__(self):
# 		super().__init__(
# 			num_classes = 2,
# 			pretrained = True,
# 			use_aux = False,
# 		)

# 	def forward(self, image, **_):
# 		return super().forward(image)


# class Seg_PSPNet_Perspective(Seg_PSPNet):

# 	def forward(self, image, perspective_scale_map, **_):
# 		return super().forward(image)

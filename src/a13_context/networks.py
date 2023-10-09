
import torch, torchvision
from torch import nn
from torch.nn.parameter import Parameter
from kornia.utils import image_to_tensor
from kornia.filters import Laplacian

from typing import List
from easydict import EasyDict
from functools import lru_cache

from ..common.util_networks import Padder

def feats_sizes(feats):
	m = '\n'.join(
		f'{k:<10} {tuple(v.shape)}' for k, v in feats.items()
	)
	print(m)

class FeatureExtractorResNet(nn.Module):
	ResNet_layers_all = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
	ResNet_layers_extract = set(['maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
	
	
	def __init__(self, backbone = torchvision.models.resnext50_32x4d, freeze=True):
		super().__init__()

		# TODO cache
		if isinstance(backbone, torchvision.models.resnet.ResNet):
			pass
		else:
			backbone = backbone(pretrained=True)
				
		# extract the needed layers
		self.resnet = torch.nn.Module()
		for layer_name in self.ResNet_layers_all:
			setattr(self.resnet, layer_name, getattr(backbone, layer_name))
		self.resnet = self.resnet.to('cuda')
		
		if freeze:
			for param in self.resnet.parameters():
				param.requires_grad = False

				
	def forward(self, image, **_):
		results = EasyDict()
		value = image
		
		for layer_name in self.ResNet_layers_all:
			value = getattr(self.resnet, layer_name)(value)
			
			if layer_name in self.ResNet_layers_extract:
				results[layer_name] = value

		return results

	@classmethod
	@lru_cache()
	def get(cls, name):
		backbone = {
			'rn50': torchvision.models.resnext50_32x4d,
			'rn101': torchvision.models.resnext101_32x8d,
		}[name]

		return cls(backbone=backbone)




class FeatureLaplacianSingle:
	def __init__(self, extractor, kernel_size, feature_name):
		self.laplacian = Laplacian(kernel_size).to('cuda')

		self.feature_name = feature_name
		# self.magnitude = magnitude

		if isinstance(extractor, str):
			extractor = FeatureExtractorResNet.get(extractor)

		self.extractor = extractor


	@property
	def kernel_size(self):
		return self.laplacian.kernel_size


	def forward_features(self, feat_tensor):
		lap_result = self.laplacian(feat_tensor)
	
		lap_sum = lap_result.abs().mean(1, keepdim=True)
			
		lap_sum_np = lap_sum.to('cpu').numpy()[0, 0]
		
		# print(np.min(lap_sum_np), np.mean(lap_sum_np), np.max(lap_sum_np))
		return lap_sum_np
	

	def forward_image(self, image_np):
		img_tr = image_to_tensor(image_np)
		img_tr = img_tr.float()
		img_tr *= 1./255.
		
		feats = self.extractor(img_tr.to('cuda')[None])

		out = self.forward_features(feats[self.feature_name])

		# out *= self.magnitude

		return out






class FeatContextDiffBlock(nn.Sequential):
	class SumOfAbs(nn.Module):
		""" 
		Applied on the filter ouput
		abs -> mean
		"""

		def forward(self, value):
			return value.abs().mean(1, keepdim=True)


	class SumOfAbsWeighted(nn.Module):
		"""
		Learned weighted average applied on the filter ouput
		abs -> learnable weighted average
		"""

		def __init__(self, feats_num_channels):
			super().__init__()
			self.weight = Parameter(torch.Tensor(feats_num_channels))
			self.reset_parameters()

		@property
		def feats_num_channels(self):
			return self.weight.shape[0]

		def reset_parameters(self):
			"""
			Initialized to perform mean: weight = 1/n
			"""
			nn.init.constant_(self.weight, 1. / self.feats_num_channels)

		def forward(self, value):
			# print(f'SumOfAbsWeighted, nch = {self.feats_num_channels} vs value {tuple(value.shape)}')

			diff_abs = value.abs() # abs of difference
			out = (self.weight[None, :, None, None] * diff_abs).sum(1, keepdim=True)

			# print(f'	out {tuple(out.shape)}, diffabs {tuple(diff_abs.shape)}')

			return out

	def __init__(self, arch_filter_type : str, arch_filter_size : str, arch_distance_type : str, feats_num_channels : int, **_):
		"""
		@param arch_filter_type: fixed-laplacian, learn-1ch-initlap, learn-1ch
		@param arch_filter_size
		@param arch_distance_type: l1, learn-abs
		@param feats_num_channels
		"""

		mod_filter = self.construct_filter(
			arch_filter_type = arch_filter_type,
			arch_filter_size = arch_filter_size
		)


		mod_distance = self.construct_distance(
			arch_distance_type = arch_distance_type,
			feats_num_channels = feats_num_channels,
		)

		super().__init__(mod_filter, mod_distance)

		self.arch_filter_type = arch_filter_type
		self.arch_filter_size = arch_filter_size
		self.arch_distance_type = arch_distance_type
		self.feats_num_channels = feats_num_channels
		
	def construct_filter(self, arch_filter_type : str, arch_filter_size : int):
		laplacian = Laplacian(
			kernel_size = arch_filter_size,
			normalized = True,
		)

		if arch_filter_type == 'fixed-laplacian':
			...

		elif arch_filter_type == 'learn-1ch-initlap':
			# make the kernel trainable, starting with laplacian initialization
			laplacian.kernel = Parameter(laplacian.kernel)

		elif arch_filter_type == 'learn-1ch':
			# make the kernel trainable, starting with random
			laplacian.kernel = Parameter(torch.laplacian.kernel)
			nn.init.kaiming_normal_(laplacian.kernel, a=2.236) #sqrt(5) like in torch.nn.Conv2d::reset_parameters

		else:
			raise NotImplementedError(f'arch_filter_type = {arch_filter_type}')
		
		return laplacian
		
	def construct_distance(self, feats_num_channels, arch_distance_type):
		if arch_distance_type == 'l1':
			return self.SumOfAbs()
		elif arch_distance_type == 'learn-abs':
			return self.SumOfAbsWeighted(feats_num_channels=feats_num_channels)
		else:
			raise NotImplementedError(f'arch_distance_type = {arch_distance_type}')
			


class UpFusionBlock(nn.Module):

	def __init__(self, num_ch_previous, num_ch_here, num_ch_next, arch_upsample_type, arch_mix_type, **_):

		# concat(previous, here)
		# 1x1 conv
		# up-conv to next

		super().__init__()

		self.num_ch_previous = num_ch_previous
		self.arch_upsample_type = arch_upsample_type

		if num_ch_previous:
			if arch_upsample_type == 'UPconv':
				self.upconvolution = nn.ConvTranspose2d(num_ch_previous, num_ch_previous, kernel_size=2, stride=2)
			elif arch_upsample_type == 'UPbilinear':
				self.upconvolution = nn.UpsamplingBilinear2d(scale_factor=2)
			else:
				raise NotImplementedError(f'arch_upsample_type {arch_upsample_type}')

		if arch_mix_type == 'mix1':
			self.mix = nn.Conv2d(num_ch_previous + num_ch_here, num_ch_next, kernel_size=1, bias=True)
		elif arch_mix_type == 'mix3':
			self.mix = nn.Conv2d(
				num_ch_previous + num_ch_here, 
				num_ch_next, 
				kernel_size=3, 
				bias=True,
				padding=1,
				padding_mode='reflect',
			)

		self.activation = nn.SELU(inplace=True)


	def forward(self, value_prev, value_here):

		if self.num_ch_previous:

			prev_up = self.upconvolution(value_prev)

			concat = torch.cat([
				prev_up,
				value_here,
			], 1) 
		
		else:

			concat = value_here

		out = self.mix(concat)
		out = self.activation(out)
		return out


class PadderMixin:
	def forward_padded(self, image):
		raise NotImplementedError()

	def forward(self, image, **_):
		if not self.training:
			padder = Padder(image.shape, 16)
			image = padder.pad(image)

		logits = self.forward_padded(image)

		if not self.training:
			logits = padder.unpad(logits)

		return logits



class ContextDetectorSlim(nn.Module, PadderMixin):
	def __init__(self, **arch_opts):
		"""
		@param arch_freeze_backbone
		@param arch_filter_type: fixed-laplacian, learn-1ch-initlap, learn-1ch
		@param arch_filter_size
		@param arch_distance_type: l1, learn-abs
		@param arch_upsample_type: upconv, bilinear
		"""
		super().__init__()

		self.extractor = FeatureExtractorResNet(
			freeze = arch_opts['arch_freeze_backbone'],
		)

		feat_depths = [256, 512, 1024, 2048]
		
		num_inter_chans = arch_opts.get('arch_inter_depth', 6)

		self.context_block_4 = FeatContextDiffBlock(
			feats_num_channels = feat_depths[3], 
			**arch_opts,
		)

		for li in range(3, 0, -1):
			context_block = FeatContextDiffBlock(
				feats_num_channels=feat_depths[li-1], 
				**arch_opts,
			)

			fuse_block = UpFusionBlock(
				num_ch_previous = 1 if li == 3 else num_inter_chans,
				num_ch_here = 1,
				num_ch_next = num_inter_chans,
				**arch_opts,
			)

			setattr(self, f'context_block_{li}', context_block)
			setattr(self, f'fuse_block_{li}', fuse_block)

		self.final = nn.Conv2d(num_inter_chans, 2, kernel_size=1, bias=True)


	def forward_padded(self, image, **_):

		feats = self.extractor(image)

		#feats_sizes(feats)

		value_prev = self.context_block_4(feats['layer4'])

		for li in range(3, 0, -1):
			context_block = getattr(self, f'context_block_{li}')
			fuse_block = getattr(self, f'fuse_block_{li}')
			feat = feats[f'layer{li}']

			# print(f'Layer {li}')
			# feats_sizes({
			# 	'value_prev': value_prev,
			# 	'feat': feat,
			# })

			val_context = context_block(feat)

			value_prev = fuse_block(value_prev, val_context)

		logits = self.final(value_prev)

		# features are 512 wide, but image is 2048 wide
		if logits.shape[2:] != image.shape[2:]:
			logits = nn.functional.interpolate(
				logits, 
				size = image.shape[2:],
				mode = 'bilinear', 
				align_corners = False,
			)

		return logits


class BackboneLaplacianMixin():
	def init_laplacian(self):
		if self.cfg.lap:
			self.mod_laplacian = Laplacian(
				kernel_size = self.cfg.lap.filter_size,
				normalized = True,
			).to('cuda')

	def laplacian(self, feats):
		if self.cfg.lap:
			feat_lap = self.mod_laplacian(feats)
			if self.cfg.lap.abs:
				torch.abs(feat_lap)
			return feat_lap
		else:
			return feats


from ..pytorch_semantic_segmentation import models as ptseg_models

#@LapNetRegistry.register_class()
class PSPNetLap(PadderMixin, ptseg_models.PSPNet, BackboneLaplacianMixin):


	def __init__(self, cfg):
		self.cfg = cfg
		num_classes = 2

		super().__init__(
			num_classes = num_classes, 
			pretrained = True, 
			use_aux = False,
		)

		self.init_laplacian()

	def forward_padded(self, image):
		img_sz = image.size()
		value = image

		for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
			value = layer(value)

		if self.cfg.lap:
			value = self.laplacian(value)

		value = self.final(self.ppm(value))

		return nn.functional.interpolate(value, img_sz[2:], mode='bilinear')

		# x = image
		# x_size = x.size()
		# x = self.layer0(x)
		# x = self.layer1(x)
		# x = self.layer2(x)
		# x = self.layer3(x)
		# if self.training and self.use_aux:
		# 	aux = self.aux_logits(x)
		# x = self.layer4(x)
		# x = self.ppm(x)
		# x = self.final(x)
		# if self.training and self.use_aux:
		# 	return F.interpolate(x, x_size[2:], mode='bilinear'), F.interpolate(aux, x_size[2:], mode='bilinear')
		# return F.interpolate(x, x_size[2:], mode='bilinear')



from src.erfnet_pytorch.train.erfnet import Net as ErfNet

class ErfNetOrig(PadderMixin, ErfNet, BackboneLaplacianMixin):
	def __init__(self, cfg):
		self.cfg = cfg
		num_classes = 2

		super().__init__(
			num_classes = num_classes, 
		)

		self.init_laplacian()

	def forward_padded(self, image):
		feat = self.encoder(image)

		if self.cfg.lap:
			feat = self.laplacian(feat)

		return self.decoder.forward(feat)

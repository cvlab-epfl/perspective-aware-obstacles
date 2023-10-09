
from easydict import EasyDict

import torch
from torch import nn
from . import Resnet
from .PosEmbedding import PosEmbedding2D
from .HANet import HANet_Conv
from .mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights, RandomPosVal_Masking, RandomVal_Masking, Zero_Masking, RandomPosZero_Masking
import torchvision.models as models

from .deepv3 import _AtrousSpatialPyramidPoolingModule

from road_anomaly_benchmark.datasets.dataset_registry import Registry

ModsFeatureProcessors = Registry()

@ModsFeatureProcessors.register_class()
class FeatPersDirect(nn.Module):
	configs = [
		dict(
			name = 'persDirect',
			modulate = 1/400,
		),
		dict(
			name = 'persDirectLayer1',
			modulate = 1/400,
			limit_to_layers = [0],
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)
		self.b_zeroed = False

	def build(self, in_channels, layer_level):
		"""
		@return: num output channels
		"""

		print(f'in ch {in_channels} layer level {layer_level}')

		limit_to_layers = self.cfg.get('limit_to_layers')
		if limit_to_layers and layer_level not in limit_to_layers:
			self.b_zeroed = True
			print('zeroed!')

		return 1

	def forward(self, feats, perspective_scale_map, **_):

		h, w = feats.shape[2:4]

		if self.b_zeroed:
			return torch.zeros((feats.shape[0], 1, h, w), dtype=feats.dtype, device=feats.device)

		#perspective_scale_map = perspective_scale_map[:, None] # add channel dimension

		psm_resized = nn.functional.interpolate(
			perspective_scale_map,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)

		return psm_resized * self.cfg.modulate

	def __repr__(self):
		return f'FeatPersDirect[{self.cfg.name}]'


@ModsFeatureProcessors.register_class()
class FeatPersEncoder(nn.Module):
	configs = [
		dict(
			name = 'persEncoder',
			num_ft = 16,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""

		self.encoder = nn.Sequential(
			nn.Conv2d(1, self.cfg.num_ft, kernel_size=1),
			nn.SELU(inplace=True),	
			nn.Conv2d(self.cfg.num_ft, self.cfg.num_ft, kernel_size=1),	
		)

		return self.cfg.num_ft

	def forward(self, feats, perspective_scale_map, **_):

		h, w = feats.shape[2:4]
		#perspective_scale_map = perspective_scale_map[:, None] # add channel dimension

		psm_resized = nn.functional.interpolate(
			perspective_scale_map,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		) * (1./400.)

		return self.encoder(psm_resized)

	def __repr__(self):
		return f'FeatPersEncoder[{self.cfg.name}]'


@ModsFeatureProcessors.register_class()
class FeatPersZoneEncoder(nn.Module):
	configs = [
		dict(
			name = 'zoneFixed8p',
			pass_direct = True,
			num_zones = 8,
			frozen = True,
			init_inv_width = 4.,
			init_centers_linspace = [0.1, 1.6],
		),
		dict(
			name = 'zoneTrain8p',
			pass_direct = True,
			num_zones = 8,
			frozen = False,
			init_inv_width = 4.,
			init_centers_linspace = [0.1, 1.6],
		),
	]

	@staticmethod
	def zone_encode(centers, inv_widths, pmap):
		dists_sq = ((pmap - centers[None, :, None, None]) * inv_widths[None, :, None, None]).pow(2)
		return torch.sigmoid(dists_sq) - 0.5

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""
		num_feat = self.cfg.num_zones
		zc_from, zc_to = self.cfg.init_centers_linspace
		
		self.zone_centers = nn.Parameter(torch.linspace(zc_from, zc_to, num_feat))
		self.zone_inv_widths = nn.Parameter(torch.full((num_feat,), fill_value=self.cfg.init_inv_width))

		if self.cfg.frozen:
			for p in [self.zone_centers, self.zone_inv_widths]:
				p.requires_grad = False

		if self.cfg.pass_direct:
			return num_feat+1
		else:
			return num_feat

	def forward(self, feats, perspective_scale_map, **_):
		h, w = feats.shape[2:4]
		psm_resized = nn.functional.interpolate(
			perspective_scale_map,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		) * (1./400.)

		zones = self.zone_encode(
			centers = self.zone_centers,
			inv_widths = self.zone_inv_widths,
			pmap = psm_resized,
		)

		if self.cfg.pass_direct:
			return torch.cat([
				psm_resized,
				zones,
			], dim=1)
		else:
			return zones



class DeepV3PlusPerspective(nn.Module):
	"""
	Implement DeepLab-V3 model
	A: stride8
	B: stride16
	with skip connections
	"""

	def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None, variant='D', skip='m1', skip_num=48, args=None, pers_enc_variant=None):
		super().__init__()
		self.criterion = criterion
		self.criterion_aux = criterion_aux
		self.variant = variant
		self.args = args
		self.trunk = trunk
					  
		if not trunk.startswith('resnet'):
			raise NotImplementedError(f'Backbone {trunk}')
	   
		channel_1st = 3
		channel_2nd = 64
		channel_3rd = 256
		channel_4th = 512
		prev_final_channel = 1024
		final_channel = 2048

		if trunk == 'resnet-50':
			resnet = Resnet.resnet50()
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		elif trunk == 'resnet-101': # three 3 X 3
			resnet = Resnet.resnet101()
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
										resnet.conv2, resnet.bn2, resnet.relu2,
										resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
		elif trunk == 'resnet-152':
			resnet = Resnet.resnet152()
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		elif trunk == 'resnext-50':
			resnet = models.resnext50_32x4d(pretrained=True)
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		elif trunk == 'resnext-101':
			resnet = models.resnext101_32x8d(pretrained=True)
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		elif trunk == 'wide_resnet-50':
			resnet = models.wide_resnet50_2(pretrained=True)
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		elif trunk == 'wide_resnet-101':
			resnet = models.wide_resnet101_2(pretrained=True)
			resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		else:
			raise ValueError("Not a valid network arch")

		self.layer0 = resnet.layer0
		self.layer1, self.layer2, self.layer3, self.layer4 = \
			resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

		if self.variant == 'D':
			for n, m in self.layer3.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
			for n, m in self.layer4.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
		elif self.variant == 'D4':
			for n, m in self.layer2.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
			for n, m in self.layer3.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
			for n, m in self.layer4.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
		elif self.variant == 'D16':
			for n, m in self.layer4.named_modules():
				if 'conv2' in n:
					m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
				elif 'downsample.0' in n:
					m.stride = (1, 1)
		else:
			# raise 'unknown deepv3 variant: {}'.format(self.variant)
			print("Not using Dilation ")

		if self.variant == 'D':
			os = 8
		elif self.variant == 'D4':
			os = 4
		elif self.variant == 'D16':
			os = 16
		else:
			os = 32

		if pers_enc_variant:
			self.perspective_encoder = ModsFeatureProcessors.get(pers_enc_variant)

			final_channel += self.perspective_encoder.build(final_channel, 0)

		else:
			self.perspective_encoder = None

		self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256, output_stride=os)


		self.bot_fine = nn.Sequential(
			nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
			Norm2d(48),
			nn.ReLU(inplace=True),
		)

		self.bot_aspp = nn.Sequential(
			nn.Conv2d(1280, 256, kernel_size=1, bias=False),
			Norm2d(256),
			nn.ReLU(inplace=True),
		)

		self.final1 = nn.Sequential(
			nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
			Norm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
			Norm2d(256),
			nn.ReLU(inplace=True),
		)

		self.final2 = nn.Sequential(
			nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
		)

		if self.args.aux_loss is True:
			self.dsn = nn.Sequential(
				nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
				Norm2d(512),
				nn.ReLU(inplace=True),
				nn.Dropout2d(0.1),
				nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
			)
			initialize_weights(self.dsn)

		initialize_weights(self.aspp)
		initialize_weights(self.bot_aspp)
		initialize_weights(self.bot_fine)
		initialize_weights(self.final1)
		initialize_weights(self.final2)

	def forward(self, x, gts=None, aux_gts=None, pos=None, attention_map=False, attention_loss=False):

		x_size = x.size()  # 800

		x = self.layer0(x)  # 400
		x = self.layer1(x)  # 400
		low_level = x
		x = self.layer2(x)  # 100

		x = self.layer3(x)  # 100

		aux_out = x
		x = self.layer4(x)  # 100

		# features finished

		# features inject!

		if self.perspective_encoder is not None:
			pmap = pos[0]
			pmap = pmap[:, None] # add channel dim
			# print('PMAP shape', pmap.shape)

			enc = self.perspective_encoder(
				feats = x,
				perspective_scale_map = pmap,
			)

			x = torch.cat([x,enc], dim=1)

		x = self.aspp(x)
			
		dec0_up = self.bot_aspp(x)

		dec0_fine = self.bot_fine(low_level)
		dec0_up = Upsample(dec0_up, low_level.size()[2:])

		dec0 = [dec0_fine, dec0_up]
		dec0 = torch.cat(dec0, 1)
		dec1 = self.final1(dec0)

		dec2 = self.final2(dec1)

		main_out = Upsample(dec2, x_size[2:])

		if self.training:
			loss1 = self.criterion(main_out, gts)

			if self.args.aux_loss is True:
				aux_out = self.dsn(aux_out)
				if aux_gts.dim() == 1:
					aux_gts = gts
				aux_gts = aux_gts.unsqueeze(1).float()
				aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
				aux_gts = aux_gts.squeeze(1).long()
				loss2 = self.criterion_aux(aux_out, aux_gts)
				
				return (loss1, loss2)
			else:
				return loss1
		else:
			return main_out



def DeepR101V3PlusD_persDirect(args, num_classes, criterion, criterion_aux):
	return DeepV3PlusPerspective(
		num_classes, 
		trunk='resnet-101', variant='D', skip='m1',
		criterion=criterion, criterion_aux=criterion_aux,
		args=args,
		pers_enc_variant='persDirect',
	)

def DeepR101V3PlusD_zoneFixed8p(args, num_classes, criterion, criterion_aux):
	return DeepV3PlusPerspective(
		num_classes, 
		trunk='resnet-101', variant='D', skip='m1',
		criterion=criterion, criterion_aux=criterion_aux,
		args=args,
		pers_enc_variant='zoneFixed8p',
	)

def DeepR101V3PlusD_persEncoder(args, num_classes, criterion, criterion_aux):
	return DeepV3PlusPerspective(
		num_classes, 
		trunk='resnet-101', variant='D', skip='m1',
		criterion=criterion, criterion_aux=criterion_aux,
		args=args,
		pers_enc_variant='persEncoder',
	)





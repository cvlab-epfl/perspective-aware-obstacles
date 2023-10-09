
from collections import OrderedDict
from easydict import EasyDict

import torch, torchvision
from torch.nn.modules.conv import Conv2d
from torch import nn

from ..common.registry import Registry

ModsBackbone = Registry()

@ModsBackbone.register_class()
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

	configs = [
		dict(
			name = 'resnext50_32x4d',
			source = 'torchvision',
			layers_to_extract = ['relu', 'layer1', 'layer2', 'layer3'],
			freeze = True,
		),
		dict(
			name = 'resnext101_32x8d',
			source = 'torchvision',
			layers_to_extract = ['relu', 'layer1', 'layer2', 'layer3'],
			freeze = True,
		),
		dict(
			name = 'resnext101_32x8d-skip1',
			source = 'torchvision',
			layers_to_extract = ['layer1', 'layer2', 'layer3'],
			freeze = True,
		),
		dict(
			name = 'resnext101_32x8d-skip2',
			source = 'torchvision',
			layers_to_extract = ['layer2', 'layer3'],
			freeze = True,
		),
	]

	def __init__(self, cfg):
		# backbone = torchvision.models.resnext50_32x4d, freeze=True, layers_to_extract = None):
		super().__init__()
		self.cfg = EasyDict(cfg)

		if self.cfg.source == 'torchvision':
			backbone_cls = getattr(torchvision.models, self.cfg.name.split('-')[0])
			backbone = backbone_cls(pretrained=True)

		else:
			raise NotImplementedError(f'Backbone source {self.cfg.source}')

		# extract layers
		self.layers_to_extract = self.cfg.get('layers_to_extract', self.ResNet_layers_extract_default)
		self.copy_layers_from_pretrained(backbone)
		
		if self.cfg.freeze:
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
	
	@staticmethod
	def eval_feature_list(backbone, names_to_execute, names_to_extract, inputs):
		results = OrderedDict()
		value = inputs
		
		for layer_name in names_to_execute:
			value = getattr(backbone, layer_name)(value)
			
			if layer_name in names_to_extract:
				results[layer_name] = value

		return results

	def forward(self, image, **_):
		return self.eval_feature_list(
			backbone = self.resnet, 
			names_to_execute = self.layers_sequential_names, 
			names_to_extract = self.layers_to_extract,
			inputs = image,
		)
		

@ModsBackbone.register_class()
class FeatureExtractorResNet_Pmap4ch(FeatureExtractorResNet):

	configs = [
		dict(
			name = 'resnext101_32x8d-pmap4Ch',
			source = 'torchvision',
			layers_to_extract = ['relu', 'layer1', 'layer2', 'layer3'],
			freeze = False,
		),
	]
	
	def __init__(self, cfg):
		super().__init__(cfg)
		
		# modify first conv to get 4 channels
		weight_backup = self.resnet.conv1.weight.detach()
		newconv = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		newconv.weight.data[:, :3] = weight_backup
		newconv.weight.data[:, 3] = weight_backup.mean(dim=1)
		self.resnet.conv1 = newconv

	def forward(self, image, perspective_scale_map, **_):
		pmap = perspective_scale_map * (1./400.) - 0.5

		input_4ch = torch.cat([
			image,
			pmap,
		], dim=1)

		return super().forward(input_4ch)

@ModsBackbone.register_class()
class FeatureExtractorResNet_PmapBranch(FeatureExtractorResNet):

	configs = [
		dict(
			name = 'resnext101_32x8d-pmapBranch',
			source = 'torchvision',
			layers_to_extract = ['relu', 'layer1', 'layer2', 'layer3'],
			freeze = True,
		),
	]
	
	def __init__(self, cfg):
		super().__init__(cfg)
		
		# create branch and unfreeze it
		from copy import deepcopy
		self.resnet_pmap_branch = deepcopy(self.resnet)
		for param in self.resnet_pmap_branch.parameters():
			param.requires_grad = False

		# modify first conv to get 1 channel
		weight_backup = self.resnet_pmap_branch.conv1.weight.detach()
		newconv = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		newconv.weight.data[:, 0] = weight_backup.mean(dim=1)
		self.resnet_pmap_branch.conv1 = newconv

		# we concat 2 branches so get 2x number of channels
		self.feat_channels = [2*c for c in self.feat_channels]


	def forward(self, image, perspective_scale_map, **_):
		pmap = perspective_scale_map * (1./400.) - 0.5

		feats_image = self.eval_feature_list(
			backbone = self.resnet, 
			names_to_execute = self.layers_sequential_names, 
			names_to_extract = self.layers_to_extract,
			inputs = image,
		)

		feats_perspective = self.eval_feature_list(
			backbone = self.resnet_pmap_branch, 
			names_to_execute = self.layers_sequential_names, 
			names_to_extract = self.layers_to_extract,
			inputs = pmap,
		)

		result = OrderedDict()
		for name, feat_img in feats_image.items():
			result[name] = torch.cat([
				feat_img,
				feats_perspective[name],
			], dim=1)

		return result


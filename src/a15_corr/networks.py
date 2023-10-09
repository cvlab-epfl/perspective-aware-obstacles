

from functools import partial
import torch, torchvision
from torch import nn

from easydict import EasyDict

from ..common.registry import Registry
from ..common.util_networks import Padder

from .networks_backbone import ModsBackbone
from .networks_features import ModsFeatureProcessors
from .networks_classifiers import ModsClassifiers


class UpBlock(nn.Sequential):
	def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True, b_upsample_act=False, dilated_fraction=0.):

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
			if b_upsample_act:
				modules += [
					nn.SELU(inplace=True),
				]

		super().__init__(*modules)

	def forward(self, x):
		
		for i in range(4):
			x = self[i](x)

		out_for_classifier = x
		out_for_pyramid = x

		if self.__len__() > 4:
			for i in range(4, self.__len__()):
				out_for_pyramid = self[i](out_for_pyramid)
	
		return EasyDict(
			out_for_classifier = out_for_classifier,
			out_for_pyramid = out_for_pyramid,
		)


from math import floor

class UpBlockWithDilation(nn.Module):
	def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True, b_upsample_act=False, dilated_fraction=0.):
		super().__init__()

		mid_channels_dilated = floor(dilated_fraction*middle_channels)
		mid_channels_nondil = middle_channels - mid_channels_dilated

		self.conv1 = nn.Conv2d(in_channels, mid_channels_nondil, kernel_size=3, padding=1)

		if mid_channels_dilated > 0:
			self.conv_dil = nn.Conv2d(in_channels, mid_channels_dilated, kernel_size=3, dilation=2, padding=2)
		else:
			self.conv_dil = None

		self.mid = nn.Sequential(
			nn.SELU(inplace=True),
			nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
		)

		ups = []
		if b_upsample:
			ups += [
				nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
			]
			if b_upsample_act:
				ups += [
					nn.SELU(inplace=True),
				]

		self.ups = nn.Sequential(*ups)

	def forward(self, x):
		
		if self.conv_dil is not None:
			x = torch.cat([
				self.conv1(x),
				self.conv_dil(x),
			], dim=1)
		else:
			x = self.conv1(x)

		x = self.mid(x)

		out_for_classifier = x
		out_for_pyramid = self.ups(x)
	
		return EasyDict(
			out_for_classifier = out_for_classifier,
			out_for_pyramid = out_for_pyramid,
		)


ModsObstacleNet = Registry()

def resize_to_match(x, hw):
	if x.shape[2:4] == hw:
		return x
	else:
		return nn.functional.interpolate(
			x,
			hw,
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)


@ModsObstacleNet.register_class()
class ObstacleNetworkCore(nn.Module):
	"""
	- gets features from a backbone
	- applies feature processors
	- deconvolution pyramid
	- fusion of outputs
	"""

	configs = [
		dict(
			name = 'v1',
			intermediate_channels = [256, 256, 128, 64],
			pyramid_up_operation = 'convTr',
		),
		dict(
			name = 'v2Upinterp',
			intermediate_channels = [256, 256, 128, 64],
			pyramid_up_operation = 'interpolate',
		),
		dict(
			name = 'v3',
			intermediate_channels = [256, 256, 128, 64],
			pyramid_up_operation = 'convTrSelu',
		),
		dict(
			name = 'v4split',
			intermediate_channels = [256, 256, 128, 64],
			pyramid_up_operation = 'convTrSplit',
		),
		dict(
			name = 'v4splitDil025',
			intermediate_channels = [256, 256, 128, 64],
			pyramid_up_operation = 'convTrSplit',
			dilated_feat_fract = 0.25,
		),
	]

	#def __init__(self, num_outputs=2, freeze=True, correlation=True, separate_gen_image=True, perspective=False, skips_start_from=0, backbone=None):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

		if self.cfg.pyramid_up_operation not in ['convTr', 'convTrSelu', 'interpolate', 'convTrSplit']:
			raise NotImplementedError(f'pyramid up {self.cfg.pyramid_up_operation}')

	def build(self, backbone, feat_processor_names, classifier, feat_processor_names_sticky=[]):
		self.backbone = backbone
		self.feat_processor_names = feat_processor_names
		self.feat_processor_names_sticky = set(feat_processor_names_sticky)
		self.feat_processor_sticky_indices = [i for i, n in enumerate(self.feat_processor_names) if n in self.feat_processor_names_sticky]
		self.classifier = classifier

		self.construct_pyramid(
			backbone_channels = self.backbone.feat_channels,
			intermediate_channels = self.cfg.intermediate_channels,
		)


	def construct_pyramid(self, backbone_channels, intermediate_channels):
		prev_chans = 0

		self.feat_processors_layers = nn.ModuleList()
		self.pyramid_layers = nn.ModuleList()
		self.pyramid_out_nchan = []

		up_block_class = UpBlock
		dilated_feat_fract = self.cfg.get('dilated_feat_fract', 0)
		if dilated_feat_fract > 0:
			up_block_class = partial(UpBlockWithDilation, dilated_fraction = dilated_feat_fract)

		for i, nch_bg, nch_inter in zip(range(intermediate_channels.__len__()), reversed(backbone_channels), intermediate_channels):

			nch_proc = 0
			nch_proc_sticky = 0
			list_fp = nn.ModuleList()
			
			for fpn in self.feat_processor_names:
				mod_fp = ModsFeatureProcessors.get(fpn, cache=False)
				num_feats = mod_fp.build(nch_bg, layer_level=i)
				nch_proc += num_feats
				if fpn in self.feat_processor_names_sticky:
					nch_proc_sticky += num_feats
				list_fp.append(mod_fp)
			
			self.feat_processors_layers.append(list_fp)
			mod_pyr = up_block_class(
				prev_chans + nch_proc, nch_inter, nch_inter, 
				b_upsample = self.cfg.pyramid_up_operation.startswith('convTr'),
				b_upsample_act = self.cfg.pyramid_up_operation.startswith('convTrSelu'),
			)
			prev_chans = nch_inter + nch_proc_sticky

			self.pyramid_layers.append(mod_pyr)
			self.pyramid_out_nchan.append(prev_chans)


		self.classifier.build(self.pyramid_out_nchan)


	def forward_feat_proc(self, i, feat_backbone, **extra):
		feats_processed = [
			fp_mod(feat_backbone, **extra)
			for fp_mod in self.feat_processors_layers[i]
		]

		feats_all = feats_processed[0] if feats_processed.__len__() == 1 else torch.cat(feats_processed, 1)
		feats_sticky = [feats_processed[i] for i in self.feat_processor_sticky_indices]

		if feats_sticky.__len__() == 1:
			feats_sticky = feats_sticky[0]
		elif feats_sticky.__len__() > 1:
			feats_sticky = torch.cat(feats_sticky, 1)

		return dict(
			all = feats_all,
			sticky = feats_sticky,
		)


	def forward(self, image, **extra):
		# fix psm batch dimension
		pmap = extra.get('perspective_scale_map')
		if pmap is not None:
			extra['perspective_scale_map'] = pmap[:, None]

		if not self.training:
			image_sh_orig = image.shape[2:4]
			padder = Padder(image.shape, 16)
			image = padder.pad(image)

			for k in extra.keys():
				v = extra[k]
				if isinstance(v, torch.Tensor) and v.shape[2:4] == image_sh_orig:
					extra[k] = padder.pad(v)

		# run backbone
		feats_backbone = list(self.backbone(image, **extra).values())

		#print('backbone shapes', ' | '.join([f'{i}->{tuple(v.shape)}' for i, v, in enumerate(feats_backbone)]))

		# deconv pyramid
		layer_classifications = []
		layer_feats_prev = None

		for i, pyr_mod in enumerate(self.pyramid_layers):
			pyramid_depth = self.pyramid_layers.__len__() - i - 1

			# layer_out = pyr_mod(
			# 	prev_layer = layer_feats[-1] if layer_feats else None,
			# 	feats = self.forward_feat_proc(i, feats_backbone[pyramid_depth]),
			# )

			#print(i, pyramid_depth, '/', self.pyramid_layers.__len__(), 'backbone', tuple(feats_backbone[pyramid_depth].shape))

			new_feats = self.forward_feat_proc(i, feats_backbone[pyramid_depth], **extra)
			new_feats_all = new_feats['all']
			new_feats_sticky = new_feats['sticky']

			if layer_feats_prev is not None:
				f = torch.cat([
					layer_feats_prev,
					new_feats_all
				], 1)
			else:
				f = new_feats_all

			b_upblock_split = 'Split' in self.cfg.pyramid_up_operation

			layer_out = pyr_mod(f)
			layer_out_for_pyramid = layer_out.out_for_pyramid
			layer_out_for_classifier = layer_out.out_for_classifier if b_upblock_split else layer_out_for_pyramid

			if new_feats_sticky.__len__():
				if b_upblock_split:
					layer_out_for_classifier = torch.cat([layer_out_for_classifier, new_feats_sticky], 1)

				new_feats_sticky = resize_to_match(new_feats_sticky, layer_out_for_pyramid.shape[2:4])
				layer_out_for_pyramid = torch.cat([layer_out_for_pyramid, new_feats_sticky], 1)

				if not b_upblock_split:
					layer_out_for_classifier = layer_out_for_pyramid

			if self.cfg.pyramid_up_operation == 'interpolate':
				layer_out_for_pyramid = resize_to_match(layer_out_for_pyramid, tuple(d*2 for d in layer_out_for_pyramid.shape[2:4]))

			layer_classifications.append(self.classifier.process_layer(
				layer_idx = i,
				pyramid_depth = pyramid_depth,
				layer_out = layer_out_for_classifier
			))

			layer_feats_prev = layer_out_for_pyramid
			del layer_out, layer_out_for_classifier, layer_out_for_pyramid

		result = self.classifier.fuse(layer_classifications, **extra)

		# if out size is too small, upsample
		result = resize_to_match(result, image.shape[2:4])

		if not self.training:
			result = padder.unpad(result)

		return result




class SoupParallelBlock(nn.Sequential):
	def __init__(self, in_channels, out_channels):

		# nch_1 = int(in_channels * 0.66 + out_channels * 0.33)
		# nch_2 = int(in_channels * 0.33 + out_channels * 0.66)

		nch_mid = (in_channels + out_channels) // 2

		modules = [
			# nn.Conv2d(in_channels, nch_1, kernel_size=1),
			# nn.SELU(inplace=True),
			nn.Conv2d(in_channels, nch_mid, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(nch_mid, out_channels, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
		]

		super().__init__(*modules)


class SoupFuserBlock(nn.Sequential):
	def __init__(self, in_channels, out_channels):

		nch_2 = in_channels // 2

		modules = [
			nn.Conv2d(in_channels, nch_2, kernel_size=1),
			nn.SELU(inplace=True),
			nn.Conv2d(nch_2, nch_2, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(nch_2, out_channels, kernel_size=1),
		]

		super().__init__(*modules)

ModsSoupCat = Registry()

@ModsSoupCat.register_class()
class SoupCatModule(nn.Module):
	configs = [
		dict(
			name = 'resizeAndCat',
		)
	]

	def __init__(self, cfg):
		super().__init__()

	def forward(self, soup_to_cat, **_):
		out_shape = soup_to_cat[0].shape[2:4]

		print('Soup imgsize changed to first layer', out_shape)

		soup_to_cat = [
			resize_to_match(sc, out_shape)
			for sc in soup_to_cat
		]
		return torch.cat(soup_to_cat, 1)


@ModsSoupCat.register_class()
class SoupCatWeighted(nn.Module):
	defaults = dict(
		bias_init = 2.,
		bias_lr = 0.002,
		conv_lr = 0.002,
	)

	configs = [
		dict(
			name = 'catWeighted1x1Sig',
			**defaults,
		),
		dict(
			name = 'catWeightedZonesSig',
			pmap_process = 'zoneFixed8p',
			**defaults,
		),
	]

	# def named_modules(self, *a, **kw):
	# 	named_mods = list(super().named_modules(*a, **kw))
		
	# 	print('SOUP NAMED MODULES', [k for (k, v) in named_mods])

	# 	return named_mods

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		num_weights = 4

		pmap_process = cfg.get('pmap_process')

		num_ch = 1

		if pmap_process is not None:
			self.pmap_proc = ModsFeatureProcessors.get(pmap_process)
			num_ch = self.pmap_proc.build(1)
		else:
			self.pmap_proc = None

		conv_1x1 = nn.Conv2d(num_ch, num_weights, kernel_size=1, bias=True)

		# initialize
		with torch.no_grad():
			init_bias = self.cfg.get('bias_init')
			if init_bias:
				conv_1x1.bias.data[:] = init_bias
		# 	conv_1x1.weight.data *= 0

		self.weight_mod = nn.Sequential(
			conv_1x1,
			nn.Sigmoid(),
		)

		self.lr_overrides = {
			'weight_mod.0.weight': self.cfg.get('conv_lr'),
			'weight_mod.0.bias': self.cfg.get('bias_lr'),
		}

		print('SOUP CAT PARAMS', [k for (k, v) in self.named_parameters()])


	def calc_weights(self, perspective_scale_map):
		if self.pmap_proc is not None:
			pmap_processed = self.pmap_proc(None, perspective_scale_map=perspective_scale_map)
		else:
			pmap_processed = perspective_scale_map * (1./400.)

		return self.weight_mod(pmap_processed)

	def forward(self, soup_to_cat, perspective_scale_map, **_):
		out_shape = soup_to_cat[0].shape[2:4]

		weights = self.calc_weights(
			resize_to_match(perspective_scale_map, out_shape),
		)

		# print('sh weights', weights.shape, 'soups', [s.shape for s in soup_to_cat])
		soup_to_cat = [
			resize_to_match(sc, out_shape) * weights[:, i:i+1]
			for i, sc in enumerate(soup_to_cat)
		]
		return torch.cat(soup_to_cat, 1)


@ModsObstacleNet.register_class()
class ObstacleNetworkSoup(nn.Module):
	"""
	- gets features from a backbone
	- applies feature processors
	- put all features together
	"""

	feat_nums_default = dict(
		num_intermediate_channels_fraction = 0.25,
		num_intermediate_channels_minmax = (32, 318),
	)

	configs = [
		dict(
			name = 'soup1',
			cat_module = 'resizeAndCat',
			**feat_nums_default,
		),
		dict(
			name = 'wsoup-1x1sig',
			cat_module = 'catWeighted1x1Sig',
			**feat_nums_default,
		),
		dict(
			name = 'wsoup-zonesig',
			cat_module = 'catWeightedZonesSig',
			**feat_nums_default,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, backbone, feat_processor_names, classifier, feat_processor_names_sticky=[], feat_processor_names_global=[]):
		self.backbone = backbone

		self.feat_processor_names = feat_processor_names
		self.feat_processor_names_sticky = set(feat_processor_names_sticky)
		self.feat_processor_sticky_indices = [i for i, n in enumerate(self.feat_processor_names) if n in self.feat_processor_names_sticky]
		self.classifier = classifier

		self.cat_mod = ModsSoupCat.get(self.cfg.cat_module, cache=False)

		self.construct_pyramid(
			backbone_channels = self.backbone.feat_channels,
		)

	def calc_num_channels_from_features(self, num_features):
		nch_out = round(self.cfg.num_intermediate_channels_fraction * num_features)
		nch_min, nch_max = self.cfg.num_intermediate_channels_minmax
		nch_out = min(nch_out, nch_max)
		nch_out = max(nch_out, nch_min)
		return nch_out


	def construct_pyramid(self, backbone_channels):
		self.feat_processors_layers = nn.ModuleList()
		self.parallel_layers = nn.ModuleList()
		self.pyramid_out_nchan = []

		nch_soup = 0

		for i, nch_bg in enumerate(backbone_channels):
			nch_feat = 0
			nch_feat_sticky = 0
			list_fp = nn.ModuleList()

			for fpn in self.feat_processor_names:
				mod_fp = ModsFeatureProcessors.get(fpn, cache=False)
				num_feats = mod_fp.build(nch_bg)
			
				nch_feat += num_feats
				if fpn in self.feat_processor_names_sticky:
					nch_feat_sticky += num_feats
				list_fp.append(mod_fp)
					
			nch_to_fuse = self.calc_num_channels_from_features(nch_feat)
			nch_soup += nch_to_fuse

			self.feat_processors_layers.append(list_fp)		
			self.parallel_layers.append(SoupParallelBlock(nch_feat, nch_to_fuse))

		self.soup_fuser = SoupFuserBlock(nch_soup, 2)


	def forward_feat_proc(self, i, feat_backbone, **extra):
		feats_processed = [
			fp_mod(feat_backbone, **extra)
			for fp_mod in self.feat_processors_layers[i]
		]

		feats_all = feats_processed[0] if feats_processed.__len__() == 1 else torch.cat(feats_processed, 1)
		feats_sticky = [feats_processed[i] for i in self.feat_processor_sticky_indices]

		if feats_sticky.__len__() == 1:
			feats_sticky = feats_sticky[0]
		elif feats_sticky.__len__() > 1:
			feats_sticky = torch.cat(feats_sticky, 1)

		return dict(
			all = feats_all,
			sticky = feats_sticky,
		)


	def forward(self, image, **extra):
		# fix psm batch dimension
		pmap = extra.get('perspective_scale_map')
		if pmap is not None:
			extra['perspective_scale_map'] = pmap[:, None]

		if not self.training:
			image_sh_orig = image.shape[2:4]
			padder = Padder(image.shape, 16)
			image = padder.pad(image)

			for k in extra.keys():
				v = extra[k]
				if isinstance(v, torch.Tensor) and v.shape[2:4] == image_sh_orig:
					extra[k] = padder.pad(v)


		# run backbone
		feats_backbone = list(self.backbone(image).values())

		soup_to_cat = []

		for i, bg_feat in enumerate(feats_backbone):
			feats = self.forward_feat_proc(i, bg_feat, **extra)
			feats = feats['all']
			feats = self.parallel_layers[i](feats)
			soup_to_cat.append(feats)
			# print(f'feats {i} {tuple(feats.shape)}')
		
		# soup_to_cat = [
		# 	resize_to_match(sc, soup_to_cat[-1].shape[2:4])
		# 	for sc in soup_to_cat
		# ]
		# soup = torch.cat(soup_to_cat, 1)
		soup = self.cat_mod(soup_to_cat, **extra)
		del soup_to_cat

		logits = self.soup_fuser(soup)
		del soup
		
		# if out size is too small, upsample
		logits = resize_to_match(logits, image.shape[2:4])

		if not self.training:
			logits = padder.unpad(logits)

		return logits




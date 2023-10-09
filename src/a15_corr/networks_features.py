
from easydict import EasyDict
import torch, torchvision
from torch import nn
import numpy as np

from ..common.registry import Registry

ModsFeatureProcessors = Registry()

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

@ModsFeatureProcessors.register_class()
class FeatPassthrough(nn.Module):
	configs = [
		dict(
			name = 'pass',
		)
	]

	def __init__(self, cfg):
		super().__init__()

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""
		return in_channels

	def forward(self, feats, **_):
		return feats


N_OFFSETS = [
	(1, 0),
	(1, 1),
	(0, 1),
	(-1, 1),
	(-1, 0),
	(-1, -1),
	(0, -1),
	(1, -1),
]

def feat_neighbourhood_corr_operator_slice(feat, dilation=1):
	"""
	@param feat: C x H x W feature map
	@param roi_mask: full resolution roi mask
	"""
	
	_, c_feat, h_feat, w_feat = feat.shape

	#feat_tr = torch.from_numpy(feat)[None]
	feat_tr = feat
	feat_tr_normed = feat_tr / feat_tr.norm(dim=1, keepdim=True)
	feat_tr_padded = torch.nn.functional.pad(
		feat_tr_normed,
		pad=(dilation, dilation, dilation, dilation),
		mode='reflect',
	)
	
	dxy = np.array(N_OFFSETS, dtype=np.int32) * dilation

	corrs = torch.cat([
		torch.sum(feat_tr_normed * feat_tr_padded[:, :, dilation-dx : h_feat + dilation -dx , dilation-dy : w_feat + dilation -dy], dim=1, keepdim=True)
		for (dx, dy) in dxy
	], dim=1)
			
	return corrs

	#corrs_sum = torch.sum(corrs, dim=1, keepdim=True)
	#return torch.cat([corrs, corrs_sum], dim=1)


def idfunc(x):
	return x

@ModsFeatureProcessors.register_class()
class FeatNeighbourCorr(nn.Module):
	configs = [
		dict(
			name = 'neighboursD1',
			dilation = 1,
		),
		dict(
			name = 'neighboursD2',
			dilation = 2,
		),
		dict(
			name = 'neighboursD1Sq',
			dilation = 1,
			absSqrt = True,
		),
		dict(
			name = 'neighboursD2Bn',
			dilation = 2,
			norm = 'batchnorm',
		),
		dict(
			name = 'neighboursD2BnEnc',
			dilation = 2,
			norm = 'batchnorm',
			encoder_num_channels = 16,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""
		normtype = self.cfg.get('norm')
		
		if normtype is None:
			self.norm = idfunc
		elif normtype == 'batchnorm':
			self.norm = nn.BatchNorm2d(8)
		else:
			raise NotImplementedError(f"Normtype {normtype}")
		
		self.encoder = idfunc
		nch_enc = self.cfg.get('encoder_num_channels')
		if nch_enc is not None:
			self.encoder = nn.Sequential(
				nn.Conv2d(8, nch_enc, kernel_size=1),
				nn.SELU(),
				nn.Conv2d(nch_enc, nch_enc, kernel_size=1),
			)
			return nch_enc
		else:
			return 8

	def forward(self, feats, **_):
		corrs = feat_neighbourhood_corr_operator_slice(feats, dilation = self.cfg.dilation)
		
		if self.cfg.get('absSqrt', False):
			corrs = torch.sqrt(torch.abs(corrs))

		corrs = self.norm(corrs)
		corrs = self.encoder(corrs)

		return corrs

	def __repr__(self):
		return f'FeatNeighbourCorr[{self.cfg.name}]'


from kornia.filters import BoxBlur


@ModsFeatureProcessors.register_class()
class FeatNeighbourLearnableCorr(nn.Module):
	configs = [
		dict(
			name = 'neighboursLcor',
			box_size = 1,
			nch_out_fraction = 0.25,
		),
		dict(
			name = 'neighboursLcorBox5',
			box_size = 5,
			nch_out_fraction = 0.25,
		),
	]


	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""

		nch_out = self.cfg.get('nch_out')
		nch_out_fraction = self.cfg.get('nch_out_fraction')

		if nch_out is None:
			nch_out = round(nch_out_fraction * in_channels)

		elif nch_out_fraction is not None:
			raise NotImplementedError(f"Conflicting nch_out {nch_out} and nch_out_fraction {nch_out_fraction}")

		self.nch_in = in_channels
		self.nch_out = nch_out
		if self.cfg.box_size > 1:
			self.box = BoxBlur((self.cfg.box_size, self.cfg.box_size))

		self.encoder = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.Conv2d(in_channels, nch_out, kernel_size=1),
			nn.SELU(),
			nn.Conv2d(nch_out, nch_out, kernel_size=1),
		)

		return nch_out

	def forward(self, feats, **_):

		if self.cfg.box_size > 1:
			feats_corr = feats * self.box(feats)
		else:
			feats_corr = feats.pow(2)
		
		return self.encoder(feats_corr)


	def __repr__(self):
		return f'FeatNeighbourChannelVariance[{self.cfg.name}]({self.nch_in}->{self.nch_out})'



@ModsFeatureProcessors.register_class()
class FeatNeighbourPoolCorr(nn.Module):
	configs = [
		dict(
			name = 'poolcor',
			nch_out_fraction = 0.125,
			product = 'dot',
		),
		dict(
			name = 'poolcorPrc',
			nch_out_fraction = 0.125,
			precoder = '1x1bn',
			product = 'dot',
		),
		dict(
			name = 'poolcorPrcCat',
			nch_out_fraction = 0.125,
			precoder = '1x1bn',
			product = 'cat',
		),
		# dict(
		# 	name = 'poolcorPrc',
		# 	nch_out_fraction = 0.125,
		# 	precoder = '1x1bn',
		# 	mix = 'cat',
		# ),
	]


	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)


	def make_scaler(self, ks, in_channels, out_channels):
		if ks > 1:
			pool = [nn.AvgPool2d(ks, count_include_pad=False)]
		else:
			pool = []

		cfg_enc = self.cfg.get('precoder')
		if cfg_enc is None:
			enc = []
		elif cfg_enc == '1x1bn':
			enc = [
				nn.Conv2d(in_channels, out_channels, 1),
				nn.SELU(True),
				nn.BatchNorm2d(out_channels),
			]
		else:
			raise NotImplementedError(f"Precoder {cfg_enc}")

		return nn.Sequential(*(pool + enc))

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""

		nch_out = self.cfg.get('nch_out')
		nch_out_fraction = self.cfg.get('nch_out_fraction')

		if nch_out is None:
			nch_out = round(nch_out_fraction * in_channels)
		elif nch_out_fraction is not None:
			raise NotImplementedError(f"Conflicting nch_out {nch_out} and nch_out_fraction {nch_out_fraction}")

		if in_channels > 500:
			scales = (1, 2, 4, 8)
		else:
			scales = (1, 4, 8, 16, 24)

		num_scales = scales.__len__()

		self.scales = nn.ModuleList([
			self.make_scaler(ks, in_channels, nch_out)
			for ks in scales
		])

		#nch_inter = (nch_out // num_scales) * num_scales

		#cfg_mix = self.cfg.get('mix')
		if self.cfg.product == 'dot':
			nch_inter = num_scales - 1
		elif self.cfg.product == 'cat':
			nch_inter = nch_out * num_scales

		self.encoder = nn.Sequential(
			nn.BatchNorm2d(nch_inter),
			nn.Conv2d(nch_inter, nch_out, kernel_size=1),
			nn.ReLU(True),
			#nn.Conv2d(nch_inter, nch_out, kernel_size=1),
		)

		return nch_out

	def scc(self, feats):
		resize_to_match(sc(feats), feats.shape[2:4])


	def forward(self, feats, **_):
		if self.cfg.product == 'dot':
			sc1 = self.scales[0](feats)

			return self.encoder(torch.cat([
				torch.sum(sc1 * resize_to_match(sc(feats), sc1.shape[2:4]), dim=1, keepdim=True)
				for sc in self.scales[1:]
			], dim=1))

		elif self.cfg.product == 'cat':
			return self.encoder(torch.cat([
				resize_to_match(sc(feats), feats.shape[2:4])
				for sc in self.scales
			], dim=1))



# self.pool1 = nn.AdaptiveAvgPool2d(1)
# self.pool2 = nn.AdaptiveAvgPool2d(2)
# self.pool3 = nn.AdaptiveAvgPool2d(3)
# self.pool4 = nn.AdaptiveAvgPool2d(6)

# out_channels = int(in_channels/4)
# self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
# 							norm_layer(out_channels),
# 							nn.ReLU(True))
# self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
# 							norm_layer(out_channels),
# 							nn.ReLU(True))
# self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
# 							norm_layer(out_channels),
# 							nn.ReLU(True))
# self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
# 							norm_layer(out_channels),
# 							nn.ReLU(True))
# # bilinear interpolate options
# self._up_kwargs = up_kwargs

# def forward(self, x):
# _, _, h, w = x.size()
# feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
# feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
# feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
# feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
# return torch.cat((x, feat1, feat2, feat3, feat4), 1)




@ModsFeatureProcessors.register_class()
class FeatNeighbourChannelVariance(nn.Module):
	configs = [
		dict(
			name = 'neighboursChanVar05',
			dilation = 1,
			nch_out_fraction = 0.5,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""

		nch_out = self.cfg.get('nch_out')
		nch_out_fraction = self.cfg.get('nch_out_fraction')

		if nch_out is None:
			nch_out = round(nch_out_fraction * in_channels)

		elif nch_out_fraction is not None:
			raise NotImplementedError(f"Conflicting nch_out {nch_out} and nch_out_fraction {nch_out_fraction}")

		self.nch_in = in_channels
		self.nch_out = nch_out
		self.box = BoxBlur((3, 3))
		self.covariance_comp_weight = nn.Conv2d(in_channels, nch_out, (1, 1), bias=False)
		
		print(f'Cov created inch {in_channels} outch {nch_out}')

		if self.cfg.dilation != 1:
			raise NotImplementedError(f"Box filter has dilation 1 atm")

		return nch_out

	def forward(self, feats, **_):
		# 1 subtract neighbourhood average
		# 2 conv1d
		# 3 sum neighbourhood

		feat_zero_avg = feats - self.box(feats)

		# (W @ x_avg)^2
		Wxsq = self.covariance_comp_weight(feat_zero_avg).pow(2)

		Wxsq_sum_sqrt = torch.sqrt(self.box(Wxsq))
				
		return Wxsq_sum_sqrt


	def __repr__(self):
		return f'FeatNeighbourChannelVariance[{self.cfg.name}]({self.nch_in}->{self.nch_out})'






@ModsFeatureProcessors.register_class()
class FeatPerswaveFlatfixed(nn.Module):
	configs = [
		dict(
			name = 'perswaveFlatfixed',
			num_ft = 16,
			wavenumber_range = [1000, 6000],
			modulate = None,
		),
		dict(
			name = 'perswaveFlatfixedModpsm',
			num_ft = 16,
			wavenumber_range = [1000, 6000],
			modulate = 'psm',
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

		wn_from, wn_to = self.cfg.wavenumber_range
		self.wavenumbers = torch.linspace(wn_from, wn_to, self.cfg.num_ft)[None, :, None, None]


	def build(self, in_channels, layer_level=None):
		"""
		@return: num output channels
		"""
		return self.cfg.num_ft

	def forward(self, feats, perspective_scale_map, **_):

		h, w = feats.shape[2:4]

		#perspective_scale_map = perspective_scale_map[:, None] # add channel dimension

		#print('Feats', feats.shape, 'psm', perspective_scale_map.shape, perspective_scale_map.device)

		psm_resized = nn.functional.interpolate(
			perspective_scale_map,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)


		psm_inv = 1./torch.clamp(psm_resized, min=1.)
		self.wavenumbers = self.wavenumbers.to(perspective_scale_map.device)
		waves = torch.sin(psm_inv * self.wavenumbers)

		if self.cfg.modulate == 'psm':
			waves *= psm_resized
		elif self.cfg.modulate is not None:
			raise NotImplementedError(f'Modulate: {self.cfg.modulate}')

		#print('Waves shape', waves.shape, 'for feat', feats.shape)

		return waves

	def __repr__(self):
		return f'FeatPerswaveFlatfixed[{self.cfg.name}]'


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
class FeatPosEncoding(nn.Module):
	configs = [
		dict(
			name = 'YXdirect',
			modulate = 1/400,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level):
		return 2

	def forward(self, feats, pos_encoding_Y, pos_encoding_X, **_):

		h, w = feats.shape[2:4]

		penc = torch.stack([pos_encoding_Y, pos_encoding_X], dim=1)
		penc_resized = nn.functional.interpolate(
			penc,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)

		return penc_resized * self.cfg.modulate

	def __repr__(self):
		return f'FeatPersDirect[{self.cfg.name}]'


@ModsFeatureProcessors.register_class()
class Attentropy(nn.Module):
	configs = [
		dict(
			name = 'attentropy_SETR1',
			modulate = 1/20,
			num_channels = 24,
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level):
		return self.cfg.num_channels

	def forward(self, feats, attentropy, **_):
	# def forward(self, feats, **k):
		# print(k.keys())

		# print('Attentropy feature runs, ', attentropy.shape)

		h, w = feats.shape[2:4]
		# print('attentropy incoming size', attentropy.shape)
		# there is a batch dim, so we add 1 to all dims
		penc = attentropy.permute(0, 3, 1, 2)
		# print('attentropy out size', penc.shape)

		penc_resized = nn.functional.interpolate(
			penc,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)

		return penc_resized * self.cfg.modulate

	def __repr__(self):
		return f'Attentropy[{self.cfg.name}]'


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
		h, w = perspective_scale_map.shape[-2:]
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


@ModsFeatureProcessors.register_class()
class FeatNormals(nn.Module):
	configs = [
		dict(
			name = 'normals1',
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)

	def build(self, in_channels, layer_level):
		"""
		@return: num output channels
		"""
		return 3

	def forward(self, feats, normals, **_):
		h, w = feats.shape[2:4]

		#perspective_scale_map = perspective_scale_map[:, None] # add channel dimension

		return nn.functional.interpolate(
			normals,
			size = (h, w),			
			mode = 'bilinear',
			align_corners = False, # to get rid of warning
		)

	def __repr__(self):
		return f'FeatNormals[{self.cfg.name}]'

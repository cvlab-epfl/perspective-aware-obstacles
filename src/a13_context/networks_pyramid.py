from collections import OrderedDict
from functools import partial

from easydict import EasyDict

import torch
from torch import nn
from kornia.filters.kernels import get_gaussian_kernel1d
from kornia.filters import filter2D, Laplacian
from kornia.geometry.transform import pyrdown
from kornia.utils import image_to_tensor

from ..common.util_networks import Padder
from ..a12_inpainting.vis_imgproc import image_montage_same_shape


# class LapImagePyramid_Blur(LapImagePyramidBase):
# 	num_levels = 6

# 	def __init__(self, b_normalize_brightness=False):
# 		super().__init__()

# 		self.blur_kernel = get_gaussian_kernel1d(5, 1.)

# 	def blur(self, value):
# 		value = filter2D(value, self.blur_kernel[None, :, None])
# 		value = filter2D(value, self.blur_kernel[None, None, :])
# 		return value

# 	def bdown(self, image, b_downsample=True):
# 		image_blur = self.blur(image)
# 		lap = torch.sum(torch.abs(image - image_blur), dim=1, keepdim=True)
			
# 		if self.b_normalize_brightness:
# 			blur_abs = torch.sum(torch.abs(image_blur), dim=1, keepdim=True)
# 			lap /= blur_abs

# 		return (image_half, laplacian)




class LapImagePyramid(nn.Module):
	num_levels = 6

	def __init__(self, b_normalize_brightness=False, type="blur"):
		super().__init__()
		self.b_normalize_brightness = b_normalize_brightness	
		self.type = type

		if self.type == 'blur':
			self.blur_kernel = get_gaussian_kernel1d(5, 1.5)
		elif self.type == 'kernel':
			self.laplacian = Laplacian(5)
		else:
			raise NotImplementedError(f'Lap Image Pyramid with type {self.type}')

	def blur(self, value):
		value = filter2D(value, self.blur_kernel[None, :, None])
		value = filter2D(value, self.blur_kernel[None, None, :])
		return value

	def pyramid_down_blur(self, image, b_downsample=True):
		image_blur = self.blur(image)
		lap = torch.sum(torch.abs(image - image_blur), dim=1, keepdim=True)
			
		if self.b_normalize_brightness:
			blur_abs = torch.sum(torch.abs(image_blur), dim=1, keepdim=True)
			lap /= blur_abs

		if b_downsample:
			image_half = image_blur[:, :, ::2, ::2]
		else:
			image_half = None

		return (image_half, lap)

	def pyramid_down_kernel(self, image, b_downsample=True):
		# lap = self.laplacian(image)
		lap = torch.sum(torch.abs(self.laplacian(image)), dim=1, keepdim=True)


		if self.b_normalize_brightness:
			lap /= torch.sum(torch.abs(image_blur), dim=1, keepdim=True)

		if b_downsample:
			image_half = pyrdown(image)
		else:
			image_half = None

		return (image_half, lap)

	def forward(self, image):	
		padder = Padder(image.shape, 1 << self.num_levels-1)
		image = padder.pad(image)
		
		laps = []
		laps_full = []
	
		laps_full_sum = None
		# b, c, h, w = image.shape
		# laps_full_sum = torch.zeros((b, 1, h, w))
		
		inv_scale = 1
		
		downfunc = {
			'blur': self.pyramid_down_blur,
			'kernel': self.pyramid_down_kernel,
		}[self.type]

		for level in range(self.num_levels):

			image_half, lap = downfunc(image, b_downsample = (level != self.num_levels-1))

			laps.append(lap)
			
			if inv_scale > 1:
				lp_full = torch.repeat_interleave(torch.repeat_interleave(lap, inv_scale, dim=2), inv_scale, dim=3)
			else:
				lp_full = lap
			laps_full.append(padder.unpad(lp_full))
			
			if laps_full_sum is None:
				laps_full_sum = torch.abs(lp_full)
			else:
				laps_full_sum += torch.abs(lp_full)

			image = image_half
			inv_scale *= 2
		
		return EasyDict(
			laps = laps,
			laps_full = laps_full,
			laps_full_sum = padder.unpad(laps_full_sum),
		)
	

class LapPyramidBasic(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.pyramid = LapImagePyramid(
			type = cfg.get('pyr_laplacian_type', 'blur'),
			b_normalize_brightness=cfg.get('pyr_normalize_brightness', False),
		)
		
	def forward(self, image, **_):
		res = self.pyramid(image)
		return res.laps_full_sum

class LapPyramidTMix(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.pyramid = LapImagePyramid(b_normalize_brightness=cfg.get('pyr_normalize_brightness', False))
		self.conv_mix = nn.Conv2d(self.pyramid.num_levels, 2, (1, 1), bias=True)

	def forward(self, image, **_):
		res = self.pyramid(image)
		pyr_stacked = torch.cat(res.laps_full, dim=1)
		return self.conv_mix(pyr_stacked)





class LapPyramidTMix2(nn.Module):

	# class Conv1Block(nn.Module):
	# 	def __init__(self, nci, nco):
	# 		super().__init__()
	# 		self.nco = nco
	# 		self.nci = nci

	# 		if nco > nci:
	# 			self.conv = nn.Conv2d(nci, nco-nci, (1, 1), bias=True)
	# 		else:
	# 			self.conv = nn.Conv2d(nci, nco, (1, 1), bias=True)
	# 		self.activation = nn.SELU()

	# 	def forward(self, v):
	# 		if self.nco > self.nci:
	# 			res = torch.cat([
	# 				v,
	# 				self.activation(self.conv(v)),
	# 			], dim=1)
	# 		else:
	# 			res = self.activation(self.conv(v))

	# 		return res

	class Conv1Block(nn.Module):
		def __init__(self, nci, nco, b_residual=True):
			super().__init__()
			self.b_residual = b_residual
	
			self.conv = nn.Conv2d(nci, nco, (1, 1), bias=True)
			self.activation = nn.SELU()

		def forward(self, v):

			convres = self.conv(v)

			if self.b_residual:
				convres[:, :v.shape[1]] += v

			return self.activation(convres)


	def __init__(self, cfg):
		super().__init__()
		self.pyramid = LapImagePyramid(b_normalize_brightness=cfg.get('pyr_normalize_brightness', False))
		self.classifier = nn.Sequential(
			self.Conv1Block(self.pyramid.num_levels-1, 8, b_residual=False),
			self.Conv1Block(8, 16),

			# these blocks should have a few convs, bypassed by residual, followed by activation
			self.Conv1Block(16, 24),
			self.Conv1Block(24, 32),
			self.Conv1Block(32, 48),
			self.Conv1Block(48, 64),
			self.Conv1Block(64, 64),
			nn.Conv2d(64, 2, (1, 1), bias=True),
		)

	def forward(self, image, **_):
		res = self.pyramid(image)
		pyr_stacked = torch.cat(res.laps_full[1:], dim=1)
		return self.classifier(pyr_stacked)


class LapPyramidFeat(nn.Module):

	class ImageFeat(nn.Sequential):
		def __init__(self, num_chan_in, num_chan_mid, num_chan_out, b_laplacian=False):
			super().__init__(
				nn.Conv2d(num_chan_in, num_chan_mid, (3, 3), stride=2, padding=1, bias=True),
				nn.SELU(),
				nn.Conv2d(num_chan_mid, num_chan_out, (3, 3), stride=2, padding=1, bias=True),
				nn.SELU(),
			)

			self.b_laplacian = b_laplacian

			if self.b_laplacian:
				self.laplacian = Laplacian(5)


		def forward(self, v):
			res = super().forward(v)

			if self.b_laplacian:
				res = self.laplacian(res)

			return res


	class Conv1BlockResidual(nn.Module):
		def __init__(self, nc):
			super().__init__()
			self.conv = nn.Conv2d(nc, nc, (1, 1), bias=True)
			self.activation = nn.SELU()

		def forward(self, v):
			return self.activation(self.conv(v) + v)


	def __init__(self, cfg):
		super().__init__()

		self.b_laplacian = cfg.get('pyr_use_laplacian', True)

		self.feat1 = self.ImageFeat(3, 8, 16, b_laplacian=self.b_laplacian)

		self.feat2 = self.ImageFeat(3, 8, 16, b_laplacian=self.b_laplacian)

		self.feat3 = self.ImageFeat(3, 8, 16, b_laplacian=self.b_laplacian)

		self.classifier = nn.Sequential(
			self.Conv1BlockResidual(48),
			self.Conv1BlockResidual(48),
			nn.Conv2d(48, 2, (1, 1), bias=True),
		)

	def forward(self, image, **_):
		f1 = self.feat1(image)

		im2 = pyrdown(image)
		f2 = self.feat2(im2)

		im3 = pyrdown(im2)
		f3 = self.feat3(im3)

		f_all = torch.cat([
			f1,
			nn.functional.interpolate(f2, f1.shape[2:], mode='bilinear'),
			nn.functional.interpolate(f3, f1.shape[2:], mode='bilinear'),
		], dim=1)


		cl = self.classifier(f_all)
		del f1, f2, f3, f_all

		return nn.functional.interpolate(cl, image.shape[2:], mode='bilinear')





def image_to_tr(image_np):
	img_tr = image_to_tensor(image_np)
	img_tr = img_tr.float()
	img_tr *= 1./255.
	return img_tr
		

# from ..paths import DIR_DATA
# lap_pyr = LapPyramid()
# DIR_DEMO = DIR_DATA / '1306_LapPyramid'

def demo1(fr, b_normalize_brightness=True):
	img = fr.image[32:, 32:]
	image_tr = image_to_tr(img)[None]

	
	lap_pyr.b_normalize_brightness = b_normalize_brightness
	pyr_res = lap_pyr(image_tr)
	
	lap_demo = [img, pyr_res.laps_full_sum[0, 0].numpy()]
	
	show(lap_demo)
	
	lap_scales = [lp[0, 0].numpy() for lp in pyr_res.laps_full]
	
	show(lap_scales)
	
	demo_img = image_montage_same_shape(
		lap_demo + [np.zeros_like(img)] + lap_scales, 
		num_cols=3, downsample=2, 
		border=4, border_color=(255, 255, 255), 
		captions = ['', 'sum', '', '1/1', '1/2', '1/4', '1/8', '1/16', '1/32'],			
	)
	
# 	show(demo_img)
	
	var = 'BriNorm' if b_normalize_brightness else 'NoNorm'
	
	imwrite(DIR_DEMO / 'Simple' / f'{fr.fid}_LapPyr{var}.webp', demo_img)
	
	#show(pyr_res.laps_full[0][0, 0].numpy())
	
# conv= nn.Conv2d(3, 1, (5, 1), bias=True, padding=(1, 0))
# s1 = conv.weight.shape
# s2 = get_gaussian_kernel1d(5, 1.).shape
#print(s1, s2)


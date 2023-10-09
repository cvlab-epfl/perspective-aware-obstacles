
import numpy as np
import torch
from torch import nn
from easydict import EasyDict
from ..common.registry import Registry

from ..a12_inpainting.vis_imgproc import image_montage_same_shape
from ..common.jupyter_show_image import adapt_img_data

ModsClassifiers = Registry()

@ModsClassifiers.register_class()
class ClassifierLastLayer(nn.Module):
	configs = [
		dict(
			name = 'last',
		)
	]

	def __init__(self, cfg):
		super().__init__()

	def build(self, layer_sizes):
		self.final = nn.Conv2d(
			layer_sizes[-1], 
			2, 
			kernel_size=1,
		)	

	def process_layer(self, pyramid_depth, layer_idx, layer_out):
		if pyramid_depth == 0:
			return self.final(layer_out)
		else:
			return None

	def fuse(self, layer_classifications, **extra):
		return layer_classifications[-1]


@ModsClassifiers.register_class()
class ClassifierSum(nn.Module):
	configs = [
		dict(
			name = 'sumSimple',
			activation = False,
		),
		dict(
			name = 'sumAct2',
			activation = '1-selu',
		)
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)
		self.visualizer_callback = None

	def build_classifier_for_level(self, layer_size):
		activation = self.cfg.activation

		if activation == False:
			return nn.Conv2d(layer_size, 2, kernel_size=1)

		elif activation == '1-selu':
			return nn.Sequential(
				nn.Conv2d(layer_size, layer_size, kernel_size=1),
				nn.SELU(inplace=True),	
				nn.Conv2d(layer_size, 2, kernel_size=1),	
			)
		else:
			raise NotImplementedError(f"Classifier activation {activation}")

	def build(self, layer_sizes):
		self.feats_to_logits = nn.ModuleList([
			self.build_classifier_for_level(layer_size)
			for layer_size in layer_sizes 
		])

	def process_layer(self, pyramid_depth, layer_idx, layer_out):
		return self.feats_to_logits[layer_idx](layer_out)

	def fuse(self, layer_classifications, **extra):
		
		logit_sum = layer_classifications[0]

		for layer_cls in layer_classifications[1:]:

			logit_sum = nn.functional.interpolate(
				logit_sum,
				size = layer_cls.shape[2:4],			
				mode = 'bilinear',
				align_corners = False, # to get rid of warning
			) + layer_cls

		if self.visualizer_callback is not None:
			self.visualizer(layer_classifications, logit_sum, **extra)

		return logit_sum


	def visualizer(self, layer_classifications, logit_sum, **_):
		out_size = logit_sum.shape[2:4]

		layer_classifications = [
			nn.functional.interpolate(lc, size=out_size, mode='bilinear', align_corners=False)
			for lc in layer_classifications
		]

		# concat along height axis
		logits_raw = torch.cat(layer_classifications, dim=2)[:, ::2, ::2]
		logits_raw = logits_raw[0].cpu().numpy()

		logits_raw_neg = logits_raw[0]
		logits_raw_pos = logits_raw[1]
		logits_together = np.concatenate([logits_raw_neg, logits_raw_pos], axis=1)

		logits_together = adapt_img_data(logits_together)

		sffunc = torch.nn.Softmax2d()

		softmaxes = [
			sffunc(lc[:, :, ::2, ::2]) for lc in layer_classifications
		]

		torch.cat(softmaxes, dim=2)
		
@ModsClassifiers.register_class()
class ClassifierFuser(nn.Module):
	configs = [
		dict(
			name = 'fuserSimple',
		),
	]

	def __init__(self, cfg):
		super().__init__()
		self.cfg = EasyDict(cfg)
		self.visualizer_callback = None


	def build_classifier_for_level(self, layer_size):
		return nn.Conv2d(layer_size, 2, kernel_size=1)

	def build(self, layer_sizes):
		self.feats_to_logits = nn.ModuleList([
			self.build_classifier_for_level(layer_size)
			for layer_size in layer_sizes 
		])

		ls = layer_sizes.__len__() * 2 + 1

		self.final = nn.Sequential(
			nn.Conv2d(ls, ls-1, kernel_size=1),
			nn.SELU(inplace=True),	
			nn.Conv2d(ls-1, ls-1, kernel_size=1),
			nn.SELU(inplace=True),	
			nn.Conv2d(ls-1, 2, kernel_size=1),	
		)

	def process_layer(self, pyramid_depth, layer_idx, layer_out):
		return self.feats_to_logits[layer_idx](layer_out)

	def fuse(self, layer_classifications, perspective_scale_map, **extra):
		
		out_sz = layer_classifications[-1].shape[2:4]

		channels = [
			nn.functional.interpolate(
				lg,
				size = out_sz,			
				mode = 'bilinear',
				align_corners = False, # to get rid of warning
			)
			for lg in layer_classifications[:-1]
		] + [layer_classifications[-1]] + [
			nn.functional.interpolate(
				perspective_scale_map[:, None],  # add channel dimension
				size = out_sz,			
				mode = 'bilinear',
				align_corners = False, # to get rid of warning
			)
		]
		channels = torch.cat(channels, dim=1)
		
		return self.final(channels)

		# if self.visualizer_callback is not None:
		# 	self.visualizer(layer_classifications, logit_sum, **extra)






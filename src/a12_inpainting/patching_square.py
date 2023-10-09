
from .demo_case_selection import DIR_INP_DEMO_CASES, DIR_INP_DEMO_CASES_IN, demo_case_selection

from pathlib import Path
from functools import lru_cache
from operator import itemgetter
import multiprocessing
import json

import numpy as np
import cv2 as cv
from easydict import EasyDict
from tqdm import tqdm

from .vis_imgproc import image_montage_same_shape

from ..common.jupyter_show_image import show
from ..datasets.dataset import imread, imwrite


@lru_cache(maxsize=1)
def load_deepfill_inpainter():
	import sys
	inp_source_dir = str((Path(__file__).parent / '../../inpainter/generative_inpainting').resolve())
	if not inp_source_dir in sys.path:
		sys.path.append(inp_source_dir)

	from inferent import InpaintInferent

	inpinf = InpaintInferent()
	# inpinf.load_nn(input_size_wh=(imp.context_size, imp.context_size), batch_size=2)
	inpinf.load_nn(input_size_wh=(256, 256), batch_size=1)

	return inpinf

from kornia.utils import tensor_to_image


"""
https://github.com/open-mmlab/mmagic/issues/1945
"""
class InpainterMmagic:
	
	CHECKPOINT_PATHS = {
		"deepfillv2_8xb2_places-256x256": "https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth",
	}

	batch_size = 32
	
	def __init__(self, variant):
		from mmagic.apis import MMagicInferencer

		if variant in self.CHECKPOINT_PATHS:    
			self.inferencer = MMagicInferencer(
				model_name = variant.split('_')[0],
				model_config = variant,
				model_ckpt = self.CHECKPOINT_PATHS[variant],
			)
		else:
			self.inferencer = MMagicInferencer(variant)
	
	def infer_batch(self, images_bhwc, masks_bhw):
		results, _ = self.inferencer.infer(
			img=images_bhwc, 
			# invert mask to satisfy mmagic convention
			mask=[np.logical_not(mask[:, :, None]) for mask in masks_bhw], 
			is_batch=True,
		)
		return [tensor_to_image(r['infer_results'].output.fake_img) for r in results]
		
	
	def __call__(self, images_hwc_list, masks_hw_list):
		batch_size = self.batch_size
		num_frames = images_hwc_list.__len__()
		outputs_list = []
	
		for batch_start in range(0, num_frames, batch_size):
	
			num_orig_frames_in_batch = min(num_frames - batch_start, batch_size)
			batch_end = batch_start + num_orig_frames_in_batch
	
			# batch_img = images_hwc_list[batch_start:batch_end]
			# batch_mask = masks_hw_list[batch_start:batch_end]
			# resize
			# batch_img, batch_mask, orig_sizes_or_null = self.resize_batch(batch_img, batch_mask, self.input_size_wh)
			# orig_sized_or_null = null
			
			batch_out_list = self.infer_batch(
				images_bhwc = images_hwc_list[batch_start:batch_end],
				masks_bhw = masks_hw_list[batch_start:batch_end],
			)
		
			outputs_list += batch_out_list
	
		return outputs_list


@lru_cache(maxsize=4)
def fusion_weights_square_patch(size_xy):
	w, h = size_xy

	gx, gy = [
		1. - np.abs(np.linspace(-1, 1, d))
		for d in [w, h]
	]

	w = np.minimum(gy[:, None], gx[None, :])

	return w.astype(np.float32)


def try_construct_fusion_weight():
	from matplotlib import pyplot as plt
	
	g = 1 - np.abs(np.linspace(-1, 1, 400))
	w = np.minimum(g[None, :], g[:, None])
	
	plt.plot(w[100, :])
	plt.plot(w[200, :])
	
	show(w)

def try_construct_fusion_weight_2():
	from matplotlib import pyplot as plt
	
	w = fusion_weights_square_patch(size_xy = (400, 200))
	
	plt.plot(w[50, :])
	plt.plot(w[100, :])
	plt.plot(w[:, 150])

	print(w.shape, w.dtype)

	show(w)


class ImpatcherSlidingSquares(EasyDict):
	"""
	Divides the specified area of an image into patches, which are then inpainted, and fused to form the reconstructed image.

	This variant uses square patches with square context areas.
	They are arranged in a regular grid with overlap.

	The fusion is weighted by the following factor:
		1 - max(|x - x_center|, |y - y_center|) / (0.5*patch_side)

	"""

	@staticmethod
	def boxes_to_bounds(box_tls, box_side, bounds_tl, bounds_br):
		box_tls, bounds_tl, bounds_br = (np.array(x, dtype=np.int32) for x in (box_tls, bounds_tl, bounds_br))
		
		#print(box_tls, bounds_tl, bounds_br)
		
		box_centers = box_tls + box_side // 2
		box_centers_in_bounds = np.clip(box_centers, a_min = bounds_tl + box_side//2, a_max=bounds_br - box_side//2 - 1)
		box_tls_in_bounds = box_centers_in_bounds - box_side // 2
		return box_tls_in_bounds
	
	@staticmethod
	def draw_patching(canvas, patch_grid, context_grid, patch_size, context_size, colors=None, **_):
		num_patches = patch_grid.__len__()
		if colors is None:
			colors = np.random.uniform(size = (num_patches, 3), low=128, high=255).astype(dtype=np.uint8)	
		else:
			colors = np.array(colors)

		cv_pt = lambda p: tuple(map(int, p))
		
		
		for i, p_tl, ctx_tl in zip(range(num_patches), patch_grid, context_grid):
			# canvas = cv.rectangle(
			# 	canvas, 
			# 	pt1 = tuple(p_tl), 
			# 	pt2 = tuple(p_tl + patch_size), 
			# 	color = tuple(colors[i]),
			# 	thickness = 2,
			# )

			crop = slice(p_tl[1], p_tl[1] + patch_size), slice(p_tl[0], p_tl[0] + patch_size)
			canvas[crop] = canvas[crop].astype(np.float32) * 0.5 + colors[i].astype(np.float32)[None, None] * 0.5

			canvas = cv.rectangle(
				canvas, 
				cv_pt(ctx_tl), 
				cv_pt(ctx_tl + context_size), 
				cv_pt(colors[i]),
				thickness = 2,
			)
			
		return canvas
	
	@classmethod
	def divide_area(cls, patch_size, context_size, patch_overlap, area_tl_xy, area_size_xy, img_size_xy, context_restriction = False, **_):
		area_tl_xy, area_size_xy, img_size_xy = (np.array(x, dtype=np.int32) for x in (area_tl_xy, area_size_xy, img_size_xy))

		if context_restriction:
			# When context restriction is on, the context box is confined to the selected area
			# If the area is smaller than the box, the box gets pushed up above the area.
			# This is undesirable, as usually we have the top-halves of objects above the box. 
			# Their visibility defeats the context restriction purpose.
			# So we extend the area down to contain at least the context box size
			area_size_xy  = np.maximum(area_size_xy, context_size)


		# how many patches fit
		# patched_area_width = num_patches * patch_size - patch_overlap * patch_size * (num_patches - 1)
		# patched_area_width = patch_size * (num_patches - patch_overlap * (num_patches - 1))
		# patched_area_width = patch_size * (num_patches - patch_overlap * num_patches + patch_overlap)
		# patched_area_width = patch_size * (num_patches * (1 - patch_overlap) + patch_overlap)
		# num_patches * (1 - patch_overlap) + patch_overlap = patched_area_width / patch_size
		# num_patches * (1 - patch_overlap)  = patched_area_width / patch_size - patch_overlap
		# num_patches = (patched_area_width / patch_size - patch_overlap) / (1 - patch_overlap)
		

		num_patches = (area_size_xy / patch_size - patch_overlap) / (1 - patch_overlap)	
		num_patches = np.round(np.ceil(num_patches)).astype(np.int32)
		# ensure there is at least 1 row/column
		num_patches = np.maximum(num_patches, [1, 1])

		
		area_size_xy_adjusted = np.ceil(patch_size * (num_patches * (1-patch_overlap) + patch_overlap)).astype(np.int32)
				
		patch_grid = np.mgrid[0:num_patches[0], 0:num_patches[1]].reshape(2, -1).transpose([1, 0])
		patch_grid = patch_grid.astype(np.float32) * (patch_size * (1-patch_overlap))
		patch_grid = np.round(patch_grid).astype(np.int32)
		
		patch_grid += area_tl_xy

		# bounds
		bounds_img = [
			[0, 0],
			img_size_xy,
		]

		if context_restriction:
			bounds = [
				area_tl_xy,
				area_tl_xy + area_size_xy_adjusted,
			]
		else:
			bounds = bounds_img	
		
		bounds_tl, bounds_br = bounds
		patch_grid = cls.boxes_to_bounds(patch_grid, patch_size, bounds_tl, bounds_br)
			
		context_grid = patch_grid - (context_size - patch_size) // 2
		context_grid = cls.boxes_to_bounds(context_grid, context_size, bounds_tl, bounds_br)
		
		# again ensure they are in the image
		patch_grid, context_grid = [
			cls.boxes_to_bounds(grid, size, bounds_img[0], bounds_img[1])
			for (grid, size) in [
				(patch_grid, patch_size),
				(context_grid, context_size),
			]
		]

		return EasyDict(
			patch_size = patch_size, 
			context_size = context_size,
			patch_overlap = patch_overlap,
			patch_grid = patch_grid,
			context_grid = context_grid,
		)
	
	@classmethod
	def extract_crops_and_masks(cls, image_hwc, patch_grid, context_grid, patch_size, context_size, area_mask=None, **_):
		num_patches = patch_grid.shape[0]

		# offset of patch from its context
		patch_offset_from_context = patch_grid - context_grid

		img_crops = []
		masks = np.ones(shape=(num_patches, context_size, context_size), dtype=bool)

		for p_idx in range(num_patches):
			# context patches out of images
			c_tl = context_grid[p_idx] # context top left
			#context_crop_slice = (slice(c_tl[1], c_tl[1]+context_size), slice(c_tl[0], c_tl[0]+context_size))
			#img_crop = image_hwc[context_crop_slice]
			img_crop = image_hwc[c_tl[1]:c_tl[1]+context_size, c_tl[0]:c_tl[0]+context_size]
			# .copy() - if we want to black out the inpainted areas
			img_crops.append(img_crop)

			# black out the inpainted areas
			#img_crops[p_idx][p_rel_tl_y+5:p_rel_tl_y+patch_size-5, p_rel_tl_x+5:p_rel_tl_x+patch_size-5, :] = 0
			#img_crops[p_idx][p_rel_tl_y+1:p_rel_tl_y+patch_size-1, p_rel_tl_x+1:p_rel_tl_x+patch_size-1, :] = 0

			# inpainting mask
			p_rel_tl_x, p_rel_tl_y = patch_offset_from_context[p_idx]
			patch_inside_ctx_crop = (slice(p_rel_tl_y, p_rel_tl_y+patch_size), slice(p_rel_tl_x, p_rel_tl_x+patch_size))
			
			if area_mask is None:
				masks[(p_idx, ) + patch_inside_ctx_crop] = False

			else:
				p_tl = patch_grid[p_idx]
				area_mask_crop = area_mask[p_tl[1]:p_tl[1]+patch_size, p_tl[0]:p_tl[0]+patch_size]

				# inpaint the subset of the patch that is inside the mask
				masks[(p_idx, ) + patch_inside_ctx_crop] = np.logical_not(area_mask_crop)
			
		return EasyDict(
			patches_context_bhwc = img_crops,
			patches_mask_bhw = masks,
		)

	@staticmethod
	def filter_out_empty_masks(context_grid, patch_grid, patches_context_bhwc, patches_mask_bhw, area_threshold=0.05, **_):

		indices_to_keep = [i for i, mask in enumerate(patches_mask_bhw) if np.count_nonzero(mask) / np.product(mask.shape) > area_threshold]

		return EasyDict(
			context_grid = context_grid[indices_to_keep],
			patch_grid = patch_grid[indices_to_keep],
			patches_mask_bhw = patches_mask_bhw[indices_to_keep],
			patches_context_bhwc = (
				[patches_context_bhwc[i] for i in indices_to_keep] 
				if isinstance(patches_context_bhwc, list) 
				else patches_context_bhwc[indices_to_keep]
			),
		)


	@classmethod
	def extract_patches_given_box(cls, image_hwc, area_tl_xy, area_size_xy, patch_size, context_size, patch_overlap, context_restriction = False):

		# formulate the grid
		fr = cls.divide_area(
			patch_size = patch_size, context_size=context_size, patch_overlap=patch_overlap, context_restriction=context_restriction,
			area_tl_xy = area_tl_xy, area_size_xy = area_size_xy,
			img_size_xy = image_hwc.shape[:2][::-1],
		)

		# later: perhaps filter out some grid points based on a label mask

		# extract image crops and inpainting masks 
		fr.update(cls.extract_crops_and_masks(image_hwc = image_hwc, **fr))

		# out: patch_size, context_size, patch_grid, context_grid, patches_context_bhwc, patches_mask_bhw
		return fr

	@classmethod
	def extract_patches_given_road_mask(cls, image_hwc, area_mask, patch_size, context_size, patch_overlap, patch_mask_intersection_min_value=0.05, context_restriction = False):
		tl_x, tl_y, bb_w, bb_h = cv.boundingRect(area_mask.astype(np.uint8))

		# formulate the grid
		fr = cls.divide_area(
			patch_size = patch_size, context_size=context_size, patch_overlap=patch_overlap, context_restriction=context_restriction,
			area_tl_xy = (tl_x, tl_y), area_size_xy = (bb_w, bb_h),
			img_size_xy = image_hwc.shape[:2][::-1],
		)

		margin = 5
		area_mask_without_margin = cv.erode(
			area_mask.astype(np.uint8), 
			np.ones((margin, margin), dtype=np.uint8),
			iterations = 1,
		).astype(bool)
		fr.area_mask_without_margin = area_mask_without_margin

		# extract image crops and inpainting masks 
		fr.update(cls.extract_crops_and_masks(
			image_hwc = image_hwc,
			area_mask = area_mask_without_margin,
			**fr,
		))

		# filter out some grid points based on a label mask
		fr.update(cls.filter_out_empty_masks(**fr, area_threshold=patch_mask_intersection_min_value))

		# out: patch_size, context_size, patch_grid, context_grid, patches_context_bhwc, patches_mask_bhw
		return fr

	@staticmethod
	def fuse_patches_max(image_background, patch_grid, context_grid, patch_size, context_size, patches_value_bhwc, patches_mask_bhw, **_):
		num_patches = patch_grid.shape[0]

		out_img = image_background.astype(np.float32)

		perpix_patch_depth = np.zeros(out_img.shape[:2], dtype=np.float32)

		patch_offset_from_context = patch_grid - context_grid
		
		# The weight is from 0 to 1, we add the 0.01 so that there is no 0/0 case
		fusion_weight = fusion_weights_square_patch((patch_size, patch_size)) + 0.01

		for p_idx in range(num_patches):
			p_tl = patch_grid[p_idx]
			p_rel_tl_x, p_rel_tl_y = patch_offset_from_context[p_idx]

			patch_rec = patches_value_bhwc[p_idx]
			rec_crop_slice = (slice(p_rel_tl_y, p_rel_tl_y+patch_size), slice(p_rel_tl_x, p_rel_tl_x+patch_size))
			inpainting_crop = patch_rec[rec_crop_slice]
			context_mask_crop = patches_mask_bhw[p_idx][rec_crop_slice]

			mask = 1 - context_mask_crop

			img_slice = (slice(p_tl[1], p_tl[1]+patch_size), slice(p_tl[0], p_tl[0]+patch_size))
	
			#print(f'max({out_img[img_slice].shape}, {inpainting_crop.shape} * {inpainting_mask_crop.shape})')	

			out_img[img_slice] = np.maximum(
				out_img[img_slice],
				inpainting_crop * mask[:, :, None],
			)
			
			perpix_patch_depth[img_slice] += mask

		return EasyDict(
			image_fused = out_img,
			fusion_weight_map = perpix_patch_depth,
		)

	@classmethod
	def fuse_patches(cls, 
			image_background, patch_grid, context_grid, patch_size, context_size, patches_reconstruction, patches_mask_bhw, 
			out_dtype=np.uint8,
			**_):
		num_patches = patch_grid.shape[0]

		out_img = image_background.astype(np.float32)

		perpix_patch_depth = np.zeros(out_img.shape[:2], dtype=np.float32)

		patch_offset_from_context = patch_grid - context_grid
		
		# The weight is from 0 to 1, we add the 0.01 so that there is no 0/0 case
		fusion_weight = fusion_weights_square_patch((patch_size, patch_size)) + 0.01

		for p_idx in range(num_patches):
			p_tl = patch_grid[p_idx]
			p_rel_tl_x, p_rel_tl_y = patch_offset_from_context[p_idx]

			patch_rec = patches_reconstruction[p_idx]
			rec_crop_slice = (slice(p_rel_tl_y, p_rel_tl_y+patch_size), slice(p_rel_tl_x, p_rel_tl_x+patch_size))
			inpainting_crop = patch_rec[rec_crop_slice]

			# this is the mask of the CONTEXT not the inpainting
			context_mask_crop = patches_mask_bhw[p_idx][rec_crop_slice]

			img_slice = (slice(p_tl[1], p_tl[1]+patch_size), slice(p_tl[0], p_tl[0]+patch_size))

			# TODO the weighted fusion can be optimized with a numba loop
			
			w_rec = fusion_weight *  np.logical_not(context_mask_crop)
			w_background = perpix_patch_depth[img_slice] + context_mask_crop
			w_sum = w_background + w_rec
			
			#print(f'({w_before.shape} * {out_img[img_slice].shape} + {fusion_weight.shape} * {inpainting_crop.shape}) / {w_after.shape}')

			out_img[img_slice] = (
					w_background[:, :, None] * out_img[img_slice] 
					+ 
					w_rec[:, :, None] * inpainting_crop
				) / w_sum[:, :, None]
			
			perpix_patch_depth[img_slice] += w_rec

		if out_dtype is not None:
			out_img = out_img.astype(out_dtype)

		return EasyDict(
			image_fused = out_img,
			fusion_weight_map = perpix_patch_depth,
		)

	@classmethod
	def draw_fusion_weights_and_context(cls, canvas, patch_grid, context_grid, patch_size, context_size, patches_mask_bhw, **_):
		num_patches = patch_grid.__len__()
		canvas_size_hw = canvas.shape[:2]
		colors = np.random.uniform(size = (num_patches, 3), low=128, high=255).astype(dtype=np.uint8)	
		
		cv_pt = lambda p: tuple(map(int, p))
		

		fusing_result = cls.fuse_patches(
			image_background = np.zeros(canvas_size_hw + (3,), dtype=np.uint8),
			patch_size = patch_size, context_size = context_size,
			patch_grid = patch_grid, context_grid = context_grid,
			patches_mask_bhw = patches_mask_bhw,
			patches_reconstruction = [
				np.tile(color, reps=(context_size, context_size, 1))
				for color in colors
			]
		)

		canvas = canvas // 2 + fusing_result.image_fused // 2

		for i, p_tl, ctx_tl in zip(range(num_patches), patch_grid, context_grid):
			# canvas = cv.rectangle(
			# 	canvas, 
			# 	pt1 = tuple(p_tl), 
			# 	pt2 = tuple(p_tl + patch_size), 
			# 	color = tuple(colors[i]),
			# 	thickness = 2,
			# )

			#crop = slice(p_tl[1], p_tl[1] + patch_size), slice(p_tl[0], p_tl[0] + patch_size)
			#canvas[crop] = canvas[crop].astype(np.float32) * 0.5 + colors[i].astype(np.float32)[None, None] * 0.5

			canvas = cv.rectangle(
				canvas, 
				cv_pt(ctx_tl), 
				cv_pt(ctx_tl + context_size), 
				cv_pt(colors[i]),
				thickness = 2,
			)
			
		return canvas

	@staticmethod
	def draw_patch_reconstructions_and_masks(patches_context_bhwc, patches_mask_bhw, patches_reconstruction, demo_colors=None, **_):

		cv_pt = lambda p: tuple(map(int, p))

		num_patches = patches_context_bhwc.__len__()
		if demo_colors is None:
			demo_colors = np.random.uniform(size = (num_patches, 3), low=128, high=255).astype(dtype=np.uint8)	
			

		demo_patches = []
		
		# limit the number of displayed patches, prevent huge images
		num_patches_override = min(num_patches, 25)

		for idx, context_img, mask, inpainting_img, color in zip(range(num_patches_override), patches_context_bhwc, patches_mask_bhw, patches_reconstruction, demo_colors):
			mask_contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)

			color = cv_pt(color)

			mask_on_input, mask_on_rec = [
				cv.drawContours(img.copy(), mask_contours, -1, color=color, thickness=1)
				for img in [context_img, inpainting_img]
			]
			
# 			m = np.logical_not(mask).astype(np.uint8)[:, :, None] * 128
# 			mask_on_input = context_img // 2 + m
# 			mask_on_rec = inpainting_img // 2 + m

			demo_patch = np.concatenate([
				mask_on_input, 
				np.zeros((mask_on_input.shape[0], 2, 3), dtype=np.uint8),
				mask_on_rec,
			], axis=1)

			demo_patches.append(demo_patch)

		demo_patches_montage = image_montage_same_shape(
			imgs = demo_patches,
			num_cols = int(np.floor(np.sqrt(demo_patches.__len__()))) // 2, 
			border=16,
		)

		#print(demo_patches.__len__(), demo_patches_montage.shape)
		
		return EasyDict(
			demo_patches = demo_patches,
			demo_patches_montage = demo_patches_montage,
		)


	def __init__(self, patch_size, inpaint_func=None, inpainter_variant_name = None, context_size = None, patch_overlap = 0, context_restriction = False, b_demo=False):
		""" 
		impatching algorithm config 
		
		inpaint_func(images_hwc_list=..., masks_hw_list=...) returns a list/stack of inpainted images, same shape as images_hwc_list

		b_demo = generate images
		"""
		self.patch_size = patch_size
		self.context_size = context_size or 2*patch_size
		self.patch_overlap = float(patch_overlap)
		self.context_restriction = context_restriction

		self.b_demo = b_demo
		self.inpaint_func = inpaint_func
		if inpaint_func:
			self.inpainter_variant_name = inpaint_func.__name__
		else:
			self.inpainter_variant_name = inpainter_variant_name


	def process_frame_v1(self, image_data, area_tl_xy, area_size_xy, **_):

		# fr = EasyDict(
		# 	image_data = image_data,
		# 	area_tl_xy = area_tl_xy, 
		# 	area_size_xy = area_size_xy,
		# 	img_size_xy = image_data.shape[:2][::-1],
		# )

		fr = self.extract_patches_given_box(
			image_hwc=image_data,
			area_tl_xy = area_tl_xy, area_size_xy = area_size_xy,
			patch_size = self.patch_size, context_size = self.context_size,
			patch_overlap = self.patch_overlap, context_restriction = self.context_restriction,
		)

		fr.patches_reconstruction = self.inpaint_func(
			images_hwc_list = fr.patches_context_bhwc, 
			masks_hw_list = fr.patches_mask_bhw,
		)

		fr.update(self.fuse_patches(image_background = image_data.copy(), **fr))

		if self.b_demo:
			fr.demo_fusion_weights = self.draw_fusion_weights_and_context(canvas = image_data.copy(), **fr)

			fr.demo_image = np.concatenate([
				fr.image_fused, fr.demo_fusion_weights,
			], axis=1)

			fr.update(self.draw_patch_reconstructions_and_masks(**fr))

		return fr

	def process_frame_v2(self, image_data, labels_road_mask, debug_mask = False, **_):

		# fr = EasyDict(
		# 	image_data = image_data,
		# 	area_tl_xy = area_tl_xy, 
		# 	area_size_xy = area_size_xy,
		# 	img_size_xy = image_data.shape[:2][::-1],
		# )

		fr = self.extract_patches_given_road_mask(
			image_hwc = image_data,
			area_mask = labels_road_mask,
			patch_size = self.patch_size, context_size = self.context_size,
			patch_overlap = self.patch_overlap, context_restriction = self.context_restriction,
		)

		if debug_mask:
			fr.demo_fusion_weights = self.draw_fusion_weights_and_context(canvas = image_data.copy(), **fr)
			return fr
			# early exit!
		
		fr.patches_reconstruction = self.inpaint_func(
			images_hwc_list = fr.patches_context_bhwc, 
			masks_hw_list = fr.patches_mask_bhw,
		)

		fr.update(self.fuse_patches(image_background = image_data.copy(), **fr))

		if self.b_demo:
			fr.demo_fusion_weights = self.draw_fusion_weights_and_context(canvas = image_data.copy(), **fr)

			# fr.demo_image = np.concatenate([
			# 	fr.image_fused, fr.demo_fusion_weights,
			# ], axis=1)

			fr.demo_image = image_montage_same_shape(
				imgs = [image_data, fr.image_fused, fr.demo_fusion_weights],
				num_cols = 2,
				border=5,
			)

			fr.update(self.draw_patch_reconstructions_and_masks(**fr))

		return fr




	@staticmethod
	def image_inpaint_and_fuse(inpaint_func, image_data, area_tl_xy, area_size_xy, b_demo=False):
		"""
		inpaint_func(images_hwc_list=..., masks_hw_list=...) returns a list/stack of inpainted images, same shape as images_hwc_list
		"""
		fr = EasyDict(
			area_tl_xy = area_tl_xy, 
			area_size_xy = area_size_xy,
			img_size_xy = image_data.shape[:2][::-1],
		)

		res_inp = imp.divide_area(**fr, **imp)
		fr.update(res_inp)

		res_extr = imp.extract_crops_and_masks(**fr, image_hwc=image_data)
		fr.update(res_extr)

		inp_results = inferent.predict_auto_batch(
			images_hwc_list = list(fr.patches_context_bhwc), 
			masks_hw_list = list(fr.patches_mask_bhw),
		)
	
		fr.patches_reconstruction = inp_results

		print([a.shape for a in fr.patches_context_bhwc])

		print([a.shape for a in fr.patches_reconstruction])

		fr.fused_image = imp.fuse_patches(image_background=image_data, **fr)


		patch_img = np.concatenate([
			np.concatenate(list(fr.patches_context_bhwc), axis=1),
			np.concatenate(list(fr.patches_reconstruction), axis=1),
		], axis=0)

		fr.demo_patches = patch_img
		fr.demo_patch_loc = imp.draw_patching(canvas=image_data.copy(), **fr)
		
		#show(list(fr.patches_context_bhwc), fr.patches_reconstruction, list(fr.patches_mask_bhw))
		#show(fr.fused_image)

		return fr


	def load_default_inpainter_net(self, variant=None):
		variant = variant or self.inpainter_variant_name
		self.inpainter_variant_name = variant

		if variant == 'dummy':
			self.inpaint_func = lambda images_hwc_list, masks_hw_list: [a//2 for a in images_hwc_list]	

		elif variant == 'deep-fill-tf2':
			# TF graph mode crashes when loaded 2nd time
			self.inpainter_net = load_deepfill_inpainter()
			self.inpaint_func = self.inpainter_net.predict_auto_batch

		elif variant.startswith('mmagic.'):
			self.inpaint_func = InpainterMmagic(variant.removeprefix('mmagic.'))

		else:
			raise NotImplementedError(f'Inpainter: {variant}')

	@staticmethod
	def imwrite_with_dir(path, img):
		path.parent.mkdir(parents=True, exist_ok=True)
		imwrite(path, img)

	def gen__worker(self, frame_sampler, task_queue : multiprocessing.Queue, solution_queue : multiprocessing.Queue, dir_out : Path):
		print('process started')
		
		self.load_default_inpainter_net()

		while not task_queue.empty():
			frame_idx = task_queue.get()
			frame = frame_sampler[frame_idx]

			fid = frame['fid']
			img = frame['image']
			labels_road_mask = frame['labels_road_mask']

			result = self.process_frame_v2(
				image_data = img,
				labels_road_mask = labels_road_mask,
			)

			inp_path = dir_out / 'images' / f'{fid}__inpainted.webp'
			self.imwrite_with_dir(inp_path, result.image_fused)
			mask_path = dir_out / 'labels' / f'{fid}__road_mask.png'
			self.imwrite_with_dir(mask_path, labels_road_mask)

			result.fid = fid
			result.labels_road_mask = labels_road_mask
			result.image_path_inpainted = inp_path
			result.label_path_road_mask = mask_path

			solution_queue.put(result)


	

	def gen__run(self, in_frame_sampler, dir_out, num_workers=3):
		"""
		sampler is indexed with integers 0....sampler.__len__()-1
		for each index, it returns:
			image - HxWx3 uint8
			labels_road_mask - HxW bool
			fid - string from which to generate file paths
		"""
		
		(dir_out / 'images').mkdir(exist_ok=True, parents=True)
		(dir_out / 'labels').mkdir(exist_ok=True, parents=True)

		num_tasks = in_frame_sampler.__len__()
		tasks = range(num_tasks)

		task_queue = multiprocessing.Queue()
		for i in range(num_tasks):
			task_queue.put(i)

		print('qsize', task_queue.qsize())
			
		solution_queue = multiprocessing.Queue()

		worker_kwargs = dict(
			frame_sampler = in_frame_sampler,
			task_queue = task_queue,
			solution_queue = solution_queue,
			dir_out = dir_out,
		)

		workers = [
			multiprocessing.Process(target=self.gen__worker, kwargs = worker_kwargs, daemon=True)
			for i in range(num_workers)
		]

		json_frames = []

		try:
			for w in workers:
				w.start()

			for i in tqdm(range(num_tasks)):
				out_fr = solution_queue.get()

				json_frames.append(dict(
					fid = out_fr.fid,
					label_path_road_mask = str(out_fr.label_path_road_mask.relative_to(dir_out)),
					image_path_inpainted = str(out_fr.image_path_inpainted.relative_to(dir_out)),
				))
		finally:
			for w in workers:
				if w.is_alive():
					w.terminate()

		json_frames.sort(key=itemgetter('fid'))

		json_index = dict(
			cfg = dict(
				patch_size = self.patch_size,
				context_size = self.context_size,
				patch_overlap = self.patch_overlap,
				context_restriction = self.context_restriction,
				inpainter_variant_name = self.inpainter_variant_name,
			),
			frames = json_frames,
		)

		with (dir_out / 'index.json').open('w') as index_file:
			json.dump(json_index, index_file, indent='	')




def test_inpatcher():
	imp = Impatcher(patch_size = 128, patch_overlap=0.5)
	imp.divide_area([100, 50], [600, 300], [2048, 1024])

	def patch_case(dc, restrict=False):
		img_data = imread(DIR_INP_DEMO_CASES_IN / dc.img)
		area_tl = dc.pos
		area_size = dc.size
			
		imp = Impatcher(
			patch_size = 100,
			patch_overlap = 0.5,
			context_restriction=restrict,
		)
		
		imp.divide_area(
			area_tl_xy = dc.pos,
			area_size_xy = dc.size,
			img_size_xy = img_data.shape[:2][::-1],
		)

		demo = imp.draw_patching(img_data.copy())
		
		show(demo)

		crops = imp.extract_patches(img_data)

	patch_case(demo_case_selection[0])

def test_inpatcher_2():
	imp = Impatcher(
		patch_size = 100, 
		context_size = 200,
		patch_overlap = 0,
		context_restriction= False,
	)

	def patch_case(dc, restrict=False):

		imp = Impatcher(
			patch_size = 100, 
			context_size = 200,
			patch_overlap = 0,
			context_restriction= False,
		)

		img_data = imread(DIR_INP_DEMO_CASES_IN / dc.img)
		area_tl = dc.pos
		area_size = dc.size

		fr = EasyDict(
			area_tl_xy = area_tl, 
			area_size_xy = area_size,
			img_size_xy = img_data.shape[:2][::-1],
		)

		res_inp = imp.divide_area(**fr, **imp)
		fr = EasyDict(**fr, **res_inp)

		res_extr = imp.extract_crops_and_masks(**fr, image_hwc=img_data)
		fr = EasyDict(**fr, **res_extr)

		out = imp.draw_patching(canvas=img_data, **fr)
		show(out)





	patch_case(demo_case_selection[0])


"""

mcedit ~/programs/main10/lib/python3.10/site-packages/mmagic/apis/inferencers/inpainting_inferencer.py

	

"""
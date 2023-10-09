

from functools import partial
from pathlib import Path

import numpy as np
import cv2 as cv
from skimage.segmentation import mark_boundaries
from imageio import imread, imwrite

from src.paths import DIR_DATA
from src.a12_inpainting.demo_case_selection import load_demo_cases, DIR_INP_DEMO_CASES, DIR_INP_DEMO_CASES_IN, DIR_INP_DEMO_CASES_OUT, demo_case_selection
from src.a12_inpainting.demo_harness import demo_harness


def extract_patches_superpix(image_data, patch_size, context_size, area_tl_xy=None, area_size_xy, **_):

	#smoothness_factor = 10.
	
	spix = cv.ximgproc.createSuperpixelSLIC(
		image = image_data,
		algorithm = cv.ximgproc.SLIC,
		region_size = patch_size,
		ruler = patch_size, # no effect on SLICO
	)
	
	spix.iterate()
	
	spix_ids = spix.getLabels()
	
	canvas = mark_boundaries(image_data, spix_ids)
	
	#show(canvas)
	
	return dict(
		demo_image = canvas,
	)
	

def try_superpix_patches(dc):
	image_data = imread(DIR_INP_DEMO_CASES_IN / dc.img)
	area_tl = dc.pos
	area_size = dc.size

	fr = extract_patches_superpix(
		patch_size = 100,
		context_size = 256,
		image_data = image_data,
		area_tl_xy = area_tl, 
		area_size_xy = area_size,
	)
	
	#show(fr.demo_patches)
	#show([fr.demo_patch_loc, fr.fused_image])
	
	#demo_whole = np.concatenate([fr.demo_patch_loc, fr.fused_image], axis=1)
	
	#rc = '-restrict' if imp.context_restriction else ''


def visualize_superpixels(image_data, patch_size, algorithm = cv.ximgproc.SLICO, smoothness_multiplier = 1, **_):
	spix = cv.ximgproc.createSuperpixelSLIC(
		image = image_data,
		algorithm = algorithm,
		region_size = patch_size,
		ruler = patch_size * smoothness_multiplier, # no effect on SLICO
	)
	
	spix.iterate()
	spix_labels = spix.getLabels()
		
	canvas = mark_boundaries(image_data, spix_labels)
	canvas = (canvas * 255).astype(np.uint8)
	
	return dict(
		superpix_labels = spix_labels,
		demo_image = canvas,
	)


def demo_superpix():
	samples = load_demo_cases()

	methods = {
		'SLICO 96': partial(visualize_superpixels, algorithm = cv.ximgproc.SLICO, patch_size=96),
		'SLIC 96': partial(visualize_superpixels, algorithm = cv.ximgproc.SLIC, patch_size=96,  smoothness_multiplier = 1),
		'MSLIC 96': partial(visualize_superpixels, algorithm = cv.ximgproc.MSLIC, patch_size=96,  smoothness_multiplier = 1),
		
		'SLICO 160': partial(visualize_superpixels, algorithm = cv.ximgproc.SLICO, patch_size=160),
		'SLIC 160': partial(visualize_superpixels, algorithm = cv.ximgproc.SLIC, patch_size=160,  smoothness_multiplier = 1),
		'MSLIC 160': partial(visualize_superpixels, algorithm = cv.ximgproc.MSLIC, patch_size=160,  smoothness_multiplier = 1),		
	}
	
# 	f = list(methods.values())[0]
# 	r = f(**samples[0])
# 	du = r['demo_image']
# 	du //= 2
# 	show(du)
	
	demo_harness(methods, samples = samples, out_dir = DIR_DATA / 'comp_superpix_for_inpainting_v1', multiproc_allowed = True)
	
	
	
	#show(fr.demo_patches)
	#show([fr.demo_patch_loc, fr.fused_image])
	
	#demo_whole = np.concatenate([fr.demo_patch_loc, fr.fused_image], axis=1)
	
	#rc = '-restrict' if imp.context_restriction else ''

if __name__ == '__main__':
	try_superpix_patches(demo_case_selection[1])

	demo_superpix()
	


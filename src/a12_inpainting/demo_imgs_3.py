from pathlib import Path
import numpy as np
import cv2 as cv
from functools import lru_cache

from ..common.jupyter_show_image import show, imread, imwrite
from ..paths import DIR_DATA, DIR_EXP, DIR_EXP2

@lru_cache()
def make_stripes(sz, thickness=2, spacing_x=20, b_float=False):
	w, h = sz
	
	canvas = np.zeros((h, w), dtype=np.uint8)
	
	
	for x in range(-h, w, spacing_x):
		cv.line(canvas, (x, 0), (x+h, h), color=(255, 255, 255), thickness=thickness)
	
	if b_float:
		canvas = canvas.astype(np.float32)
		canvas *= (1/255)
		
	return canvas




def mark_area(canvas_f, mask, color, flip_stripes=False, thickness=1):
	h, w = mask.shape
	color = np.array(color) * (1./255)

	#canvas_f = canvas.astype(np.float32) * (1./255)
	mask_f = mask.astype(np.float32)

	# contour by erosion
	mask_inset = cv.erode(
		src = mask.astype(np.uint8),
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)),
	).astype(bool)
	contour = mask ^ mask_inset
	
	stripe_tex_f = make_stripes((w, h), b_float=True, thickness=thickness, spacing_x=thickness*8)
	if flip_stripes:
		stripe_tex_f = stripe_tex_f[:, ::-1]
		
	color_mask_f = np.maximum(stripe_tex_f * mask_f, contour)
	
	fused = canvas_f * (1.-color_mask_f[:, :, None]) + color_mask_f[:, :, None] * color[None, None, :]
	
	return fused
	
def demo_annotate_frame_roi(image, labels, fid, crop_up=0, b_save=False, **_):
	
	image = image[crop_up:]
	labels = labels[crop_up:]
	
	img = cv.pyrDown(cv.pyrDown(image))
	labels = labels[::4, ::4]
	
	mask_road = labels == 0
	mask_obstacle = labels == 1
	
	canvas = img.astype(np.float32) * (1./255.)
	
# 	canvas = mark_area(canvas, mask_road, (2, 124, 255))
	canvas = mark_area(canvas, mask_road, (57 , 255, 163))
	canvas = mark_area(canvas, mask_obstacle, (255, 85, 0), flip_stripes=True)
	
	
	
	if b_save:
		imwrite( DIR_DATA / '0000_interactive_out' / f'roi_demo_{fid}.webp', (canvas * 255).astype(np.uint8))
	
	show(canvas)
	

# from src.a12_inpainting.demo_case_selection import DatasetRegistry
# ds_ro = DatasetRegistry.get_implementation('RoadObstacles-v003')
# ds_laf = DatasetRegistry.get_implementation('FishyLAF-LafRoi')
# demo_annotate_frame_roi(**ds_ro['paving_boot_1'], b_save=True, crop_up=500)
# demo_annotate_frame_roi(**ds_laf['01_Hanns_Klemm_Str_45_000000_000260'], b_save=True, crop_up=0)

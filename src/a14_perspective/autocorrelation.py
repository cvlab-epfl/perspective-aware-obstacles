
from functools import partial
from multiprocessing.dummy import Pool as Pool_thread

import numpy as np
import cv2 as cv
from tqdm import tqdm

from ..common.jupyter_show_image import show, imwrite
from ..paths import DIR_DATA
from ..a12_inpainting.vis_imgproc import image_montage_same_shape

from .warp import affine_scale

def patch_metrics_worker(image, ofs, mask_clipped=None, mask_area=None, RADIUS=32):
	box = image[ofs[1] - RADIUS:ofs[1] + RADIUS, ofs[0] - RADIUS:ofs[0] + RADIUS]
	match = cv.matchTemplate(image, box, cv.TM_CCORR_NORMED)

	area = np.count_nonzero(mask_clipped & (match > 0.98)) * (1./mask_area)
	avg = np.mean(match, where=mask_clipped)
	return ofs, area, avg


def patch_autocorr_uniqueness_metrics(image, roi_mask, radius=32, step=16, b_pbar=False):
	"""
	For each RxR patch within the roi_mask, calculate its uniqueness using auto-correlation.

	@return dict with
		score_avg: 1. - average cos-distance of this patch to others
		score_area: 1. - area fraction with similarity above 0.98
	"""

	h, w = roi_mask.shape

	# patch grid	
	g = np.stack(np.mgrid[radius:h-radius:step, radius:w-radius:step], axis=2)
	g = g.reshape(-1, 2)	
	g = g[roi_mask[g[:, 0], g[:, 1]] == 1]
	g_yx = g
	g_xy = g_yx[:, ::-1]
	
	# preproc mask for worker
	roi_mask_clipped = roi_mask[radius:-radius+1,radius:-radius+1].copy().astype(bool)
	roi_mask_area = np.count_nonzero(roi_mask_clipped)
		
	worker = partial(
		patch_metrics_worker, 
		image, 
		#mask_full = roi_mask.astype(np.uint8),
		mask_clipped = roi_mask_clipped, 
		mask_area = roi_mask_area, 
		RADIUS = radius,
	)
	# progress bar
	pbar = tqdm if b_pbar else (lambda x, **_: x)
	
	# score arrays
	result_area = np.zeros((h//step, w//step), dtype=np.float32)
	result_avg = np.zeros((h//step, w//step), dtype=np.float32)
	#result_avg -= np.median(result_avg)
	#result_avg[result_avg < 0] = 0

	with Pool_thread() as pool:
		jobs = pool.imap(worker, g_xy, chunksize=48)

		for ofs, area_ratio, avg_corr in pbar(jobs, total=g_xy.__len__()):
			rr, rc = ofs[1] // step, ofs[0] // step
			result_area[rr, rc] = 1. - area_ratio
			result_avg[rr, rc] = 1. - avg_corr
	
	return dict(
		score_avg = result_avg,
		score_area = result_area,
	)



def stats(name, arr):
	print(f'{name}: min {np.min(arr)} avg {np.mean(arr)} max {np.max(arr)}')
	

def warp_score_back(H, score_unw, out_sz_xy):
	"""
	out_sz_xy: fr.image.shape[:2][::-1]
	"""

	score = cv.warpPerspective(
		score_unw, 
		H, 
		tuple(out_sz_xy), 
		flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR,
	)
	return score
	

def interactive_patch_metrics(fr, use_unwarp=True, radius=32, step=16, b_pbar=False, b_show=False, b_save=False):

	if use_unwarp:
		image = fr.unwarp_image
		road_mask = fr.unwarp_mask
	else:
		road_mask = fr.label_pixel_gt < 255
		image = fr.image

		road_top = max(0, np.where(np.any(road_mask, axis=1))[0][0] - radius // 2)
		image = fr.image[road_top:].copy()
		road_mask = road_mask[road_top:]

		
	acmet = patch_autocorr_uniqueness_metrics(
		image = image,
		roi_mask = road_mask, 
		radius = radius,
		step = step,
		b_pbar = b_pbar,
	)

	result_area, result_avg = acmet['score_area'], acmet['score_avg']
	
	out_sz_xy = road_mask.shape[::-1]

	if use_unwarp:
		H_with_scale = affine_scale(1./step, 1./step) @ fr.unwarp_H
		# H_with_scale =  fr.unwarp_H

		for k in ['score_area', 'score_avg']:
			score_unwarped = acmet[k]
			# score = warp_score_back(H_with_scale, score_unwarped, fr.image.shape[:2][::-1])
			score = cv.resize(score_unwarped, image.shape[:2][::-1])
			score = warp_score_back(fr.unwarp_H, score, fr.image.shape[:2][::-1])

			acmet[k] = score
			acmet[f'{k}_unw'] = score_unwarped
		
	else:
		for k in ['score_area', 'score_avg']:
			score_small = acmet[k]
			score = cv.resize(score_small, out_sz_xy)

			score = np.concatenate([
				np.zeros((road_top, road_mask.shape[1]), dtype=score.dtype),
				score
			], axis=0)

			acmet[k] = score
			acmet[f'{k}_small'] = score_small

	if b_save or b_show:
		# stats('results area', result_area)
		# stats('results avg', result_avg)

		# rbig_area = np.repeat(np.repeat(result_area, step, axis=1), step, axis=0)
		# rbig_avg = np.repeat(np.repeat(result_avg, step, axis=1), step, axis=0)
	
		# rbig_area = cv.resize(result_area, out_sz)
		# rbig_avg = cv.repeat(result_avg, out_sz)
	
		dimg = image_montage_same_shape(
			[fr.image, fr.label_pixel_gt < 255, acmet['score_avg'], acmet['score_area']],
			captions = ['image', 'mask', '1 - avg cos', 'area uniq'],
			downsample= 2 ,
		)

		if use_unwarp:
			dimg_perpsective = image_montage_same_shape([
				image, road_mask,
				cv.resize(acmet['score_avg_unw'], road_mask.shape[::-1]),
				cv.resize(acmet['score_area_unw'], road_mask.shape[::-1]),
			], captions=[
				'image unw', 'mask unw',
				'1 - avg cos', 'area uniq',
			], caption_size= 2)

			dimg = np.concatenate([
				dimg, dimg_perpsective
			], axis=0)

			name = 'unw'

		else:
			
			name = 'flat'
		
		
		if b_show:
			show(dimg)
		if b_save:
			imwrite(DIR_DATA / '1411_autocorr_area' / f'{fr.fid}_uniq_{name}.webp', dimg)
	
		return acmet


	#return result
		

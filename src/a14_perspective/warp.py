
import numpy as np
import cv2 as cv
from .cityscapes_pitch_angles import cam_info_to_transform
from ..common.geometry import affine_translation, affine_scale, homography_apply_rowvec, rot_around_x

def z_weights_from_horizon_level(horiz_level, center_y):
	"""
	Find a linear function $w0 + [wx wy] * [x y]$ such that it is 0 at horizon and 1 at our chosen center.
	This is used as the perspective scale map.
	In this version we assume no roll, so wx == 0.

	System of equations
		w0 + wy horiz_level = 0
		w0 + wy center_y = 1

	w0 = - wy horiz_level
	wy( horiz_level - center_y ) = 1

	Solution
		wy = 1/(horiz_level - center_y)
		w0 = - horiz_level wy
	"""

	if center_y <= horiz_level:
		raise ValueError(f"Center ({center_y}) above horizon {horiz_level}")

	wx = 0.
	wy = 1. / (horiz_level - center_y)
	w0 = - horiz_level * wy

	return np.array([
		w0, wx, wy
	], dtype=np.float64)


def linear_z_to_homography(linear_z_weights, mid_pt):
	w0, wx, wy = linear_z_weights
	mid_pt = np.array(mid_pt)
	z_center = float(mid_pt.reshape(1, -1) @ linear_z_weights[1:].reshape(-1, 1) + w0)
	
	H = np.array([
		[-1, 0, 0], # negatives here somehow prevent image from flipping around center
		[0, -2, 0],
		[wx, wy, z_center],
	])

	H = affine_translation(mid_pt) @ H @ affine_translation(-mid_pt)

	return H

def unwarp_homography_for_horizon(horiz_level, img_size_xy):
	mid_pt = np.array(img_size_xy) * 0.5
	z_weights = z_weights_from_horizon_level(horiz_level, mid_pt[1])
	H = linear_z_to_homography(z_weights, mid_pt)
	return H


def unwarp_homography_for_camera(camera_info):
	angle = -(np.pi * 0.5 + camera_info['extrinsic']['pitch'])
	R = rot_around_x(angle)
	t = np.array([0, 0, -camera_info['extrinsic']['z']])
	H = R.copy()
	H[:, 2] = R @ t

	cam_matrix = cam_info_to_transform(camera_info).cam_matrix
	H = cam_matrix @ H
	H = np.linalg.inv(H)
	# flip vertical
	H = affine_scale(1, -1) @ H
	#print('R\n', R, '\nH\n', H, '\nt\n', t)
	
	return H

# Adjusting homography for road hull

def hull_around_binary_mask(mask):
	"""
		Find convex hull of a mask
	"""
	mask_u8 = mask.astype(np.uint8).copy()

	# open to remove small noise points
	#plane_mask_u8 = cv2.morphologyEx(plane_mask_u8, cv2.MORPH_OPEN, MORPH_KERNEL)

	# find contours of the plane-mask, calculate convex hull of all contour points
	contour_list, hierarchy = cv.findContours(
		mask_u8,
		mode=cv.RETR_EXTERNAL,
		method=cv.CHAIN_APPROX_SIMPLE
	)

	all_contour_pts = np.concatenate(contour_list, axis=0)
	hull = cv.convexHull(all_contour_pts)[:, 0, :]

	return hull


def homography_fit_hull_unwspace(H, hull_unw, desired_max_dim):
	top_left = np.min(hull_unw, axis=0)
	bot_right = np.max(hull_unw, axis=0)

	size = np.abs(bot_right - top_left)
	scale = float(desired_max_dim) / np.max(size)

	# print(f'{affine_scale(scale, scale)} @ {affine_translation(-top_left)} @ {H}')

	H = affine_scale(scale, scale) @ affine_translation(-top_left) @ H


	out_size = size*scale
	return H, out_size, scale

def homography_fit_hull(H, hull_pts, desired_max_dim):
	hull_unw = homography_apply_rowvec(H, hull_pts)

	return homography_fit_hull_unwspace(H, hull_unw, desired_max_dim)

def homography_adjust_to_mask(H, mask, out_dimension):
	#H, out_size = homography_top_left_to_zero(H, get_img_size(inv_sc_map), 1024)

	mask_hull = hull_around_binary_mask(mask)
	H, out_size, scale = homography_fit_hull(H, mask_hull, out_dimension)

	return H, out_size, scale



def unwarp_road_frame(fr, horiz_margin=64, bottom_margin=32, b_show=False, size=2048):
	horiz_level = fr.persp_info.horizon_level
	img_sz_xy = fr.image.shape[:2][::-1]
	out_sz = (size, size)
	
	if 'label_pixel_gt' in fr:
		# LAF
		mask_road_unwarp = fr.label_pixel_gt <= 2
	else:
		# cityscapes
		mask_road_unwarp = (fr.labels_source == 6) | (fr.labels_source == 7)
	
	roi = mask_road_unwarp.copy()
	zero_above_level = int(horiz_level)+horiz_margin
	if zero_above_level > 0:
		roi[:zero_above_level] = False
	roi[roi.shape[0]-bottom_margin:] = False

	if 'camera_info' in fr:
		H = unwarp_homography_for_camera(fr.camera_info)
	else:
		H = unwarp_homography_for_horizon(horiz_level, img_sz_xy)	

	H, out_size, scale = homography_adjust_to_mask(H, roi, out_sz[0])
		
	fr.unwarp_H = H
	fr.unwarp_size = tuple(np.ceil(out_size).astype(np.int32))
	
	img_unw = cv.warpPerspective(fr.image, H, fr.unwarp_size)

	fr.unwarp_image = img_unw
	fr.unwarp_mask = cv.warpPerspective(
		mask_road_unwarp.astype(np.uint8), 
		H, 
		fr.unwarp_size,
		flags = cv.INTER_NEAREST,
	)
	
	if b_show:
		from ..common.jupyter_show_image import show
		show(
			[fr.image, mask_road_unwarp],
			[img_unw, fr.unwarp_mask],
		)
	
	return fr

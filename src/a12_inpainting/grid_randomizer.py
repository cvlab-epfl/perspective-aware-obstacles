import numpy as np
from easydict import EasyDict
from math import ceil, floor

import matplotlib, matplotlib.patches, matplotlib.pyplot

from ..a14_perspective.cityscapes_pitch_angles import transform_points_world_to_camera, read_cam_info, perspective_info_from_camera_info, draw_perspective_markers

def square_grid_randomizer(size_xy, grid_step, item_radius, random_offset=0):
	size_xy = np.array(size_xy, dtype=np.int32)
	
	size_xy_without_margin = np.maximum(
		size_xy - ceil(2 * item_radius),
		0,
	)
	
	# +1 we always make 1 grid row/column, then we see how many more we can add
	num_grid_pts_xy = np.floor(size_xy_without_margin / grid_step).astype(np.int32) + 1
	num_grid_pts = np.prod(num_grid_pts_xy)
	
	# allocate the points on the grid
	grid_pts_xy_base = np.mgrid[0:num_grid_pts_xy[0], 0:num_grid_pts_xy[1]].reshape(2, num_grid_pts).transpose([1, 0]) * grid_step

	# random offsets
	random_offsets_xy = np.random.normal(loc = 0., scale = 0.5, size=(num_grid_pts, 2)) * random_offset
	
	# offset the grid but force it to fall inside the bounding rect
	grid_pts_xy = np.clip(
		grid_pts_xy_base + random_offsets_xy,
		[0, 0],
		size_xy_without_margin-1,
	)
			
	# add the margin back
	grid_center_xy = grid_pts_xy + item_radius

	for dim in [0, 1]:
		if size_xy[dim] <= 2 * item_radius:
			grid_center_xy[:, dim] = size_xy[dim] * 0.5
	
	return grid_center_xy
	

def see_grid_randomizer():

	area_size_xy = np.array([800, 500])
	area_tl = np.array([200, 100])

	grid_centers = square_grid_randomizer(size_xy = area_size_xy, grid_step = 100, item_radius=50, random_offset=60)
	
	patch_size = np.array([80, 80])
	patch_tls = area_tl + grid_centers - 0.5*patch_size
		
	fig, plot = matplotlib.pyplot.subplots(1)
	plot.set_xlim([0, 1200])
	plot.set_ylim([0, 700])
	

	def draw_rect(tl, size, color='b'):
		plot.add_patch(matplotlib.patches.Rectangle(xy=tl, width=size[0], height=size[1], linewidth=1, edgecolor=color, facecolor='none'))

	draw_rect(area_tl, area_size_xy, color='r')
	
	for tl in patch_tls:
		draw_rect(tl, patch_size, color='b')


def generate_floor_grid_on_image(perspective_info, image_hw, grid_gap_m = [1, 1], grid_size_xy = [20, 10], first_offset = 5, random_offset = 0., constrain=True, **_):
	"""
	@param perspective_info: see `cityscapes_pitch_angles.py` to create perspective info
	"""
	
	num_grid_pts_xy = np.array(grid_size_xy, dtype=np.int32)
	num_grid_pts = np.prod(num_grid_pts_xy)
	
	# allocate the points on the grid
	grid_pts_xy = np.mgrid[0:num_grid_pts_xy[0], 0:num_grid_pts_xy[1]].reshape(2, num_grid_pts).transpose([1, 0])
	grid_pts_xy = grid_pts_xy.astype(np.float64)
	# center in x
	grid_pts_xy[:, 0] += -0.5*num_grid_pts_xy[0]

	# separate points by grid gap
	grid_pts_xy *= np.array(grid_gap_m, dtype=np.float32)[None, :]

	# add random offsets
	random_offsets_xy = np.random.normal(loc = 0., scale = 0.5, size=(num_grid_pts, 2)) * random_offset
	grid_pts_xy += random_offsets_xy

	# offset forward
	grid_pts_xy[:, 1] += first_offset

	# promote to 3D on flat ground Z=0
	grid_pts_xyz = np.concatenate([grid_pts_xy, np.zeros(num_grid_pts)[:, None]], axis=1)

	# transform to image space
	pts_img = transform_points_world_to_camera(
		perspective_info.cam_matrix,
		perspective_info.extrinsic_matrix, 
		grid_pts_xyz,
	)
	pts_img = pts_img[:, :2] # drop the Z

	# constrain to image space

	h, w = image_hw

	if constrain:
		# start from 1 and end with w-1 because these will be rounded to ints
		pts_mask_in_img = (
			(1 <= pts_img[:, 0]) & (pts_img[:, 0] < w-1) 
			&
			(1 <= pts_img[:, 1]) & (pts_img[:, 1] < h-1)
		)
		pts_img = pts_img[pts_mask_in_img, :]
	
	return pts_img


def show_frame_with_perspective_grid(fr, **grid_opts):
	"""
	@param grid_opts: args for generate_floor_grid_on_image 
	"""
	cam_info = read_cam_info(fr.dset, fr.fid)
	fr.camera_info = cam_info
	persp_info = perspective_info_from_camera_info(cam_info)
	fr.perspective_info = persp_info

	canvas = fr.image.copy()
	draw_perspective_markers(canvas, persp_info)
		
	print('K=\n', persp_info.cam_matrix, '\n RT =\n', persp_info.extrinsic_matrix)
	
	pts_img = generate_floor_grid_on_image(persp_info, image_hw=fr.image.shape[:2], **grid_opts)
	# print(pts_img)

	fig, plot = matplotlib.pyplot.subplots(1, 1, figsize=(14, 12))
	plot.imshow(canvas)
	plot.scatter(pts_img[:, 0], pts_img[:, 1], color='r')

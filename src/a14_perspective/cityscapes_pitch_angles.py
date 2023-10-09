
import json
from pathlib import Path
from math import ceil
import numpy as np
from matplotlib import pyplot
from easydict import EasyDict
import cv2 as cv
from ..paths import DIR_DATA
from ..common.geometry import rot_around_x, spatial_transform, projection_apply_rowvec, extend_with_neutral_row
from ..common.jupyter_show_image import show, imread, imwrite


def read_cam_info(dset, fid):
	cam_path = dset.dir_root / 'camera' / dset.split  / f'{fid}_camera.json'
	cam_content = cam_path.read_text()
	return json.loads(cam_content)

def read_cam_info_laf(dset, fr):
	cam_path = dset.dir_root / 'camera' / dset.split / f'{fr.scene_id:02d}_{fr.scene_name}' / f'{fr.fid}_camera.json'
	cam_content = cam_path.read_text()
	return json.loads(cam_content)


def world_to_camera_transform_matrix(cam_h, cam_pitch):
	"""
	@param cam_h: height of camera above road in meters
	@param cam_pitch: pitch of camera axis, pointing down
	"""
	# lower by h
	s1_lower_by_h = spatial_transform([0, 0, -cam_h])
	# rotate -pitch around x axis
	s2_rot = spatial_transform(t=[0, 0, 0], r=rot_around_x(-cam_pitch))
	# swap zy
	s3_swap = np.array([
		[1, 0, 0, 0],
		[0, 0, 1, 0],
		[0, -1, 0, 0],
		[0, 0, 0, 1],
	], dtype=np.float64)
	
	world_to_cam = s3_swap @ s2_rot @ s1_lower_by_h
	
	return world_to_cam


def cam_info_to_transform(cam_info):
	intr = cam_info['intrinsic']
	cx = intr['u0']
	cy = intr['v0']
	fx = intr['fx']
	fy = intr['fy']
	
	extr = cam_info['extrinsic']
	camera_height_m = extr['z']
	pitch = extr['pitch']
	
	cam_matrix = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1],
	])
	
	extrinsic_matrix = world_to_camera_transform_matrix(cam_h = camera_height_m, cam_pitch = pitch)
	
	return EasyDict(
		cam_matrix = cam_matrix,
		extrinsic_matrix = extrinsic_matrix,
	)

def transform_points_world_to_camera(cam_matrix, extrinsic_matrix, points):
	in_colvec = points.T
	in_homog = extend_with_neutral_row(in_colvec)
	cam_homog = extrinsic_matrix @ in_homog
	cam_norm = cam_homog[:2, :] / cam_homog[2:3, :]
	points_img = (cam_matrix @ extend_with_neutral_row(cam_norm))

	out_rowvec = points_img.T
	return out_rowvec


def perspective_info_from_camera_info(cam_info):
	intr = cam_info['intrinsic']
	cx = intr['u0']
	cy = intr['v0']
	fx = intr['fx']
	fy = intr['fy']
	
	extr = cam_info['extrinsic']
	camera_height_m = extr['z']
	pitch = extr['pitch']
	
	horiz = cy + fy * np.tan(-pitch)
	pix_per_meter_at_midpoint = fx * np.sin(pitch) / camera_height_m
	
	pix_per_meter_slope = pix_per_meter_at_midpoint / (cy - horiz)
	
	return EasyDict(
		horizon_level = horiz,
		pix_per_meter_at_midpoint = pix_per_meter_at_midpoint,
		pix_per_meter_slope = pix_per_meter_slope,
		midpoint = [cx, cy],
		**cam_info_to_transform(cam_info),
	)
	
def gen_perspective_scale_map(image_hw, horizon_level, pix_per_meter_slope):
	h, w = image_hw

	heatmap_top_pixel = max(0, ceil(horizon_level))
	scale_at_top = (heatmap_top_pixel - horizon_level) * pix_per_meter_slope
	scale_at_bottom = (h - horizon_level) * pix_per_meter_slope

	scale_map = np.zeros((h, w), dtype=np.float32)
	scale_map[heatmap_top_pixel:, :] = np.linspace(scale_at_top, scale_at_bottom, h-heatmap_top_pixel)[:, None]

	return scale_map

def draw_perspective_markers(canvas, persp_info):
	horiz_level = persp_info.horizon_level
	horiz_level_int = round(horiz_level)
	canvas[horiz_level_int-4:horiz_level_int, :] = (30, 250, 0)
	
			
	mcol = canvas.shape[1] // 2
	for row in range(0, canvas.shape[0], 100):
		if row > horiz_level:
			
			# inv_h0=np.cos(cam_info['extrinsic']['pitch']) / cam_info['extrinsic']['z']
			# v = 512 - row
			# P = inv_h0 * (cam_info['intrinsic']['fy'] * np.tan(cam_info['extrinsic']['pitch']) - v)
			
			below_horiz = row - horiz_level
			pix_per_m = below_horiz * persp_info.pix_per_meter_slope
			
			# print(pix_per_m, P)
			# pix_per_m = P
			
			cv.rectangle(
				canvas,
				(round(mcol - 0.5*pix_per_m), round(row - 2)),
				(round(mcol + 0.5*pix_per_m), round(row + 2)),
				(250, 100, 10),
				2,
			)
			
	
	mid = persp_info.midpoint
	pix_per_m = persp_info.pix_per_meter_at_midpoint
	
	# cv.rectangle(
	# 	canvas,
	# 	(round(mid[0] - pix_per_m), round(mid[1] - 3)),
	# 	(round(mid[0] + pix_per_m), round(mid[1] + 3)),
	# 	(10, 40, 250),
	# 	2,
	# )

	return canvas

def show_frame_with_horizon(dset, idx_or_fid):
	fr = dset[idx_or_fid]

	try:
		cam_info = read_cam_info(dset, fr.fid)
	except:
		cam_info = read_cam_info_laf(dset, fr)

	fr.camera_info = cam_info
	persp_info = perspective_info_from_camera_info(cam_info)
	fr.perspective_info = persp_info

	canvas = fr.image.copy()
	draw_perspective_markers(canvas, persp_info)

	horiz_level = persp_info.horizon_level
	horiz_level_int = round(horiz_level)

		
	h, w = fr.image.shape[:2]
	perspective_feature = np.zeros((h, w), dtype=np.float32)
	horiz_to_bottom = h - horiz_level
	scale_at_bottom = horiz_to_bottom * persp_info.pix_per_meter_slope
	perspective_feature[horiz_level_int+1:, :] = np.linspace(0, scale_at_bottom, h-horiz_level_int-1)[:, None]
	
	fig, plot = pyplot.subplots(1, 1)
	e1 = plot.imshow(perspective_feature)
	plot.set_title('Pixels per meter')
	fig.colorbar(e1)
	fig.tight_layout()
	
	show(canvas)
	
	fr.image_with_horizon = canvas

	dir_out = DIR_DATA / '1401_Ctc_Perspective' / f'{dset.name}-{dset.split}'
	dir_out.mkdir(exist_ok=True, parents=True)
	fn = fr.fid.replace('/', '__')
	imwrite(dir_out / f'{fn}_horiz.webp', canvas)
	fig.savefig(dir_out / f'{fn}_levels.jpg')


	return fr




def find_top_road_line(road_mask, horiz_above_road=16, DOWNSAMPLE=4):
	
	road_small = road_mask[::DOWNSAMPLE, ::DOWNSAMPLE]
	
	road_small_open = cv.morphologyEx(
		road_small.astype(np.uint8), 
		cv.MORPH_OPEN,
		np.ones((7, 7), dtype=np.uint8),
	)
	
	num_components, labels, stats, centroids = cv.connectedComponentsWithStats(road_small_open)
	
	areas = stats[:, 4]
	
	# ignore the 0 index which represents the background (0) value
	top_cc = np.argmax(areas[1:])+1
	
	tl_wh = stats[top_cc, :4] * DOWNSAMPLE
	horz_top = tl_wh[1] - horiz_above_road
	
	# print(tl_wh, areas[top_cc], 'at', top_cc)
	
	# canvas = fr.image.copy()
	# canvas = cv.line(canvas, (0, horz_top), (fr.image.shape[1], horz_top), (255, 0, 0), 4)
	
	# show([canvas, labels == top_cc])

	return horz_top


def invent_cam_and_persp_from_horizon(img_shape, horiz_level):
	h, w = img_shape[:2]
	cx = w * 0.5
	cy = h * 0.5
	fx = 2262
	fy = 2265

	if horiz_level > cy:
		horiz_level = cy - 20
	
	camera_height_m = 1.5
	
# 	print(horiz_level_int)
	
	# horiz = cy + fy * np.tan(-pitch)
	# (horiz - cy) / fy = tan(-pitch)
	# pitch = atan( (cy - horiz) / fy)
	
	
	
	pitch = np.arctan( (cy - horiz_level) / fy )
	
# 	print(pitch)
	
	pix_per_meter_at_midpoint = fx * np.sin(pitch) / camera_height_m
	
	pix_per_meter_slope = pix_per_meter_at_midpoint / (cy - horiz_level)
	
	cam_info = EasyDict(
		intrinsic = dict(
			u0 = cx,
			v0 = cy,
			fx = fx,
			fy = fy,
		),
		extrinsic = dict(
			z = camera_height_m,
			pitch = np.arctan((cy - horiz_level) / fy),
		),
	)

	pers_info = EasyDict(
		horizon_level = horiz_level,
		pix_per_meter_at_midpoint = pix_per_meter_at_midpoint,
		pix_per_meter_slope = pix_per_meter_slope,
		midpoint = [cx, cy],
	)

	return cam_info, pers_info



def invent_horizon_above_main_road(road_mask, horiz_above_road=16):
	horiz_level = find_top_road_line(road_mask, horiz_above_road=horiz_above_road)
	return invent_cam_and_persp_from_horizon(road_mask.shape[:2], horiz_level)



def perspective_scale_from_road_mask(labels_road_mask, **_):
	perspective_info = invent_horizon_above_main_road(labels_road_mask)

	sm = gen_perspective_scale_map(labels_road_mask.shape, perspective_info.horizon_level, perspective_info.pix_per_meter_slope)

	return dict(
		perspective_scale_map = sm,
	)





def load_frame_with_perspective_info(dset, idx_or_fid):
	fr = dset[idx_or_fid]
	try:
		cam_info = read_cam_info(dset, fr.fid)
	except:
		cam_info = read_cam_info_laf(dset, fr)

	fr.camera_info = cam_info
	fr.persp_info = perspective_info_from_camera_info(cam_info)

	#print(fr.persp_info.horizon_level)

	return fr


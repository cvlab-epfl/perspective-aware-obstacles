

from easydict import EasyDict
from road_anomaly_benchmark.datasets.dataset_registry import DatasetRegistry
from .cityscapes_pitch_angles import read_cam_info_laf, gen_perspective_scale_map, perspective_info_from_camera_info, invent_horizon_above_main_road
from ..a12_inpainting.sys_road_area import ModuleRegistry # road segmentation

# def load_perspective_info(**fields):
# 	#print('Load perspective scale', fields.keys(), fields['dset'])

# 	if 'perspective_scale_map' in fields:
# 		return

# 	dsname, dssplit = fields['dset_name'].split('-')

# 	if dsname == 'LostAndFound':
# 		cam_info = read_cam_info_laf(EasyDict(
# 			dir_root = dset.cfg.dir_root,
# 			split = dssplit,
# 		), EasyDict(fields))

# 		perspective_info = perspective_info_from_camera_info(cam_info)

# 	elif dsname == 'ObstacleTrack':
# 		sys_roadarea = ModuleRegistry.get('RoadAreaSystem', 'semcontour-roadwalk-v1')
# 		sys_roadarea.init_storage()

# 		roadarea = sys_roadarea.load_values(fields)
# 		road_mask = roadarea['labels_road_mask']
# 		cam_info, perspective_info = invent_horizon_above_main_road(road_mask)

# 	psm = gen_perspective_scale_map(
# 		fields['image'].shape[:2], 
# 		perspective_info.horizon_level, 
# 		perspective_info.pix_per_meter_slope,
# 	)

# 	return dict(
# 		camera_info = cam_info,
# 		persp_info = perspective_info,
# 		perspective_scale_map = psm,
# 	)

def pmap_loader_for_segmi_dset(dset):

	dsname, dssplit = dset.cfg.name_for_persistence.split('-')

	def make_output(cam_info, perspective_info, frame_size_hw):
		psm = gen_perspective_scale_map(
			frame_size_hw,
			perspective_info.horizon_level, 
			perspective_info.pix_per_meter_slope,
		)

		return EasyDict(
			camera_info = cam_info,
			persp_info = perspective_info,
			perspective_scale_map = psm,
		)

	if dsname == 'LostAndFound':

		def load_perspective_info_for_frame(**fields):
			cam_info = read_cam_info_laf(EasyDict(
				dir_root = dset.cfg.dir_root,
				split = dssplit,
			), EasyDict(fields))
			perspective_info = perspective_info_from_camera_info(cam_info)
			return make_output(
				cam_info, perspective_info, fields['image'].shape[:2], 
			)

	elif dsname == 'ObstacleTrack':
		sys_roadarea = ModuleRegistry.get('RoadAreaSystem', 'semcontour-roadwalk-v1')
		sys_roadarea.init_storage()

		def load_perspective_info_for_frame(**fields):
			roadarea = sys_roadarea.load_values(EasyDict(
				fid = fields['fid'],
				dset = dict(name = dsname, split = dssplit),
			))
			road_mask = roadarea['labels_road_mask']
			cam_info, perspective_info = invent_horizon_above_main_road(road_mask)
			return make_output(
				cam_info, perspective_info, fields['image'].shape[:2], 
			)
		
	return load_perspective_info_for_frame

	
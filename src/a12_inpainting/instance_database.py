
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import cv2 as cv
from easydict import EasyDict
from tqdm import tqdm
import h5py

from ..paths import DIR_DATA
from ..datasets.dataset import imread
from ..a11_instance_detector.bbox_transform import labels_to_bbox, bbox_calc_dimensions



def lift_class_area_to_instances(instance_map_crop, cls_id):

	cls_mask = instance_map_crop == cls_id

	num_components, component_map = cv.connectedComponents(cls_mask.astype(np.uint8), connectivity=4)

	instance_map_crop[cls_mask] = component_map[cls_mask] + (cls_id*1000)
	


def generate_instances_from_connected_components(instance_map, class_ids_to_lift):

	instance_map_lifted = instance_map.copy().astype(np.uint32)

	bbox_labels, bbox_coords = labels_to_bbox(instance_map)

	for cls_id in class_ids_to_lift:
		# find bounding box around the chosen class
		box_idx = np.searchsorted(bbox_labels, cls_id)
		class_present = bbox_labels[box_idx] == cls_id

		if class_present:
			tl_x, tl_y, br_x, br_y = bbox_coords[box_idx]

			lift_class_area_to_instances(
				instance_map_crop = instance_map_lifted[tl_y:br_y+1, tl_x:br_x+1],
				cls_id = cls_id,
			)

	return instance_map_lifted


def extract_instance_properties_from_label_map(instance_label_map):
	"""
	[inst id, bbox, pixel area]
	"""

	inst_ids, inst_bboxes = labels_to_bbox(instance_label_map)
	num_instances = inst_ids.__len__()
	inst_pixel_areas = np.zeros(num_instances, dtype=np.uint32)

	# get areas
	for idx, inst_id, inst_bbox in zip(range(num_instances), inst_ids, inst_bboxes):
		tl_x, tl_y, br_x, br_y = inst_bbox

		map_crop = instance_label_map[tl_y:br_y+1, tl_x:br_x+1]
		inst_pixel_areas[idx] = np.count_nonzero(map_crop == inst_id) # PERF: could result the np.unique from bbox

	return EasyDict(
		instance_ids = inst_ids,
		instance_bboxes = inst_bboxes,
		instance_pixel_areas = inst_pixel_areas,
	)


def instance_db__ingest_ctc_frame(fid_and_inst_path, class_ids_to_lift, **_):
	fid, inst_file_path = fid_and_inst_path
	inst_map = imread(inst_file_path)

	inst_map_lifted = generate_instances_from_connected_components(inst_map, class_ids_to_lift)
	inst_map_lifted[inst_map_lifted < 1000] = 0

	inst_info = extract_instance_properties_from_label_map(inst_map_lifted)
	inst_info.instance_classes = (inst_info.instance_ids // 1000).astype(np.uint8)

	return EasyDict(
		fid = fid,
		instance_map_lifted = inst_map_lifted.astype(np.uint16),
		**inst_info,
	)


def instance_db__process_ctc(dset, dir_out, cls_to_lift, frames=None, filename_prefix=''):
	"""
	cls_to_lift = [
		dset.label_info.name2id[name]
		for name in ['traffic light', 'traffic sign', 'pole']
	]

	instance_db__process_ctc(
		dset = dset,
		dir_out = DIR_DATA / '1203_instance_db' / f'{dset.name}-{dset.split}',
		filename_prefix = f'{dset.name}-{dset.split}_',
		#frames = dset.frames[:10],
		cls_to_lift = cls_to_lift,
	)
	"""
	dir_out = Path(dir_out)
	dir_out.mkdir(exist_ok=True, parents=True)
	frames = frames or dset.frames

	worker = partial(instance_db__ingest_ctc_frame, class_ids_to_lift=cls_to_lift)

	task_specs = [
		(fr.fid, fr.dset.path_for_channel('instances', fr.fid))
		for fr in frames
	]

	# FrameID index
	fids_all = [t[0] for t in task_specs]
	fids_all.sort()
	fid_to_index = {fid: i for i, fid in enumerate(fids_all)}
	
	inst_infos = []

	with h5py.File(dir_out / f'{filename_prefix}instance-maps-lifted.hdf5', 'w') as file_instance_maps:
		with Pool() as pool:
			for res in tqdm(pool.imap(worker, task_specs, chunksize=15), total=task_specs.__len__()):

				# write the instance map
				file_instance_maps.create_dataset(res.fid, data=res.instance_map_lifted, compression=8)
				
				del res['fid']
				res.frame_idx = np.full_like(res.instance_ids, fid_to_index[res.fid])
				inst_infos.append(res)

	num_instances = sum(r.instance_ids.__len__() for r in inst_infos)

	print(f'Processed {num_instances} instances in {inst_infos.__len__()} frames')

	with h5py.File(dir_out / f'{filename_prefix}instance-index.hdf5', 'w') as file_inst_info:
		g_fname = file_inst_info.create_group('frame_names')
		for i, fid in enumerate(fids_all):
			g_fname.attrs[fid] = i

		for name in ['instance_ids', 'instance_classes', 'frame_idx', 'instance_bboxes', 'instance_pixel_areas']:
			file_inst_info.create_dataset(
				name,
				data = np.concatenate([r[name] for r in inst_infos]),
				compression = 5,
			)



# def instance_db__load(index_file_path):
# 	index_file_path = Path(index_file_path)


# 	with h5py.File(index_file_path, 'r') as file_inst_info:

# 		inst_info = EasyDict({
# 			name: file_inst_info[name][:] 
# 			for name in ['instance_ids', 'instance_classes', 'frame_idx', 'instance_bboxes', 'instance_pixel_areas']
# 		})

# 		fid_to_idx = dict(file_inst_info['frame_names'].attrs)
# 		idx_to_fid = [None]*(max(fid_to_idx.values())+1)
# 		for fid, idx in fid_to_idx.items():
# 			idx_to_fid[idx] = fid
		
# 		inst_info.frame_names = idx_to_fid
	
# 		return inst_info

def instance_db__read_lifted_instances(fid,  inst_map_file_path):

	with h5py.File(inst_map_file_path, 'r') as file_instance_maps:
		return file_instance_maps[fid][:]


class InstDb:

	#def __init__(self, index_file_path, inst_map_file_path):

	def __init__(self, dset, margin=0):
		self.set_margin(margin)

		if isinstance(dset, str):
			from .demo_case_selection import DatasetRegistry
			dset_key = dset
			dset_obj = DatasetRegistry.get_implementation(dset_key)
		else:
			dset_key = f'{dset.name}-{dset.split}'
			dset_obj = dset


		dir_out = DIR_DATA / '1203_instance_db' / dset_key
		file_path_inst = dir_out / f'{dset_key}_instance-index.hdf5'
		file_path_maps = dir_out / f'{dset_key}_instance-maps-lifted.hdf5'

		self.dset = dset_obj
		self.index_file_path = Path(file_path_inst)
		self.inst_map_file_path = Path(file_path_maps)

		with h5py.File(self.index_file_path, 'r') as file_inst_info:

			for name in ['instance_ids', 'instance_classes', 'frame_idx', 'instance_bboxes', 'instance_pixel_areas']:
				setattr(self, name, file_inst_info[name][:])

			fid_to_idx = dict(file_inst_info['frame_names'].attrs)
			idx_to_fid = [None]*(max(fid_to_idx.values())+1)
			for fid, idx in fid_to_idx.items():
				idx_to_fid[idx] = fid
			
			self.frame_names = idx_to_fid

	def set_margin(self, margin):
		self.margin = margin

	def __len__(self):
		return self.instance_ids.__len__()

	
	def fid_to_inst_map(self, fid):
		return instance_db__read_lifted_instances(fid, self.inst_map_file_path)

	def fid_to_image(self, fid):
		return self.dset[fid].image


	def __getitem__(self, inst_idx_in_db):

		inst_id = self.instance_ids[inst_idx_in_db]
		inst_bbox = self.instance_bboxes[inst_idx_in_db]
		inst_class = self.instance_classes[inst_idx_in_db]

		fid = self.frame_names[self.frame_idx[inst_idx_in_db]]
		instance_map = self.fid_to_inst_map(fid)
		image = self.fid_to_image(fid)

		# bounding box with margin
		img_h, img_w = instance_map.shape
		tl_x, tl_y, br_x, br_y = inst_bbox

		tl_x = max(0, tl_x - self.margin)
		tl_y = max(0, tl_y - self.margin)
		br_x = min(img_w-1, br_x + self.margin)
		br_y = min(img_h-1, br_y + self.margin)

		slc = (slice(tl_y, br_y+1), slice(tl_x, br_x+1))

		# crop and calculate mask
		image_crop = image[slc]
		map_crop = instance_map[slc]
		mask = map_crop == inst_id

		return EasyDict(
			image_crop = image_crop, 
			mask = mask,
			instance_id = inst_id,
			instance_class = inst_class,
			pixel_area = self.instance_pixel_areas[inst_idx_in_db],
		)

	@staticmethod
	def filter_obstacle_instances(inst_db, dim_min_max = [10, 150], area_min_max=[100, 5000]):
		bbox_dims = bbox_calc_dimensions(inst_db.instance_bboxes)
		
		max_dim = np.maximum(bbox_dims.width, bbox_dims.height)
		min_dim = np.minimum(bbox_dims.width, bbox_dims.height)
		
		cond_area = (area_min_max[0] < inst_db.instance_pixel_areas) & (inst_db.instance_pixel_areas < area_min_max[1])
		
		cond_dim = (dim_min_max[0] < min_dim) & (max_dim < dim_min_max[1])
		
		cond_all = cond_area & cond_dim
		
		filtered_obstacle_ids = np.where(cond_all)[0]
		
		print(f'filtered obstacles {filtered_obstacle_ids.__len__()} / {inst_db.instance_bboxes.shape[0]}')
		
		return filtered_obstacle_ids

	def filter_obstacles_and_store(self, dim_min_max, area_min_max):

		self.filtered_obstacle_ids = self.filter_obstacle_instances(
			self, dim_min_max=dim_min_max, area_min_max=area_min_max,
		)

		bbox_dims = bbox_calc_dimensions(self.instance_bboxes)
		
		# avgsize = average of width, height, sqrt(area)
		
		self.filtered_obstacles_avgsize = (bbox_dims.width[self.filtered_obstacle_ids] 
		+ bbox_dims.height[self.filtered_obstacle_ids] 
		+ np.sqrt(self.instance_pixel_areas[self.filtered_obstacle_ids])) * (1./3.)
		#self.filtered_obstacles_avgsize = np.sqrt(self.instance_pixel_areas[self.filtered_obstacle_ids])
				
		filtered_obstacles_avgsize_argsort = np.argsort(self.filtered_obstacles_avgsize)

		self.filtered_obstacles_avgsize_sorted = self.filtered_obstacles_avgsize[filtered_obstacles_avgsize_argsort]
		self.filtered_obstacle_sorted_to_main_index = self.filtered_obstacle_ids[filtered_obstacles_avgsize_argsort]

	def random_filtered_obstacle_in_size_range(self, sz_min, sz_max, min_num_candidates = 1):
		left = np.searchsorted(self.filtered_obstacles_avgsize_sorted, sz_min)
		right = np.searchsorted(self.filtered_obstacles_avgsize_sorted, sz_max)

		#print(f'sz {sz_min:0.1f}-{sz_max:0.1f} range {left}-{right} / {self.filtered_obstacle_ids.__len__()}')

		num_obj_total = self.filtered_obstacles_avgsize_sorted.__len__()

		if right - left < min_num_candidates:
			candidate_deficit = min_num_candidates - (right - left) 
			right = min(right + candidate_deficit // 2, num_obj_total)
			left = max(left - candidate_deficit // 2, 0)

		if left == num_obj_total:
			left -= 1

		i = np.random.randint(left, right-1)
		i = self.filtered_obstacle_sorted_to_main_index[i]
		return self[i]




# def instance_db__get_instance(inst_idx_in_db,  inst_infos, fid_to_inst_map, fid_to_image, ):

# 	fid = inst_infos.frame_names[inst_infos.frame_idx[inst_idx_in_db]]

# 	instance_map = fid_to_inst_map(fid)
# 	image = fid_to_image(fid)

# 	bbox = inst_infos.instance_bboxes[inst_idx_in_db]
# 	tl_x, tl_y, br_x, br_y = bbox

# 	slc = (slice(tl_y, br_y+1), slice(tl_x, br_x+1))
# 	#slc = (slice(tl_y-50, br_y+51), slice(tl_x-50, br_x+51)) # to view

# 	image_crop = image[slc]
# 	map_crop = instance_map[slc]

# 	mask = map_crop == inst_infos.instance_ids[inst_idx_in_db]

# 	return EasyDict(
# 		image_crop = image_crop, 
# 		mask = mask,
# 	)



#from ..paths import DIR_DATA

# def instance_db__loader_for_dset(dset):
# 	dir_out = DIR_DATA / '1203_instance_db' / f'{dset.name}-{dset.split}'

# 	file_path_inst = dir_out / f'{dset.name}-{dset.split}_instance-index.hdf5'
# 	file_path_maps = dir_out / f'{dset.name}-{dset.split}_instance-maps-lifted.hdf5'
 
# 	inst_db = instance_db__load(file_path_inst)

# 	inst_loader = partial(instance_db__get_instance,
# 		inst_infos = inst_db,
# 		fid_to_inst_map = partial(
# 			instance_db__read_lifted_instances,
# 			inst_map_file_path=file_path_maps,
# 		),
# 		fid_to_image = lambda fid: dset[fid].image,
# 	)

# 	return EasyDict(
# 		inst_db = inst_db,
# 		inst_loader = inst_loader,
# 	)

# TODO inst db class
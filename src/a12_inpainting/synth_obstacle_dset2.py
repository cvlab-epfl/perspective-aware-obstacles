
from src.common.jupyter_show_image import adapt_img_data
from pathlib import Path
from operator import itemgetter
import multiprocessing, json
from functools import partial

import cv2 as cv
import numpy as np
import h5py
from tqdm import tqdm
from easydict import EasyDict

from .file_io import imread, imwrite
from .grid_randomizer import square_grid_randomizer, generate_floor_grid_on_image
from ..common.registry import ModuleRegistry2

from ..paths import DIR_DATA
from ..common.multiprocess_with_init import run_parallel
from ..pipeline.config import extend_config
from ..a14_perspective.cityscapes_pitch_angles import gen_perspective_scale_map



class ObjectStatistics:
	"""
	Records the dimensions (width, height, area) and injection placement of objects
	"""

	FIELDS_FLOAT = ['area', 'width', 'height', 'position_x', 'position_y', 'perspective_scale']
	FIELDS_INT = ['semantic_class', 'frame_idx']
	FIELDS = FIELDS_FLOAT + FIELDS_INT

	def __init__(self):
		self.values = {
			k: []
			for k in self.FIELDS
		}

	def add_obstacle(self, position, mask, semantic_class, perspective_scale=-1):
		entry = dict(
			area = np.count_nonzero(mask),
			width = np.count_nonzero(np.any(mask, axis=1)),
			height = np.count_nonzero(np.any(mask, axis=0)),
			position_x = position[0], 
			position_y = position[1],
			semantic_class = semantic_class,
			perspective_scale = perspective_scale,
		)

		for k, v in entry.items():
			self.values[k].append(v)
		

	def finalize_frame(self, frame_idx):
		self.values = {
			k: np.array(v, dtype=np.int32 if k in self.FIELDS_INT else np.float32)
			for k, v in self.values.items()
		}
		if self.values['frame_idx'].__len__() == 0:
			self.values['frame_idx'] = np.full(self.values['area'].__len__(), frame_idx, dtype=np.int32)


	@classmethod
	def union(cls, stat_objects):
		unified = cls()

		for k in cls.FIELDS:
			unified.values[k] = np.concatenate([
				stat.values[k] for stat in stat_objects
			], axis=0)

		return unified

	def save(self, path):
		with h5py.File(path, 'w') as fh:
			for k in self.FIELDS:
				fh[k] = self.values[k]

	@classmethod
	def load(cls, path):
		unified = cls()
		with h5py.File(path, 'r') as fh:
			for k in cls.FIELDS:
				unified.values[k] = fh[k][:]
		return unified
		



# class DsetSynthObstacles_Top:
# 	...

# 	# variant - perspective 1


# class DsetSynthObstacles_PlacementGenerator:
# 	...

# 	# submod - obstacle choice
# 	# submod - obstacle transform and insertion
# 	# submod - placement grid


# class DsetSynthObstacles_Fusion:
# 	...


# class DsetSynthObstacles_PatchSampler:
# 	...
# 	# generate
# 	# sample


@ModuleRegistry2('1230_SynthObstacle_Placement')
class DsetSynthObstacles_Placement:
	# TODO split generator into a module, since we prefix with gen__

	@classmethod
	def configs(cls):

		dset_variants = [
			dict(
				background_dataset = f'cityscapes-{split}',
				road = dict(
					road_like_class_ids = [7],
				),
			)
			for split in ['train', 'val']
		] + [
			dict(
				background_dataset = f'IndiaDriving-{split}',
				road = dict(
					road_like_class_ids = [0, 1, 2, 3],
				),
			)
			for split in ['train', 'val']
		]		

		cfgs = []
		for var in dset_variants:
			bg_dset = var['background_dataset']
			bg_name, bg_split = bg_dset.split('-')
			
			opt = EasyDict(extend_config(var, dict(
				split = bg_split,
				#background_dataset = bg_dset,
				obstacles = dict(
					# For now using Cityscapes instances everywhere
					database = f'cityscapes-{bg_split}', 
					margin = 11,
					filter = dict(
						# using the big-object by default
						dim_min_max = [10, 250], 
						area_min_max = [100, 35e3],
						# small objects:
						# dim_min_max = [10, 150], 
						# area_min_max = [100, 5000],
					),
					
				),
			)))

			# the C variant is for re-generating the dset without overriding the one used for previous experiments
			# in order to measure the object statistics
			for name in [f'v2b_{bg_dset}', f'v2c_{bg_dset}']:
				cfgs.append(EasyDict(
					name = name,
					display_name = 'Uniform',
					storage_dir = f'1230_SynthObstacle/placement_{name}',
					# perspective = None,
					perspective = True,
					
					injection = dict(
						obstacle_choice = 'uniform',
						obstacle_anchor = 'center',

						grid_type = 'uniform',

						grid_step = 150,
						item_radius = 75,
						random_offset = 60,
						injection_probability_per_slot = 0.2,
					),
					**opt,
				))

			# name = f'v3persp1_{bg_dset}'
			# cfgs.append(EasyDict(
			# 	name = name,
			# 	storage_dir = f'1230_SynthObstacle/placement_{name}',
			# 	perspective = 'select',
			# 	**opt,
			# ))

			name = f'v3persp2_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_scale',
					obstacle_anchor = 'center',

					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 2.,
					random_offset = 0.5,
					injection_probability_per_slot = 0.15,
					obs_size_mean_std = [0.5, 0.15]
				),
				**opt,
			))

			name = f'v3persp2_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_scale',
					obstacle_anchor = 'center',

					grid_type = 'ground',

					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 2.,
					random_offset = 0.5,
					injection_probability_per_slot = 0.15,
					obs_size_mean_std = [0.5, 0.15]
				),
				**opt,
			))
	


			name = f'v3persp3A_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_scale',
					obstacle_anchor = 'center',

					grid_type = 'ground',
					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 0.5,
					random_offset = 0.2,
					injection_probability_per_slot = 0.15,
					obs_size_mean_std = [0.5, 0.15]
				),
				**opt,
			))

			name = f'v3persp3B_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_scale',
					obstacle_anchor = 'center',

					grid_type = 'uniform',

					grid_step = 150,
					item_radius = 75,
					random_offset = 60,
					injection_probability_per_slot = 0.2,
					
					obs_size_mean_std = [0.5, 0.23]
				),
				**opt,
			))

			name = f'v3persp3C_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_select',
					obstacle_anchor = 'center',

					grid_type = 'uniform',

					grid_step = 150,
					item_radius = 75,
					random_offset = 60,
					injection_probability_per_slot = 0.2,
					
					obs_size_mean_std = [0.4, 0.15]
				),
				**opt,
			))

			name = f'v3persp3D_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				display_name = 'Perspective-aware size',
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_select',
					obstacle_anchor = 'center',
					grid_type = 'ground',
					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 0.25,
					random_offset = 0.5,
					injection_probability_per_slot = 0.12,
					
					obs_size_mean_std = [0.4, 0.15],
				),
				**opt,
			))

			size_variants = {
				'02': [0.2, 0.1],
				'05': [0.55, 0.2],
				'07': [0.7, 0.2],
				'10': [1.0, 0.25],
				'6w': [0.675, 0.575],
			}

			for sz_name, obj_size in size_variants.items():
				name = f'v3Dsz{sz_name}_{bg_dset}'
				c1 = EasyDict(
					name = name,
					display_name = 'Perspective-aware size',
					storage_dir = f'1230_SynthObstacle/placement_{name}',
					perspective = True,
					injection = dict(
						obstacle_choice = 'perspective_select',
						obstacle_anchor = 'center',
						grid_type = 'ground',
						grid_gap_m = [1, 3.5],
						grid_size_xy = [12, 9],
						first_offset = 0.25,
						random_offset = 0.5,
						injection_probability_per_slot = 0.12,
						
						obs_size_mean_std = obj_size,
					),
					**opt,
				)
				cfgs.append(c1)

				c2 = EasyDict(c1)
				c2.name = f'v3Dsz{sz_name}-attnSETR_{bg_dset}'
				c2.extra_attention = 'SETR'
				cfgs.append(c2)


			name = f'v3p5bs_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				display_name = 'Perspective-aware size',
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_select',
					obstacle_anchor = 'base',
					grid_type = 'ground',
					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 0.25,
					random_offset = 0.5,
					injection_probability_per_slot = 0.12,
					
					obs_size_mean_std = [0.4, 0.15]
				),
				**opt,
			))

			name = f'v3p4sc_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				display_name = 'Perspective-aware size',
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_scale',
					obstacle_anchor = 'center',
					grid_type = 'ground',
					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 0.25,
					random_offset = 0.5,
					injection_probability_per_slot = 0.12,
					
					obs_size_mean_std = [0.4, 0.15]
				),
				**opt,
			))


			name = f'v3persp3E_{bg_dset}'
			cfgs.append(EasyDict(
				name = name,
				display_name = 'Perspective-aware size',
				storage_dir = f'1230_SynthObstacle/placement_{name}',
				perspective = True,
				injection = dict(
					obstacle_choice = 'perspective_select',
					grid_type = 'ground',
					grid_gap_m = [1, 3.5],
					grid_size_xy = [12, 9],
					first_offset = 0.25,
					random_offset = 0.5,
					injection_probability_per_slot = 0.12,
					
					obs_size_mean_std = [0.7, 0.3]
				),
				**opt,
			))

		return cfgs
	
	
	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.dir_dataset = DIR_DATA / self.cfg.storage_dir
	
	@property
	def split(self):
		return self.cfg.split

	def img_path(self, channel, frame_id):
		return self.dir_dataset / 'images' / f'img{frame_id:04d}_{channel}.webp'
	
	def gen__prepare_instdb(self):
		inst_db_name = self.cfg.obstacles.database

		# TODO inst db get_implementation
		from src.a12_inpainting.instance_database import InstDb
		
		idb = InstDb(inst_db_name, margin=self.cfg.obstacles.margin)
		idb.filter_obstacles_and_store(**self.cfg.obstacles.filter)
		return idb
		
	def gen__prepare_dataset(self):
		from src.a12_inpainting.demo_case_selection import DatasetRegistry
		return DatasetRegistry.get_implementation(self.cfg.background_dataset)

	def gen__prepare(self):
		""" init components needed to generate the dataset 
		* database of instances to inject
		"""
		self.background_dataset = self.gen__prepare_dataset()
		self.instance_db = self.gen__prepare_instdb()
	

	@staticmethod
	def gen__inject_obstacle_into_image(image_margin_ref, image_bb_ref, instmap_ref, obstacle_image, obstacle_mask, obstacle_id_for_map, obstacle_center_pt, margin=0):
		obstacle_h, obstacle_w = obstacle_mask.shape
		obstacle_size_xy = np.array([obstacle_w, obstacle_h], dtype=np.int32)
		image_h, image_w = instmap_ref.shape
		image_size_xy = np.array([image_w, image_h], dtype=np.int32)

		obstacle_tl_xy = obstacle_center_pt.astype(np.int32) - obstacle_size_xy // 2
		obstacle_br_xy = obstacle_tl_xy + obstacle_size_xy
		
		overdraft_tl = -np.minimum(obstacle_tl_xy, 0)
		overdraft_br = np.maximum(obstacle_br_xy - image_size_xy, 0)

		obstacle_tl_xy += overdraft_tl
		obstacle_br_xy -= overdraft_br

		crop_slice = (slice(obstacle_tl_xy[1], obstacle_br_xy[1]), slice(obstacle_tl_xy[0], obstacle_br_xy[0]))

		source_slice = (slice(overdraft_tl[1], obstacle_h - overdraft_br[1]), slice(overdraft_tl[0], obstacle_w - overdraft_br[0]))

		#print(source_slice)

		obstacle_image = obstacle_image[source_slice]
		obstacle_mask = obstacle_mask[source_slice]


		# labels
		map_crop = instmap_ref[crop_slice]	
		map_crop[obstacle_mask] = obstacle_id_for_map
		
		# image bb
		if image_bb_ref is not None:
			image_bb_ref[crop_slice] = obstacle_image

		# image masked
		if image_margin_ref is not None:
			mask_with_margin = cv.dilate(
				obstacle_mask.astype(np.uint8), 
				kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (margin*2, margin*2)),
			).astype(bool)
			
			image_margin_ref[crop_slice][mask_with_margin] = obstacle_image[mask_with_margin]

	def gen__get_random_obstacle_uniform(self, perspective_info=None, obstacle_position_xy=None):
		idb = self.instance_db
		idx_in_db = idb.filtered_obstacle_ids[np.random.randint(idb.filtered_obstacle_ids.__len__())]
		obstacle_info = idb[idx_in_db]

	
		if perspective_info is not None and obstacle_position_xy is not None:
			obs_y = obstacle_position_xy[1]
			below_horiz = obs_y - perspective_info.horizon_level
			pix_per_m = perspective_info.pix_per_meter_slope * below_horiz
			obstacle_info.perspective_scale = pix_per_m

		return obstacle_info
					
	def gen__produce_frame_with_obstacles_uniform(self, image_data, labels_road_mask, cfg=None, f_get_random_obstacle=None):

		f_get_random_obstacle = f_get_random_obstacle or self.gen__get_random_obstacle_uniform
		cfg = cfg or self.cfg		
	
		# extract image and road mask
		#image_data = fr.image
		img_h, img_w, _ = image_data.shape
		#labels_road_mask = fr.labels_source == 7

		if cfg.injection.grid_type == 'uniform':
			# road area
			tl_x, tl_y, bb_w, bb_h = cv.boundingRect(labels_road_mask.astype(np.uint8))
			# reduce by the margin of 5, which is also used by inpaining
			tl_x += 5
			tl_y += 5
			bb_w -= 10
			bb_h -= 10
			
			obstacle_grid_centers = square_grid_randomizer(
				size_xy = [bb_w, bb_h], 
				grid_step = cfg.injection.grid_step, 
				item_radius = cfg.injection.item_radius, 
				random_offset = cfg.injection.random_offset,
			)

		else:
			raise NotImplementedError(f"Grid type {cfg.injection.grid_type}")


		# grid centers to full image coords
		obstacle_grid_centers += np.array([tl_x, tl_y])
		obstacle_grid_centers = np.rint(obstacle_grid_centers).astype(np.int32)
		num_obstacle_slots = obstacle_grid_centers.__len__()
		
		# discard grid positions outside of road


		# choose which slots are occupied
		obstacle_coin_flips = np.random.uniform(0, 1, size=num_obstacle_slots) < cfg.injection.injection_probability_per_slot
		
		obstacle_grid_centers = obstacle_grid_centers[obstacle_coin_flips]
		num_obstacle_slots = obstacle_grid_centers.__len__()
		
		# data for the generated frame
		obstacle_instance_map = np.zeros_like(labels_road_mask, dtype=np.uint16)
		image_with_margin = np.zeros((img_h, img_w, 3), dtype=np.uint8)
		image_with_bb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
				
		obstacle_classes = [0]
		inst_id = 1
		
		object_stats = ObjectStatistics()

		for grid_pos in obstacle_grid_centers:
			obstacle_info = f_get_random_obstacle(obstacle_position_xy = grid_pos)
			
			if obstacle_info is not None:
				#try:
				self.gen__inject_obstacle_into_image(
					image_margin_ref = image_with_margin, 
					image_bb_ref = image_with_bb,
					instmap_ref = obstacle_instance_map, 
					obstacle_image = obstacle_info.image_crop, 
					obstacle_mask = obstacle_info.mask, 
					obstacle_id_for_map = inst_id, 
					obstacle_center_pt = grid_pos,
					margin = self.cfg.obstacles.margin,
				)
				
				inst_id += 1
				obstacle_classes.append(obstacle_info.instance_class)

				object_stats.add_obstacle(
					position = grid_pos, 
					mask = obstacle_info.mask,
					semantic_class = obstacle_info.instance_class,
				)

				#except IndexError as e:
				#	print('Failed to inject obstacle:', e)
		
		obstacle_classes = np.array(obstacle_classes, dtype=np.uint16)

		# TODO constrain obstacle_instance_map to inpainted area
		
		return EasyDict(
			image_background = image_data,
			image_injections_with_margin = image_with_margin,
			image_injections_with_bb = image_with_bb,
			obstacle_instance_map = obstacle_instance_map,
			road_mask = labels_road_mask,
			#inpainting_mask = inpainting_result.area_mask_without_margin,
			obstacle_classes = obstacle_classes,
			object_stats = object_stats,
		)


	def gen__get_random_obstacle_perspective_select(self, perspective_info, obstacle_position_xy, obs_size_mean_std):
		"""
		- Determine the desired obstacle size at given point
		- Choose from InstanceDB only obstacles within size range
		- No scaling
		"""
		idb = self.instance_db

		obs_y = obstacle_position_xy[1]
		below_horiz = obs_y - perspective_info.horizon_level
		if below_horiz <= 0:
			print('Obstacle y={obs_y} above horizon ({perspective_info.horizon_level})')
			return None


		pix_per_m = perspective_info.pix_per_meter_slope * below_horiz
		#print(f'y={obs_y:0.1f} pix_per_m={pix_per_m:0.1f}')

		mean, std = obs_size_mean_std

		obstacle_info = idb.random_filtered_obstacle_in_size_range(
			pix_per_m * (mean - std*1.5), 
			pix_per_m * (mean + std*1.5),
			min_num_candidates = 16,
		)
		obstacle_info.perspective_scale = pix_per_m
		return obstacle_info

	def gen__get_random_obstacle_perspective_scale(self, perspective_info, obstacle_position_xy, obs_size_mean_std):
		"""
		- Determine the desired obstacle size at given point
		- Choose uniform random obstacle from whole InstanceDB 
		- Scale to desired size
		"""
		idb = self.instance_db

		# perspective size at location
		obs_y = obstacle_position_xy[1]
		below_horiz = obs_y - perspective_info.horizon_level
		if below_horiz <= 0:
			print('Obstacle y={obs_y} above horizon ({perspective_info.horizon_level})')
			return None

		pix_per_m = perspective_info.pix_per_meter_slope * below_horiz
		# print(f'y={obs_y:0.1f} pix_per_m={pix_per_m:0.1f}')

		# get uniformly random obstacle
		idb = self.instance_db
		idx_in_db = idb.filtered_obstacle_ids[np.random.randint(idb.filtered_obstacle_ids.__len__())]
		obstacle_info = idb[idx_in_db]

		# resize obstacle
		obs_m, obs_std = obs_size_mean_std
		desired_size = np.clip(np.random.normal(loc=obs_m, scale=obs_std), 0.2, 10) * pix_per_m
		obstacle_orig_size = np.max(obstacle_info.mask.shape) - 2 * idb.margin
		scale = desired_size / obstacle_orig_size
		# print(f'obstacle sz {obstacle_orig_size} -> scale {scale}')

		# resize image
		# area for downscaling, cubic for upscaling
		interp = cv.INTER_CUBIC if scale >= 1 else cv.INTER_AREA
		obstacle_info.image_crop = cv.resize(
			obstacle_info.image_crop, 
			(0, 0), 
			fx=scale, fy=scale, 
			interpolation=interp,
		)
		# mask - nearest interpolation
		obstacle_info.mask = cv.resize(
			obstacle_info.mask.astype(np.uint8),
			(0, 0), 
			fx=scale, fy=scale, 
			interpolation = cv.INTER_NEAREST,
		).astype(bool)

		obstacle_info.perspective_scale = pix_per_m
		
		return obstacle_info

	def gen__produce_frame_with_obstacles_perspective(self, image_data, labels_road_mask, perspective_info, cfg=None):
		cfg = cfg or self.cfg		
		cfgi = cfg.injection

		choice = cfgi.get('obstacle_choice', 'uniform')

		if choice == 'uniform':
			f_get_random_obstacle = partial(self.gen__get_random_obstacle_uniform, perspective_info = perspective_info)

		elif choice == 'perspective_select':
			f_get_random_obstacle = partial(
				self.gen__get_random_obstacle_perspective_select, 
				perspective_info = perspective_info,
				obs_size_mean_std = cfg.injection.obs_size_mean_std,
			)

		elif choice == 'perspective_scale':
			f_get_random_obstacle = partial(
				self.gen__get_random_obstacle_perspective_scale, 
				perspective_info = perspective_info,
				obs_size_mean_std = cfg.injection.obs_size_mean_std,
			)
		else:
			raise NotImplementedError(f"Obstacle choice method {choice}")

		# perspective grid points
		img_h, img_w = labels_road_mask.shape

		if cfg.injection.grid_type == 'uniform':

			tl_x, tl_y, bb_w, bb_h = cv.boundingRect(labels_road_mask.astype(np.uint8))

			# force box below horizon
			# if top-y has moved, adjust bbox size
			bottom_y = tl_y + bb_h
			tl_y = max(tl_y, perspective_info.horizon_level + 25)
			bb_h = bottom_y - tl_y

			# reduce by the margin of 5, which is also used by inpaining
			tl_x += 5
			tl_y += 5 
			bb_w -= 10
			bb_h -= 10
			
			grid_pts = square_grid_randomizer(
				size_xy = [bb_w, bb_h], 
				grid_step = cfg.injection.grid_step, 
				item_radius = cfg.injection.item_radius, 
				random_offset = cfg.injection.random_offset,
			)
			# grid centers to full image coords
			grid_pts += np.array([tl_x, tl_y])

		elif cfg.injection.grid_type == 'ground':

			grid_pts = generate_floor_grid_on_image(
				perspective_info = perspective_info,
				image_hw = (img_h, img_w),
				grid_gap_m = cfg.injection.grid_gap_m,
				grid_size_xy = cfg.injection.grid_size_xy,
				first_offset = cfg.injection.first_offset,
				random_offset = cfg.injection.random_offset,
				constrain = True,
			)

		else:
			raise NotImplementedError(f"Grid type {cfg.injection.grid_type}")

		# choose only points on the road
		
		grid_pts_int = np.rint(grid_pts).astype(np.int32)
		mask_pts_onroad = labels_road_mask[grid_pts_int[:, 1], grid_pts_int[:, 0]]
		# coin flips
		obstacle_coin_flips = np.random.uniform(0, 1, size=grid_pts_int.__len__()) < cfg.injection.injection_probability_per_slot
		# choose points
		mask_pts_chosen = mask_pts_onroad & obstacle_coin_flips
		grid_pts_int = grid_pts_int[mask_pts_chosen, :]

		# data for the generated frame
		obstacle_instance_map = np.zeros_like(labels_road_mask, dtype=np.uint16)
		image_with_margin = np.zeros((img_h, img_w, 3), dtype=np.uint8)
		image_with_bb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
				
		obstacle_classes = [0]
		object_stats = ObjectStatistics()
		inst_id = 1
		
		for grid_pos in grid_pts_int:
			obstacle_info = f_get_random_obstacle(
				obstacle_position_xy = grid_pos, 
			)
			
			if cfgi.obstacle_anchor == 'center':
				obstacle_center = grid_pos			
			elif cfgi.obstacle_anchor == 'base':
				obstacle_height = obstacle_info.mask.shape[0]
				obstacle_center = grid_pos + [0, obstacle_height // 2] 
			else:
				raise NotImplementedError(f'Injection anchor strategy, obstacle_anchor={cfgi.obstacle_anchor}')

			if obstacle_info is not None:
				#try:
				self.gen__inject_obstacle_into_image(
					image_margin_ref = image_with_margin, 
					image_bb_ref = image_with_bb,
					instmap_ref = obstacle_instance_map, 
					obstacle_image = obstacle_info.image_crop, 
					obstacle_mask = obstacle_info.mask, 
					obstacle_id_for_map = inst_id, 
					obstacle_center_pt = obstacle_center,
					margin = self.cfg.obstacles.margin,
				)
				
				inst_id += 1
				obstacle_classes.append(obstacle_info.instance_class)

				object_stats.add_obstacle(
					position = obstacle_center, 
					mask = obstacle_info.mask,
					semantic_class=obstacle_info.instance_class,
					perspective_scale = obstacle_info.get('perspective_scale', -1)
				)
					
				# except IndexError as e:
				# 	print('Failed to inject obstacle:', e)
		
		obstacle_classes = np.array(obstacle_classes, dtype=np.uint16)
		
		# TODO constrain obstacle_instance_map to inpainted area
		
		return EasyDict(
			image_background = image_data,
			image_injections_with_margin = image_with_margin,
			image_injections_with_bb = image_with_bb,
			obstacle_instance_map = obstacle_instance_map,
			road_mask = labels_road_mask,
			obstacle_classes = obstacle_classes,
			perspective_info = perspective_info,
			object_stats = object_stats,
		)
		


	def gen__process_frame_ctc(self, frame_id, fid, image, labels_source, dset=None, **_):

		np.random.seed(np.uint32(frame_id*frame_id*74 + frame_id*419 + 823))

		# TODO get range of road mask from cfg

		road_like_ids = self.cfg.road.road_like_class_ids
		labels_road_mask = labels_source == road_like_ids[0]
		for clsid in road_like_ids[1:]:
			labels_road_mask |= labels_source == clsid

		if self.cfg.perspective:
			from ..a14_perspective.cityscapes_pitch_angles import read_cam_info, perspective_info_from_camera_info
			from ..a14_perspective.cityscapes_pitch_angles import draw_perspective_markers
			cam_info = read_cam_info(dset, fid)
			perspective_info = perspective_info_from_camera_info(cam_info)

			#image = draw_perspective_markers(image.copy(), perspective_info)

			gen_result = self.gen__produce_frame_with_obstacles_perspective(
				image_data = image,
				labels_road_mask = labels_road_mask,
				perspective_info = perspective_info,
			)

		else:
			gen_result = self.gen__produce_frame_with_obstacles_uniform(
				image_data = image,
				labels_road_mask = labels_road_mask,
			)
		
		gen_result.frame_id = frame_id
		gen_result.frame_id_str = f'{frame_id:05d}'
		gen_result.object_stats.finalize_frame(frame_id)

		gen_result.update(
			source_frame_id = fid,
		)

		self.gen__write_frame_image(gen_result)
		
		# remove images so we don't send them over IPC
		# list to avoid "dictionary changed size during iteration"
		for k in list(gen_result.keys()):
			if k.startswith('image') and not 'path' in k:
				del gen_result[k]
		
		return gen_result
	
	def gen__write_frame_image(self, fr_data):
		fr_data.image_path_injections_with_margin = self.img_path('image_injections_with_margin', fr_data.frame_id)
		imwrite(fr_data.image_path_injections_with_margin, fr_data.image_injections_with_margin)

		fr_data.image_path_injections_with_bb = self.img_path('image_injections_with_bb', fr_data.frame_id)
		imwrite(fr_data.image_path_injections_with_bb, fr_data.image_injections_with_bb)

	def gen__write_frame_data(self, fr_data, hdf_file):
		g = hdf_file.create_group(fr_data.frame_id_str)
		g.attrs['source_frame_id'] = fr_data.source_frame_id
		g.create_dataset('road_mask', data=fr_data.road_mask.astype(bool), compression=3)
		#g.create_dataset('inpainting_mask', data=fr_data.inpainting_mask.astype(bool), compression=3)
		g.create_dataset('obstacle_instance_map', data=fr_data.obstacle_instance_map.astype(np.uint16), compression=8)
		g.create_dataset('obstacle_classes', data=fr_data.obstacle_classes.astype(np.uint16))

		if self.cfg.perspective:
			pi = fr_data.perspective_info
			gp = g.create_group('perspective')
			gp.attrs['horizon_level'] = pi.horizon_level
			gp.attrs['pix_per_meter_slope'] = pi.pix_per_meter_slope
			gp.attrs['pix_per_meter_at_midpoint'] = pi.pix_per_meter_at_midpoint
			gp.create_dataset('midpoint', data=np.array(pi.midpoint, dtype=np.float32))
			gp.create_dataset('cam_matrix', data=pi.cam_matrix)
			gp.create_dataset('extrinsic_matrix', data=pi.extrinsic_matrix)

	def gen__worker(self, task_queue : multiprocessing.Queue, solution_queue : multiprocessing.Queue):
		print('process started')
		
		self.gen__prepare()

		import os
		np.random.seed(os.getpid())

		while not task_queue.empty():
			frame_idx = task_queue.get()
			frame = self.background_dataset[frame_idx]

			result = self.gen__process_frame_ctc(frame_id = frame_idx, **frame)

			solution_queue.put(result)

	def gen__run(self, num_workers=4):
		background_dataset = self.gen__prepare_dataset()
		num_tasks = background_dataset.__len__()
		tasks = range(num_tasks)
		
		json_frames = []

		self.dir_dataset.mkdir(exist_ok=True, parents=True)
		(self.dir_dataset / 'images').mkdir(exist_ok=True, parents=True)

		task_queue = multiprocessing.Queue()
		for i in range(num_tasks):
			task_queue.put(i)

		print('qsize', task_queue.qsize())
			
		solution_queue = multiprocessing.Queue()

		worker_kwargs = dict(
			#frame_sampler = in_frames,
			task_queue = task_queue,
			solution_queue = solution_queue,
		)

		workers = [
			multiprocessing.Process(target=self.gen__worker, kwargs = worker_kwargs, daemon=True)
			for i in range(num_workers)
		]
		for w in workers:
			w.start()

		try:
			obj_stats = []

			with h5py.File(self.dir_dataset / 'labels.hdf5', 'w') as labels_file:
				for i in tqdm(range(num_tasks)):
					out_fr = solution_queue.get()

					self.gen__write_frame_data(out_fr, labels_file)

					json_frames.append(dict(
						frame_id = out_fr.frame_id,
						frame_id_str = out_fr.frame_id_str,
						source_frame_id = out_fr.source_frame_id,
						**{
							path_key: str(out_fr[path_key].relative_to(self.dir_dataset))
							for path_key in out_fr.keys()
							if path_key.startswith('image_path')
						}
					))

					obj_stats.append(out_fr.object_stats)

			obj_stats = ObjectStatistics.union(obj_stats)
			obj_stats.save(self.dir_dataset / 'objects.hdf5')

		finally:
			for w in workers:
				if w.is_alive():
					w.terminate()

		json_frames.sort(key=itemgetter('frame_id'))

		json_index = dict(
			cfg = self.cfg,
			frames = json_frames,
		)

		with (self.dir_dataset / 'index.json').open('w') as index_file:
			json.dump(json_index, index_file, indent='	')

	
	# loading

	def discover(self):
		with (self.dir_dataset / 'index.json').open('r') as index_file:
			index_data = json.load(index_file)

		self.cfg = EasyDict(index_data['cfg'])
		self.frames = index_data['frames']
		# sort frames as they may be unsorted in the JSON
		self.frames.sort(key=itemgetter('frame_id'))

		# background dataset for images
		self.background_dataset = self.gen__prepare_dataset()

	def load_frame_images(self, fr_info):
		fr_bg = self.background_dataset[fr_info['source_frame_id']]
		
		fr_bg.update({
			f'image_{ch}': imread(self.dir_dataset / fr_info[f'image_path_{ch}'])
			for ch in ('injections_with_margin', 'injections_with_bb')
		})

		return fr_bg

	def load_frame_labels(self, fr_info):
		with h5py.File(self.dir_dataset / 'labels.hdf5', 'r') as file_labels:
			g = file_labels[fr_info['frame_id_str']]

			fr_data = EasyDict({
				ch: g[ch][:]
				for ch in ('road_mask', 'obstacle_instance_map', 'obstacle_classes')
			})

			gp = g.get('perspective')
			if gp:
				pi =  EasyDict({
					name: val[:]
					for (name, val) in gp.items()
				})
				pi.update(gp.attrs)

				fr_data['perspective_info'] = pi			
				fr_data['perspective_scale_map'] = gen_perspective_scale_map(
					fr_data['road_mask'].shape, pi.horizon_level, pi.pix_per_meter_slope,
				)

			return fr_data


	def __len__(self):
		return self.frames.__len__()

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			subset = self.__class__(self.cfg)
			subset.frames = self.frames[idx]
			return subset
		else:
			fr_info = self.frames[idx]
			frame_id_str = fr_info['frame_id_str']
			fr = EasyDict(frame_id = fr_info['frame_id'], frame_id_str = frame_id_str, fid = frame_id_str)
			fr.update(self.load_frame_images(fr_info))
			fr.update(self.load_frame_labels(fr_info))
			fr.update(self.load_extra_features(fr_info))
			fr.dset = self
			return fr



	def calculate_class_distribution(self):
		#class_areas = np.zeros(255, dtype=np.float64)

		class_areas = []

		from multiprocessing.dummy import Pool as Pool_thread

		def worker(i):
			fr = self[i]
			gt_map = (fr.obstacle_instance_map > 0).astype(np.uint8)
			gt_map[~ fr.road_mask ] = 255
			return np.bincount(gt_map.reshape(-1), minlength=255)

		with Pool_thread() as p:
			num_fr = self.__len__()
			for bc in tqdm(p.imap(worker, list(range(num_fr)), chunksize=8), total=num_fr):	
				class_areas.append(bc)
		
		class_areas_sum = np.sum(class_areas, axis=0)
		areas_all = np.sum(class_areas_sum)
		class_areas_relative = class_areas_sum / areas_all

		# https://arxiv.org/pdf/1606.02147.pdf
		# w_class = 1/ ln(c+p_class)
		# c = 1.02

		training_samples_relative_p = class_areas_relative[0:2] / np.sum(class_areas_relative[0:2])
		recommended_weights = 1/np.log(1.02 + training_samples_relative_p)

		return EasyDict(
			class_areas_sum = class_areas_sum,
			class_areas_relative = class_areas_relative,
			recommended_weights = recommended_weights,
		)

		# recommended weights
		# 1.43138301, 34.96345657
	

	def plot_object_stats(self):
		obj_stats = ObjectStatistics.load(self.dir_dataset / 'objects.hdf5')

		import matplotlib
		matplotlib.use('Agg')
		from matplotlib import pyplot
		import seaborn
		dir_plots = self.dir_dataset / 'plots'
		dir_plots.mkdir(exist_ok=True)

		pos_from_bot_edge = 1024 - obj_stats.values['position_y']

		data = {
			'obj edge to image edge': pos_from_bot_edge - obj_stats.values['height']*0.5,
			'sqrt area': np.sqrt(obj_stats.values['area']),
			'perspective scale': obj_stats.values['perspective_scale'],
			'generalsize': (1./3.) * (obj_stats.values['height'] + obj_stats.values['width'] + np.sqrt(obj_stats.values['area'])),
		}


		fig, plots = pyplot.subplots(2, 2, figsize=(10, 10))
		fig.suptitle(self.cfg.get('display_name', self.cfg.name))

		#plot.set_title(self.cfg.name)
		#plot.hist(pos_from_bot_edge)

		#fig = seaborn.PairGrid(data, x_vars = ['obj edge to image edge', 'area'])
		#fig.map(seaborn.displot)

		seaborn.histplot(data, x='obj edge to image edge', ax = plots[0, 0])
		seaborn.histplot(data, x='sqrt area', ax = plots[0, 1])

		plots[1, 0].scatter(data['obj edge to image edge'], data['sqrt area'], s=0.75)
		plots[1, 0].set_xlabel('obj edge to image edge')
		plots[1, 0].set_ylabel('sqrt area')

		plots[1, 1].scatter(data['perspective scale'], data['sqrt area'], s=0.75)
		plots[1, 1].set_xlabel('perspective scale')
		plots[1, 1].set_ylabel('sqrt area')


		# seaborn.scatterplot(data=data, x='obj edge to image edge', y='sqrt area', ax=plots[1, 0])

		#fig.displot(data, x='area')
		#fig.set(title = self.cfg.name)

		fig.tight_layout()
		fig.savefig(dir_plots / f'plot_placementVsSize_{self.cfg.name}.png')


		# fig, plot = pyplot.subplots(1, 1, figsize=(3.5, 3.5))
		fig, plot = pyplot.subplots(1, 1, figsize=(4, 2.5))

		#fig.suptitle(self.cfg.name)
		#plot.set_title(self.cfg.name)
		#plot.hist(pos_from_bot_edge)

		#fig = seaborn.PairGrid(data, x_vars = ['obj edge to image edge', 'area'])
		#fig.map(seaborn.displot)

		# fig.suptitle(self.cfg.get('display_name', self.cfg.name))
		downsample = 5
		plot.scatter(data['perspective scale'][::downsample], data['generalsize'][::downsample], s=0.9)
		plot.set_xlabel('perspective scale [pix/m]')
		plot.set_ylabel('general size [pix]')
		fig.tight_layout()
		fig.savefig(dir_plots / f'plot_generalsize_{self.cfg.name}.png')
		
		with matplotlib.rc_context({'text.usetex': True}):
			fig.savefig(dir_plots / f'plot_generalsize_{self.cfg.name}.eps')
			fig.savefig(dir_plots / f'plot_generalsize_{self.cfg.name}.pdf')	
		print('Plots written to ', dir_plots / f'plot_generalsize_{self.cfg.name}.*')


	
# gener = SynthDsetV2()
# gener.gen__prepare()
# gener = ModuleRegistry2.get_implementation('DsetSynthObstacles_Placement', 'v2_cityscapes-train') 


from pathlib import Path
import json

from tqdm import tqdm
import cv2 as cv

def fuse_sharp(image_bg, image_objects, mask):
	image_out = image_bg.copy()
	image_out[mask] = image_objects[mask]
	return image_out

def fuse_blur(image_bg, image_objects, mask, blur_ksize):
	
	mask_f = mask.astype(np.float32)
	mask_smooth = cv.GaussianBlur(mask_f, ksize=(blur_ksize, blur_ksize), sigmaX=0)[:, :, None]
	
	img_fused = (
		image_bg.astype(np.float32) * (1.-mask_smooth)
		+
		image_objects.astype(np.float32) * mask_smooth
	).astype(np.uint8)
	
	return img_fused



def DsetSynthObstacles_Fusion__configs():
	cfgs = []
	placement_cfgs = DsetSynthObstacles_Placement.configs()

	# PLACEMENT_CFG_NOT_IN_DIRNAME = [
	# 	'-attnSETR',
	# ]

	def name_to_storage_dir(n):
		sd = f'1230_SynthObstacle/fusion_{n}'
		# for x in PLACEMENT_CFG_NOT_IN_DIRNAME:
		# 	sd = sd.replace(x, '')
		return sd

	for placement_cfg in placement_cfgs:
		name = f'Fsharp-{placement_cfg.name}'

		cfgs.append(EasyDict(
			name = name,
			cls = 'flat',
			mod_placement = placement_cfg.name,
			storage_dir = name_to_storage_dir(name),
			split = placement_cfg.split,			
			fusion_blur = 1,
		))
		
		for b in (3, 5):
			name = f'Fblur{b}-{placement_cfg.name}'

			cfgs.append(EasyDict(
				name = name,
				cls = 'flat',
				mod_placement = placement_cfg.name,
				storage_dir = name_to_storage_dir(name),
				split = placement_cfg.split,
				fusion_blur = b,
			))

		# Opt
		name = f'FOpt1-{placement_cfg.name}'
		cfgs.append(EasyDict(
			name = name,
			cls = 'opt',	
			mod_placement = placement_cfg.name,
			storage_dir = name_to_storage_dir(name),
			split = placement_cfg.split,
		))

		# Unwarp
		name = f'Fblur5unwarp1-{placement_cfg.name}'
		cfgs.append(EasyDict(
			name = name,
			cls = 'unwarp',	
			mod_placement = placement_cfg.name,
			storage_dir = name_to_storage_dir(name),
			split = placement_cfg.split,
			fusion_blur = 5,
		))

	return cfgs



class DsetSynthObstacles_FusionFlat:
	# TODO split generator into a module, since we prefix with gen__

	# @classmethod
	# def configs(cls):
	# 	cfgs = []
	# 	placement_cfgs = DsetSynthObstacles_Placement.configs()

	# 	for placement_cfg in placement_cfgs:
	# 		name = f'Fsharp-{placement_cfg.name}'

	# 		cfgs.append(EasyDict(
	# 			name = name,
	# 			mod_placement = placement_cfg.name,
	# 			storage_dir = f'1230_SynthObstacle/fusion_{name}',
	# 			split = placement_cfg.split,			
	# 			fusion_blur = 1,
	# 		))
			
	# 		for b in (3, 5):
	# 			name = f'Fblur{b}-{placement_cfg.name}'

	# 			cfgs.append(EasyDict(
	# 				name = name,
	# 				mod_placement = placement_cfg.name,
	# 				storage_dir = f'1230_SynthObstacle/fusion_{name}',
	# 				split = placement_cfg.split,
	# 				fusion_blur = b,
	# 			))
	
	# 	return cfgs
	
	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.dir_dataset = DIR_DATA / self.cfg.storage_dir

		self.extra_features = []

	# TODO this is duplicated
	@property
	def split(self):
		return self.cfg.split
	@property
	def name(self):
		return self.cfg.name

	@property
	def dset_key(self):
		return f'1230_SynthObstacle_Fusion_{self.cfg.name}'

	def img_path(self, channel, frame_id, type='image'):
		fmt = {'image': 'webp', 'label': 'png', 'demo': 'webp'}[type]
		dir = {'image': 'images', 'label': 'labels', 'demo': 'demo'}[type]
		return self.dir_dataset / dir / f'img{frame_id:04d}_{channel}.{fmt}'
	
	def gen__process_frame(self, fr):
		if self.cfg.fusion_blur == 1:
			f = fuse_sharp
		else:
			f = partial(fuse_blur, blur_ksize=self.cfg.fusion_blur)
			
		img_fused = f(
			image_bg = fr.image,
			image_objects = fr.image_injections_with_margin,
			mask = fr.obstacle_instance_map > 0,
		)
		
		imwrite(self.img_path('image_fused', fr.frame_id), img_fused)
		#imwrite(self.img_path('image_fused'), img_fused)
		fr['image_fused'] = img_fused
		return fr
	
	def gen__init(self):
		self.placement_dset = ModuleRegistry2.get_implementation('1230_SynthObstacle_Placement', self.cfg.mod_placement)
		self.placement_dset.discover()
	
	def gen__iter(self, idx):
		self.gen__process_frame(self.placement_dset[idx])
		return None

	def gen__run(self, num_workers=8):
		placement_dset = ModuleRegistry2.get_implementation('1230_SynthObstacle_Placement', self.cfg.mod_placement)
		placement_dset.discover()

		run_parallel(
			worker_init_func = self.gen__init,
			worker_task_func = self.gen__iter,
			tasks = range(placement_dset.__len__()),
			num_workers = num_workers,
			progress_bar = True,
		)

		# for fr in tqdm(placement_dset):
		# 	self.gen__process_frame(fr)
	
	def discover(self):
		self.placement_dset = ModuleRegistry2.get_implementation('1230_SynthObstacle_Placement', self.cfg.mod_placement)
		self.placement_dset.discover()

	def register_extra_feature(self, loader):
		self.extra_features.append(loader)	
	
	def __len__(self):
		return self.placement_dset.__len__()
	
	def load_frame_images(self, frame_id):
		return {
			ch: imread(self.img_path(ch, frame_id))
			for ch in ('image_fused',)
		}
	

	def load_extra_features(self, fr_info, fr):
		out_data = dict()

		for loader in self.extra_features:
			out_data.update(loader(
				fr_info = fr_info,
				frame = fr,
				dset = self,
			))

		return out_data


	def __getitem__(self, idx):
		fr_info = self.placement_dset.frames[idx]
		frame_id_str = fr_info['frame_id_str']
		fr = EasyDict(frame_id = fr_info['frame_id'], frame_id_str = frame_id_str, fid = frame_id_str)
		fr.update(self.placement_dset.load_frame_labels(fr_info))
		fr.update(self.load_frame_images(fr_info['frame_id']))
		# api for training
		fr.image = fr.image_fused
		fr.labels_road_mask = fr.road_mask	

		fr.update(self.load_extra_features(fr_info, fr))
		
		
		return fr
		
from ..a14_perspective.cityscapes_pitch_angles import load_frame_with_perspective_info
from ..a14_perspective.warp import unwarp_road_frame
import json

class DsetSynthObstacles_FusionUnwarp(DsetSynthObstacles_FusionFlat):

	def discover(self):
		super().discover()
		self.idx_mapping = json.loads((self.dir_dataset / 'idx_mapping.json').read_text())


	def __len__(self):
		return self.idx_mapping.__len__()

	def gen__process_frame(self, fr):

		# fuse image in the usual way
		if self.cfg.fusion_blur == 1:
			f = fuse_sharp
		else:
			f = partial(fuse_blur, blur_ksize=self.cfg.fusion_blur)

		img_fused = f(
			image_bg = fr.image,
			image_objects = fr.image_injections_with_margin,
			mask = fr.obstacle_instance_map > 0,
		)
		fr.image = img_fused

		# load perspective
		dset_bg = self.placement_dset.background_dataset	
		fr_perspective = load_frame_with_perspective_info(dset_bg, fr.frame_id)
		fr.camera_info = fr_perspective.camera_info
		fr.persp_info = fr_perspective.persp_info
	
		# warp
		fr.label_pixel_gt = np.logical_not(fr.road_mask).astype(np.uint8) * 255

		try:
			unwarp_road_frame(fr)
		except ValueError:
			return None

		# write warped image
		fr['image_fused'] = fr.unwarp_image
		imwrite(self.img_path('image_fused', fr.frame_id), fr['image_fused'])

		# warp label channels:
		for ch in 'road_mask', 'obstacle_instance_map', 'obstacle_classes':
			tp = fr[ch].dtype
			fr[ch] = cv.warpPerspective(
				fr[ch].astype(np.uint8), 
				fr.unwarp_H, 
				tuple(fr.unwarp_size),
				flags = cv.INTER_NEAREST,
			).astype(tp)
			imwrite(self.img_path(ch, fr.frame_id, type='label'), fr[ch])
				
		#pb = self.img_path('demoobs', fr.frame_id, fmt='png')
		#pdemo = pb.parents[1] / 'demo' / pb.name
		#imwrite(pdemo, adapt_img_data(fr.obstacle_instance_map))

		return fr

	def __getitem__(self, idx):
		idx = self.idx_mapping[idx]

		fr_info = self.placement_dset.frames[idx]
		frame_id = fr_info['frame_id']
		frame_id_str = fr_info['frame_id_str']
		fr = EasyDict(frame_id = frame_id, frame_id_str = frame_id_str, fid = frame_id_str)

		fr.update({
			ch: imread(self.img_path(ch, frame_id, type='image'))
			for ch in ('image_fused',)
		})

		fr.update({
			ch: imread(self.img_path(ch, frame_id, type='label'))
			for ch in ('road_mask', 'obstacle_instance_map', 'obstacle_classes')
		})

		# api for training
		fr.image = fr.image_fused
		fr.labels_road_mask = fr.road_mask	
		return fr

	def gen__iter(self, idx):
		fr = self.gen__process_frame(self.placement_dset[idx])
		if fr is None:
			return None
		else:
			return EasyDict(frame_id = fr.frame_id)

	def gen__collect(self, fr):
		if fr is not None:
			self.idx_mapping.append(fr.frame_id)

	def gen__run(self, num_workers=8):
		placement_dset = ModuleRegistry2.get_implementation('1230_SynthObstacle_Placement', self.cfg.mod_placement)
		placement_dset.discover()

		self.idx_mapping = []

		run_parallel(
			worker_init_func = self.gen__init,
			worker_task_func = self.gen__iter,
			tasks = range(placement_dset.__len__()),
			num_workers = num_workers,
			progress_bar = True,
			host_collect_func = self.gen__collect,
		)
		(self.dir_dataset / 'idx_mapping.json').write_text(json.dumps(self.idx_mapping))







def DsetSynthObstacles_Fusion(cfg):
	""" 
	Using a func instead of class to allow polymoprhism with old module registry
	TODO use new module registry
	"""

	if cfg.cls == 'flat':
		c = DsetSynthObstacles_FusionFlat
	elif cfg.cls == 'unwarp':
		c = DsetSynthObstacles_FusionUnwarp
	elif cfg.cls == 'opt':
		from .synth_obstacle_dset_opt import DsetSynthObstacles_FusionOpt
		c = DsetSynthObstacles_FusionOpt

	return c(cfg)

DsetSynthObstacles_Fusion.configs = DsetSynthObstacles_Fusion__configs
DsetSynthObstacles_Fusion = ModuleRegistry2('1230_SynthObstacle_Fusion')(DsetSynthObstacles_Fusion)

import click

@click.group()
def main():
	...

@main.command()
@click.argument('module')
@click.argument('name')
@click.option('--num-workers', type=int, default=12)
def gen(module, name, num_workers):
	""" Generate the synthetic dataset for a given name """

	mod = ModuleRegistry2.get_implementation(module, name)
	mod.gen__run(num_workers=num_workers)

	if hasattr(mod, 'plot_object_stats'):
		mod.plot_object_stats()


@main.command()
@click.argument('module')
@click.argument('name')
def distribution(module, name):
	""" Gather object distribution """

	mod = ModuleRegistry2.get_implementation(module, name)
	mod.discover()
	res = mod.calculate_class_distribution()
	print(res)


@main.command()
@click.argument('module')
@click.argument('name')
def object_stats(module, name):
	""" Plot object distribution """

	import matplotlib
	# matplotlib.rcParams['pdf.fonttype'] = 42
	# matplotlib.rcParams['ps.fonttype'] = 42
	# # matplotlib.rcParams['text.usetex'] = True

	mod = ModuleRegistry2.get_implementation(module, name)
	res = mod.plot_object_stats()



if __name__ == '__main__':
	main()


# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v2_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion v2sharp_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion v2blur3_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion v2blur5_cityscapes-train


# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v3persp2_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5-v3persp2_cityscapes-train

# Perspective dset

# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v3persp2_cityscapes-val --num-workers 12
# python -m src.a12_inpainting.synth_obstacle_dset2 object-stats 1230_SynthObstacle_Placement v3persp2_cityscapes-val

# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5-v3persp2_cityscapes-val --num-workers 12
# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-val
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-val


# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v3persp2_cityscapes-train --num-workers 12
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5-v3persp2_cityscapes-train --num-workers 12
# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-train
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-train

# python -m src.a12_inpainting.synth_obstacle_dset2 distribution 1230_SynthObstacle_Placement v3persp2_cityscapes-train


# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v2b_IndiaDriving-train --num-workers 30
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Placement v2b_IndiaDriving-val --num-workers 10

# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5-v2b_IndiaDriving-train --num-workers 30
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5-v2b_IndiaDriving-val --num-workers 10

# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-train
# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-val

# Unwarp

# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5unwarp1-v2b_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5unwarp1-v2b_cityscapes-val

# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5unwarp1-v2b_cityscapes-train
# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5unwarp1-v2b_cityscapes-val


# python -m src.a12_inpainting.synth_obstacle_dset2 distribution 1230_SynthObstacle_Placement v2b_cityscapes-train



#	python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5unwarp1-v3persp3D_cityscapes-val
#	python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion Fblur5unwarp1-v3persp3D_cityscapes-train
# 	python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5unwarp1-v3persp3D_cityscapes-train
# 	python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1-768 1230_SynthObstacle_Fusion_Fblur5unwarp1-v3persp3D_cityscapes-val

# bash 1504_gen_dset.sh v3persp3E_cityscapes-val
# bash 1504_gen_dset.sh v3persp3E_cityscapes-train

"""
python -m src.a12_inpainting.synth_obstacle_dset2 object-stats 1230_SynthObstacle_Placement v3persp3D_cityscapes-train
python -m src.a12_inpainting.synth_obstacle_dset2 object-stats 1230_SynthObstacle_Placement v2c_cityscapes-train
python -m src.a12_inpainting.synth_obstacle_dset2 object-stats 1230_SynthObstacle_Placement v2b_cityscapes-train

python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion FOpt1-v3persp3D_cityscapes-val --num-workers 1
python -m src.a12_inpainting.synth_obstacle_dset2 gen 1230_SynthObstacle_Fusion FOpt1-v3persp3D_cityscapes-train
"""

# bash 1504_gen_dset.sh v3persp3Dbs_cityscapes-val
# bash 1504_gen_dset.sh v3persp3Dbs_cityscapes-train

# bash 1504_gen_dset.sh v3p5bs_cityscapes-val
# bash 1504_gen_dset.sh v3p5bs_cityscapes-train

# bash 1504_gen_dset.sh v3p4sc_cityscapes-val
# bash 1504_gen_dset.sh v3p4sc_cityscapes-train
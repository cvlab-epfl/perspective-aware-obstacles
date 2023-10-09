

from pathlib import Path
import json
import multiprocessing
from operator import itemgetter
from copy import deepcopy

#from multiprocessing.dummy import Pool as Pool_thread

import numpy as np
import cv2 as cv
from easydict import EasyDict
import h5py
from tqdm import tqdm

from .file_io import hdf5_read_hierarchy_from_group
from .grid_randomizer import square_grid_randomizer
from ..paths import DIR_DATA
from ..datasets.dataset import imwrite, imread
from ..pipeline.config import extend_config

def hdf5_load_recursive(g):
	return EasyDict({
		k: hdf5_load_recursive(v) 
		if isinstance(v, h5py.Group) else v[:]
		for k, v in g.items()
	})


class SynthObstacleDset:

	default_cfg = EasyDict(
		name = '1204-SynthObstacleDset-v1-Ctc',
		split = 'train',

		background_dataset = 'cityscapes',

		obstacles = dict(
			database = 'cityscapes',
			filter = dict(
				dim_min_max = [10, 150], 
				area_min_max = [100, 5000],
			),
		),

		injection = dict(
			grid_step = 150,
			item_radius = 75,
			random_offset = 60,
			injection_probability_per_slot = 0.2,
		),

		inpainter_cfg = dict(
			#name = 'dummy',
			name = 'deep-fill-tf2',
		),

		inpainter_sliding_window_cfg = dict(
			patch_size = 200, context_size = 400, patch_overlap = 0.7, context_restriction=False,
		),
	)

	CFGS_BY_NAME = {}

	@classmethod
	def add_config(cls, cfg_override, splits=('train', 'val')):

		new_cfg = EasyDict(extend_config(cls.default_cfg, cfg_override))
		name = new_cfg.name

		for split in splits:
			c = EasyDict(deepcopy(new_cfg)) # copy
			c.split = split
			c.obstacles.database += f'_{split}' # cityscapes_split
			c.background_dataset += f'-{split}'
			cls.CFGS_BY_NAME[f'{name}-{split}'] = c

	@classmethod
	def get_implementation(cls, name_and_split):
		cfg = cls.CFGS_BY_NAME[name_and_split]
		return cls(cfg, try_load=False)
	
	def __init__(self, cfg, try_load=True):
		self.set_cfg(cfg)

		if try_load:
			self.load_from_storage()		

	def set_cfg(self, cfg):
		self.cfg = EasyDict(self.default_cfg)
		if cfg is not None:
			self.cfg.update(cfg)
	
		self.name = self.cfg.name
		self.split = self.cfg.split

		self.dir_dataset = Path(self.cfg.get(
			'dir_dataset',
			DIR_DATA / self.cfg.name / self.cfg.split, # default path from name
		))

	def img_path(self, channel, frame_id):
		# suffix = {
		# 	'image_with_obstacles': 'injected',
		# 	'image_inpainted': 'inpainted',
		# }[channel]

		return self.dir_dataset / 'images' / f'img{frame_id:04d}_{channel}.webp'

	def load_from_storage(self, dir_dataset_override=None):
		if dir_dataset_override is not None:
			self.dir_dataset = Path(dir_dataset_override)

		with (self.dir_dataset / 'index.json').open('r') as index_file:
			index_data = json.load(index_file)
		
		cfg = EasyDict(index_data['cfg'])
		cfg.dir_dataset = self.dir_dataset
		self.set_cfg(cfg)

		self.frames = index_data['frames']
		# sort frames as they may be unsorted in the JSON
		self.frames.sort(key=itemgetter('frame_id'))

	def discover(self):
		self.load_from_storage()

	@classmethod
	def from_new_config(cls, cfg):
		return cls(cfg, try_load=False)

	@classmethod
	def from_disk(cls, dir_dataset):
		return cls(dict(dir_dataset = dir_dataset), try_load=True)

	def __len__(self):
		return self.frames.__len__()

	def load_frame_images(self, fr_info):
		return {
			f'image_{ch}': imread(self.dir_dataset / fr_info[f'image_path_{ch}'])
			for ch in ('injected', 'inpainted')
		}

	def load_frame_labels(self, fr_info):
		with h5py.File(self.dir_dataset / 'labels.hdf5', 'r') as file_labels:
			g = file_labels[fr_info['frame_id_str']]

			return {
				ch: g[ch][:]
				for ch in ('road_mask', 'inpainting_mask', 'obstacle_instance_map', 'obstacle_classes')
			}

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			subset = SynthObstacleDset(self.cfg)
			subset.frames = self.frames[idx]
			return subset
		else:
			fr_info = self.frames[idx]
			frame_id_str = fr_info['frame_id_str']
			fr = EasyDict(frame_id = fr_info['frame_id'], frame_id_str = frame_id_str, fid = frame_id_str)
			fr.update(self.load_frame_images(fr_info))
			fr.update(self.load_frame_labels(fr_info))
			fr.image = fr.image_injected
			fr.dset = self
			return fr

	


	def gen__prepare(self):
		""" init components needed to generate the dataset 
		* database of instances to inject
		* inpainter
		* patch divider for inpainting
		"""

		self.instance_db = self.gen__prepare_instdb()
		self.instance_db.filter_obstacles_and_store(**self.cfg.obstacles.filter)

		self.gen__prepare_inpainter()

		from .patching_square import ImpatcherSlidingSquares
		self.inpainting_sliding_window = ImpatcherSlidingSquares(
			b_demo = False,
			inpaint_func = self.inpainter_func,
			**self.cfg.inpainter_sliding_window_cfg,
		)
		self.inpainting_sliding_window_func = self.inpainting_sliding_window.process_frame_v2

		self.background_dataset = self.gen__prepare_dataset()

	def gen__prepare_dataset(self):
		from .demo_case_selection import DatasetRegistry
		return DatasetRegistry.get_implementation(self.cfg.background_dataset)

	def gen__prepare_instdb(self):
		inst_db_name = self.cfg.obstacles.database

		if inst_db_name.startswith('cityscapes_'):
			print(inst_db_name)
			_, split = inst_db_name.split('_')

			from ..datasets.cityscapes import DatasetCityscapes
			dset_ctc = DatasetCityscapes(split=split)
			dset_ctc.discover()

			from .instance_database import InstDb
			inst_db = InstDb(dset_ctc)

			return inst_db
		
		else:
			raise NotImplementedError(f'Inst db: {inst_db_name}')
		
	def gen__prepare_inpainter(self):
		# TODO: use sys_reconstruction module
		inpainter_name = self.cfg.inpainter_cfg.name

		if inpainter_name == 'dummy':
			self.inpainter_net = None
			self.inpainter_func = lambda images_hwc_list, masks_hw_list: [a//2 for a in images_hwc_list]
		
		elif inpainter_name == 'deep-fill-tf2':
			import sys
			inp_source_dir = str(Path('inpainter/generative_inpainting').resolve())
			if not inp_source_dir in sys.path:
				sys.path.append(inp_source_dir)

			from inferent import InpaintInferent

			inpinf = InpaintInferent()
			# inpinf.load_nn(input_size_wh=(imp.context_size, imp.context_size), batch_size=2)
			inpinf.load_nn(input_size_wh=(256, 256), batch_size=1)

			self.inpainter_net = inpinf
			self.inpainter_func = self.inpainter_net.predict_auto_batch

		else:
			raise NotImplementedError(f'Inpainter: {inpainter_name}')


	def gen__get_random_obstacle(self):
		idb = self.instance_db
		idx_in_db = idb.filtered_obstacle_ids[np.random.randint(idb.filtered_obstacle_ids.__len__())]
		obstacle_info = idb[idx_in_db]
		return obstacle_info

	@staticmethod
	def gen__inject_obstacle_into_image(image_ref, instmap_ref, obstacle_image, obstacle_mask, obstacle_id_for_map, obstacle_center_pt):
		obstacle_h, obstacle_w = obstacle_mask.shape
		obstacle_size_xy = np.array([obstacle_w, obstacle_h], dtype=np.int32)
		
		obstacle_tl_xy = obstacle_center_pt.astype(np.int32) - obstacle_size_xy // 2
		obstacle_br_xy = obstacle_tl_xy + obstacle_size_xy
		
		crop_slice = (slice(obstacle_tl_xy[1], obstacle_br_xy[1]), slice(obstacle_tl_xy[0], obstacle_br_xy[0]))
		canvas_crop = image_ref[crop_slice]
		canvas_crop[obstacle_mask] = obstacle_image[obstacle_mask]

		map_crop = instmap_ref[crop_slice]	
		map_crop[obstacle_mask] = obstacle_id_for_map
		

	def gen__produce_frame_with_obstacles(self, image_data, labels_road_mask, cfg=None, f_get_random_obstacle=None):

		f_get_random_obstacle = f_get_random_obstacle or self.gen__get_random_obstacle
		cfg = cfg or self.cfg		
	
		# extract image and road mask
		#image_data = fr.image
		img_h, img_w, _ = image_data.shape
		#labels_road_mask = fr.labels_source == 7

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
		# grid centers to full image coords
		obstacle_grid_centers += np.array([tl_x, tl_y])
		obstacle_grid_centers = np.rint(obstacle_grid_centers).astype(np.int32)
		num_obstacle_slots = obstacle_grid_centers.__len__()
		
		# choose which slots are occupied
		obstacle_coin_flips = np.random.uniform(0, 1, size=num_obstacle_slots) < cfg.injection.injection_probability_per_slot
		
		obstacle_grid_centers = obstacle_grid_centers[obstacle_coin_flips]
		num_obstacle_slots = obstacle_grid_centers.__len__()
		
		# data for the generated frame
		obstacle_instance_map = np.zeros_like(labels_road_mask, dtype=np.uint16)
		image_modified = image_data.copy()
		
		obstacle_classes = [0]
		inst_id = 1
		
		for grid_pos in obstacle_grid_centers:
			obstacle_info = f_get_random_obstacle()
			
			try:
				self.gen__inject_obstacle_into_image(
					image_ref = image_modified, instmap_ref = obstacle_instance_map, 
					obstacle_image = obstacle_info.image_crop, obstacle_mask = obstacle_info.mask, 
					obstacle_id_for_map = inst_id, obstacle_center_pt = grid_pos,
				)
				
				inst_id += 1
				obstacle_classes.append(obstacle_info.instance_class)
				
			except Exception as e:
				print('Failed to inject obstacle')
		
		obstacle_classes = np.array(obstacle_classes, dtype=np.uint16)
		
		# inpaint
		inpainting_result = self.inpainting_sliding_window_func(
			image_data = image_modified,
			labels_road_mask = labels_road_mask,
		)
		
		# TODO constrain obstacle_instance_map to inpainted area
		#obstacle_instance_map patching_info.area_mask_without_margin.astype(np.uint8)
		
		return EasyDict(
			image_with_obstacles = image_modified,
			image_inpainted = inpainting_result.image_fused,
			road_mask = labels_road_mask,
			inpainting_mask = inpainting_result.area_mask_without_margin,
			obstacle_instance_map = obstacle_instance_map,
			obstacle_classes = obstacle_classes,
		)

	def gen__process_frame_ctc(self, frame_id, fid, image, labels_source, **_):

		gen_result = self.gen__produce_frame_with_obstacles(
			image_data = image,
			labels_road_mask = labels_source == 7,
		)
		gen_result.frame_id = frame_id
		gen_result.frame_id_str = f'{frame_id:05d}'

		gen_result.update(
			source_frame_id = fid,
		)

		self.gen__write_frame_image(gen_result)

		return gen_result


	def gen__write_frame_image(self, fr_data):
		fr_data.image_path_injected = self.img_path('injected', fr_data.frame_id)
		imwrite(fr_data.image_path_injected, fr_data.image_with_obstacles)

		fr_data.image_path_inpainted = self.img_path('inpainted', fr_data.frame_id)
		imwrite(fr_data.image_path_inpainted, fr_data.image_inpainted)

	def gen__write_frame_data(self, fr_data, hdf_file):
		
		g = hdf_file.create_group(fr_data.frame_id_str)

		g.attrs['source_frame_id'] = fr_data.source_frame_id

		g.create_dataset('road_mask', data=fr_data.road_mask.astype(bool), compression=3)
		g.create_dataset('inpainting_mask', data=fr_data.inpainting_mask.astype(bool), compression=3)
		g.create_dataset('obstacle_instance_map', data=fr_data.obstacle_instance_map.astype(np.uint16), compression=8)
		g.create_dataset('obstacle_classes', data=fr_data.obstacle_classes.astype(np.uint16))

	
	def gen__worker(self, task_queue : multiprocessing.Queue, solution_queue : multiprocessing.Queue):
		print('process started')
		
		self.gen__prepare()

		while not task_queue.empty():
			frame_idx = task_queue.get()
			frame = self.background_dataset[frame_idx]

			result = self.gen__process_frame_ctc(frame_id = frame_idx, **frame)

			solution_queue.put(result)

	def gen__run(self, num_workers=3):
		background_dataset = self.gen__prepare_dataset()
		num_tasks = background_dataset.__len__()
		tasks = range(num_tasks)
		#worker = lambda i: self.gen__process_frame_ctc(frame_id = i, **in_frames[i])
		
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
			with h5py.File(self.dir_dataset / 'labels.hdf5', 'w') as labels_file:
				for i in tqdm(range(num_tasks)):
					out_fr = solution_queue.get()

	#			for out_fr in tqdm(map(worker, tasks), total=num_tasks):
					self.gen__write_frame_data(out_fr, labels_file)

					json_frames.append(dict(
						frame_id = out_fr.frame_id,
						frame_id_str = out_fr.frame_id_str,
						source_frame_id = out_fr.source_frame_id,
						image_path_injected = str(out_fr.image_path_injected.relative_to(self.dir_dataset)),
						image_path_inpainted = str(out_fr.image_path_inpainted.relative_to(self.dir_dataset)),
					))
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


	def calculate_class_distribution(self):
		#class_areas = np.zeros(255, dtype=np.float64)

		class_areas = []

		for fr in tqdm(self):
			gt_map = (fr.obstacle_instance_map > 0).astype(np.uint8)
			gt_map[~ fr.inpainting_mask ] = 255

			class_areas.append(
				np.bincount(gt_map.reshape(-1), minlength=255),
			)
		
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

		
# default
SynthObstacleDset.add_config({}, ('train', 'val'))
# big objects
SynthObstacleDset.add_config(dict(
	name = '1204-SynthObstacleDset-v2MoreObj-Ctc',
	obstacles = dict(
		database = 'cityscapes',
		filter = dict(
			dim_min_max = [10, 250], 
			area_min_max = [100, 35e3],
		),
	),
), ('train', 'val'))



class SynthObstacleDset_PytorchSampler:

	def __init__(self, reported_len, start_at=0):
		self.reported_len = reported_len
		self.position = start_at

	def __len__(self):
		return self.reported_len

	def __iter__(self):
		for i in range(self.__len__()):
			yield self.position
			self.position += 1


class SynthObstacleDsetTrainingSampler:
	default_cfg = dict(
		crop_size_xy = [768, 384],
		grid_step = 20,
		grid_random_offset = 100,
		shuffled = True,

		mod_dataset = dict(
			name = 'SET_DSET_NAME',
			split = 'SET_SPLIT'
		)
	)
		
	CFGS_BY_NAME = {}

	@classmethod
	def add_config(cls, cfg_override):
		new_cfg = EasyDict(extend_config(cls.default_cfg, cfg_override))
		key = f'{new_cfg.mod_dataset.name}-PatchSampler-{new_cfg.mod_dataset.split}'
		cls.CFGS_BY_NAME[key] = new_cfg

	@classmethod
	def get_implementation(cls, name_and_split):
		try:
			cfg = cls.CFGS_BY_NAME[name_and_split]
		except KeyError:
			raise KeyError(f'Requested {name_and_split} but available {cls.CFGS_BY_NAME.keys()}')
		return cls(cfg)

	def __init__(self, cfg=None, dset=None):
		self.cfg = EasyDict(self.default_cfg)
		if cfg is not None:
			self.cfg.update(cfg)

		self.dset = dset
		self.name = f'{self.cfg.mod_dataset.name}-PatchSampler'
		self.split = self.cfg.mod_dataset.split
		self.extra_channels_to_load = {}
		self.preproc_func = None

	def discover(self):
		if self.dset is None:
			self.dset = SynthObstacleDset.from_disk(DIR_DATA / self.cfg.mod_dataset.name / self.cfg.mod_dataset.split)

		#DIR_DATA / f'dataset_{self.cfg.name}', # default path from name
		self.sampling_info = None
	
		path_sampling_data = self.dset.dir_dataset / 'sampling_v1.hdf5'

		if path_sampling_data.is_file():
			with h5py.File(path_sampling_data, 'r') as file_sampling:
				self.epoch_len = file_sampling['sampling/frame_idx'].shape[1]
		else:
			self.epoch_len = self.dset.__len__()

	def __len__(self):
		return self.epoch_len

	def __getitem__(self, idx):
		if isinstance(idx, tuple):
			epoch_idx, sample_idx = idx
			return self.sampling_get_frame(epoch_idx, sample_idx)
		elif isinstance(idx, int):
			epoch_idx = idx // self.__len__()
			sample_idx = idx % self.__len__()
			return self.sampling_get_frame(epoch_idx, sample_idx)
		else:
			raise NotImplementedError(f'Index with type {type(idx)}: {idx}')

	def make_pytorch_sampler(self, start_at = 0, short_epoch=None):
		return SynthObstacleDset_PytorchSampler(
			reported_len = short_epoch or self.__len__(), 
			start_at = start_at,
		)

	def sampling_generate_for_frame(self, fr_info):
		sampling_cfg = self.cfg
		
		crop_size_xy = np.array(sampling_cfg.crop_size_xy, dtype=np.int32)
		crop_radius = np.linalg.norm(crop_size_xy)

		fr = EasyDict(frame_id = fr_info['frame_id'], frame_id_str = fr_info['frame_id_str'])
		fr.update(self.dset.load_frame_labels(fr_info))

		frame_size_xy = np.array(fr.inpainting_mask.shape[::-1], dtype=np.int32)

		tl_x, tl_y, bb_w, bb_h = cv.boundingRect(fr.inpainting_mask.astype(np.uint8))
		
		sample_grid_centers = square_grid_randomizer(
			size_xy = [bb_w, bb_h], 
			grid_step = sampling_cfg.grid_step,
			#item_radius = crop_radius * 0.5, 
			item_radius = min(crop_size_xy) // 2,
			random_offset = sampling_cfg.grid_random_offset,
		)

		# grid centers to full image coords
		sample_grid_centers = np.rint(sample_grid_centers + np.array([tl_x, tl_y])).astype(np.int32)

		patch_tls = sample_grid_centers - crop_size_xy // 2
		patch_brs = patch_tls + crop_size_xy

		# drop patches outside of image
		is_valid = (
			(patch_tls[:, 0] >= 0) 
			& (patch_tls[:, 1] >= 0)
			& (patch_brs[:, 0] <= frame_size_xy[0])
			& (patch_brs[:, 1] <= frame_size_xy[1])
		)

		#print('a', is_valid)
		#print(patch_tls, patch_brs, frame_size_xy)

		# drop patches outside of ROI
		for i in range(patch_tls.__len__()):
			if is_valid[i]:
				try:
					tl = patch_tls[i]
					br = patch_brs[i]
					crop_slice = (slice(tl[1], br[1]), slice(tl[0], br[0]))
					roi_mask_crop = fr.inpainting_mask[crop_slice]
				except:
					print(crop_slice)

				roi_fraction = np.count_nonzero(roi_mask_crop) / np.prod(roi_mask_crop.shape)

				if roi_fraction < 0.75:
					is_valid[i] = False

		#print('b', is_valid)

		patch_tls = patch_tls[is_valid, :]
		patch_brs = patch_brs[is_valid, :]

		# add flips
		sample_tls = np.tile(patch_tls, (2, 1))
		sample_brs = np.tile(patch_brs, (2, 1))
		sample_flip_x = np.zeros(sample_tls.__len__(), dtype=bool)
		sample_flip_x[sample_tls.__len__() // 2:] = True

		return EasyDict(
			sample_tls = sample_tls.astype(np.uint16),
			sample_brs = sample_brs.astype(np.uint16),
			sample_flip_x = sample_flip_x.astype(bool),
		)

	def sampling_generate(self):

		num_epochs = 300
		num_patches_per_frame = np.zeros(self.dset.frames.__len__(), dtype=np.uint16)

		with h5py.File(self.dset.dir_dataset / 'sampling_v1.hdf5', 'w') as file_sampling:
			frs = self.dset.frames
			for i, fr_info in tqdm(enumerate(frs), total=frs.__len__()):
				patch_info = self.sampling_generate_for_frame(fr_info)

				num_patches_per_frame[i] = patch_info.sample_tls.__len__()

				g = file_sampling.create_group(fr_info['frame_id_str'])
				for k, v in patch_info.items():
					g[k] = v

			print(num_patches_per_frame)

			# find frames with non-zero patches
			valid_frame_ids = np.where(num_patches_per_frame > 0)[0].astype(np.uint16)
			num_frames = valid_frame_ids.__len__()

			# permute patches
			rng = np.random.default_rng()

			patch_indices_all = np.tile(np.arange(0, num_epochs, dtype=np.uint16), (frs.__len__(), 1))
			if self.cfg.shuffled:
				for frid in range(num_frames):
					rng.shuffle(patch_indices_all[frid])
			patch_indices_all[valid_frame_ids] %= num_patches_per_frame[valid_frame_ids, None]

			# permute frames
			frame_ids_permutations = np.tile(valid_frame_ids, (num_epochs, 1))
			if self.cfg.shuffled:
				for ep in range(num_epochs):
					rng.shuffle(frame_ids_permutations[ep])		

			file_sampling['sampling/frame_idx'] = frame_ids_permutations
			file_sampling['sampling/patch_idx'] = patch_indices_all

	def gen__run(self, **_):
		self.discover()
		self.sampling_generate()

	def sampling_load(self):
		with h5py.File(self.dset.dir_dataset / 'sampling_v1.hdf5', 'r') as file_sampling:
			self.sampling_info = hdf5_read_hierarchy_from_group(file_sampling)
			self.epoch_len = self.sampling_info.sampling.frame_idx.shape[1]

	def sampling_get_frame(self, epoch_idx, sample_idx):
		if self.sampling_info is None:
			self.sampling_load()

		fr_idx = self.sampling_info.sampling.frame_idx[epoch_idx, sample_idx]
		patch_idx = self.sampling_info.sampling.patch_idx[fr_idx, epoch_idx]

		fr_data = self.dset[fr_idx]

		# extra channels to load
		for ch_name, ch_obj in self.extra_channels_to_load.items():
			fr_data[ch_name] = ch_obj.read_value(**fr_data)

		sampling_frame = self.sampling_info[fr_data['frame_id_str']]
		szs = sampling_frame.sample_brs - sampling_frame.sample_tls
		# print({k: v.shape for k, v in fr_data.items() if isinstance(v, np.ndarray)})
		# print(np.unique(szs))
		# print(np.min(sampling_frame.sample_tls, axis=0), np.max(sampling_frame.sample_brs, axis=0))

		tl = sampling_frame.sample_tls[patch_idx]
		br = sampling_frame.sample_brs[patch_idx]
		b_flip = sampling_frame.sample_flip_x[patch_idx]

		#print(tl, br, b_flip)

		# Integrating the flip into the crop slice (with step -1) proved to cause constant bugs
		# First [brx:tlx:-1] fails if brx is 2048 (equal to image width)
		# then [brx-1:tlx-1:-1] fails if tlx is 0
		# Instead, we will [:, ::-1] the whole image later

		crop = (
			slice(tl[1], br[1]), 
			slice(tl[0], br[0]),
		)

		channels_to_crop = set([
			'image_inpainted', 
			'image_injected', 
			'obstacle_instance_map', 
			'inpainting_mask',
			*self.extra_channels_to_load.keys(),
		])

		fr = EasyDict({
			k: fr_data[k][crop] 
			for k in channels_to_crop
		})
		if b_flip:
			fr = EasyDict({
				k: v[:, ::-1]
				for k, v in fr.items()
			})

		fr.frame_id = fr_data.frame_id

		if self.preproc_func is not None:
			fr = self.preproc_func(**fr)
		
		return fr

SynthObstacleDsetTrainingSampler.add_config(dict(
	shuffled = True,
	mod_dataset = dict(name = '1204-SynthObstacleDset-v1-Ctc', split='train'),
))
SynthObstacleDsetTrainingSampler.add_config(dict(
	shuffled = True,
	mod_dataset = dict(name = '1204-SynthObstacleDset-v1-Ctc', split='val'),
))
SynthObstacleDsetTrainingSampler.add_config(dict(
	shuffled = True,
	mod_dataset = dict(name = '1204-SynthObstacleDset-v2MoreObj-Ctc', split='train'),
))
SynthObstacleDsetTrainingSampler.add_config(dict(
	shuffled = False,
	mod_dataset = dict(name = '1204-SynthObstacleDset-v2MoreObj-Ctc', split='val'),
))


import click

@click.command()
@click.argument('dset_name')
@click.option('--num-workers', type=int, default=3)
def main(dset_name, num_workers):
	""" Generate the synthetic dataset for a given name """

	try:
		gen = SynthObstacleDset.get_implementation(dset_name)
	except KeyError:
		gen = SynthObstacleDsetTrainingSampler.get_implementation(dset_name)
	gen.gen__run(num_workers=num_workers)
	
if __name__ == '__main__':
	main()


# How frames are prepared for training

# from src.paths import DIR_DATA
# from src.a12_inpainting.synth_obstacle_dset import SynthObstacleDset
# from src.a12_inpainting.discrepancy_experiments import Exp1205_Discrepancy_ImgVsInpaiting
# dset = SynthObstacleDset.from_disk(DIR_DATA / '1204-SynthObstacleDset-v1-Ctc' / 'train')
# frt = dset[5]
# frt.update(Exp1205_Discrepancy_ImgVsInpaiting.translate_from_inpainting_dset_to_gen_image_terms(**frt))
# show(frt.semseg_errors_label)

# python -m src.a12_inpainting.synth_obstacle_dset 1204-SynthObstacleDset-v2MoreObj-Ctc-val --num-workers 1
# python -m src.a12_inpainting.synth_obstacle_dset 1204-SynthObstacleDset-v2MoreObj-Ctc-PatchSampler-val

# python -m src.a12_inpainting.synth_obstacle_dset 1204-SynthObstacleDset-v2MoreObj-Ctc-train --num-workers 3

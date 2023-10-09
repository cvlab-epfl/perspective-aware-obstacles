from pathlib import Path
from functools import partial, lru_cache

import cv2 as cv
import numpy as np
import h5py
from tqdm import tqdm
from easydict import EasyDict

from ..paths import DIR_DATA
from .file_io import imread, imwrite
from ..common.registry import ModuleRegistry2
from .demo_case_selection import DatasetRegistry
from .file_io import hdf5_read_hierarchy_from_group
from .grid_randomizer import square_grid_randomizer


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


@lru_cache(4)
def get_yx_maps(height_width):
	H, W = height_width
	pos_h = np.arange(H, dtype=np.float32)
	pos_w = np.arange(W, dtype=np.float32)
	pos_h = np.tile(pos_h[:, None], [1, W])
	pos_w = np.tile(pos_w[None, :], [H, 1])
	return np.stack([pos_h, pos_w], axis=0)

@ModuleRegistry2('1230_SynthObstacle_PatchSampler')
class SynthObstacleDsetTrainingSampler:
	cfg_768 = EasyDict(
		crop_size_xy = [768, 384],
		grid_step = 20,
		grid_random_offset = 100,
		shuffled = True,
		num_epochs = 300,
	)

	configs = [
		EasyDict(name = 'v1-768',
			**cfg_768,
		),
		EasyDict(name = 'v1-768-entSETR',
			extra_features = ['attentropy/SETR'],
			storage_name = 'v1-768',
			**cfg_768,
		),
		EasyDict(name = 'v1-512',
			crop_size_xy = [512, 256],
			grid_step = 20,
			grid_random_offset = 30,
			shuffled = True,
			num_epochs = 300,
		),
		EasyDict(name = 'v1-480',
			crop_size_xy = [480, 480],
			grid_step = 20,
			grid_random_offset = 100,
			shuffled = True,
			num_epochs = 300,
		),
		EasyDict(name = 'v1-384',
			crop_size_xy = [384, 384],
			grid_step = 20,
			grid_random_offset = 100,
			shuffled = True,
			num_epochs = 300,
		),
	]
	
	CHANNELS_TO_SAMPLE = [
		'image_fused', 
		'obstacle_instance_map', 
		'road_mask',
	]

	def storage_path(self, dset=None):
		dset = dset or self.dset
		dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')
		storage_name = self.cfg.get('storage_name', self.cfg.name)
		return DIR_DATA / '1230_SynthObstacle' / f'patchSampler_{storage_name}' / f'{dk}.hdf5'

	def __init__(self, cfg=None):
		self.cfg = EasyDict(self.configs[0])
		if cfg is not None:
			self.cfg.update(cfg)

		# self.dset = dset
		# self.name = f'{self.cfg.mod_dataset.name}-PatchSampler'
		# self.split = self.cfg.mod_dataset.split
		self.extra_channels_to_load = {}
		self.preproc_func = None

		extra_channels = {}
		for ef in self.cfg.get('extra_features', []):
			extra_channels.update(self.init_extra_features(ef))

		self.set_channels(
			channel_names_to_sample = self.CHANNELS_TO_SAMPLE,
			extra_channels_to_load = extra_channels,
			renames = {},
		)

	def set_channels(self, channel_names_to_sample : list = None, extra_channels_to_load : dict = None, renames : dict = None):
		"""
		@param channel_names_to_sample: channels from the sampled dataset
		@param extra_channels_to_load: {name: channel_obj}
		@param renames: {old_name: new_name}
		"""
		if channel_names_to_sample is not None:
			self.channels_to_sample = list(set(channel_names_to_sample))

		if extra_channels_to_load is not None:
			self.extra_channels_to_load = extra_channels_to_load

		if renames is not None:
			self.renames = renames


	def init_extra_features(self, name):

		cat, inst = name.split('/')

		if cat == 'attentropy':			
			from .attentropy_loader import ExtraLoaderAttentropyForSampler
			return dict(
				attentropy = ExtraLoaderAttentropyForSampler(inst),
			)

		return {}

	# def load_dataset(self, dset):
	# 	self.dset = dset

	# 	#DIR_DATA / f'dataset_{self.cfg.name}', # default path from name
	# 	self.sampling_info = None
	
	# 	path_sampling_data = self.storage_path()

	# 	if path_sampling_data.is_file():
	# 		with h5py.File(path_sampling_data, 'r') as file_sampling:
	# 			self.epoch_len = file_sampling['sampling/frame_idx'].shape[1]
	# 	else:
	# 		self.epoch_len = self.dset.__len__()


	def load(self, dset):
		self.dset = dset

		with h5py.File(self.storage_path(), 'r') as file_sampling:
			self.sampling_info = hdf5_read_hierarchy_from_group(file_sampling)
		
		self.epoch_len = self.sampling_info.sampling.frame_idx.shape[1]


	def sampling_generate_for_frame(self, fr):
		sampling_cfg = self.cfg
		
		crop_size_xy = np.array(sampling_cfg.crop_size_xy, dtype=np.int32)
		crop_radius = np.linalg.norm(crop_size_xy)

		#fr = EasyDict(frame_id = fr_info['frame_id'], frame_id_str = fr_info['frame_id_str'])
		#fr.update(self.dset.load_frame_labels(fr_info))

		frame_size_xy = np.array(fr.road_mask.shape[::-1], dtype=np.int32)

		tl_x, tl_y, bb_w, bb_h = cv.boundingRect(fr.road_mask.astype(np.uint8))
		
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
					roi_mask_crop = fr.road_mask[crop_slice]
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

	def gen__run(self, dset, **_):
		self.dset = dset

		num_epochs = self.cfg.num_epochs
		# num_patches_per_frame = np.zeros(self.dset.frames.__len__(), dtype=np.uint16)
		num_patches_per_frame = np.zeros(self.dset.__len__(), dtype=np.uint16)

		sp = self.storage_path()
		sp.parent.mkdir(parents=True, exist_ok=True)
		with h5py.File(sp, 'w') as file_sampling:
			#frs = self.dset.frames
			for i, fr in tqdm(enumerate(dset), total=dset.__len__()):
				patch_info = self.sampling_generate_for_frame(fr)

				num_patches_per_frame[i] = patch_info.sample_tls.__len__()

				g = file_sampling.create_group(fr['frame_id_str'])
				for k, v in patch_info.items():
					g[k] = v

			print(num_patches_per_frame)

			# find frames with non-zero patches
			valid_frame_ids = np.where(num_patches_per_frame > 0)[0].astype(np.uint16)
			num_frames = valid_frame_ids.__len__()

			# permute patches
			rng = np.random.default_rng()

			patch_indices_all = np.tile(np.arange(0, num_epochs, dtype=np.uint16), (dset.__len__(), 1))
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

	def sampling_get_frame(self, epoch_idx, sample_idx):
		if self.sampling_info is None:
			self.sampling_load()

		fr_idx = self.sampling_info.sampling.frame_idx[epoch_idx, sample_idx]
		patch_idx = self.sampling_info.sampling.patch_idx[fr_idx, epoch_idx]

		fr_data = self.dset[fr_idx]

		# print('extra channels', fr_data, self.dset)

		# extra channels to load
		for ch_name, ch_obj in self.extra_channels_to_load.items():
			fr_data[ch_name] = ch_obj.read_value(**fr_data, dset=self.dset)

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

		channels_to_crop = set(self.channels_to_sample).union(self.extra_channels_to_load.keys())
		#print(channels_to_crop, 'vs', fr_data.keys())

		fr = EasyDict({
			k: fr_data[k][crop] 
			for k in channels_to_crop
		})
		if b_flip:
			fr = EasyDict({
				k: v[:, ::-1].copy() # copy because torch doesn't like negative strides
				for k, v in fr.items()
			})


		# print(fr.keys())
		pos_YX = get_yx_maps(fr.image_fused.shape[:2])
		fr.pos_encoding_X = pos_YX[1] + tl[1]
		fr.pos_encoding_Y = pos_YX[0] + tl[0]

		fr.frame_id = fr_data.frame_id

		renamed = {
			new_name: fr[old_name]
			for (old_name, new_name) in self.renames.items()
		}
		fr.update(renamed)

		# print('frame produced', fr.keys())

		if self.preproc_func is not None:
			fr = self.preproc_func(**fr)
		
		# print('frame preproc', fr.keys())

		return fr

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

		
import click

@click.group()
def main():
	...

@main.command()
@click.argument('name')
@click.argument('dset')
def gen(name, dset):
	""" Generate the synthetic dataset for a given name """
	
	dset_obj = DatasetRegistry.get_implementation(dset)

	mod = ModuleRegistry2.get_implementation('1230_SynthObstacle_PatchSampler', name)
	mod.gen__run(dset_obj)
	
if __name__ == '__main__':
	main()


# python -m src.a12_inpainting.synth_obstacle_patch_sampler gen v1_768 1230_SynthObstacle_Fusion_v2sharp_cityscapes-val

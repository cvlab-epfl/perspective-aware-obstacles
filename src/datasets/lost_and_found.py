
import logging
from pathlib import Path
import numpy as np
import os, re
from .dataset import DatasetBase, ChannelLoaderImage, ChannelLoaderHDF5, imread
from ..paths import DIR_DSETS, DIR_DATA

log = logging.getLogger('exp')

DIR_LOST_AND_FOUND = Path(os.environ.get('DIR_LAF', DIR_DSETS / 'dataset_LostAndFound' / '2048x1024_webp'))
DIR_LOST_AND_FOUND_SMALL = Path(os.environ.get('DIR_LAF_SMALL', DIR_DSETS / 'dataset_LostAndFound' / '1024x512'))

LAF_CLASS_NAMES = """Background
Free space
Crate (black) 
Crate (black - 2x stacked)
Crate (black - upright)
Crate (gray)
Crate (gray - 2x stacked)
Crate (gray - upright)
Bumper
Cardboard box
Crate (blue)
Crate (blue - small)
Crate (green)
Crate (green - small)
Exhaust pipe
Headlight
Euro pallet
Pylon
Pylon (large)
Pylon (white)
Rearview mirror
Tire
Cardboard box
Plastic bag (bloated)
Styrofoam
Ball
Bicycle
Dog (black)
Dog (white)
Kid Dummy
Bobby car (gray)
Bobby car (red)
Bobby car (yellow)
Marker pole (lying)
Post (red - lying)
Post stand
Timber (small)
Timber (squared)
Wheel cap
Wood (thin)
Kid (walking)
Kid (on a bobby car)
Kid (on a small bobby car)
Kid (crawling)""".split('\n')


class DatasetLostAndFound(DatasetBase):

	# invalid frames are those where np.count_nonzero(labels_source) is 0
	INVALID_LABELED_FRAMES = {
		'train': [44,  67,  88, 109, 131, 614],
		'test': [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
	}

	name = 'LostAndFound'

	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']

	class_names = LAF_CLASS_NAMES

	def __init__(self, dir_root=DIR_LOST_AND_FOUND, split='train', only_interesting=True, only_valid=True, b_cache=False):
		"""
		:param split: Available splits: "train", "test"
		:param only_interesting: means we only take the last frame from each sequence:
			in that frame the object is the closest to the camera
		"""
		super().__init__(b_cache=b_cache)

		self.dir_root = Path(dir_root)
		self.split = split
		self.only_interesting = only_interesting
		self.only_valid = only_valid

		# https://github.com/mcordts/cityscapesScripts#dataset-structure

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_labelIds{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_instanceIds{channel.img_ext}',
			),

			# disparity precomputed disparity depth maps
			# To obtain the disparity values, compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256., while a value p = 0 is an invalid measurement. 
			# Warning: the images are stored as 16-bit pngs, which is non-standard and not supported by all libraries.
			disparity = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/disparity/{dset.split}/{fid}_disparity{channel.img_ext}',
			),
		)

		self.channel_disable('disparity')

	def load_roi(self):
		"""
		Load a ROI which excludes the ego-vehicle and registration artifacts
		"""
		try:
			self.roi = imread(self.dir_root / 'LAF_roi.png') > 0
		except Exception as e:
			log.warning(f'LAF failed to load roi {self.dir_root / "LAF_roi.png"}')

	@staticmethod
	def tr_get_anomaly_gt(labels_source, **_):
		return dict(
			anomaly_gt = labels_source >= 2,
		)

	@staticmethod
	def calc_dir_img(dset):
		return dset.dir_root / 'leftImg8bit' / dset.split

	@staticmethod
	def calc_dir_label(dset):
		return dset.dir_root / 'gtCoarse' / dset.split

	def discover(self):
		for img_ext in self.IMG_FORMAT_TO_CHECK:
			self.frames_all = self.discover_directory_by_suffix(
				self.dir_root / 'leftImg8bit' / self.split,
				suffix = f'_leftImg8bit{img_ext}',
			)
			if self.frames_all:
				log.info(f'LAF: found images in {img_ext} format')
				break

		log.info(f'LAF: found images in {img_ext} format')
		self.channels['image'].img_ext = img_ext

		# LAF's PNG images contain a gamma value which makes them washed out, ignore it
		# if img_ext == '.png':
			# self.channels['image'].opts['ignoregamma'] = True

		# parse names to determine scenes, sequences and timestamps
		for fr in self.frames_all:
			fr.apply(self.laf_name_to_sc_seq_t)

		if self.only_valid:
			invalid_indices = self.INVALID_LABELED_FRAMES[self.split]
			valid_indices = np.delete(np.arange(self.frames_all.__len__()), invalid_indices)
			self.frames_all = [self.frames_all[i] for i in valid_indices]

		# orgnize frames into hierarchy:
		# fr = scenes_by_id[fr.scene_id][fr.scene_seq][fr.scene_time]
		scenes_by_id = dict()

		for fr in self.frames_all:
			scene_seqs = scenes_by_id.setdefault(fr.scene_id, dict())

			seq_times = scene_seqs.setdefault(fr.scene_seq, dict())

			seq_times[fr.scene_time] = fr

		self.frames_interesting = []

		# Select the last frame in each sequence, because thats when the object is the closest
		for sc_name, sc_sequences in scenes_by_id.items():
			for seq_name, seq_times in sc_sequences.items():
				#ts = list(seq_times.keys())
				#ts.sort()
				#ts_sel = ts[-1:]
				#self.frames_interesting += [seq_times[t] for t in ts_sel]


				if isinstance(self.only_interesting, bool):
					t_last = max(seq_times.keys())
					self.frames_interesting.append(seq_times[t_last])
				else:
					times_sorted = list(seq_times)
					times_sorted.sort()

					for t in times_sorted[-self.only_interesting:]:
						self.frames_interesting.append(seq_times[t])


		# set self.frames to the appropriate collection
		self.use_only_interesting(self.only_interesting)

		self.load_roi()

		super().discover()

	RE_LAF_NAME = re.compile(r'([0-9]{2})_.*_([0-9]{6})_([0-9]{6})')

	@staticmethod
	def laf_name_to_sc_seq_t(fid, **_):
		m = DatasetLostAndFound.RE_LAF_NAME.match(fid)

		return dict(
			scene_id = int(m.group(1)),
			scene_seq = int(m.group(2)),
			scene_time = int(m.group(3))
		)

	def use_only_interesting(self, only_interesting):
		self.only_interesting = only_interesting
		self.frames = self.frames_interesting if only_interesting else self.frames_all

class DatasetLostAndFoundSmall(DatasetLostAndFound):
	def __init__(self, dir_root=DIR_LOST_AND_FOUND_SMALL, **kwargs):
		super().__init__(dir_root=dir_root, **kwargs)

	def load_roi(self):
		"""
		Load a ROI which excludes the ego-vehicle and registration artifacts
		"""
		self.roi = imread(DIR_DATA / 'cityscapes/roi.png').astype(bool)

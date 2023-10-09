
from pathlib import Path
import os
from functools import partial

from .dataset import DatasetBase, ChannelLoaderImage
from .generic_sem_seg import DatasetLabelInfo
from ..paths import DIR_DSETS

from .india_driving_dataset_labels import labels as idd_labels
IDDLabelInfo = DatasetLabelInfo(idd_labels)

# Labels as defined by the dataset
DIR_IDD_BASE = Path(os.environ.get('DIR_IDD', DIR_DSETS / 'dataset_IDD_IndiaDrivingDataset'))

# Loader
class DatasetIndiaDriving(DatasetBase):
	name = 'IndiaDriving'
	label_info = IDDLabelInfo
	class_names = IDDLabelInfo.class_names

	img_subdir_by_ext = {
		'.webp': 'leftImg8bit_webp',
		'.png': 'leftImg8bit',
	}

	def __init__(self, dir_root=DIR_IDD_BASE / 'idd20k_segmentation', split='train', b_cache=False, img_ext='.webp'):
		super().__init__(b_cache=b_cache)

		self.dir_root = Path(dir_root)
		self.split = split
		self.img_ext = img_ext

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = img_ext,
				file_path_tmpl = '{dset.dir_root}/' + self.img_subdir_by_ext[img_ext] + '/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_labelids{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_instanceids{channel.img_ext}',
			),
		)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(
			self.dir_root / self.img_subdir_by_ext[self.img_ext] / self.split,
			suffix = '_leftImg8bit' + self.img_ext,
		)

		# fid = 144/770875 --> seq = 144, time = 770875
		for fr in self.frames:
			fr.scene_seq, fr.scene_time = map(int, fr.fid.split('/'))

		super().discover()

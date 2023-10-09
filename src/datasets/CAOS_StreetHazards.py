
import logging
from pathlib import Path
import os
from .dataset import DatasetBase, ChannelLoaderImage
from ..paths import DIR_DSETS

# Labels as defined by the dataset
# from .cityscapes_labels import labels as cityscapes_labels
# CityscapesLabelInfo = DatasetLabelInfo(cityscapes_labels)

log = logging.getLogger('exp')

DIR_CAOS_StreetHazards = Path(os.environ.get('DIR_CAOS_StreetHazards', DIR_DSETS / 'dataset_CAOS_StreetHazards'))

# Loader
class DatasetCaosStreetHazards(DatasetBase):
	name = 'CaosStreetHazards'
	# label_info = CityscapesLabelInfo
	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']

	def __init__(self, dir_root=DIR_CAOS_StreetHazards, split='test', b_cache=False):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/{dset.split}/images_webp/{dset.split}/{fid}{channel.img_ext}',
			),
			# labels_source = ChannelLoaderImage(
			# 	img_ext = '.png',
			# 	file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_labelIds{channel.img_ext}',
			# ),
			# instances = ChannelLoaderImage(
			# 	img_ext = '.png',
			# 	file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_instanceIds{channel.img_ext}',
			# ),
		)

		# self.channel_disable('instances')

	def discover(self):
		for img_ext in self.IMG_FORMAT_TO_CHECK:
			img_dir = self.dir_root / self.split / 'images_webp' / self.split
			log.info(f'img dir {img_dir}')
			self.frames = self.discover_directory_by_suffix(
				img_dir,
				suffix = img_ext,
			)
			if self.frames:
				break

		self.channels['image'].img_ext = img_ext
		super().discover()


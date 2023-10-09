
from pathlib import Path
import os, json
from functools import partial

from .dataset import DatasetBase, ChannelLoaderImage
from ..paths import DIR_DSETS
from ..pipeline.frame import Frame


# Labels as defined by the dataset
DIR_WILDDASH_BASE = Path(os.environ.get('DIR_WILDDASH', DIR_DSETS / 'dataset_WildDash/wd_public_02'))

# Loader
class DatasetWildDash(DatasetBase):
	name = 'WildDash'

	img_subdir_by_ext = {
		'.webp': 'leftImg8bit_webp',
		'.png': 'leftImg8bit',
	}

	def __init__(self, dir_root=DIR_WILDDASH_BASE, split='val', b_cache=False):
		super().__init__(b_cache=b_cache)

		self.dir_root = Path(dir_root)
		self.split = split

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.jpg',
				file_path_tmpl = '{dset.dir_root}/images/{fid}{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/labels/{fid}{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/instances/{fid}{channel.img_ext}',
			),
		)

	def discover(self):
		with (self.dir_root / 'panoptic.json').open('r') as fin:
			frame_info = json.load(fin)

		with (self.dir_root / 'wilddash2-meta.json').open('r') as fin:
			class_info = json.load(fin)


		self.frames = [Frame(
				extra_id = entry['id'],
				fid = entry['file_name'].split('.')[0],
			)
			for entry in frame_info['images']
		]

		self.id2class = {
			label['id']: label['name']
			for label in class_info['labels']
		}

		self.class_names = ['NULL'] * (1 + max(self.id2class.keys()))
		for id, name in self.id2class.items():
			self.class_names[id] = name

		super().discover()

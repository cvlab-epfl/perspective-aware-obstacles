from pathlib import Path
import os, json
import numpy as np
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain
from .dataset import DatasetBase, ChannelLoaderImage
from ..paths import DIR_DSETS

DIR_SMALL_OBSTACLE = Path(os.environ.get('DIR_SMALL_OBSTACLE', DIR_DSETS / 'dataset_SmallObstacleDataset' ))

class DatasetSmallObstacle(DatasetBase):
	name = 'SmallObstacleDataset'
	#class_names = ['background', 'road', 'anomaly']

	SPLITS = {
		'test': ['stadium_3', 'vindhya_2'],
		'train': ['file_1', 'file_2', 'file_3', 'file_5', 'seq_1', 'seq_2', 'seq_5', 'seq_6', 'stadium_4'],
		'val': ['seq_3', 'seq_4', 'stadium_1', 'vindhya_1'],
	}

	def __init__(self, dir_root=DIR_SMALL_OBSTACLE, split='train', b_cache=False):
		super().__init__(b_cache=b_cache)

		self.split = split
		self.dir_root = dir_root

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/{dset.split}/{frame.fid_sequence}/image/{frame.fid_index}.png',
			),
			labels_color = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/{dset.split}/{frame.fid_sequence}/labels/{frame.fid_index}.png',
			),
			# TODO sensors
		)
		#self.channel_disable('instances')
		#self.tr_output = self.tr_make_labels_for_anomaly
		self.tr_output = TrsChain(
			self.tr_remove_alpha,
			self.tr_make_labels_for_anomaly,
		)

	def discover(self):
		# choose frames which have labels, many images don't have labels!

		dir_seqs = self.dir_root / self.split

		seqs = [p.stem for p in dir_seqs.iterdir()]
		seqs.sort()

		frames = []

		for seq in seqs:
			dir_labels = dir_seqs / seq / 'labels'

			frame_ids = [p.stem for p in dir_labels.iterdir()]
			frame_ids.sort()

			frames += [
				Frame(
					fid = seq + '_' + n,
					fid_index = n,
					fid_sequence = seq,
				)
				for n in frame_ids
			]
		
		self.frames = frames

		super().discover()

	@staticmethod
	def tr_make_labels_for_anomaly(labels_color, **_):
		# labels = labels_source.copy()
		labels = np.full_like(labels_color, fill_value=255, dtype=np.uint8)
		labels[labels_color == 1] = 0
		labels[labels_color >= 2] = 1
		return dict(labels=labels)

	@staticmethod
	def tr_remove_alpha(image, **_):
		return dict(
			image = image[:, :, :3]
		)


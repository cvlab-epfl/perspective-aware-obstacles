from pathlib import Path
import os, json
import numpy as np
from ..pipeline.frame import Frame
from .dataset import DatasetBase, ChannelLoaderImage
from ..paths import DIR_DSETS

DIR_ROAD_OBSTACLES = Path(os.environ.get('DIR_ROAD_OBSTACLES', DIR_DSETS / 'dataset_RoadObstacle' ))

class DatasetRoadObstacles(DatasetBase):
	name = 'RoadObstacles'
	class_names = ['background', 'road', 'anomaly']


	def __init__(self, dir_root = DIR_ROAD_OBSTACLES, split='sample2', name_override=None): #dir_root=DIR_ROAD_ANOMALY
		super().__init__(b_cache=False)

		if name_override is not None:
			self.name = name_override

		self.split = split
		self.dir_root = dir_root

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/frames/{fid}.webp',
			),
			labels_source = ChannelLoaderImage(
				file_path_tmpl='{dset.dir_root}/frames/{fid}.labels/labels_semantic.png',
			),
			instances = ChannelLoaderImage(
				file_path_tmpl='{dset.dir_root}/frames/{fid}.labels/labels_instance.png',
			),
		)
		#self.channel_disable('instances')

		self.tr_output = self.tr_make_labels_for_anomaly

	def discover(self):
		fid_list = json.loads((self.dir_root / 'splits.json').read_text())[self.split]
		self.frames = [Frame(fid = fid) for fid in fid_list]

		for fr in self.frames:
			p = Path(self.channels['image'].resolve_file_path(dset=self, frame=fr))
			if not p.is_file():
				raise FileNotFoundError(str(p))

		super().discover()

	@staticmethod
	def tr_make_labels_for_anomaly(labels_source, **_):
		# labels = labels_source.copy()
		labels = np.full_like(labels_source, fill_value=255, dtype=np.uint8)
		labels[labels_source == 1] = 0
		labels[labels_source == 2] = 1
		return dict(labels=labels)


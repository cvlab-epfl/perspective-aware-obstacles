from pathlib import Path
import os, json
from ..pipeline.frame import Frame
from .dataset import DatasetBase, ChannelLoaderImage
from ..paths import DIR_DSETS

DIR_ROAD_ANOMALY = Path(os.environ.get('DIR_ROAD_ANOMALY', DIR_DSETS / 'dataset_RoadAnomaly' ))

class DatasetRoadAnomaly(DatasetBase):
	name = 'RoadAnomaly'
	split = 'test'
	class_names = ['background', 'road', 'anomaly']

	def __init__(self, dir_root=DIR_ROAD_ANOMALY, b_cache=False, must_have_instances=False):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.must_have_instances = must_have_instances

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/frames/{fid}{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				file_path_tmpl='{dset.dir_root}/frames/{fid}.labels/labels_semantic.png',
			),
			instances = ChannelLoaderImage(
				file_path_tmpl='{dset.dir_root}/frames/{fid}.labels/labels_instance.png',
			),
		)
		self.channel_disable('instances')
		self.tr_output = self.tr_make_labels_for_anomaly

	def discover(self):
		img_list = json.loads((self.dir_root / 'frame_list.json').read_text())
		self.img_ext = Path(img_list[0]).suffix
		self.channels['image'].img_ext = self.img_ext

		self.frames_all = [Frame(fid = Path(img_filename).stem) for img_filename in img_list]
		self.frames = self.frames_all

		if self.must_have_instances:
			self.frames = [
				fr for fr in self.frames 
				if Path(self.channels['instances'].resolve_file_path(dset=self, frame=fr)).is_file()
			]
		else:
			self.frames = self.frames_all
		
		super().discover()

	@staticmethod
	def tr_make_labels_for_anomaly(labels_source, **_):
		labels = labels_source.copy()
		labels[labels == 2] = 1
		return dict(labels=labels)

	@staticmethod
	def tr_get_anomaly_gt(labels_source, **_):
		return dict(
			anomaly_gt = labels_source >= 2,
		)
	

from pathlib import Path
import numpy as np
import cv2 as cv
from easydict import EasyDict
from tqdm import tqdm

from ..common.registry import ModuleRegistry
from .sys_segmentation import SemSegSystem


from ..paths import DIR_DATA
from ..datasets.dataset import ChannelLoaderImage, imread
from ..datasets.cityscapes import CityscapesLabelInfo



class RoadAreaSystem:

	def __init__(self, cfg):
		self.cfg = cfg

	def init_storage(self):
		out_dir = DIR_DATA / '1207roadar-{channel.ctx.cfg.name}' / '{dset.name}-{dset.split}' 
	
		self.storage = dict(
			road_area_label = ChannelLoaderImage(out_dir / 'labels' / '{fid_no_slash}_roadAreaLabel.png'),
			road_area_demo = ChannelLoaderImage(out_dir / 'demo' / '{fid_no_slash}_roadAreaDemo.webp'),
		)
		for c in self.storage.values(): c.ctx = self

	def load_values(self, frame):
		return self.interpret_labels(
			self.storage['road_area_label'].read_value(**frame),
		)

	def load_into_frame(self, frame):
		channels = self.load_values(frame)
		frame.update(channels)
		return channels

	def tr_load(self, **fr):
		return self.load_into_frame(fr)

	def measure_covered_objects_frame(self, dset, idx):
		fr = dset[idx]
		self.load_into_frame(fr)
		gt_obstacle = fr.label_pixel_gt == 1
		
		num_common = np.count_nonzero(gt_obstacle & fr.labels_road_mask)
		num_missed = np.count_nonzero(gt_obstacle & (~fr.labels_road_mask))

		return dict(
			num_common = num_common,
			num_missed = num_missed,
		)


	def measure_covered_objects(self, dset):
		num_common = 0
		num_missed = 0

		for i in tqdm(range(dset.__len__())):
			r = self.measure_covered_objects_frame(dset, i)

			num_common += r['num_common']
			num_missed += r['num_missed']

		print(f'common: {num_common} | missed: {num_missed}')
		print(f'TPR = {num_common / (num_common + num_missed)}')

		return dict(
			num_common = num_common,
			num_missed = num_missed,
		)



	@staticmethod
	def interpret_labels(road_area_label, **_):

		roi = road_area_label <  255

		return EasyDict(
			labels_road_mask = (0 < road_area_label) & roi,
			roi = roi,
			obstacle_from_sem = road_area_label == 2,
		)

	@classmethod
	def get_implementation(cls, name):
		return ModuleRegistry.get(cls, name)


def calc_road_area_contour(sem_class_prediction, selected_classes, roi = None, **_):
	"""
	0 - non-road
	1 - road area (except for holes)
	2 - holes inside of road area
	255 - out of roi
	"""
	road_mask_with_holes = np.zeros_like(sem_class_prediction, dtype=bool)
	
	for c in selected_classes:
		road_mask_with_holes |= sem_class_prediction == c
	
	if roi is not None:
		road_mask_with_holes &= roi
	
	contours, _ = cv.findContours(
		image = road_mask_with_holes.astype(np.uint8),
		mode = cv.RETR_EXTERNAL,
		#method = cv.CHAIN_APPROX_TC89_L1,
		method = cv.CHAIN_APPROX_SIMPLE,
	)
	
	road_mask_filled = cv.drawContours(
		image = np.zeros_like(sem_class_prediction, dtype=np.uint8),
		contours = contours,
		contourIdx = -1, # all
		color = 1,
		thickness = -1, # fill
	)
	
	road_label = np.zeros_like(sem_class_prediction, dtype=np.uint8)
	road_label[road_mask_filled.astype(bool)] = 2
	road_label[road_mask_with_holes] = 1 
	
	if roi is not None:
		road_label[np.logical_not(roi)] = 255
	
	return road_label


#TODO write the name only once
@ModuleRegistry(RoadAreaSystem, 'semcontour-roadwalk-v1')
class RoadAreaSemContour(RoadAreaSystem):

	ROAD_AREA_CLASSES_DEFAULT = [CityscapesLabelInfo.name2id[c] for c in ['road', 'sidewalk']]

	@classmethod
	def configs(cls):

		base = dict(
			roi_name = 'LAF',
			road_area_classes = cls.ROAD_AREA_CLASSES_DEFAULT,
		)

		return [
			EasyDict(
				name = 'semcontour-roadwalk-v1',
				semseg_name = 'gluon-psp-ctc',
				**base,
			),
			EasyDict(
				name = 'semcontour-roadwalk-gDeeplab101',
				semseg_name = 'gluon-deeeplab101-ctc',
				**base,
			),
			EasyDict(
				name = 'semcontour-roadwalk-gDeeplabW',
				semseg_name = 'gluon-deeplabW-ctc',
				**base,
			),
			EasyDict(
				name = 'semcontour-roadwalk-DeepLab3W',
				semseg_name = 'DeepLab3W-ctc',
				**base,
			),
			EasyDict(
				name = 'semcontour-roadwalk-fastscnn',
				semseg_name = 'gluon-fastscnn-ctc',
				**base,
			),
		]

	def __init__(self, cfg):
		super().__init__(cfg)

		self.sys_semseg =  SemSegSystem.get_implementation(self.cfg.semseg_name)
		self.sys_semseg.init_storage()

		# if self.cfg.roi_name == 'LAF':
		# 	from ..datasets.lost_and_found import DatasetLostAndFound
		# 	dset = DatasetLostAndFound()
		# 	dset.load_roi()
		# 	self.roi = dset.roi
		# else:
		# 	self.roi = None
	

	def process_and_save(self, frame):
		sem_class_prediction = self.sys_semseg.storage['sem_class_prediction'].read_value(**frame)

		frame.sem_class_prediction = sem_class_prediction

		frame.road_area_label = calc_road_area_contour(
			sem_class_prediction = sem_class_prediction, 
			roi = self.roi, 
			selected_classes = self.cfg.road_area_classes,
		)

		for k in ['road_area_label']:
			self.storage[k].write_value(frame[k], **frame)

	def process_dset(self, dset):
		if 'LAF' in dset.name or dset.name == 'LostAndFound':
			from ..datasets.lost_and_found import DatasetLostAndFound
			self.roi = DatasetLostAndFound().load_roi()
		else:
			self.roi = None

		for fr in tqdm(dset):
			self.process_and_save(fr)


@ModuleRegistry(RoadAreaSystem, 'gt')
class RoadAreaGt(RoadAreaSystem):

	default_cfg = EasyDict(
		name = 'gt',
	)

	def process_and_save(self, frame):
		frame.road_area_label = calc_road_area_contour(
			sem_class_prediction = frame.label_pixel_gt, 
			selected_classes = [0, 1, 2],
		)

		for k in ['road_area_label']:
			self.storage[k].write_value(frame[k], **frame)

	def process_dset(self, dset):
		for fr in tqdm(dset):
			self.process_and_save(fr)



@ModuleRegistry(RoadAreaSystem, 'LAFroi')
class RoadAreaRoiImage(RoadAreaSystem):
	default_cfg = EasyDict(
		name = 'LAFroi',
		islands_from = 'semcontour-roadwalk-v1',
		roi_img = Path(__file__).with_name('LAF_roi_2048.png'),
	)

	def __init__(self, cfg):
		self.cfg = cfg

		if self.cfg.islands_from:
			self.sys_ra_islands = RoadAreaSystem.get_implementation(self.cfg.islands_from)
		else:
			self.sys_ra_islands = None
		
		self.roi_mask = imread(self.cfg.roi_img).astype(bool)

	def load_values(self, frame):

		if self.sys_ra_islands is not None:
			labels_for_islands = self.sys_ra_islands.load_values(frame)
			obstacle_from_sem = labels_for_islands['obstacle_from_sem']
		else:
			obstacle_from_sem = np.zeros_like(self.roi_mask)

		return dict(
			labels_road_mask = self.roi_mask,
			roi = self.roi_mask,
			obstacle_from_sem = obstacle_from_sem,
		)


import click
from .demo_case_selection import DatasetRegistry


@click.group()
def main():
	...

@main.command()
@click.argument('sys_name')
@click.argument('dset_name')
def generate(sys_name, dset_name):
	system = RoadAreaSystem.get_implementation(sys_name)
	system.init_storage()
	# system.load()

	for dsn in dset_name.split(','):
		print(f'Processing: {sys_name} - {dsn}')
		dset = DatasetRegistry.get_implementation(dsn)
		system.process_dset(dset)


@main.command()
@click.argument('sys_name')
@click.argument('dset_name')
def measure_covered_objects(sys_name, dset_name):
	system = RoadAreaSystem.get_implementation(sys_name)
	system.init_storage()
	# system.load()

	for dsn in dset_name.split(','):
		dset = DatasetRegistry.get_implementation(dsn)
		print(f'Measure covered objects: {sys_name} - {dsn}')
		system.measure_covered_objects(dset)


if __name__ == '__main__':
	main()


# python -m src.a12_inpainting.sys_road_area generate semcontour-roadwalk-v1 LostAndFound-test
# python -m src.a12_inpainting.sys_road_area generate semcontour-roadwalk-v1 LostAndFound-train
# python -m src.a12_inpainting.sys_road_area generate semcontour-roadwalk-v1 RoadAnomaly-test
# python -m src.a12_inpainting.sys_road_area generate semcontour-roadwalk-v1 FishyLAF-val

# python -m src.a12_inpainting.sys_road_area generate gt LostAndFound-train

# python -m src.a12_inpainting.sys_road_area measure-covered-objects semcontour-roadwalk-v1 $dsets
# python -m src.a12_inpainting.sys_road_area generate semcontour-roadwalk-DeepLab3W $dsets
# python -m src.a12_inpainting.sys_road_area measure-covered-objects semcontour-roadwalk-DeepLab3W $dsets
 
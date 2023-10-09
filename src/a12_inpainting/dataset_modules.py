
import click
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path

from ..paths import DIR_DATA, DIR_DSETS
from ..pipeline.frame import Frame
from ..common.registry import ModuleRegistry2

#import sys
# sys.path.append(str(Path(__file__).absolute().parents[3] / 'wuppertal-dataset-obstacles'))
#print(str(Path(__file__).absolute().parents[2]))
#sys.path.append(str(Path(__file__).absolute().parents[2] / 'wuppertal-dataset-obstacles'))

from road_anomaly_benchmark.datasets import DatasetRegistry as wp_dsets


class WrapWP:
	tr_post_load_pre_cache = None


	def __init__(self, dset_name):
		self.dset_name = dset_name
		self.dset_key = dset_name
		try:
			self.name, self.split = dset_name.split('-')
		except Exception as e:
			print(dset_name)
			raise e


	def __len__(self):
		return self.dset.__len__()

	def discover(self):
		self.dset = wp_dsets.get(self.dset_name)
		self.dset_key = self.dset.cfg.get('name_for_persistence', self.dset.cfg.name)
		print('Discoverred key ', self.dset_key)

		try:
			self.name, self.split = self.dset_key.split('-')
		except Exception as e:
			print('Splitting dset key', self.dset_key)
			raise e

	def __getitem__(self, i):
		try:
			fr = Frame(self.dset[i])
		except Exception as e:
			fr = Frame(self.dset.get_frame(i, 'image'))

		# this can perhaps be serialized by multiprocessing
		dset_key = fr.get('dset_name')
		if dset_key:
			name, split = dset_key.split('-', maxsplit=1)
			fr.dset = EasyDict(
				dset_key = dset_key,
				name = name,
				split = split,
			)
		else:
			fr.dset = EasyDict(
				dset_key = self.dset_key,
				name = self.name,
				split = self.split,
			)

		if 'label_pixel_gt' in fr:
			fr.labels = fr.label_pixel_gt
		elif 'semantic_class_gt' in fr and 'classes' in self.dset.cfg:
			fr.labels = np.full_like(fr.semantic_class_gt, 255)
			fr.labels[fr.semantic_class_gt == self.dset.cfg.classes.usual] = 0
			fr.labels[fr.semantic_class_gt == self.dset.cfg.classes.anomaly] = 1
		else:
			# make fake labels
			h, w = fr.image.shape[:2]
			h_half = h//2
			w_half = w//2
			fr.labels = np.full((h, w), 255, dtype=np.uint8)
			fr.labels[h_half:] = 0
			fr.labels[h_half+100:h_half+150, w_half-50:w_half+50] = 1


		if self.tr_post_load_pre_cache is not None:
			fr.update(self.tr_post_load_pre_cache(frame=fr, **fr))

		return fr

	def __iter__(self):
		for i in range(self.__len__()):
			yield self[i]

	@classmethod
	def list_dsets(cls):
		# print(wp_dsets.list_available_dsets())
		return [
			cls(name)
			for name in wp_dsets.list_available_dsets()
		]
		

	# @property
	# def name(self):
	# 	name, split = self.cfg.name.split('-')
	# 	return name

	# @property
	# def split(self):
	# 	name, split = self.cfg.name.split('-')
	# 	return split


class DatasetRegistry:
	
	from ..datasets.cityscapes import DatasetCityscapes
	from ..datasets.lost_and_found import DatasetLostAndFound
	from ..datasets.india_driving_dataset import DatasetIndiaDriving
	from ..datasets.road_anomaly import DatasetRoadAnomaly
	from ..datasets.road_anomaly_2 import DatasetRoadObstacles
	from ..datasets.fishyscapes import DatasetFishyscapesConvertedLAF
	from ..datasets.small_obstacle_dataset import DatasetSmallObstacle
	from .synth_obstacle_dset import SynthObstacleDset, SynthObstacleDsetTrainingSampler
	from . import synth_obstacle_dset2

	dsets = {
		getattr(d, 'dset_key', f'{d.name}-{d.split}'): d
		for d in [
			# DsetSelectionV1(),

			DatasetCityscapes(split='train'),
			DatasetCityscapes(split='val'),
			DatasetLostAndFound(split='train', only_interesting=False),
			DatasetLostAndFound(split='test', only_interesting=False),

			DatasetIndiaDriving(split='train'),
			DatasetIndiaDriving(split='val'),

			DatasetRoadAnomaly(),

			# DatasetRoadAnomaly2(split='sample1'),
			# DatasetRoadAnomaly2(split='test'),
			
			DatasetRoadObstacles(split='v003'),
			DatasetRoadObstacles(split='UnusualSurface'),
			DatasetRoadObstacles(name_override='RoadObstacles2048p', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048_padding', split='full'),
			DatasetRoadObstacles(name_override='RoadObstacles2048p', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048_padding', split='nodog'),
			DatasetRoadObstacles(name_override='RoadObstacles2048p', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048_padding', split='nobig'),

			#DatasetRoadObstacles(name_override='RoadObstacles2048b3', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048b3', split='full'),
			DatasetRoadObstacles(name_override='RoadObstacles2048', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048', split='full'),
			
			#DatasetRoadObstacles(name_override='RoadObstacles2048b7', dir_root = DIR_DSETS / 'dataset_RoadObstacle_2048b7', split='full'),

			#DatasetRoadAnomaly2(split='cwebp'),
			#DatasetRoadAnomaly2(split='pyrDown'),

			DatasetFishyscapesConvertedLAF(split='val'),
			DatasetFishyscapesConvertedLAF(split='LafRoi'),

			SynthObstacleDset.get_implementation('1204-SynthObstacleDset-v1-Ctc-train'),
			SynthObstacleDset.get_implementation('1204-SynthObstacleDset-v1-Ctc-val'),
			SynthObstacleDset.get_implementation('1204-SynthObstacleDset-v2MoreObj-Ctc-train'),
			SynthObstacleDset.get_implementation('1204-SynthObstacleDset-v2MoreObj-Ctc-val'),

			SynthObstacleDsetTrainingSampler.get_implementation('1204-SynthObstacleDset-v1-Ctc-PatchSampler-train'),
			SynthObstacleDsetTrainingSampler.get_implementation('1204-SynthObstacleDset-v1-Ctc-PatchSampler-val'),
			SynthObstacleDsetTrainingSampler.get_implementation('1204-SynthObstacleDset-v2MoreObj-Ctc-PatchSampler-train'),
			SynthObstacleDsetTrainingSampler.get_implementation('1204-SynthObstacleDset-v2MoreObj-Ctc-PatchSampler-val'),

			DatasetSmallObstacle(split='test'),
			DatasetSmallObstacle(split='train'),
			DatasetSmallObstacle(split='val'),
		] + [
			ModuleRegistry2.get_implementation('1230_SynthObstacle_Fusion', name) for name in ModuleRegistry2.list_entries('1230_SynthObstacle_Fusion')	
			# 	'v2sharp_cityscapes-train', 'v2sharp_cityscapes-val',
			# 	'v2blur3_cityscapes-train', 'v2blur3_cityscapes-val',
			# 	'v2blur5_cityscapes-train', 'v2blur5_cityscapes-val',
			# 'Fblur5-v2b_cityscapes-train', 'Fblur5-v2b_cityscapes-val',
			# 'Fblur5-v3persp2_cityscapes-train', 'Fblur5-v3persp2_cityscapes-val', 
			# 'Fblur5-v2b_IndiaDriving-train', 'Fblur5-v2b_IndiaDriving-val',
			# 'Fblur5unwarp1-v2b_cityscapes-train', 'Fblur5unwarp1-v2b_cityscapes-val',
			#]
		] + WrapWP.list_dsets()
	}

	@classmethod
	def get_implementation(cls, name):
		try:
			ds = cls.dsets[name]
		except KeyError as e:
			print(f'Requested {name} but dset keys are', '\n '.join(cls.dsets.keys()))
			raise e

		ds.discover()
		
		return ds






@click.group()
def main():
	...

@main.command()
@click.argument('dset_name', type=str)
def class_distribution(dset_name):
	dset = DatasetRegistry.get_implementation(dset_name)

	class_areas = []


	for fr in tqdm(dset):
		labels = fr['labels']

		class_areas.append(
			np.bincount(labels.reshape(-1), minlength=255).astype(np.float64),
		)
		
	class_areas_sum = np.sum(class_areas, axis=0)
	areas_all = np.sum(class_areas_sum)
	class_areas_relative = class_areas_sum / areas_all

	print(dset_name)

	for cls_id, area_total, area_relative in zip(range(256), class_areas_sum, class_areas_relative):
		if area_total > 0.1:
			print(f'	{cls_id:03d} - {area_relative}')

	print('	obstacles / road', class_areas_sum[1] / class_areas_sum[0])

	return EasyDict(
		class_areas_sum = class_areas_sum,
		class_areas_relative = class_areas_relative,
	)

if __name__ == '__main__':
	main()

# python -m src.a12_inpainting.demo_case_selection class-distribution  RoadObstacles-v003

# RoadObstacles-v003
# 	000 - 0.38771536825396824
# 	001 - 0.0009895111111111112
# 	255 - 0.6112951206349206
# 	obstacles / road 0.0025521585991477745

# python -m src.a12_inpainting.demo_case_selection class-distribution  FishyLAF-LafRoi

# FishyLAF-LafRoi
# 	000 - 0.14972510814666748
# 	001 - 0.0017223501205444337
# 	255 - 0.848552541732788
# 	obstacles / road 0.01150341543822601

from bdlb.fishyscapes.benchmark_road import FishyscapesLafOnRoad, FishyscapesOnRoad_LafFid
from bdlb.fishyscapes.benchmark import calculate_metrics_perpixAP

from tqdm import tqdm

class FishyscapesOnRoad_LafFid(FishyscapesLafOnRoad):
	def evaluate(self, estimator, dataset=None, name=None, num_points=50, ret_frames=False):
		if dataset is None:
			dataset = self.get_dataset()

		# predict uncertainties over the dataset
		labels = []
		uncertainties = []
		for batch in tqdm(dataset):
			#print(batch.keys())
			fid = str(batch['basedata_id'].numpy(), 'ascii')
			#print(fid)

			labels.append(batch['mask'].numpy())
		
			uncertainties.append(estimator(
				image = batch[self.IMAGE_KEY].numpy(),
				fid = fid,
			))

		metrics = calculate_metrics_perpixAP(
			labels,
			uncertainties,
			num_points=num_points,
		)

		if ret_frames:
			return dict(
				labels = labels,
				uncertainties = uncertainties,
				metrics = metrics
			)
		else:
			return metrics


import numpy as np
from ..paths import DIR_DATA
from ..common.jupyter_show_image import imread

class CachedScoreReader:
	"""
	out_dir_base / 'score_as_image' / f'{fid}__score_as_image.webp'
		selfname = self.cfg['name']
		save_dir_name = f'1209discrep-{selfname}'
		out_dir_base = DIR_DATA / save_dir_name / f'{dset.name}-{dset.split}'

	"""

	@staticmethod
	def pattern_helper(expname, dset_name_split):
		return str(DIR_DATA / '1209discrep' / expname / dset_name_split / 'score_as_image' / '{fid}__score_as_image.webp')


	def __init__(self, pattern):
		self.pattern = pattern

	def __call__(self, image=None, fid=None):
		"""Assigns a random uncertainty per pixel."""

		cached_path = self.pattern.format(fid=fid)
		#print(cached_path)

		uncertainty = imread(cached_path)[:, :, 0].astype(np.float32) * (1./255.)


		# uncertainty = tf.random.uniform(image.shape[:-1])

		# imwrite(
		# 	DIR_DATA / '12XX_ROD' / f'{self.fr_counter}.webp',
		# 	image.numpy(),
		# )
		# self.fr_counter += 1

		return uncertainty
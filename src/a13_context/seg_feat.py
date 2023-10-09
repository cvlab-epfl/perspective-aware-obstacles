from pathlib import Path
import json

import numpy as np
from easydict import EasyDict
import cv2 as cv
# import h5py
from pandas import DataFrame

from ..common.jupyter_show_image import imwrite
from .seg_feat_extractors import SegFeatExtractorRegistry
from ..paths import DIR_DATA
from ..a12_inpainting.dataset_modules import DatasetRegistry


class FrameMultiFeatTester:
	
	def __init__(self, mask_target, mask_roi = None):
		self.mask_target = mask_target.astype(bool)
		self.mask_roi = mask_roi.astype(bool) if mask_roi is not None else np.ones_like(self.mask_target)
	
		self.mask_pyr_levels = self.mask_pyramid_build(self.mask_target, self.mask_roi)
		
		self.mask_pyr_by_width = {
			pyr.mask_target_float.shape[1]: pyr
			for pyr in self.mask_pyr_levels
		}
	
	def __getstate__(self):
		raise NotImplementedError('picle of MultiFeatTester')
	
	@staticmethod
	def mask_pyramid_next(prev_pyr):
		pyr = EasyDict(
			mask_roi = prev_pyr.mask_roi[::2, ::2],
			mask_target_float = cv.resize(prev_pyr.mask_target_float, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
		)
		pyr.mask_target_inroi = pyr.mask_target_float[pyr.mask_roi]
		pyr.mask_target_norm = np.linalg.norm(pyr.mask_target_float[pyr.mask_roi])
		return pyr
	
	@classmethod
	def mask_pyramid_build(cls, mask_target, mask_roi, num_levels=5):
		
		pyr = EasyDict(
			mask_roi = mask_roi,
			mask_target_float = mask_target.astype(np.float32),
		)
		pyr.mask_target_inroi = pyr.mask_target_float[pyr.mask_roi]
		pyr.mask_target_norm = np.linalg.norm(pyr.mask_target_float[pyr.mask_roi])
		
		
		levels = [pyr]
		for i in range(num_levels-1):
			levels.append(cls.mask_pyramid_next(levels[-1]))
		
		return levels
		
	def score_features(self, feats):
		"""
		@param feats: C x H x W
		"""
		sh_c, sh_h, sh_w = feats.shape
		pyr = self.mask_pyr_by_width[sh_w]
		
		feats_in_roi = feats[:, pyr.mask_roi]
		
		length_product = np.linalg.norm(feats_in_roi, axis=1) * pyr.mask_target_norm
		dot_product = np.sum(feats_in_roi * pyr.mask_target_inroi[None], axis=1)
	
		cos_sim = dot_product / length_product	
		cos_sim[length_product < 1e-12] = 0
	
		return cos_sim
	

def mft_for_frame(fr):
	if 'label_pixel_gt' in fr:
		mask_obstacle = fr.label_pixel_gt == 1
		mask_road = fr.label_pixel_gt <= 1
	else:
		mask_obstacle = fr.road_mask
		mask_road = (fr.obstacle_instance_map != 0) | mask_obstacle

	mft = FrameMultiFeatTester(mask_obstacle, mask_road)
	return mft

class FeatTester:
	
	block_names = None
	
	def init_table(self, block_names_and_sizes):
		
		self.block_names = list(block_names_and_sizes.keys())
		self.feats_num_all = np.sum(list(block_names_and_sizes.values()))
		
		self.feats_names = []
		for bl_name, bl_size in block_names_and_sizes.items():
			self.feats_names += [
				f'{bl_name}.{i:04d}' for i in range(bl_size)
			]
		
		self.feats_scores_frames = []
		
# 		self.feats_scores = np.zeros(self.feats_num_all)
		
	
	def add_frame_scores(self, block_scores_by_name):
		
		scores_concat = np.concatenate([
			block_scores_by_name[bl_name]
			for bl_name in self.block_names
		], axis=0)
		
		self.feats_scores_frames.append(scores_concat)
		
		
	def score_frame(self, fr, extractor):
		mft = mft_for_frame(fr)
		
		
		feat_blocks = extractor.extract(fr.image)
		
		if self.block_names is None:
			self.init_table({
				block_name: block_vals.shape[0]
				for block_name, block_vals in feat_blocks.items()
			})
		
		self.add_frame_scores({
			block_name: mft.score_features(block_vals)
			for block_name, block_vals in feat_blocks.items()
		})
	

	def avg_and_save(self, dset_name, extr_name):
		
		feats_scores_avg = np.mean(np.stack(self.feats_scores_frames, axis=0), axis=0)
		
		feats_order = np.argsort(feats_scores_avg)[::-1]
		
		# extr_name = 'EXTR'
		# dset_name = 'DSET'
		
		top_feats = feats_order[:25]
		top_feats_names = [self.feats_names[i] for i in top_feats]

		table = DataFrame({
			'feat': top_feats_names,
			'cosine': feats_scores_avg[top_feats],
		})
				
		
		path_out = DIR_DATA / '1306_SegFeats' / 'cosine' / f'{extr_name}__{dset_name}.json'
		path_out.parent.mkdir(parents=True, exist_ok=True)
		
		path_out.write_text(json.dumps({
			'dset': {
				'name': dset_name,
				'num_frames': self.feats_scores_frames.__len__(),
			},
			'scores': {
				fn: float(fs)
				for fn, fs in zip(self.feats_names, feats_scores_avg)
			},
			}, indent='	',
		))
		
		return table
	

def get_feature_from_blocks(block_dict, feat_name):
	bl_name, channel = feat_name.rsplit('.', 1)
	channel = int(channel)
	return block_dict[bl_name][channel]

	
import click
from tqdm import tqdm
from ..a12_inpainting.vis_imgproc import image_montage_same_shape
from ..road_anomaly_benchmark.evaluation import Evaluation

def name_list(name_list):
	return [name for name in name_list.split(',') if name]

main = click.Group()


@main.command()
@click.argument('dset_names')
@click.argument('extractor_name')
@click.argument('feat_names')
@click.option('--num-examples', type=int, default=1024*1024)
@click.option('--save-outputs/--no-save-outputs', default=False)
def visualize(dset_names, extractor_name, feat_names, num_examples, save_outputs=False):

	extr = SegFeatExtractorRegistry.get(extractor_name)
	feat_names = name_list(feat_names)

	oracle = False
	if 'oracle' in feat_names:
		feat_names.remove('oracle')
		oracle = True
		oracle_key = f'{extractor_name}.oracle'

	for dset_name in name_list(dset_names):
		dset = DatasetRegistry.get_implementation(dset_name)


		num_examples = min(num_examples, dset.__len__())
		step = dset.__len__() // num_examples

		fr_idx_to_vis = range(0, dset.__len__(), step)

		if save_outputs:
			wu_evals = [Evaluation(
					method_name = f'{feat_name}', 
					dataset_name = dset_name,
				)
				for feat_name in (feat_names + [oracle_key] if oracle else [])
			]

		for idx in tqdm(fr_idx_to_vis):
			fr = dset[idx]
			feat_blocks = extr.extract(fr.image)

			feats = [
				get_feature_from_blocks(feat_blocks, feat_name)
				for feat_name in feat_names
			]

			if oracle:
				mft = mft_for_frame(fr)
				
				best_score = -1.
				best_feat_val = None

				for block_name, block_vals in feat_blocks.items():
					block_scores = mft.score_features(block_vals)
					block_best_score_idx = np.argmax(block_scores)
					block_best_score = block_scores[block_best_score_idx]

					if block_best_score > best_score:
						best_score = block_best_score
						best_feat_val = block_vals[block_best_score_idx]

				feats.append(best_feat_val)


			if save_outputs:
				for wu_ev, feat_val in zip(wu_evals, feats):
					wu_ev.save_output(fr, feat_val)


			feats_max_width = int(np.max([f.shape[1] for f in feats]))
			s = feats_max_width / fr.image.shape[1] 
			img_sml = cv.resize(fr.image, None, fx=s, fy=s, interpolation=cv.INTER_AREA)

			demo_img = image_montage_same_shape(
				[img_sml] + feats,
				num_cols = 3, border=2, border_color=(200, 200, 200),
			)	

			path_out = DIR_DATA / '1306_SegFeats' / 'cosine_vis' / f'{extractor_name}__{fr.fid}.webp'
			path_out.parent.mkdir(parents=True, exist_ok=True)
			imwrite(path_out, demo_img)

		if save_outputs:
			for wu_ev in wu_evals:
				wu_ev.wait_to_finish_saving()



@main.command()
@click.argument('dset_names')
@click.argument('extractor_names')
@click.option('--investigate-top', type=int, default=0)
def measure(dset_names, extractor_names, investigate_top=0):

	for extractor_name in name_list(extractor_names):
		extr = SegFeatExtractorRegistry.get(extractor_name)

		for dset_name in name_list(dset_names):
			print(f'--- {extractor_name} vs {dset_name} ---')

			dset = DatasetRegistry.get_implementation(dset_name)

			tester = FeatTester()

			for fr in tqdm(dset):
				tester.score_frame(fr, extr)

			table = tester.avg_and_save(dset_name = dset_name, extr_name = extractor_name)
			print(table)

			if investigate_top > 0:
				raise NotImplementedError()




if __name__ == '__main__':
	main()


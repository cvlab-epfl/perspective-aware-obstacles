import logging, json, gc
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import cv2 as cv
import click
from tqdm import tqdm
from easydict import EasyDict

from ..common.name_list import name_list
from ..common.jupyter_show import adapt_img_data
from ..paths import DIR_DATA, DIR_EXP
from ..pipeline.transforms import TrsChain
from ..pipeline.config import add_experiment, extend_config, EXPERIMENT_CONFIGS
from ..datasets.dataset import imwrite

from .vis_imgproc import image_montage_same_shape
from .sys_road_area import RoadAreaSystem
from .sys_reconstruction import InpaintingSystem
from .discrepancy_pipeline import Exp12xx_DiscrepancyForInpaintingBase, Exp0560_SwapFgd_ImgAndLabelsVsGen_2048
from .discrepancy_networks import morphology_cleaning_v1, morphology_cleaning_v2

from ..common.registry import ModuleRegistry
from .demo_case_selection import DatasetRegistry
from .metrics import CurveInfoClassification, EvaluatorSystem_PerPixBinaryClassification

log = logging.getLogger('exp')

class ObstaclePipelineSystem:
	default_cfgs = []
	default_cfgs_by_name = {c.name: c for c in default_cfgs}

	

	@classmethod
	def add_config(cls, cfg={}, extend=None):
		base = cls.default_cfgs_by_name[extend] if extend else {}
		
		new_cfg = EasyDict(extend_config(base, cfg))

		cls.default_cfgs.append(new_cfg)
		cls.default_cfgs_by_name[new_cfg['name']] = new_cfg

		return new_cfg

	@classmethod
	def get_implementation(cls, name):

		if name == 'Island':
			return ObstaclePipelineSystem_Island()
		elif name == 'Dummy':
			return ObstaclePipelineSystem_Dummy()
		elif name.startswith('PixelDistance'):
			for cfg in ObstaclePipelineSystem_PixelDistance.configs:
				if cfg.name == name:
					return ObstaclePipelineSystem_PixelDistance(cfg)
			
			raise KeyError(f'Requested {name} but we only have {list(c.name for c in ObstaclePipelineSystem_PixelDistance.configs)}')

		elif name.startswith('LapSimple'):
			from ..a13_context.detectors import ObstaclePipelineSystem_LaplacianSimple
			
			for cfg in ObstaclePipelineSystem_LaplacianSimple.configs:
				if cfg.name == name:
					return ObstaclePipelineSystem_LaplacianSimple(cfg)

		elif name.startswith('LapNet'):
			from ..a13_context.detectors import ObstaclePipelineSystem_LaplacianNet
			
			return ObstaclePipelineSystem_LaplacianNet.get_implementation(name)
	
		elif name.startswith('LapSeg'):
			from ..a13_context.detectors import LapNetRegistry
			return LapNetRegistry.get(name)

		elif name.startswith('15') or name.startswith('16'):
			from ..a15_corr.experiments import ModsCrDetector
			return ModsCrDetector.get(name)

		elif name.startswith('Segmi_'):
			return ObstaclePipelineSystem_Segmi(name[6:])

		else:
			cfg = cls.default_cfgs_by_name.get(name)
			
			if cfg is None:
				msg = f'Requested name {name} but we have only {list(cls.default_cfgs_by_name.keys())}'
				print(msg)
				raise KeyError(msg)
				
			cls_name = cfg.get('cls', 'ObstaclePipelineSystem_Discrepancy')

			classes_by_name = {c.__name__: c for c in [
				ObstaclePipelineSystem_Discrepancy,
				ObstaclePipelineSystem_DiscrepancyResynth,
				ObstaclePipelineSystem_Island,
				ObstaclePipelineSystem_DiscrepancyPatched,
				ObstaclePipelineSystem_DiscrepancyUnwarped,
			]}

			cls_to_create = classes_by_name[cls_name]
			return cls_to_create(cfg)
	
	def get_name(self):
		raise NotImplementedError()

	def get_display_name(self):
		return self.get_name()

	def load(self):
		pass

	def predict_frames(self, frames_iterable, limit_length_to=None):
		raise NotImplementedError()

	def write_thresholded_scores(self, curve_info : CurveInfoClassification, fid_list, labels, scores):
		dir_base = DIR_DATA / '1209discrep' / curve_info.method_name / curve_info.dataset_name

		thr_95 = curve_info.threshold_at_95_tpr

		with ThreadPoolExecutor(12) as tp:
			num_tp = 0
			num_fp = 0
			num_all = 0
			num_pos = 0

			for (fid, fr_label, fr_score) in zip(fid_list, labels, scores):
				fr_roi = fr_label < 255
				fr_gt_positives = fr_label == 1

				detection_95 = fr_score > thr_95
				detection_half = fr_score > 0.5

				tp.submit(imwrite, dir_base / 'threshold_tpr95' / f'{fid}_thrTPR95.png', detection_95.astype(np.uint8) * 255)
				tp.submit(imwrite, dir_base / 'threshold_half' / f'{fid}_thrHalf.png', detection_half.astype(np.uint8) * 255)

				num_all += np.count_nonzero(fr_roi)
				num_pos += np.count_nonzero(fr_gt_positives)

				num_tp += np.count_nonzero(detection_95 & fr_gt_positives)
				num_fp += np.count_nonzero(detection_95 & (fr_label == 0))

			summary = {
				'threshold': thr_95,
				'num_tp': num_tp,
				'num_fp': num_fp,
				'num_all': num_all,
				'num_pos': num_pos,
				'tpr': num_tp / num_pos,
				'fpr': num_fp / (num_all - num_pos),
			}
			(dir_base / 'FPR_AT_TPR95.json').write_text(json.dumps(summary, indent='	'))

			print(summary)


	def run_evaluation(self, dset_name, metric_name='perpixAP', sem_obstacle_enabled=True, limit_length_to=None, b_histogram=False, b_perframe=True, save_fpr95=False, b_morphology=False, wu_eval=False):
		dset = DatasetRegistry.get_implementation(dset_name)
		
		mod_curve = EvaluatorSystem_PerPixBinaryClassification.get_implementation(metric_name)
		
		fr_list = self.predict_frames(dset, limit_length_to=limit_length_to)

		if wu_eval:
			from road_anomaly_benchmark.evaluation import Evaluation

			wu_ev = Evaluation(
				method_name = self.get_name(), 
				dataset_name = dset_name,
			)
			for fr in fr_list:
				if 'dset_name' not in fr:
					fr.dset_name = dset_name

				wu_ev.save_output(fr, fr.anomaly_p)
			
		else:
			wu_ev = None

		# print(label_frame_list[0].shape, label_frame_list[0].dtype)

		fid_list = [fr.fid for fr in fr_list]
		labels = [fr.labels for fr in fr_list]
		# TODO Dataset provided labels_for_TASK like per-pix metrics here
		scores = [fr.anomaly_p for fr in fr_list]

		b_has_islands = 'obstacle_from_sem' in fr_list[0]

		if sem_obstacle_enabled and not b_has_islands:
			log.warning(f'Requested sem_obstacle_enabled=True, but value obstacle_from_sem not in frame')
			sem_obstacle_enabled = False

		if sem_obstacle_enabled:
			sem_obstacle = [fr.obstacle_from_sem for fr in fr_list]
		
		del fr_list
		gc.collect()

		perframe_id_list_arg = fid_list if b_perframe else None

		curve_info = mod_curve.evaluate_results(
			method_name = self.get_name(),
			dataset_name = dset_name,
			display_name = self.get_display_name(),

			label_frame_list = labels,
			prediction_frame_list = scores,

			perframe_id_list = perframe_id_list_arg,
			b_histogram = b_histogram,
		)

		if save_fpr95:
			self.write_thresholded_scores(curve_info, fid_list, labels, scores)

		# we can use the predicted fpr95 threshold to show the tpr95 threshold

		if b_morphology:
			scores_morpho = [morphology_cleaning_v1(score) for score in scores]
			mod_curve.evaluate_results(
				method_name = self.get_name() + '-morpho1',
				dataset_name = dset_name,
				display_name = self.get_display_name() + ' + Morphology1',

				label_frame_list = labels,
				prediction_frame_list = scores_morpho,
				
				perframe_id_list = perframe_id_list_arg,
			)

			scores_morpho = [morphology_cleaning_v2(score) for score in scores]
			mod_curve.evaluate_results(
				method_name = self.get_name() + '-morpho2',
				dataset_name = dset_name,
				display_name = self.get_display_name() + ' + Morphology2',

				label_frame_list = labels,
				prediction_frame_list = scores_morpho,
				
				perframe_id_list = perframe_id_list_arg,
			)


		if wu_ev is not None:
			wu_ev.wait_to_finish_saving()

		if sem_obstacle_enabled:
			
			# TODO oracle
			for score, sem_obs, label in zip(scores, sem_obstacle, labels):
				sem_oracle = sem_obs & (label == 1)
				score[sem_oracle] = 0.9999

			mod_curve.evaluate_results(
				method_name = self.get_name() + '-islandsOracle',
				dataset_name = dset_name,
				display_name = self.get_display_name() + ' + Sem-seg-islands-Oracle',

				label_frame_list = labels,
				prediction_frame_list = scores,
				
				perframe_id_list = perframe_id_list_arg,
			)

			for score, sem_obs in zip(scores, sem_obstacle):
				score[sem_obs] = 0.9999
			
			mod_curve.evaluate_results(
				method_name = self.get_name() + '-islands',
				dataset_name = dset_name,
				display_name = self.get_display_name() + ' + Sem-seg-islands',

				label_frame_list = labels,
				prediction_frame_list = scores,

				perframe_id_list = perframe_id_list_arg,
			)

	@staticmethod
	def store_func(wu_ev, dset_name_default, **fr):
		if 'dset_name' not in fr:
			fr['dset_name'] = dset_name_default
		wu_ev.save_output(fr, fr['anomaly_p'])

	def run_evaluation_store_only(self, dset_name, limit_length_to=None):
		# from road_anomaly_benchmark.evaluation import Evaluation
		from ..pipeline.frame import Frame

		dset = DatasetRegistry.get(dset_name)

		wu_ev = Evaluation(
			method_name = self.get_name(), 
			dataset_name = dset_name,
		)

		class WrapDset:
			def __init__(self, dset):
				self.dset = dset
				# self.dset_iterator = dset.__iter__()
				self.size = dset.__len__()

			@property
			def name(self):
				return self.dset.name

			# def __next__(self):
			# 	fr = self.dset_iterator.__next__()
			# 	return Frame(fr)

			def __getitem__(self, idx):
				fr = self.dset[idx]

				# Erasing code assumes save paths with {dset.name}-{dset.split} 
				# to accomodate that, we insert a 'dset' entry into frames
				# which contains the dset name split by a -
				dset_key = fr.get('dset_name')
				if dset_key:
					dsn, dssplit = dset_key.split('-', maxsplit=1)
					fr.dset = EasyDict(
						name = dsn,
						split = dssplit,
					)

				return Frame(fr)

			def __len__(self):
				return self.dset.__len__()

		# f_store = partial(self.store_func, wu_ev, getattr(dset, 'dset_key', f'{dset.name}-{dset.split}'))
		f_store = partial(self.store_func, wu_ev, getattr(dset, 'dset_key', f'{dset.name}'))

		self.predict_frames(
			WrapDset(dset),
			limit_length_to=limit_length_to, 
			return_frames=False, 
			process_func=f_store,
		)

		wu_ev.wait_to_finish_saving()



class ObstaclePipelineSystem_Island(ObstaclePipelineSystem):
	
	def __init__(self):
		self.sys_roadarea = RoadAreaSystem.get_implementation('semcontour-roadwalk-v1')()
		self.sys_roadarea.init_storage()

	def get_name(self):
		return 'Island'

	def get_display_name(self):
		return 'SemSeg Island Only'


	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):

		frames_out = []

		for i, frame in tqdm(enumerate(frames_iterable)):
			road_area_label = self.sys_roadarea.load_values(frame)
			obstacle_from_sem = road_area_label['obstacle_from_sem']

			frame.anomaly_p = obstacle_from_sem * 0.75

			if process_func is not None:
				process_func(**frame)

			del frame['image']

			if return_frames:
				frames_out.append(frame)

			if limit_length_to > 0 and i >= limit_length_to:
				break

		return frames_out
			

class ObstaclePipelineSystem_Dummy(ObstaclePipelineSystem):

	def __init__(self):
		pass

	def get_name(self):
		return 'Dummy'

	def get_display_name(self):
		return 'Random'

	def predict_frames(self, frames_iterable, limit_length_to=0, return_frames=True, process_func=None):

		frames_out = []

		for i, frame in enumerate(frames_iterable): #tqdm(enumerate(frames_iterable), total=min(frames_iterable.__len__(), limit_length_to)):
			h, w = frame['image'].shape[:2]
			frame['anomaly_p'] = np.random.uniform(size=(h, w)).astype(np.float32)

			if process_func is not None:
				process_func(**frame)

			del frame['image']

			if return_frames:
				frames_out.append(frame)

			if limit_length_to > 0 and i >= limit_length_to:
				break

		return frames_out


class ObstaclePipelineSystem_Segmi(ObstaclePipelineSystem):

	def __init__(self, name):
		self.cfg = EasyDict(
			name = name,
		)

	def load(self):
		import sys
		sys.path.append('/cvlabdata2/home/lis/dev/wuppertal-dataset-obstacles/')
		from some_methods_inference_public import METHOD_KEYS, baselines_module

		key = METHOD_KEYS[self.cfg.name]
		method_object = getattr(baselines_module, self.cfg.name)
		method = method_object(key)

		self.method = method

	def get_name(self):
		return self.cfg.name

	def get_display_name(self):
		return self.cfg.get('display_name', self.get_name())

	def predict_frames(self, frames_iterable, limit_length_to=0, return_frames=True, process_func=None):

		frames_out = []

		for i, frame in tqdm(enumerate(frames_iterable), total=min(frames_iterable.__len__(), limit_length_to)):
			h, w = frame['image'].shape[:2]
			frame['anomaly_p'] = self.method.anomaly_score(frame['image'])

			if process_func is not None:
				process_func(**frame)

			del frame['image']

			if return_frames:
				frames_out.append(frame)

			if limit_length_to > 0 and i >= limit_length_to:
				break

		return frames_out



from ..a14_perspective.cityscapes_pitch_angles import perspective_scale_from_road_mask

class ObstaclePipelineSystem_Discrepancy(ObstaclePipelineSystem):

	exp_cls = Exp12xx_DiscrepancyForInpaintingBase

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

		# calculate the effective config for discrepancy module
		cfg_overlays = [
			# tells the module to write outputs under this system's name
			dict(
				dir_checkpoint = DIR_EXP / self.cfg.discrepancy['name'],
				inference = dict(
					save_dir_name = f'1209discrep/{self.cfg.name}'
				),
			),
			# this system's discrepancy cfg
			self.cfg.discrepancy,
		]

		cfg_discrepancy = self.exp_cls.cfg
		for ovr in cfg_overlays:
			cfg_discrepancy = extend_config(cfg_discrepancy, diff=ovr, warn=True)
		
		self.mod_discrepancy = self.exp_cls(cfg_discrepancy)

		self.b_needs_inpainting = cfg_discrepancy['net']['comparator_net_name'] == 'ComparatorImageToImage'


	def get_name(self):
		return self.cfg.name

	def get_display_name(self):
		return self.cfg.display_name

	def load(self):
		self.mod_discrepancy.init_net('eval')

	def construct_pipeline(self):
		md = self.mod_discrepancy
		pipe = md.construct_default_pipeline('test_v2')

		loaders = []

		if md.sys_road is not None:
			loaders.append(md.sys_road.tr_load)

		if self.cfg.discrepancy.net.separate_gen_image:
			loaders.append(md.sys_inp.frame_load_inpainting)
		
		if self.cfg.discrepancy.net.get('perspective', False):
			loaders.append(perspective_scale_from_road_mask)

		pipe.tr_input += loaders

		return pipe


	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None, prepare_func=None):
		num_frames = frames_iterable.__len__()

		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		md = self.mod_discrepancy
		pipe = self.construct_pipeline()
		md.sampler_test.data_source = range(num_frames)

		if process_func is not None:
			pipe.tr_output.append(process_func)

		if prepare_func is not None:
			pipe.tr_batch_pre_merge.append(prepare_func)

		# run image-saving in background threads
		with ThreadPoolExecutor(6) as thread_pool:
			md.background_thread_pool = thread_pool
			
			result_frames = pipe.execute(frames_iterable, b_accumulate=return_frames, b_grad=False)

			del md.background_thread_pool

		return result_frames


	def run_training_then_eval(self, eval_dsets = ('FishyLAF-val',)):
		b_training_successful = self.mod_discrepancy.training_procedure_on_instance()

		gc.collect()

		if b_training_successful:
			self.load()

			gc.collect()

			for dsname in eval_dsets:
				#self.run_evaluation(dsname)
				self.run_evaluation_store_only(dsname)



class ObstaclePipelineSystem_DiscrepancyResynth(ObstaclePipelineSystem_Discrepancy):
	exp_cls = Exp0560_SwapFgd_ImgAndLabelsVsGen_2048



from .discrepancy_pipeline import infer_patch_fuse_disc
from tqdm import tqdm


class ObstaclePipelineSystem_DiscrepancyPatched(ObstaclePipelineSystem_Discrepancy):

	def load(self):
		super().load()

		sys_inp = self.mod_discrepancy.sys_inp
		sys_inp.load()

		if self.b_needs_inpainting:
			sys_inp.imp.load_default_inpainter_net()


	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):
		
		print('lto', limit_length_to)

		md = self.mod_discrepancy
		ra = self.mod_discrepancy.sys_road


		# with ThreadPoolExecutor(3) as thread_pool:
		# 	md.background_thread_pool = thread_pool
		# 	del md.background_thread_pool
		
		num_frames = frames_iterable.__len__()
		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		result_frames = []

		for i, fr in tqdm(zip(range(num_frames), frames_iterable), total=num_frames):
			ra.load_into_frame(fr)

			fr_before_patching = EasyDict(fr)

			fr.update(infer_patch_fuse_disc(
				self, 
				frt = fr, 
				b_make_inpainting = self.b_needs_inpainting,
				fusion_type = self.cfg['fusion_type'],
				b_show = False,
			))

			md.test_write_result_v2(**fr)

			fr_before_patching.update(
				anomaly_p = fr.anomaly_p,
			)

			if process_func is not None:
				process_func(**fr_before_patching)

			if return_frames:
				result_frames.append(fr_before_patching)
		
		return result_frames

	def run_training_then_eval(self, *a, **k):
		raise NotImplementedError(f'Training {self.get_name()}')


class ObstaclePipelineSystem_DiscrepancyUnwarped(ObstaclePipelineSystem_Discrepancy):

	def load(self):
		super().load()


	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):
		from src.a12_inpainting.demo_case_selection import DatasetRegistry
		from src.a14_perspective.cityscapes_pitch_angles import load_frame_with_perspective_info
		from src.a14_perspective.warp import unwarp_road_frame
		from .discrepancy_pipeline import discrepancy_infer
		import gc

		dset_laf = DatasetRegistry.get_implementation('LostAndFound-train')
		dset_laf.dir_root = dset_laf.dset.DIR_LAF

		dset_laf_test = DatasetRegistry.get_implementation('LostAndFound-test')
		dset_laf_test.dir_root = dset_laf.dset.DIR_LAF

		md = self.mod_discrepancy
		ra = self.mod_discrepancy.sys_road


		# with ThreadPoolExecutor(3) as thread_pool:
		# 	md.background_thread_pool = thread_pool
		# 	del md.background_thread_pool
		
		num_frames = frames_iterable.__len__()
		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		result_frames = []

		for i, fr_from_eval in tqdm(zip(range(num_frames), frames_iterable), total=num_frames):
			try:
				fr = load_frame_with_perspective_info(dset_laf, fr_from_eval.fid)
			except KeyError:
				fr = load_frame_with_perspective_info(dset_laf_test, fr_from_eval.fid)

			fr.image = fr_from_eval.image
			out_sz_xy = fr.image.shape[:2][::-1]

			unwarp_road_frame(fr)

			# infer anomaly_p
			fr.unwarp_anomaly_p = discrepancy_infer(md, fr.unwarp_image[None], batch_size=1)[0]
				
			# warp anomaly_p
			fr.anomaly_p = cv.warpPerspective(
				fr.unwarp_anomaly_p, 
				fr.unwarp_H, 
				tuple(out_sz_xy), 
				flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR,
			)

			md.test_write_result_v2(**fr)

			if process_func is not None:
				process_func(**fr)

			if return_frames:
				result_frames.append(fr)

			gc.collect()
		
		return result_frames



class ObstaclePipelineSystem_PixelDistance(ObstaclePipelineSystem):

	configs = [
		EasyDict(
			name = 'PixelDistance-L1-Inp',
			reconstruction = 'sliding-deepfill-v1',
			distance_function = 'L1',
		),
		EasyDict(
			name = 'PixelDistance-L2-Inp',
			reconstruction = 'sliding-deepfill-v1',
			distance_function = 'L2',
		),
		EasyDict(
			# Multi-Scale-Gradient-Magnitude-Similarity
			name = 'PixelDistance-MSGMS-Inp',
			reconstruction = 'sliding-deepfill-v1',
			distance_function = 'MSGMS',
		),
		EasyDict(
			name = 'PixelDistance-L1-SynthPix',
			reconstruction = 'pix2pixHD_405',
			distance_function = 'L1',
		),
		EasyDict(
			name = 'PixelDistance-L2-SynthPix',
			reconstruction = 'pix2pixHD_405',
			distance_function = 'L2',
		),
		EasyDict(
			# Multi-Scale-Gradient-Magnitude-Similarity
			name = 'PixelDistance-MSGMS-SynthPix',
			reconstruction = 'pix2pixHD_405',
			distance_function = 'MSGMS',
		),
	]

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

		from .pixel_distance_functions import PixelDistanceFunctions
		self.distance_function = getattr(PixelDistanceFunctions, self.cfg.distance_function)

		self.mod_reconstruction = InpaintingSystem.get_implementation(self.cfg.reconstruction)()
		self.mod_reconstruction.init_storage()
		self.reconstruction_cache_channel = self.mod_reconstruction.storage['image_inpainted']

	def get_name(self):
		return self.cfg.name

	def get_display_name(self):
		return self.cfg.name

	def write_visualizations(self, frame):
		fid = frame.fid
		dset = frame.dset
		anomaly_p = frame.anomaly_p
		image = frame.image

		out_dir_base = DIR_DATA / '1209discrep' / self.get_name() / getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')
		out_path_vis = out_dir_base / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		out_path_lowres = out_dir_base / 'score_as_image' / f'{fid}__score_as_image.webp'
	
		
		out_path_lowres.parent.mkdir(exist_ok=True, parents=True)
		imwrite(
			out_path_lowres,
			(anomaly_p * 255).astype(np.uint8),
		)

		out_path_vis.parent.mkdir(exist_ok=True, parents=True)
		demo_img = cv.addWeighted(
			image // 2, 0.5, 
			adapt_img_data(anomaly_p), 0.5,
			0.0,
		)
		imwrite(out_path_vis, demo_img)


	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):

		frames_out = []

		seq_len = frames_iterable.__len__() 
		if limit_length_to not in (0, None):
			seq_len = min(seq_len, limit_length_to)

		for i, frame in tqdm(zip(range(seq_len), frames_iterable), total=seq_len):
			
			img = frame['image'].astype(np.float32)
			rec_img = self.reconstruction_cache_channel.read_value(**frame)
			frame['gen_image'] = rec_img

			rec_img = rec_img.astype(np.float32)
			img *= (1./255)
			rec_img *= (1./255)

			frame['anomaly_p'] = self.distance_function(img, rec_img)
			
			# write images
			self.write_visualizations(frame)

			if process_func is not None:
				process_func(**frame)

			del frame['image']
			del frame['gen_image']

			if return_frames:
				frames_out.append(frame)

		return frames_out














def init_discrepancy_configs():
	add = ObstaclePipelineSystem_Discrepancy.add_config

	AUG_NOISE_PRESET1 = dict(
		layers = (
			(0.18, 1.0),
			(0.31, 0.5),
			(0.84, 0.2),
		),
		magnitude_range = [0.1, 0.6],
	)

	dset_opt_v2 = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
		mod_sampler = 'v1-768',
	)

	dset_opt_v3 = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-val',
		mod_sampler = 'v1-768',
	)

	dset_opt_unw = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5unwarp1-v2b_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5unwarp1-v2b_cityscapes-val',
		mod_sampler = 'v1-512',
	)

	# Image vs Inpainting
	add(dict(
		name = 'ImgVsInp-ds2',
		display_name = 'Image vs Inpainting, dset 2',
		discrepancy = dict(
			name = 'ImgVsInp-ds2', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v3, # SIC
			),
			net = dict(
				separate_gen_image = True,
				perspective = False,
			),
		),
	))
	add(dict(
		name = 'ImgVsInp-ds2b',
		display_name = 'Image vs Inpainting, dset 2b',
		discrepancy = dict(
			name = 'ImgVsInp-ds2b', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				perspective = False,
			),
		),
	))
	add(dict(
		name = 'ImgVsInp-ds3',
		display_name = 'Image vs Inpainting, dset 3',
		discrepancy = dict(
			name = 'ImgVsInp-ds3', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v3,
			),
			net = dict(
				separate_gen_image = True,
				perspective = False,
			),
		),
	))

	# Image vs image repeated
	add(dict(
		name = 'ImgVsSelf-ds2',
		display_name = 'Image vs Self, dset 2',
		discrepancy = dict(
			name = 'ImgVsSelf-ds2', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v3,
			),
			net = dict(
				separate_gen_image = False,
				perspective = False,
			),
		),
	))
	add(dict(
		name = 'ImgVsSelf-ds2b',
		display_name = 'Image vs Self, dset 2b',
		discrepancy = dict(
			name = 'ImgVsSelf-ds2b', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				perspective = False,
			),
		),
	))

	# inpaiting by blur
	for inp_blur_size in (71, 111, 201, 301):
		name = f'ImgVsBlur{inp_blur_size}-ds2b'

		add(dict(
			name = name,
			display_name = 'Image vs Blur, dset 2b',
			discrepancy = dict(
				name = name, # = the checkpoint dir name too
				train = dict(
					**dset_opt_v2,
				),
				net = dict(
					separate_gen_image = False,
					perspective = False,
				),
				gen = dict(
					inpainting_name = f'blur{inp_blur_size}',
				)
			),
		))

	add(dict(
		name = 'ImgVsSelf-ds3',
		display_name = 'Image vs Self, dset 3',
		discrepancy = dict(
			name = 'ImgVsSelf-ds3', # = the checkpoint dir name too
			train = dict(
				**dset_opt_v3,
			),
			net = dict(
				separate_gen_image = False,
				perspective = False,
			),
			sys_road_area = 'semcontour-roadwalk-v1',
		),
	))
	add(dict(
		name = 'ImgVsSelf-dsunw1',
		display_name = 'Image vs Self, dset unw 1',
		cls = 'ObstaclePipelineSystem_DiscrepancyUnwarped',
		discrepancy = dict(
			name = 'ImgVsSelf-unw1', # = the checkpoint dir name too
			train = dict(
				**dset_opt_unw,
			),
			net = dict(
				separate_gen_image = False,
				perspective = False,
			),
		),
	))


	add(dict(
		name = '1409-1-NoCorr',
		display_name = '1409-1-NoCorr',
		discrepancy = dict(
			name = '1409-1-NoCorr', # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = False,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))
	
	add(dict(
		name = '1409-2-2ndLayer',
		display_name = '1409-2-2ndLayer',
		discrepancy = dict(
			name = '1409-2-2ndLayer', # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	add(dict(
		name = '1409-3-2ndLayerNoBlur',
		display_name = '1409-3-2ndLayerNoBlur',
		discrepancy = dict(
			name = '1409-3-2ndLayerNoBlur', # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = False,
		),
	))

	n = '1409-4-2ndLayerNoBlurNoNoise'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = False,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = False,
		),
	))

	n = '1409-5-3rdLayer'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 3,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	n = '1409-6-3rdLayerNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 3,
			),
			preproc_blur = False,
		),
	))

	n = '1410-1-2L-lossBc26'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				class_weights = [1.44113989, 26.46853189],
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	n = '1410-2-2L-lossBc08'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				class_weights = [1, 8],
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	n = '1410-3-2L-lossFoc08'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				loss_name = "focal",
				class_weights = [1, 8],
				focal_loss_gamma = 3.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	n = '1410-4-2L-lossFoc'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 3.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				skips_start_from = 2,
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	loss_foc_opts = dict(
		loss_name = "focal",
		class_weights = None,
		focal_loss_gamma = 3.0, # curve param
		focal_loss_alpha = 0.5, # scale of loss
	)

	n = '1411-1-ResnetFoc'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = dict(
				kernel_size = 5,
				blur_in_training = True,
				blur_in_inference = True,
			)
		),
	))

	n = '1411-2-ResnetFocNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1411-3-Resnet2LFocNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
				skips_start_from = 2,
			),
			preproc_blur = False,
		),
	))


	n = '1411-4-ResnetNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1411-5-ResnetFocNoBlur-vsSelf'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = False, # vs Self
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1411-6-ResnetUnfrozenNaiveFocNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				optimizer = dict(
					learn_rate = 0.0001 * 0.5, # half LR
				),
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
				freeze_backbone = False,
			),
			preproc_blur = False,
		),
	))


	n = '1411-7-ResnetUnfrozenNaiveFocNoBlurNoCorr'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				optimizer = dict(
					learn_rate = 0.0001 * 0.5, # half LR
				),
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = False,
				backbone_name = 'resnext50_32x4d',
				freeze_backbone = False,
			),
			preproc_blur = False,
		),
	))

	n = '1411-8-ResnetFocNoBlurNoCorr'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				**dset_opt_v2,
				**loss_foc_opts,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = False,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	# resnet, NLLoss with class weights
	n = '1412-1-ResnetBc26NoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = [1.44113989, 26.46853189],
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	# new: resnet, foc but default params
	n = '1412-2-ResnetFocdefNoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 2.0, # curve param
				focal_loss_alpha = 0.25, # scale of loss
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1412-3-ResnetNoBlur-vsSelf'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = False,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1412-4-Resnet101NoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	n = '1412-5-ResnetNoBlurNoNoise'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = False,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
		),
	))

	n = '1412-6-ResnetNoBlur-Resynth'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext50_32x4d',
			),
			preproc_blur = False,
			gen = dict(
				inpainting_name = 'pix2pixHD_405',
			)
		),
	))


	n = '1413-0-Resnet101NoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	n = '1413-0b-Resnet101NoBlur'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))


	n = '1413-1-Resnet101NoBlur-NoNoise'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = False,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	n = '1413-2-Resnet101NoBlur-Resynth'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
			gen = dict(
				inpainting_name = 'pix2pixHD_405',
			)
		),
	))

	n = '1413-3-Resnet101NoBlur-NoCorr'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = False,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	n = '1413-4-Resnet101NoBlur-vsSelf'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = False,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))
	n = '1413-4b-Resnet101NoBlur-vsSelf'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = False,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))


	n = '1413-5-Resnet101NoBlur-gtRoad'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = '1412-4-Resnet101NoBlur', # use existing checkpoint
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				class_weights = None,
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
	
			sys_road_area = 'gt',
			gen = dict(
				inpainting_name = 'sliding-deepfill-v1-gtroad',
			)
		),
	))
	
	n = '1413-6-Resnet101NoBlur-foc'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 2.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	n = '1413-6b-Resnet101NoBlur-foc'
	add(dict(
		name = n,
		display_name = n,
		discrepancy = dict(
			name = n, # = the checkpoint dir name too
			train = dict(
				augmentation_noise = AUG_NOISE_PRESET1,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 2.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
				**dset_opt_v2,
			),
			net = dict(
				separate_gen_image = True,
				correlation_layer = True,
				backbone_name = 'resnext101_32x8d',
			),
			preproc_blur = False,
		),
	))

	cfgs_before_last = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	for cfg in cfgs_before_last:
		# perspective mode for perspective dset only
		if 'ds3' in cfg.name:
			name = cfg.name

			add(extend = name, cfg = dict(
				name = f'{name}-Pers',
				display_name = cfg.display_name + ' PerspectiveFeat',
				discrepancy = dict(
					net = dict(
						perspective = True,
					)
				)
			))

	cfg_resynth = dict(
		display_name = 'Resynthesis Orig',
		cls = 'ObstaclePipelineSystem_DiscrepancyResynth',
		discrepancy = dict(
			name = '0560_Diff_SwapFgd_ImgAndLabelVsGen_2048', # = the checkpoint dir name too

			net = dict(
				separate_gen_image = True,
				perspective = False,
			)
		),
	)
	add(dict(
		name = 'Resynth2048Orig',
		**cfg_resynth,
		sys_road_area = None,
	))
	add(dict(
		name = 'Resynth2048Orig-LAFroi',
		sys_road_area = 'LAFroi',
		**cfg_resynth,
	))

	# Training variants: focal loss, noise augmentation
	main_bases = [
		'ImgVsSelf-ds3', 'ImgVsInp-ds3',
		'ImgVsSelf-ds2', 'ImgVsInp-ds2',
		'ImgVsSelf-dsunw1',
	]

	mixin_focalW = dict(
		loss_name = 'focal',
		focal_loss_gamma = 2.0, # curve param
		focal_loss_alpha = 0.25, # scale of loss
	)
	mixin_noise = dict(
		augmentation_noise = AUG_NOISE_PRESET1,
	)

	cfgs_before_last = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	for cfg in cfgs_before_last:
		name = cfg.name

		# add(extend = name, cfg = dict(
		# 	name = f'{name}-NoiseAndFocalW',
		# 	display_name = cfg.display_name + ' NoiseAug+FocalW',
		# 	discrepancy = dict(
		# 		name = f'1218_{name}-NoiseAndFocalW',
		# 		train = dict(
		# 			**mixin_noise,
		# 			**mixin_focalW,
		# 		),
		# 	),
		# ))

		add(extend = name, cfg = dict(
			name = f'{name}-NoiseImg',
			display_name = cfg.display_name + ' NoiseAug',
			discrepancy = dict(
				name = f'{name}-NoiseImg',
				train = mixin_noise,
			),
		))

		# add(extend = name, cfg = dict(
		# 	name = f'{name}-FocalWeighted',
		# 	display_name = cfg.display_name + ' FocalW',
		# 	discrepancy = dict(
		# 		name = f'1216_{name}-FocalWeighted',
		# 		train = mixin_focalW,
		# 	),
		# ))
	
	# cfgs_before = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	# for cfg in cfgs_before:
	# 	name = cfg.name
	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-train2BigObj',
	# 		display_name = cfg.display_name + ' TrainBigObj',
	# 		discrepancy = dict(
	# 			name = f'{name}-train2BigObj',
	# 			train = dict(
	# 				dset_train = '1204-SynthObstacleDset-v2MoreObj-Ctc-PatchSampler-train',
	# 				dset_val = '1204-SynthObstacleDset-v2MoreObj-Ctc-PatchSampler-val',
	# 			),
	# 		),
	# 	))

	# Blur 
	# copy list so that adding will not cause a loop
	cfgs_before_blur = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)

	for cfg in cfgs_before_blur:
		name = cfg.name
		name_after_img = name[3:]

		if name.startswith('Img'):
			for blur in (3, 5, 7):
				
				add(extend = name, cfg = dict(


					name = f'ImgBlurInfer{blur:02d}{name_after_img}',
					display_name = cfg.display_name.replace('Image', f'Image-blur{blur}-infer'),
					blur_key = (1, blur),
					discrepancy = dict(
						name = cfg.discrepancy.name,
						preproc_blur = dict(
							kernel_size = blur,
							blur_in_training = 'no_retrain',
							blur_in_inference = True,
						)
					)
				))

				add(extend = name, cfg = dict(
					name = f'ImgBlur{blur:02d}{name_after_img}',
					display_name = cfg.display_name.replace('Image', f'Image-blur{blur}'),
					blur_key = (blur, blur), # train, inf
					discrepancy = dict(
						name = f'{cfg.discrepancy.name}-Blur{blur:02d}',

						preproc_blur = dict(
							kernel_size = blur,
							blur_in_training = True,
							blur_in_inference = True,
						)
					)
				))
				
	# 			add(extend = name, cfg = dict(
	# 				name = f'ImgBlurTrain{blur:02d}{name_after_img}',
	# 				display_name = cfg.display_name.replace('Image', f'Image-blur{blur}-train'),
	# 				blur_key = (blur, 1),
	# 				discrepancy = dict(
	# 					name = f'{cfg.discrepancy.name}-Blur{blur:02d}',
	# 					preproc_blur = dict(
	# 						kernel_size = blur,
	# 						blur_in_training = True,
	# 						blur_in_inference = False,
	# 					)
	# 				)
	# 			))
				

				# blur 3 at training, blur 5 or 7 at inference
				for blur_inf in (3, 5, 7):
					if blur_inf != blur:
						add(extend = f'ImgBlur{blur:02d}{name_after_img}', cfg = dict(
							name = f'ImgBlur{blur:02d}Inf{blur_inf:02d}{name_after_img}',
							display_name = cfg.display_name.replace('Image', f'Image-blur{blur}t{blur_inf}inf'),
							blur_key = (blur, blur_inf),
							discrepancy = dict(
								preproc_blur = dict(
									kernel_size = blur_inf,
									blur_in_training = 'no_retrain',
									blur_in_inference = True,
								)
							)
						))

	

	# cfgs_before = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	# for cfg in cfgs_before:
	# 	name = cfg.name

	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-HighW',
	# 		display_name = cfg.display_name + ' HighW',
	# 		discrepancy = dict(
	# 			name = cfg.discrepancy.name + '-HighW',
	# 			train = dict(
	# 				class_weights = [1.43138301, 34.96345657],
	# 			)
	# 		)
	# 	))

	# base_for_dsets = 'ImgVsInp-archResy-NoiseImg'
	# cfg = ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name[base_for_dsets]
	# for fusion in ['v2sharp', 'v2blur3', 'v2blur5']:
	# 	name = cfg.name
	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-trainFusion{fusion}',
	# 		display_name = cfg.display_name + ' fusion ' + fusion,
	# 		discrepancy = dict(
	# 			name = f'{name}-trainFusion{fusion}',
	# 			train = dict(
	# 				dset_train = f'1230_SynthObstacle_Fusion_{fusion}_cityscapes-train',
	# 				dset_val = f'1230_SynthObstacle_Fusion_{fusion}_cityscapes-val',
	# 			),
	# 		),
	# 	))

	to_ext_with_gtroad = ['ImgBlur05VsInp-ds2-NoiseImg', 'ImgBlur05VsInp-ds2b-NoiseImg']

	for name in to_ext_with_gtroad:
		cfg = ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name[name]

		add(extend = cfg.name, cfg = dict(
			name = f'{cfg.name}-gtRoad',
			display_name = cfg.display_name + ' GT road',
			discrepancy = dict(
				sys_road_area = 'gt',
				gen = dict(
					inpainting_name = 'sliding-deepfill-v1-gtroad',
				)
			),
		))

	for name, cfg in list(ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name.items()):
		if 'VsInp' in name:
			for INP in ['mmdeepfillv2', 'mmaotgan']:
				new_name = name.replace('VsInp', f'VsInp{INP}')
				# print('Extending', name, 'to', new_name)


				add(extend = cfg.name, cfg = dict(
					name = new_name,
					display_name = cfg.display_name + f' Inp{INP}',
					discrepancy = dict(
						name = new_name,
						gen = dict(
							inpainting_name = f'sliding-{INP}',
						),
					),
				))



	

	# cfgs_before = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	# for cfg in cfgs_before:
	# 	name = cfg.name

	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-PatchMax',
	# 		cls = 'ObstaclePipelineSystem_DiscrepancyPatched',
	# 		display_name = cfg.display_name + ' PatchMax',
	# 		fusion_type = 'max',
	# 	))

	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-PatchWeighted',
	# 		cls = 'ObstaclePipelineSystem_DiscrepancyPatched',
	# 		display_name = cfg.display_name + ' PatchhWeighted',
	# 		fusion_type = 'weighted',
	# 	))

	to_ext_with_synthpix = [
		'ImgBlur05VsInp-ds2-NoiseImg', 
		'ImgBlur05VsInp-ds2b-NoiseImg',
		'ImgBlur05VsInp-ds2b', 
		'ImgBlur05VsInp-ds3-NoiseImg', 
		'ImgBlur05VsInp-ds3', 
	]
	for name in to_ext_with_synthpix:
		cfg = ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name[name]
		add(extend = cfg.name, cfg = dict(
			name = f'{cfg.name}-SynthPix405',
			display_name = cfg.display_name + ' SynthPix405',
			discrepancy = dict(
				name = cfg.discrepancy.name + '-SynthPix405',
				gen = dict(
					inpainting_name = 'pix2pixHD_405',
				)
			)
		))

	for cfg in list(ObstaclePipelineSystem_Discrepancy.default_cfgs):
		name = cfg.name
		if name.startswith('Img'):
			add(extend = name, cfg = dict(
				name = f'{name}-rep2',
				discrepancy = dict(
					name = f'{name}-rep2',
					train = dict(
						optimizer = dict(
							learn_rate = 0.0001 * 0.5, # half LR
						),
					),
				),
			))

	cfgs_before_last = list(ObstaclePipelineSystem_Discrepancy.default_cfgs)
	for cfg in cfgs_before_last:
		name = cfg.name

		add(extend = name, cfg = dict(
			name = f'{name}-last',
			display_name = cfg.display_name + ' last-epoch',
			discrepancy = dict(
				checkpoint_for_eval = 'last',
			)
		))




	# for cfg in cfgs_before:
	# 	name = cfg.name

	# 	add(extend = name, cfg = dict(
	# 		name = f'{name}-SynthPix405',
	# 		display_name = cfg.display_name + ' SynthPix405',
	# 		discrepancy = dict(
	# 			name = cfg.discrepancy.name + '-SynthPix405',
	# 			gen = dict(
	# 				inpainting_name = 'pix2pixHD_405',
	# 			)
	# 		)
	# 	))


	



init_discrepancy_configs()	

@click.group()
def main():
	pass

@main.command()
@click.argument('methods')
@click.argument('dsets')
@click.option('--comparison', default=None)
@click.option('--islands/--no-islands', default=True)
@click.option('--limit-length', type=int, default=0)
@click.option('--perframe/--no-perframe', default=True)
@click.option('--histogram/--no-histogram', default=False)
@click.option('--fpr95/--no-fpr95', default=False)
@click.option('--morphology/--no-morphology', default=False)
@click.option('--wu-eval/--no-wu-eval', default=False)
def evaluation(methods, dsets, islands, comparison=None, limit_length=0, perframe=True, histogram=False, fpr95=False, morphology=False, wu_eval=False):
	methods = name_list(methods)
	dsets = name_list(dsets)
	
	for m_name in methods:
		print(f'-- {m_name} --')
		m = ObstaclePipelineSystem.get_implementation(m_name)
		m.load()

		for d in dsets:
			print(f'-- {m_name} vs {d} --')
			m.run_evaluation(
				d, 
				sem_obstacle_enabled=islands, 
				limit_length_to=limit_length, 
				b_histogram=histogram, 
				b_perframe=perframe,
				save_fpr95=fpr95,
				b_morphology=morphology,
				wu_eval = wu_eval,
			)

		del m
		gc.collect()

	if comparison is not None:
		curve_module = EvaluatorSystem_PerPixBinaryClassification.get_implementation('perpixAP')

		for d in dsets:
			curve_module.produce_comparison(
				comparison_name = comparison,
				dataset_name = d,
				method_names = methods,
			)
			
			if islands:
				curve_module.produce_comparison(
					comparison_name = f'{comparison}-vsIslands',
					dataset_name = d,
					method_names = methods + [f'{m}-islands' for m in methods],
				)

@main.command()
@click.argument('methods')
@click.argument('dsets')
@click.option('--limit-length', type=int, default=0)
@click.option('--metric_dsets', type=str, default='')
def eval_store(methods, dsets, metric_dsets, limit_length=0):
	methods = name_list(methods)
	dsets = name_list(dsets)
	metric_dsets = name_list(metric_dsets)
	
	for m_name in methods:
		print(f'-- {m_name} --')
		m = ObstaclePipelineSystem.get_implementation(m_name)
		m.load()

		for d in dsets:
			print(f'-- {m_name} vs {d} --')
			m.run_evaluation_store_only(
				d, limit_length_to=limit_length,
			)

		if metric_dsets:			
			from ..a15_corr.inspect_frames import run_usual_metrics
			for dset in metric_dsets:
				run_usual_metrics(m_name, dset)

		del m
		gc.collect()


from road_anomaly_benchmark.evaluation import Evaluation
from road_anomaly_benchmark.datasets.dataset_registry import DatasetRegistry
from road_anomaly_benchmark.paths import DIR_OUTPUTS

@main.command()
@click.argument('methods')
@click.option('--dsets')
@click.option('--limit-length', type=int, default=0)
@click.option('--metric_dsets', type=str, default='')
@click.option('--vis/--no-vis', default=True)
def eval_segme(methods, dsets, metric_dsets, limit_length=0, vis=True):
	methods = name_list(methods)
	dsets = name_list(dsets)
	metric_dsets = name_list(metric_dsets) or dsets
	
	for m_name in methods:
		print(f'-- {m_name} --')
		m = ObstaclePipelineSystem.get_implementation(m_name)
		m.load()

		for d in dsets:
			print(f'-- {m_name} vs {d} --')
			m.run_evaluation_store_only(
				d, limit_length_to=limit_length,
			)

		del m
		gc.collect()

		if metric_dsets:
			for dsev in metric_dsets:
				if not dsev:
					continue

				print(f""" Metrics {dsev} """)
				ev = Evaluation(
					method_name = m_name, 
					dataset_name = dsev,
				)
				ag = ev.calculate_metric_from_saved_outputs(
					'PixBinaryClass',
					frame_vis = vis,
				)
				ag = ev.calculate_metric_from_saved_outputs(
					'SegEval-ObstacleTrack',
					frame_vis = vis,
				)



@main.command()
@click.argument('methods')
@click.argument('dsets')
@click.option('--metric_dsets', type=str, default='')
@click.option('--require_dset', type=str, default='LostAndFound-testNoKnown')
def eval_store_if_has_other(methods, dsets, metric_dsets, require_dset):
	from road_anomaly_benchmark.paths import DIR_OUTPUTS

	methods = name_list(methods)
	dsets = name_list(dsets)
	metric_dsets = name_list(metric_dsets)
	
	for m_name in methods:
		print(f'-- {m_name} --')

		if (DIR_OUTPUTS / 'PixBinaryClass' / 'data' / f'PixClassCurve_{m_name}_{require_dset}.hdf5').is_file():
			print(m_name, 'yes')

			# m = ObstaclePipelineSystem.get_implementation(m_name)
			# m.load()

			# for d in dsets:
			# 	print(f'-- {m_name} vs {d} --')
			# 	m.run_evaluation_store_only(d)

			# 	if metric_dsets:
			# 		from ..a15_corr.inspect_frames import run_usual_metrics
			# 		for dset in metric_dsets:
			# 			run_usual_metrics(m_name, dset)
			# del m
			gc.collect()

		else:
			print(m_name, 'no')

@main.command()
@click.argument('method', type=str)
@click.option('--eval_dsets', type=str, default='ObstacleTrack-all,LostAndFound-testNoKnown')
@click.option('--metric_dsets', type=str, default='ObstacleTrack-test,ObstacleTrack-validation,LostAndFound-testNoKnown')
@click.option('--reps', type=int, default=0)
def train(method, eval_dsets, metric_dsets, reps=0):
	eval_dsets = name_list(eval_dsets)
	metric_dsets = name_list(metric_dsets)

	if reps > 0:
		methods = [f'{method}-rep{i}' for i in range(1, reps+1)]
	else:
		methods = [method]

	for method in methods:
		m = ObstaclePipelineSystem.get_implementation(method)
		b_training_successful = m.run_training_then_eval(eval_dsets = eval_dsets)
		b_training_successful = b_training_successful or (b_training_successful is None)
		gc.collect()

		if metric_dsets and b_training_successful:
			from ..a15_corr.inspect_frames import run_usual_metrics
			for dset in metric_dsets:
				run_usual_metrics(method, dset)
		
		gc.collect()

	

@main.command()
@click.argument('method', type=str)
def show_arch(method):
	m = ObstaclePipelineSystem.get_implementation(method)
	m.load()
	print(m.mod_discrepancy.net_mod)
	

@main.command()
@click.argument('method', type=str)
@click.option('--metric_dsets', type=str, default='ObstacleTrack-test,ObstacleTrack-validation,LostAndFound-testNoKnown')
def diag_weighted_soup(method, metric_dsets):
	metric_dsets = name_list(metric_dsets)

	m = ObstaclePipelineSystem.get_implementation(method)
	m.load()

	import torch
	torch.set_grad_enabled(False)

	from matplotlib import pyplot
	cw = m.mod_discrepancy.net_mod.cat_mod.calc_weights
	

	wmod_conv = m.mod_discrepancy.net_mod.cat_mod.weight_mod[0]
	print('Mixer weights', wmod_conv.weight.data, 'bias', wmod_conv.bias.data)

	for dsetname in metric_dsets:
		dset = DatasetRegistry.get_implementation(dsetname)
		m.init_perspective_loader(dset)

		for fr in tqdm(dset):
			fr.update(m.load_perspective_scale(**fr))
			ps = torch.from_numpy(fr.perspective_scale_map)[None, None].cuda()
			weights = cw(ps).cpu().numpy()

			lines = weights[0, :, :, 32]

			fig, plot = pyplot.subplots(1, 1)
			
			# print('shapes', 'lines', lines.shape, 'pmap', fr.perspective_scale_map.shape)

			plot.plot(fr.perspective_scale_map[:, 32] * (1./400.), label='pmap scaled')


			for i in range(4):
				plot.plot(lines[i], label = f'block {i}')

			outpath = DIR_DATA / '1560_WeightedSoup' / method / f'{fr.fid}__soupWeights.png'
			outpath.parent.mkdir(parents=True, exist_ok=True)

			fig.legend()
			fig.tight_layout()
			fig.savefig(outpath)


if __name__ == '__main__':
	main()

# python -m src.a12_inpainting.discrepancy_experiments evaluation Dummy SmallObstacleDataset-test




#python -c "from src.a12_inpainting.discrepancy_experiments import ObstaclePipelineSystem; s = ObstaclePipelineSystem('ImgVsInp-archResy'); s.load(); s.run_evaluation('RoadAnomaly-test')"
		
# python -c "from src.a12_inpainting.discrepancy_experiments import Exp0560_SwapFgd_ImgAndLabelsVsGen_2048 as exp_class; exp_class.training_procedure()"

# python -m src.a12_inpainting.discrepancy_experiments ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy  RoadAnomaly-test

# python -m src.a12_inpainting.discrepancy_experiments ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy  RoadAnomaly2-sample1

# python -m src.a12_inpainting.discrepancy_experiments ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy  RoadAnomaly2-sample1 --comparison Standard

# python -m src.a12_inpainting.discrepancy_experiments ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy  FishyLAF-LafRoi,FishyLAF-val,RoadAnomaly2-sample1 --comparison Standard

# python -m src.a12_inpainting.discrepancy_experiments Island RoadAnomaly2-sample1 --no-islands
# python -m src.a12_inpainting.discrepancy_experiments Island FishyLAF-LafRoi,RoadAnomaly2-sample1 --no-islands



# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResyFocal --eval_dsets FishyLAF-LafRoi,FishyLAF-val,RoadAnomaly2-sample1 

# 1217
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResyFocalWeighted
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResyNoiseImg

# -- Noise

# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy,ImgVsInp-archResyNoiseImg FishyLAF-LafRoi,FishyLAF-val,RoadAnomaly2-sample1

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResyNoiseImg FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison NoiseImg

# -- Focal

# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResyFocalWeighted RoadAnomaly2-sample1,FishyLAF-val,FishyLAF-LafRoi

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResyFocal,ImgVsInp-archResyFocalWeighted FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison Focal



# -- Blur level comparison

# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgBlur03VsInp-archResy,ImgBlur03VsSelf-archResy,ImgBlur05VsInp-archResy,ImgBlur05VsSelf-archResy,ImgBlur09VsInp-archResy,ImgBlur09VsSelf-archResy FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsSelf-archResy,ImgBlur03VsInp-archResy,ImgBlur03VsSelf-archResy,ImgBlur05VsInp-archResy,ImgBlur05VsSelf-archResy,ImgBlur09VsInp-archResy,ImgBlur09VsSelf-archResy FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison Blur


# Blur + focal w / noise
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgBlur03VsInp-archResyNoiseImg,ImgBlur03VsInp-archResyFocalWeighted RoadAnomaly2-sample1,FishyLAF-LafRoi,FishyLAF-val

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResyNoiseImg,ImgVsInp-archResyFocalWeighted,ImgBlur03VsInp-archResy,ImgBlur03VsInp-archResyNoiseImg,ImgBlur03VsInp-archResyFocalWeighted FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison BlurNoiseFocal


# -- Train new configs

# noise+focal
# control group noise+focal

# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResy-NoiseAndFocalW
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsSelf-archResy-NoiseAndFocalW
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsSelf-archResy-NoiseImg
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsSelf-archResy-FocalWeighted

## FishyLAF
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-NoiseAndFocalW,ImgVsSelf-archResy-NoiseAndFocalW,ImgVsSelf-archResy-NoiseImg,ImgVsSelf-archResy-FocalWeighted FishyLAF-LafRoi,FishyLAF-val



## blur for RA2
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgBlur03Inp-archResy-NoiseAndFocalW,ImgBlur03Self-archResy-NoiseAndFocalW,ImgBlur03Self-archResy-NoiseImg,ImgBlur03Self-archResy-FocalWeighted RoadAnomaly2-sample1



# -- last checkpoint
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgBlur03VsInp-archResy-last,ImgBlur03VsInp-archResy-NoiseImg-last,ImgBlur03VsInp-archResy-FocalWeighted-last,ImgBlur03VsInp-archResy-NoiseAndFocalW-last,ImgBlur03VsSelf-archResy-last,ImgBlur03VsSelf-archResy-NoiseImg-last,ImgBlur03VsSelf-archResy-FocalWeighted-last,ImgBlur03VsSelf-archResy-NoiseAndFocalW-last,ImgBlur03VsInp-archResy-FocalWeighted-last  RoadAnomaly2-sample1 &


# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-last,ImgVsInp-archResy-NoiseImg-last,ImgVsInp-archResy-FocalWeighted-last,ImgVsInp-archResy-NoiseAndFocalW-last,ImgVsSelf-archResy-last,ImgVsSelf-archResy-NoiseImg-last,ImgVsSelf-archResy-FocalWeighted-last,ImgVsSelf-archResy-NoiseAndFocalW-last  FishyLAF-LafRoi,FishyLAF-val

# python -m src.a12_inpainting.metrics ImgVsInp-archResy-last,ImgVsInp-archResy-NoiseImg-last,ImgVsInp-archResy-FocalWeighted-last,ImgVsInp-archResy-NoiseAndFocalW-last,ImgVsSelf-archResy-last,ImgVsSelf-archResy-NoiseImg-last,ImgVsSelf-archResy-FocalWeighted-last,ImgVsSelf-archResy-NoiseAndFocalW-last  FishyLAF-LafRoi,FishyLAF-val --comparison NoiseFocal4_LastCheckpoint &

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResy-last,ImgVsSelf-archResy,ImgVsSelf-archResy-last --comparison LastVsBest1

# ImgVsInp-archResy-NoiseImg-last,ImgVsInp-archResy-FocalWeighted-last,ImgVsInp-archResy-NoiseAndFocalW-last,ImgVsSelf-archResy-last,ImgVsSelf-archResy-NoiseImg-last,ImgVsSelf-archResy-FocalWeighted-last,ImgVsSelf-archResy-NoiseAndFocalW-last  FishyLAF-LafRoi,FishyLAF-val --comparison NoiseFocal4_LastCheckpoint &


# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-PatchMax FishyLAF-LafRoi
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsSelf-archResy-PatchMax FishyLAF-LafRoi


# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-SynthPix504 FishyLAF-LafRoi





# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc RoadAnomaly2-pyrDown
# python -m src.a12_inpainting.sys_road_area semcontour-roadwalk-v1 RoadAnomaly2-pyrDown
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 RoadAnomaly2-pyrDown --num_workers 1
# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-NoiseAndFocalW,ImgVsSelf-archResy-NoiseAndFocalW,ImgBlur03VsInp-archResy-NoiseAndFocalW RoadAnomaly2-pyrDown


# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc RoadAnomaly2-cwebp
# python -m src.a12_inpainting.sys_road_area semcontour-roadwalk-v1 RoadAnomaly2-cwebp
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 RoadAnomaly2-cwebp --num_workers 1
# symlink instead!

# python -m src.a12_inpainting.discrepancy_experiments evaluation ImgVsInp-archResy-NoiseAndFocalW,ImgVsSelf-archResy-NoiseAndFocalW,ImgBlur03VsInp-archResy-NoiseAndFocalW RoadAnomaly2-cwebp



# python -m src.a12_inpainting.discrepancy_experiments train Resynth2048Orig
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResy-SynthPix405



# python -m src.a12_inpainting.discrepancy_experiments evaluation Dummy SmallObstacleDataset-test --limit-length 10




# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-archResy-HighW





# 
# 
# 
#  FishyLAF-LafRoi


def get_configs_for_blur_study():
	"""
	ImgVsInp-archResy-NoiseImg-last
	ImgBlurInfer03VsInp-archResy-NoiseImg-last
	ImgBlurInfer05VsInp-archResy-NoiseImg-last
	ImgBlurInfer07VsInp-archResy-NoiseImg-last

	ImgBlurTrain03VsInp-archResy-NoiseImg-last
	ImgBlur03VsInp-archResy-NoiseImg-last
	ImgBlur03Inf05VsInp-archResy-NoiseImg-last
	ImgBlur03Inf07VsInp-archResy-NoiseImg-last

	ImgBlurTrain05VsInp-archResy-NoiseImg-last
	ImgBlur05VsInp-archResy-NoiseImg-last
	ImgBlur05Inf03VsInp-archResy-NoiseImg-last
	ImgBlur05Inf07VsInp-archResy-NoiseImg-last
	"""

	configs_for_blur_study = [c for c in ObstaclePipelineSystem_Discrepancy.default_cfgs if 'VsInp' in c.name and 'last' in c.name and 'NoiseImg' in c.name and 'SynthPix' not in c.name and c.get('blur_key') and c.blur_key[0] != 7] + [ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name['ImgVsInp-archResy-NoiseImg-last']]
	[(c.name, c.get('blur_key', (1,1))) for c in configs_for_blur_study]



# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-ds3-NoiseImg

# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-ds3-Pers-NoiseImg
# python -m src.a12_inpainting.discrepancy_experiments train ImgVsInp-ds3-Pers-NoiseImg

from easydict import EasyDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ..common.registry import Registry
from ..paths import DIR_DATA, DIR_EXP, DIR_EXP2
from ..pipeline.config import add_experiment, extend_config

from ..a12_inpainting.discrepancy_experiments import ObstaclePipelineSystem
from ..a12_inpainting.discrepancy_pipeline import Exp12xx_DiscrepancyForInpaintingBase
from ..a12_inpainting.sys_road_area import ModuleRegistry

from .networks import ModsObstacleNet, ModsBackbone, ModsFeatureProcessors, ModsClassifiers
from .experiment_variants import exp15_configs, exp15_configs_unwarp
import gc

from ..a14_perspective.warp import unwarp_road_frame
import cv2 as cv

ModsCrDetector = Registry()

class Exp150x(Exp12xx_DiscrepancyForInpaintingBase):

	def __init__(self, cfg):
		cfg = add_experiment(Exp12xx_DiscrepancyForInpaintingBase.cfg, **cfg)
		super().__init__(cfg)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = EasyDict(self.cfg['net'])

		if 'hanet' in cfg_net.arch_core.lower():
			from .hanet_networks import experiments_configs

		backbone = ModsBackbone.get(cfg_net.arch_backbone, cache=False)
		classifier = ModsClassifiers.get(cfg_net.arch_classifier, cache = False)
		core = ModsObstacleNet.get(cfg_net.arch_core, cache=False)
		core.build(
			backbone = backbone,
			feat_processor_names = cfg_net.arch_feature_procs,
			feat_processor_names_sticky = cfg_net.get('arch_feature_procs_sticky', []),
			classifier = classifier,
		)

		self.net_mod = core
		# print(self.net_mod)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])


from ..a14_perspective.cityscapes_pitch_angles import read_cam_info_laf, gen_perspective_scale_map, perspective_info_from_camera_info, invent_horizon_above_main_road
from ..a14_perspective.pos_enc_xy import get_yx_maps

@ModsCrDetector.register_class()
class ObstaclePipelineSystem_CrNet(ObstaclePipelineSystem):

	exp_cls = Exp150x

	@classmethod
	def configs(cls):
		return exp15_configs()

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.cfg.discrepancy.setdefault('name', self.cfg.name)

		# self.mod_discrepancy = LapNetRegistry.get(self.cfg.mod_discrepancy)
	
		# calculate the effective config for discrepancy module
		cfg_overlays = [
			# tells the module to write outputs under this system's name
			dict(
				dir_checkpoint = DIR_EXP / self.cfg.discrepancy['name'],
				inference = dict(
					save_dir_name = self.cfg.name,
				),
			),
			# this system's discrepancy cfg
			self.cfg.discrepancy,
		]

		# cfg_discrepancy = self.exp_cls.cfg
		cfg_discrepancy = {}
		for ovr in cfg_overlays:
			cfg_discrepancy = extend_config(cfg_discrepancy, diff=ovr, warn=True)
		
		self.mod_discrepancy = self.exp_cls(cfg_discrepancy)
		self.b_needs_inpainting = False # cfg_discrepancy['net']['comparator_net_name'] == 'ComparatorImageToImage'


	def get_name(self):
		return self.cfg.name

	def get_display_name(self):
		return self.cfg.display_name

	def load(self):
		self.mod_discrepancy.init_net('eval')

	def load_perspective_scale(self, **fields):
		#print('Load perspective scale', fields.keys(), fields['dset'])

		if 'perspective_scale_map' in fields:
			return

		if self.perspective_loader_type == 'laf':
			cam_info = read_cam_info_laf(EasyDict(
				dir_root = self.perspective_loader_dset_stub.dir_root,
				split = fields['dset']['split'],
			), EasyDict(fields))
			perspective_info = perspective_info_from_camera_info(cam_info)

		elif self.perspective_loader_type == 'sem':
			roadarea = self.sys_roadarea.load_values(fields)
			road_mask = roadarea['labels_road_mask']
			cam_info, perspective_info = invent_horizon_above_main_road(road_mask)

		psm = gen_perspective_scale_map(
			fields['image'].shape[:2], 
			perspective_info.horizon_level, 
			perspective_info.pix_per_meter_slope,
		)

		pos_YX = get_yx_maps(fields['image'].shape[:2])

		return dict(
			camera_info = cam_info,
			persp_info = perspective_info,
			perspective_scale_map = psm,
			pos_encoding_X = pos_YX[1],
			pos_encoding_Y = pos_YX[0],
		)

	def init_perspective_loader(self, dset):

		b_unwarp = self.cfg.get('unwarp', False)
		b_perspective = self.cfg.discrepancy.net.get('perspective', False)

		if b_perspective or b_unwarp:
			laf_root = getattr(dset.dset, 'DIR_LAF', None)

			if laf_root:
				self.perspective_loader_type = 'laf'
				self.perspective_loader_dset_stub = EasyDict(
					dir_root = laf_root,
				)
			else:
				self.perspective_loader_type = 'sem'
				self.sys_roadarea = ModuleRegistry.get('RoadAreaSystem', 'semcontour-roadwalk-v1')
				self.sys_roadarea.init_storage()
		

	def construct_pipeline(self):
		md = self.mod_discrepancy
		pipe = md.construct_default_pipeline('test_v2')

		loaders = []

		# if md.sys_road is not None:
		# 	loaders.append(md.sys_road.tr_load)
		
		b_unwarp = self.cfg.get('unwarp', False)
		b_perspective = self.cfg.discrepancy.net.get('perspective', False)

		if b_perspective or b_unwarp:
			loaders.append(self.load_perspective_scale)

		if b_unwarp:
			loaders.append(self.unwarp_preprocess)
			pipe.tr_output.insert(0, self.unwarp_postprocess)

		extra_features = self.cfg.discrepancy.get('extra_features', {})
		atn = extra_features.get('attentropy')
		if atn:
			from ..a12_inpainting.attentropy_loader import load_attentropy
			from functools import partial
			loaders.append(partial(load_attentropy, attn_name=atn))

		pipe.tr_input += loaders

		return pipe

	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):
		num_frames = frames_iterable.__len__()

		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		self.init_perspective_loader(frames_iterable)

		md = self.mod_discrepancy
		pipe = self.construct_pipeline()
		md.sampler_test.data_source = range(num_frames)

		if process_func is not None:
			# insert before TrKeepFields
			#pipe.tr_output.insert(-1, lambda *a, **kw: print(a, kw.keys()))
			# pipe.tr_output.append(process_func)
			pipe.tr_output.insert(-1, process_func)

		# TODO load inpainting as part of pipeline
		#md.sys_inp.decorate_dset_to_load_inp_image(frames_iterable)

		# run image-saving in background threads
		with ThreadPoolExecutor(6) as thread_pool:
			md.background_thread_pool = thread_pool
			
			result_frames = pipe.execute(frames_iterable, b_accumulate=return_frames, b_grad=False, allow_workers=False)

			del md.background_thread_pool

		return result_frames

	def run_training_then_eval(self, eval_dsets = ('FishyLAF-val',)):
		b_training_successful = self.mod_discrepancy.training_procedure_on_instance()

		gc.collect()

		if b_training_successful:
			self.load()

			gc.collect()

			for dsname in eval_dsets:
				self.run_evaluation_store_only(dsname)

		return b_training_successful


	def unwarp_preprocess(self, frame, **_):
		unwarp_road_frame(frame)
		#from ..common.jupyter_show_image import imwrite
		#imwrite(DIR_DATA/'1519_unwarp'/f'{frame.fid}_unwarp.webp', frame.unwarp_image)

		return dict(
			image = frame.unwarp_image,
			image_orig = frame.image,
		)

	def unwarp_postprocess(self, anomaly_p, unwarp_H, image_orig, **_):
		unwarp_orig_size_xy = image_orig.shape[:2][::-1]
		#print(anomaly_p.shape, unwarp_orig_size_xy)
		# warp anomaly_p
		reconstructed_anomaly_p = cv.warpPerspective(
			anomaly_p, 
			unwarp_H, 
			unwarp_orig_size_xy, 
			flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR,
		)

		return dict(
			image = image_orig,
			anomaly_p = reconstructed_anomaly_p,
		)


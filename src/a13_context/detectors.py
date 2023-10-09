
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import cv2 as cv
import torch


from .networks import FeatureLaplacianSingle, ContextDetectorSlim, ErfNetOrig, PSPNetLap
from .networks_pyramid import LapPyramidBasic, LapPyramidTMix, LapPyramidTMix2, LapPyramidFeat
from .erfnet import ErfNetBasic, ErfNetImageMultiscale

from ..pipeline.log import log
from ..paths import DIR_DATA, DIR_EXP, DIR_EXP2
from ..common.jupyter_show_image import imread, imwrite, adapt_img_data
from ..pipeline.config import add_experiment, extend_config
from ..a12_inpainting.discrepancy_experiments import ObstaclePipelineSystem
from ..a12_inpainting.discrepancy_pipeline import Exp12xx_DiscrepancyForInpaintingBase


from road_anomaly_benchmark.datasets.dataset_registry import Registry

LapNetRegistry = Registry()


def heatmap_upsample(lap_sum, size=512):
	r = size // lap_sum.shape[1]
	if r > 1:
		return np.repeat(np.repeat(lap_sum, r, 0), r, 1)
	else:
		return lap_sum
	

class ObstaclePipelineSystem_LaplacianSimple(ObstaclePipelineSystem):

	configs = [
		EasyDict(
			name = 'Laplacian-RN50-Layer1-Ker31',
			backbone = 'rn50',
			feature = 'layer1',
			kernel_size = 31,
		),
		EasyDict(
			name = 'Laplacian-RN50-Layer2-Ker31',
			backbone = 'rn50',
			feature = 'layer2',
			kernel_size = 31,
		),
		EasyDict(
			name = 'Laplacian-RN50-Layer3-Ker31',
			backbone = 'rn50',
			feature = 'layer3',
			kernel_size = 31,
		),
		EasyDict(
			name = 'Laplacian-RN50-Layer4-Ker31',
			backbone = 'rn50',
			feature = 'layer4',
			kernel_size = 31,
		),

		EasyDict(
			name = 'Laplacian-RN50-Layer3-Ker03',
			backbone = 'rn50',
			feature = 'layer3',
			kernel_size = 3,
		),

		EasyDict(
			name = 'Laplacian-RN50-Layer3-Ker07',
			backbone = 'rn50',
			feature = 'layer3',
			kernel_size = 7,
		),
	]

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

		self.mod_net = FeatureLaplacianSingle(
			extractor = cfg.backbone,
			feature_name = cfg.feature,
			# magnitude = cfg.magnitude,
			kernel_size = cfg.kernel_size,
		)


	def get_name(self):
		return self.cfg.name

	def get_display_name(self):
		return self.cfg.name

	def write_visualizations(self, frame):
		fid = frame.fid
		dset = frame.dset
		anomaly_p = frame.anomaly_p
		image = frame.image

		out_dir_base = DIR_DATA / '1302-laplacian-discrep' / self.get_name() / getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')
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


	def predict_frames(self, frames_iterable, limit_length_to=None):
		frames_out = []

		seq_len = frames_iterable.__len__() 
		if limit_length_to not in (0, None):
			seq_len = min(seq_len, limit_length_to)

		for i, frame in tqdm(zip(range(seq_len), frames_iterable), total=seq_len):
			
			img = frame['image']

			heatmap = self.mod_net.forward_image(img)

			frame['anomaly_p'] = heatmap_upsample(heatmap, size=img.shape[1])
			
			# write images
			self.write_visualizations(frame)

			del frame['image']

			frames_out.append(frame)

		return frames_out





class Exp13xx_LapNet(Exp12xx_DiscrepancyForInpaintingBase):

	cfg = add_experiment(Exp12xx_DiscrepancyForInpaintingBase.cfg, 
		name = '1302_LapNetSlim',
		net = dict(
			arch_freeze_backbone = True,
			arch_filter_type = 'fixed-laplacian',
			arch_filter_size = 21,
			arch_distance_type = 'l1',
			arch_upsample_type = 'UPconv',
			arch_mix_type = 'mix1',
			arch_inter_depth = 6,

			batch_train = 4,
			batch_eval = 2, # eval is on full images instead of crops
			batch_infer = 1,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_v2blur5_cityscapes-train',
			dset_val = '1230_SynthObstacle_Fusion_v2blur5_cityscapes-val',
			mod_sampler = 'v1_768',

			num_workers = 4,
			epoch_limit = 50,

			# loss
			loss_name = 'cross_entropy',
			class_weights = [1.45693524, 19.18586532],

			augmentation_noise = False,
		),
		preproc_blur = False,
		# we don't use it but lets not break yet
		gen = dict(
			inpainting_name = 'sliding-deepfill-v1',
		),
	)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = self.cfg['net']

		cfg_arch = EasyDict({
			k: v for k, v in cfg_net.items()
			if k.startswith('arch')
		})

		# TODO comparator

		self.net_mod = ContextDetectorSlim(**cfg_arch)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])
	

class ObstaclePipelineSystem_LaplacianNet(ObstaclePipelineSystem):

	exp_cls = Exp13xx_LapNet

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

		# calculate the effective config for discrepancy module
		cfg_overlays = [
			# tells the module to write outputs under this system's name
			dict(
				dir_checkpoint = DIR_EXP / self.cfg.discrepancy['name'],
				inference = dict(
					save_dir_name = f'13xxlapnet/{self.cfg.name}'
				),
			),
			# this system's discrepancy cfg
			self.cfg.discrepancy,
		]

		cfg_discrepancy = self.exp_cls.cfg
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

	def predict_frames(self, frames_iterable, limit_length_to=None):
		num_frames = frames_iterable.__len__()

		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		md = self.mod_discrepancy
		pipe = md.construct_default_pipeline('test_v2')
		md.sampler_test.data_source = range(num_frames)

		# TODO load inpainting as part of pipeline
		#md.sys_inp.decorate_dset_to_load_inp_image(frames_iterable)

		# run image-saving in background threads
		with ThreadPoolExecutor(6) as thread_pool:
			md.background_thread_pool = thread_pool
			
			result_frames = pipe.execute(frames_iterable, b_accumulate=True, b_grad=False)

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


	# @classmethod
	# @lru_cache(max_size=1)
	# def get_configs(cls):
	# 	...

	# 	EasyDict(
	# 		name = f'LapNet_...',
	# 		discrepancy = dict(
	# 			arch_freeze_backbone = True,
	# 			arch_filter_type = 'fixed-laplacian',
	# 			arch_filter_size = 21,
	# 			arch_distance_type = 'l1',
	# 		),
	# 	)


	@classmethod
	def arch_config_by_name(cls, name):
		cfg = EasyDict()
		
		lapnet, backbone, ker_size, ker_type, dist_type, *rest = name.split('_')

		if backbone == 'RN50fr':
			cfg.arch_freeze_backbone = True
		elif backbone == 'RN50':
			cfg.arch_freeze_backbone = False
		else:
			raise NotImplementedError(f'CFG backbone: {backbone}')

		cfg.arch_filter_size = int(ker_size[3:])

		# cfg.arch_filter_type = {
		# 	'LapFix': 'fixed-laplacian',
		# 	'LapLearn1ChInit': 'learn-1ch-initlap',

		# }

		cfg.arch_kernel_type = ker_type
		cfg.arch_distance_type = dist_type


		for r in rest:
			if r.startswith('mix'):
				cfg.arch_mix_type = r
			elif r.startswith('UP'):
				cfg.arch_upsample_type = r
			elif r.startswith('id'):
				cfg.arch_inter_depth = int(r[2:])


		return cfg

	@classmethod
	def get_implementation(cls, name):
		'LapNet_RN50fr_Ker21-LapFix-L1'

		cfg = EasyDict(
			name = name,
			display_name = name,
			discrepancy = dict(
				name = f'1302_{name}', # checkpoint name
				net = cls.arch_config_by_name(name),
			)
		)

		return cls(cfg)

		



def ext(base, warn=True, **diff):
	result = base.copy()

	for k, val in diff.items():
		if warn and k not in result:
			log.warning(f'Warning: overriding key [{k}] which is not present in the base config')
		if isinstance(val, dict):
			baseval = base.get(k, None)
			if isinstance(baseval, dict):
				# merge a dict
				result[k] = extend_config(baseval, val, warn=warn)
			else:
				# overwrite any other type of value
				result[k] = val
		else:
			result[k] = val
	return result

@LapNetRegistry.register_class()
class Exp1304_SegLap(Exp12xx_DiscrepancyForInpaintingBase):

	DEFAULTS_NET = dict(
		#batch_train = 1, # RESET
		#batch_eval = 2, # eval is on full images instead of crops
		batch_infer = 1,

		separate_gen_image = False,
		perspective = False,
	)

	DEFAULT_LAP = dict(abs=True, filter_size=5)

	BLUR_CFG = dict(
		kernel_size = 3,
		blur_in_training = True,
		blur_in_inference = True,
	)

	NOISEAUG_CFG = dict(
		layers = (
			(0.18, 1.0),
			(0.31, 0.5),
			(0.84, 0.2),
		),
		magnitude_range = [0.1, 0.6],
	)

	DEFAULTS = dict(
		net = dict(
			batch_train = 12,
			batch_eval = 6,
			batch_infer = 1,

			separate_gen_image = False,
			perspective = False,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
			mod_sampler = 'v1-768',

			num_workers = 4,
			epoch_limit = 65,

			# loss
			loss_name = 'cross_entropy',
			class_weights = [1.45693524, 19.18586532],

			augmentation_noise = NOISEAUG_CFG,
		),
		preproc_blur = False,
		# preproc_blur = dict(
		# 	kernel_size = 3,
		# 	blur_in_training = True,
		# 	blur_in_inference = True,
		# ),
	)

	configs = [EasyDict(ext(DEFAULTS,

		name = '1304_SegPSP',
		net = dict(
			arch = 'PSP',
			lap = False,
			batch_train = 4,
			batch_eval = 2,
		),
	)),
	EasyDict(ext(DEFAULTS,
		name = '1304_SegPSP_Lap',
		net = dict(
			arch = 'PSP',
			lap = DEFAULT_LAP,
			batch_train = 4,
			batch_eval = 2,
		),
	)),
	EasyDict(ext(DEFAULTS,
		name = '1304_SegErfnet',
		net = dict(
			arch = 'ErfNet',
			lap = False,
			batch_train = 12,
			batch_eval = 5,
		),
	)),
	EasyDict(ext(DEFAULTS,
		name = '1304_SegErfnet_Lap',
		net = dict(
			arch = 'ErfNet',
			lap = DEFAULT_LAP,
			batch_train = 12,
			batch_eval = 5,
		),
	)),
	
	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnetTEST',

		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp2_cityscapes-val',
			mod_sampler = 'v1-768',

			num_workers = 4,
			epoch_limit = 100,

			# loss
			loss_name = 'cross_entropy',
			class_weights = [1.45693524, 19.18586532],

			augmentation_noise = dict(
				layers = (
					(0.18, 1.0),
					(0.31, 0.5),
					(0.84, 0.2),
				),
				magnitude_range = [0.1, 0.6],
			),
		),
		net = dict(
			arch = 'ErfNetImageMultiscale',
			batch_train = 2,
			batch_eval = 1,

			separate_gen_image = False,
			perspective = False,
		),
	)),

	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3',
		net = dict(
			arch = 'ErfBasic',
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		preproc_blur = False,
	)),


	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_Msi1',
		net = dict(
			arch = 'ErfNetImageMultiscale',
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_Msi1_Blur',
		net = dict(
			arch = 'ErfNetImageMultiscale',
			train = dict(augmentation_noise=False),
			extra_blur_channels = True,
		),
		preproc_blur = False,
	)),

	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_IDD',
		net = dict(
			arch = 'ErfBasic',
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-val',
			mod_sampler = 'v1-768',
			num_workers = 4,
			epoch_limit = 40,
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnetOrig_IDD',
		net = dict(
			arch = 'ErfNetOrig',
			lap = False,
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-val',
			mod_sampler = 'v1-768',
			num_workers = 4,
			epoch_limit = 40,
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_PSP_IDD',
		net = dict(
			arch = 'PSP',
			lap = False,
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-val',
			mod_sampler = 'v1-768',
			num_workers = 4,
			epoch_limit = 40,
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_PSP_IDD_Bl3Nog',
		net = dict(
			arch = 'PSP',
			lap = False,
			train = dict(augmentation_noise=False),
			extra_blur_channels = False,
		),
		train = dict(
			dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-train',
			dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_IndiaDriving-val',
			mod_sampler = 'v1-768',
			num_workers = 4,
			epoch_limit = 40,
			augmentation_noise = NOISEAUG_CFG,
		),
		preproc_blur = BLUR_CFG,
	)),

	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_Bl3Nog',
		net = dict(
			arch = 'ErfBasic',
			extra_blur_channels = False,
		),
		train = dict(augmentation_noise = NOISEAUG_CFG),
		preproc_blur = BLUR_CFG,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_Bl3Nog_Msi1',
		net = dict(
			arch = 'ErfNetImageMultiscale',
			extra_blur_channels = False,
		),
		train = dict(augmentation_noise = NOISEAUG_CFG),
		preproc_blur = BLUR_CFG,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1305_SegErfnet3_Bl3Nog_Msi1_Blur',
		net = dict(
			arch = 'ErfNetImageMultiscale',
			extra_blur_channels = True,
		),
		train = dict(augmentation_noise = NOISEAUG_CFG),
		preproc_blur = BLUR_CFG,
	)),

	
	EasyDict(ext(DEFAULTS,
		name = '1306_PyramidFeat_Control',
		net = dict(
			arch = 'LapPyramidFeat1',
			pyr_use_laplacian = True,
		),
		train = dict(
			# epoch_limit = 50,
			optimizer = dict(learn_rate = 0.0002,),
			augmentation_noise=False
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1306_PyramidFeat_Lap',
		net = dict(
			arch = 'LapPyramidFeat1',
			pyr_use_laplacian = False,
		),
		train = dict(
			# epoch_limit = 50,
			optimizer = dict(learn_rate = 0.0002,),
			augmentation_noise=False
		),
		preproc_blur = False,
	)),


	EasyDict(ext(DEFAULTS,
		name = '1306_PyramidBl_Basic',
		net = dict(
			arch = 'LapPyramidBasic',
			pyr_laplacian_type = 'blur',
			b_normalize_brightness = False,
		),
	)),
	EasyDict(ext(DEFAULTS,
		name = '1306_PyramidLap_Basic',
		net = dict(
			arch = 'LapPyramidBasic',
			pyr_laplacian_type = 'kernel',
			b_normalize_brightness = False,
		),
	)),
	# EasyDict(ext(DEFAULTS,
	# 	name = '1306_Pyramid_BasicNorm',
	# 	net = dict(
	# 		arch = 'LapPyramidBasic',
	# 		b_normalize_brightness = True,
	# 	),
	# )),

	EasyDict(ext(DEFAULTS,
		name = '1306_Pyramid_Mix1',
		net = dict(
			arch = 'LapPyramidTMix',
			pyr_normalize_brightness = False,
		),
		train = dict(
			epoch_length_limit = 100,
			epoch_length_limit_val = 150,
			epoch_limit = 25,

			optimizer = dict(
				learn_rate = 0.01,
			),

			augmentation_noise = False,
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1306_Pyramid_Mix2',
		net = dict(
			arch = 'LapPyramidTMix2',
			pyr_normalize_brightness = False,
		),
		train = dict(
			epoch_length_limit = 1000,
			epoch_length_limit_val = 200,
			epoch_limit = 30,

			optimizer = dict(
				learn_rate = 0.0001,
			),

			augmentation_noise = False,
		),
		preproc_blur = False,
	)),

	# Correlation head - control group
	EasyDict(ext(DEFAULTS,
		name = '1315_DeepLabHeadControl',
		net = dict(
			arch = 'DeepLabObstacleHead',
			separate_gen_image = False,
			perspective = False,
		),
		train = dict(
			epoch_limit = 50,

			optimizer = dict(
				learn_rate = 0.0001,
			),
		),
		preproc_blur = False,
	)),
	EasyDict(ext(DEFAULTS,
		name = '1315_DeepLabHeadControlTEST',
		
		train = dict(
			#dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2blur5_cityscapes-train',
			#dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2blur5_cityscapes-val',
			mod_sampler = 'v1-768',

			num_workers = 4,
			epoch_limit = 100,

		),
		net = dict(
			arch = 'DeepLabObstacleHead',
			separate_gen_image = False,
			perspective = False,
			batch_train = 2,
			batch_eval = 1,
		),
	)),
	]

	def __init__(self, cfg):
		cfg = add_experiment(Exp12xx_DiscrepancyForInpaintingBase.cfg, **cfg)
		super().__init__(cfg)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = self.cfg['net']


		arch = cfg_net['arch']

		if arch == 'PSP':
			self.net_mod = PSPNetLap(cfg_net)
		elif arch == 'ErfNetOrig':
			self.net_mod = ErfNetOrig(cfg_net)
		elif arch == 'ErfBasic':
			self.net_mod = ErfNetBasic(num_class=2)
		elif arch == 'ErfNetImageMultiscale':
			self.net_mod = ErfNetImageMultiscale(
				num_class=2, 
				extra_blur_channels = cfg_net.get('extra_blur_channels'),
			)
		elif arch == 'LapPyramidBasic':
			self.net_mod = LapPyramidBasic(cfg_net)
			self.class_softmax = lambda net_out: dict(pred_prob = torch.cat([net_out, net_out], dim=1))
		elif arch == 'LapPyramidTMix':
			self.net_mod = LapPyramidTMix(cfg_net)
		elif arch == 'LapPyramidTMix2':
			self.net_mod = LapPyramidTMix2(cfg_net)
		elif arch == 'LapPyramidFeat1':
			self.net_mod = LapPyramidFeat(cfg_net)
		elif arch == 'DeepLabObstacleHead':
			from .pytorch_load_gluon import ObstacleHeadedDeeplab
			self.net_mod = ObstacleHeadedDeeplab(cfg_net)
	
		else:
			raise NotImplementedError(arch)


		#print(self.net_mod)

		# cfg_arch = EasyDict({
		# 	k: v for k, v in cfg_net.items()
		# 	if k.startswith('arch')
		# })

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])


@LapNetRegistry.register_class()
class ObstaclePipelineSystem_LaplacianSeg(ObstaclePipelineSystem):

	exp_cls = Exp1304_SegLap

	@classmethod
	def configs(cls):
		cfgs = []

		for cfg in Exp1304_SegLap.configs:
			name = f'LapSeg_{cfg.name}'

			cfgs.append(dict(
				name = name,
				discrepancy = cfg,
			))

		return cfgs



	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

		# self.mod_discrepancy = LapNetRegistry.get(self.cfg.mod_discrepancy)
	
		# calculate the effective config for discrepancy module
		cfg_overlays = [
			# tells the module to write outputs under this system's name
			dict(
				dir_checkpoint = DIR_EXP / self.cfg.discrepancy['name'],
				inference = dict(
					save_dir_name = f'13xxlapnet/{self.cfg.name}'
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

	def predict_frames(self, frames_iterable, limit_length_to=None, return_frames=True, process_func=None):
		num_frames = frames_iterable.__len__()

		if not (limit_length_to in (None, 0)):
			num_frames = min(num_frames, limit_length_to)

		md = self.mod_discrepancy
		pipe = md.construct_default_pipeline('test_v2')
		md.sampler_test.data_source = range(num_frames)

		if process_func is not None:
			pipe.tr_output.append(process_func)

		# TODO load inpainting as part of pipeline
		#md.sys_inp.decorate_dset_to_load_inp_image(frames_iterable)

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
				self.run_evaluation(dsname)





# cmd_train="python -m src.a12_inpainting.discrepancy_experiments train "

# python -m src.a12_inpainting.discrepancy_experiments train LapNet_RN50fr_Ker21_fixed-laplacian_l1

# $cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1
# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1

# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1


# LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs
# LapNet_RN50fr_Ker21_learn-1ch-initlap_learn-abs

# LapNet_RN50_Ker21_fixed-laplacian_l1
# LapNet_RN50_Ker21_learn-1ch-initlap_l1

# LapNet_RN50_Ker21_fixed-laplacian_learn-abs
# LapNet_RN50_Ker21_learn-1ch-initlap_learn-abs


# python -m src.a12_inpainting.discrepancy_experiments evaluation Laplacian-RN50-Layer3-Ker31 RoadObstacles2048p-full
# python -m src.a12_inpainting.discrepancy_experiments evaluation Laplacian-RN50-Layer3-Ker31 FishyLAF-LafRoi


# python -m src.a12_inpainting.discrepancy_experiments evaluation Laplacian-RN50-Layer1-Ker31,Laplacian-RN50-Layer2-Ker31,Laplacian-RN50-Layer3-Ker31,Laplacian-RN50-Layer4-Ker31 FishyLAF-LafRoi,RoadObstacles2048p-full --no-perframe --comparison Laplacian-Layers
# python -m src.a12_inpainting.discrepancy_experiments evaluation Laplacian-RN50-Layer3-Ker03,Laplacian-RN50-Layer3-Ker07 FishyLAF-LafRoi,RoadObstacles2048p-full --no-perframe 

# python -m src.a12_inpainting.metrics Laplacian-RN50-Layer1-Ker31,Laplacian-RN50-Layer2-Ker31,Laplacian-RN50-Layer3-Ker31,Laplacian-RN50-Layer4-Ker31 FishyLAF-LafRoi,RoadObstacles2048p-full --comparison Laplacian-Layers

# python -m src.a12_inpainting.metrics Laplacian-RN50-Layer3-Ker03,Laplacian-RN50-Layer3-Ker07,Laplacian-RN50-Layer3-Ker31 FishyLAF-LafRoi,RoadObstacles2048p-full --comparison Laplacian-Kernels


# python -m src.a12_inpainting.discrepancy_experiments train LapSeg_1304_SegErfnet
# python -m src.a12_inpainting.discrepancy_experiments train LapSeg_1304_SegErfnet_Lap

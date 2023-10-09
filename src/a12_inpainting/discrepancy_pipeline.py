import logging
log = logging.getLogger('exp')

import numpy as np
import torch
from torch.optim import Adam as AdamOptimizer
import cv2 as cv
import h5py

from ..paths import DIR_DATA
from ..common.jupyter_show import adapt_img_data
from ..pipeline.pipeline import Pipeline
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain, TrKeepFields, TrKeepFieldsByPrefix, tr_print
from ..pipeline.transforms_pytorch import TrCUDA, TrNP
from ..pipeline.transforms_pytorch import image_batch_preproc, image_batch_preproc_undo

from ..pipeline.config import add_experiment
from ..datasets.dataset import imwrite
from ..datasets.cityscapes import DatasetCityscapes
from ..a05_differences.experiments import ExperimentDifference_Auto_Base, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT
from ..a05_differences.metrics import binary_confusion_matrix


from .vis_imgproc import image_montage_same_shape
from .discrepancy_networks import ComparatorUNet, FeatureExtractorForComparator, FocalLossIgnoreInvalid, NoiseAugmentation, ModBlurInput
from ..a01_sem_seg.networks import LossCrossEntropy2d

#from .synth_obstacle_dset import SynthObstacleDsetTrainingSampler
from .demo_case_selection import DatasetRegistry
from ..common.registry import ModuleRegistry2
from .sys_reconstruction import InpaintingSystem, decorate_laf_dset_to_load_inpainted_images
from .sys_road_area import RoadAreaSystem
from .sys_segmentation import SemSegSystem
from . import synth_obstacle_patch_sampler # register

from collections import defaultdict

class ExperimentMixin_HalfFloatTraining:
	def build_optimizer(self, role, chk_optimizer=None):
		log.info('Building optimizer')

		cfg_opt = self.cfg['train']['optimizer']

		network = self.net_mod

		
		lr_overrides = {}
		for name, mod in network.named_modules():
			for ovr_name, ovr_value in getattr(mod, 'lr_overrides', {}).items():
				if ovr_value is not None:
					lr_overrides[f'{name}.{ovr_name}'] = ovr_value

		print('LR OVERRIDES', lr_overrides)

		param_groups = defaultdict(list)
		lr_default = cfg_opt['learn_rate']

		for (name, p) in network.named_parameters():
			if p.requires_grad:
				lr = lr_overrides.get(name, lr_default)
				param_groups[lr].append(p)

		for (lr, ps) in param_groups.items():
			log.info(f'Optmizer group: {ps.__len__()} at lr {lr}')

		param_groups = [
			dict(params = ps, lr = lr)
			for (lr, ps) in param_groups.items()
		]
		
		self.optimizer = AdamOptimizer(
			param_groups,
			# [p for p in network.parameters() if p.requires_grad],
			# lr=cfg_opt['learn_rate'],
			weight_decay=cfg_opt.get('weight_decay', 0),
		)

		opt_type = cfg_opt.get('opt_type', 'plateau')

		if opt_type == 'plateau':
			self.learn_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				self.optimizer,
				patience=cfg_opt['lr_patience'],
				min_lr = cfg_opt['lr_min'],
			)

		elif opt_type == 'poly':
			self.learn_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optimizer,
				lr_lambda = lambda epoch: (1-epoch / 53)**0.9
			)
		else:
			raise NotImplementedError(opt_type)

		self.amp_scaler = torch.cuda.amp.GradScaler()

		if chk_optimizer is not None:
			self.optimizer.load_state_dict(chk_optimizer['optimizer'])

	def training_start_batch(self, **_):
		self.optimizer.zero_grad()

		ac = torch.cuda.amp.autocast()
		ac.__enter__()
		return dict(
			amp_autocast = ac,
		)
	
	def training_backpropagate(self, loss, frame, **_):
		#if torch.any(torch.isnan(loss)):
		#	print('Loss is NAN, cancelling backpropagation in batch')

			#raise Exception('Stopping training so we can investigate where the nan is coming from')
		#else:

		# https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples

		if 'amp_autocast' in frame:
			frame['amp_autocast'].__exit__(None, None, None)
			del frame['amp_autocast']

		self.amp_scaler.scale(loss).backward()
		self.amp_scaler.step(self.optimizer)
		self.amp_scaler.update()


class ExperimentMixin_ModularPixelLoss:
	def init_loss(self):
		cfg_train = self.cfg['train']
		
		# class weights (how much positive class is more important than negative)
		class_weights = cfg_train.get('class_weights', None)

		if class_weights is not None:
			log.info(f'	class weights: {class_weights}')
			class_weights = torch.Tensor(class_weights)
		else:
			log.warning('	no class weights')

		# loss type

		loss_name = cfg_train.get('loss_name')

		if loss_name == 'cross_entropy':
			self.loss_mod = LossCrossEntropy2d(
				weight = class_weights,
			)

		elif loss_name == 'focal':
			self.loss_mod = FocalLossIgnoreInvalid(
				alpha = cfg_train['focal_loss_alpha'],
				gamma = cfg_train['focal_loss_gamma'],
				reduction = 'mean',
				weights = class_weights,
			)
		else:
			raise NotImplementedError(f'Loss name [{loss_name}]')

		self.cuda_modules(['loss_mod'])



class ExperimentMixin_DiscrepancyWriter:

	@staticmethod
	def test_write_result_v2_background(out_dir_base, fid, image, labels_gt, anomaly_p, roi):
		
		try:
			#out_path_img = out_dir_base / 'input_image' / f'{fid}__input_image.webp'
			out_path_vis = out_dir_base / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
			out_path_lowres = out_dir_base / 'score_as_image' / f'{fid}__score_as_image.png'
			out_path_cmats = out_dir_base / 'cmats' / f'{fid}__demo_anomaly.hdf5'

			if labels_gt is not None and roi is not None:
				cmats = binary_confusion_matrix(anomaly_p, labels_gt, roi=roi, normalize=False)['cmat']
				
				out_path_cmats.parent.mkdir(exist_ok=True, parents=True)
				with h5py.File(out_path_cmats, 'w') as file_cmat:
					file_cmat['cmats'] = cmats
			

			out_path_lowres.parent.mkdir(exist_ok=True, parents=True)
			imwrite(
				out_path_lowres,
				(anomaly_p * 255).astype(np.uint8),
			)

			out_path_vis.parent.mkdir(exist_ok=True, parents=True)
			demo_img = cv.addWeighted(
				image // 2, 0.5, 
				adapt_img_data(anomaly_p, value_range=(0., 1.)), 0.5,
				0.0,
			)
			imwrite(out_path_vis, demo_img)
			
			#imwrite(out_path_img, image)

		except Exception as e:
			log.exception(f'Exception in test_write_result_v2_background for frame {fid}')
			raise e


	def test_write_result_v2(self, fid, dset, image, anomaly_p, labels_road_mask=None, labels=None, labels_source=None, **_):
		
		if dset is None:
			return

		if labels is None:
			labels = labels_source

		selfname = self.cfg['name']
		save_dir_name = f'1209discrep-{selfname}'

		# override save dir name if the owning module wants this
		cfg_infer = self.cfg.get('inference')
		if cfg_infer:
			save_dir_name = cfg_infer.get('save_dir_name', save_dir_name)

		out_dir_base = DIR_DATA / save_dir_name / f'{dset.name}-{dset.split}'

		if labels is not None:
			# gt for FishyLAF
			roi = labels < 255 
			labels_gt = labels == 1
		else:
			roi = None
			labels_gt = None
		
		# road area
		# clear anomaly_p outside road area
		if labels_road_mask is not None:
			anomaly_p = anomaly_p * labels_road_mask

		# opt: add anomaly=1 inside holes
		# # #

		kw = dict(out_dir_base = out_dir_base,
			fid = fid,
			image = image,
			labels_gt = labels_gt,
			anomaly_p = anomaly_p,
			roi = roi,
		)

		tp = getattr(self, 'background_thread_pool', None)
		if tp:
			tp.submit(self.test_write_result_v2_background, **kw)
		else:
			self.test_write_result_v2_background(**kw)
		
		
		return dict(
			anomaly_p = anomaly_p,
			labels = labels, #unify name btw labels_source
		)

class ExperimentMixin_DiscrepancySubmodules:
	def init_discrepancy_submodules(self):
		print(self.cfg)
		sys_road_name = self.cfg['sys_road_area']
		if sys_road_name:
			self.sys_road = RoadAreaSystem.get_implementation(sys_road_name)
			self.sys_road.init_storage()
		else:
			self.sys_road = None
		
		sys_inp_name = self.cfg['gen']['inpainting_name']
		log.info(f'	ModInp: {sys_inp_name}')
		if sys_inp_name:
			self.sys_inp = InpaintingSystem.get_implementation(sys_inp_name)
			self.sys_inp.init_storage()
		else:
			self.sys_inp = None

		if sys_road_name and sys_inp_name:
			assert self.sys_road.cfg.name == self.sys_inp.cfg.road_area_name

		self.modules_preproc_train = []
		self.modules_preproc_infer = []

		cfg_blur = self.cfg.get('preproc_blur')
		if cfg_blur:
			mod_blur = ModBlurInput(cfg_blur)
			if cfg_blur['blur_in_training']:
				if cfg_blur['blur_in_training'] == 'no_retrain':
					def er(*a, **k):
						raise NotImplementedError('no_retrain specified for this inference blur')
					self.modules_preproc_train.append(er)
				else:
					self.modules_preproc_train.append(mod_blur)
	
			if cfg_blur['blur_in_inference']:
				self.modules_preproc_infer.append(mod_blur)
					
		cfg_noise = self.cfg['train'].get('augmentation_noise')
		if cfg_noise:
			log.info('	Initializing noise module')
			self.mod_noise = NoiseAugmentation(
				layer_defs = cfg_noise['layers'],
				magnitude_range = cfg_noise['magnitude_range'],
			)
		else:
			self.mod_noise = None


class Exp12xx_DiscrepancyForInpaintingBase(
		ExperimentMixin_HalfFloatTraining,
		ExperimentMixin_DiscrepancySubmodules,
		ExperimentMixin_ModularPixelLoss, 
		ExperimentMixin_DiscrepancyWriter, 
		ExperimentDifference_Auto_Base,
	):
	# Method-resolution-order is left-to-right

	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name = '1205_Discrepancy_ImgVsInpaiting',
		net = dict(
			comparator_net_name = 'ComparatorImageToImage',

			freeze_backbone = True,
			correlation_layer = True,

			batch_train = 4,
			batch_eval = 2, # eval is on full images instead of crops
			batch_infer = 1,

			separate_gen_image = True,
			perspective = False,
		),
		train = dict(
			dset_train = '1204-SynthObstacleDset-v1-Ctc-PatchSampler-train',
			dset_val = '1204-SynthObstacleDset-v1-Ctc-PatchSampler-val',
			mod_sampler = 'v1_768',

			num_workers = 4,
			epoch_limit = 65,

			# loss
			loss_name = 'cross_entropy',
			class_weights = [1.45693524, 19.18586532],

			augmentation_noise = False,
		),
		preproc_blur = False,
		gen = dict(
			inpainting_name = 'sliding-deepfill-v1',
		),
		sys_road_area = 'semcontour-roadwalk-v1',
	)

	fields_for_test = ['image', 'gen_image']
	fields_for_training = ['image', 'gen_image', 'semseg_errors_label'] 
	# ROI is communicated by setting label to 255, see ExperimentDifference_Auto_Base.tr_semseg_errors_to_label


	def init_transforms(self):
		extra_fields = []
		if self.cfg.net.separate_gen_image:
			extra_fields.append('gen_image')
		if self.cfg.net.perspective:
			extra_fields.append('perspective_scale_map')
			extra_fields += ['pos_encoding_X', 'pos_encoding_Y']



		extra_fields += self.cfg.get('extra_features', {}).keys()
		print('Loading extra features', self.cfg.get('extra_features', {}), 'extra fields', extra_fields)

		self.fields_for_test = ['image'] + extra_fields
		self.fields_for_training = ['image', 'semseg_errors_label'] + extra_fields


		super().init_transforms()

		self.init_discrepancy_submodules()
			

	def tr_augmentation_gpu_batch(self, image, **_):

		if self.mod_noise is not None:
			image = self.mod_noise(image)

		return dict(
			image = image,
		)
			
	def tr_preproc_infer(self, frame, **_):
		for mod in self.modules_preproc_infer:
			frame.update(mod(**frame))
		

	# not called with fr.apply
	def tr_preproc_train(self, **frame_fields):
		frame = self.translate_from_inpainting_dset_to_gen_image_terms(**frame_fields)

		for mod in self.modules_preproc_train:
			frame.update(mod(**frame))

		return frame


	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = self.cfg['net']

		# comparator_name = cfg_net.get('comparator_net_name')

		# comp_cls = {c.__name__: c for c in [
		# 	ComparatorImageToImage,
		# 	ComparatorImageToEmpty,
		# 	ComparatorImageToSelf,
		# 	Seg_PSPNet,
		# 	Seg_PSPNet_Perspective,
		# ]}[comparator_name]

		comp_cls = ComparatorUNet

		backbone = FeatureExtractorForComparator(
			backbone_name = cfg_net.get('backbone_name', 'vgg'),
			separate_gen_image = cfg_net.get('separate_gen_image', True),
			freeze = cfg_net['freeze_backbone'],
			perspective = cfg_net.get('perspective', False),
		)

		self.net_mod = comp_cls(
			num_outputs = 2, 
			correlation = cfg_net.get('correlation_layer', True),
			extractor = backbone,
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def tr_net(self, image, gen_image = None, perspective_scale_map = None, **extra):

		# print('tr_net extras', extra.keys())

		logits = self.net_mod(
			image = image, 
			gen_image = gen_image,
			perspective_scale_map = perspective_scale_map,
			**extra,
		)

		return dict(pred_anomaly_logits = logits)


	@staticmethod
	def translate_from_inpainting_dset_to_gen_image_terms(frame_id, image_fused, obstacle_instance_map, road_mask, image_inpainted=None, **extras):
		gt_map = (obstacle_instance_map > 0).astype(np.int64)
		gt_map[np.logical_not(road_mask)] = 255

		return Frame(
			fid = frame_id,
			image = image_fused,
			gen_image = image_inpainted,
			semseg_errors_label = gt_map,
			**extras,
		)

	def init_default_datasets(self, b_threaded=False):
		#print('INIT DEFAULT DSET')

		for split in ('train', 'val'):
			dset = DatasetRegistry.get_implementation(self.cfg.train[f'dset_{split}'])
			mod_sampler = ModuleRegistry2.get_implementation('1230_SynthObstacle_PatchSampler', self.cfg.train.mod_sampler)
			mod_sampler.load(dset)

			mod_sampler.preproc_func = self.tr_preproc_train

			# whether to use inapinted image channel
			if self.cfg.net.separate_gen_image:
				mod_sampler.set_channels(
					extra_channels_to_load = dict(
						image_inpainted = self.sys_inp.storage['image_inpainted'],
					)
				)
			
			# TODO normals
			if self.cfg.net.get('normals'):
				from ..datasets.dataset import ChannelLoaderImage
				mod_sampler.extra_channels_to_load['normals'] = ChannelLoaderImage(
					DIR_DATA / '1701_Omnidata/pred3_normals' / 'Cityscapes' / '{fid}_normal_normal.png',
				)


			# whether to use perspective scale channel
			if self.cfg.net.perspective:
				mod_sampler.set_channels(mod_sampler.channels_to_sample + ['perspective_scale_map'])

			self.set_dataset(split, mod_sampler)

		self.frames_to_log = set([0, 1, 2, 3, 6, 8, 9])

	def init_log(self, frames_to_log=None):
		if frames_to_log is not None:
			self.frames_to_log = set(frames_to_log)

		self.init_log__tboard()

		ds = self.datasets['val']

		#chans_backup = ds.channels_enabled
		#ds.set_channels_enabled('image', 'semseg_errors')

		# Write the ground-truth for comparison
		for fid in self.frames_to_log:
			fid_no_slash = str(fid).replace('/', '__')
			fr = ds[fid] #ds.get_frame_by_fid(fid)

			imwrite(self.train_out_dir / f'gt_image_{fid_no_slash}.webp', fr.image)

			mask_roi = fr.semseg_errors_label < 255
			mask_obstacle = fr.semseg_errors_label == 1

			gt_labels_vis = np.zeros_like(fr.image, dtype=np.uint8)
			gt_labels_vis[mask_roi] = 128
			gt_labels_vis[mask_obstacle] = 255

			imwrite(self.train_out_dir / f'gt_labels_{fid_no_slash}.png', gt_labels_vis)
			

			rows = [fr.image, gt_labels_vis]
			if self.cfg.net.separate_gen_image: #fr.get('gen_image') is not None:
				rows.append(fr.gen_image)
			if self.cfg.net.perspective: #fr.get('perspective_scale_map') is not None:
				rows.append(adapt_img_data(fr.perspective_scale_map))

			gt_vis_all = image_montage_same_shape(rows, num_cols=2)

			imwrite(self.train_out_dir / f'gt_all_{fid_no_slash}.webp', gt_vis_all)


			self.tboard_img.add_image(
				'{0}_img'.format(fid),
				fr.image.transpose((2, 0, 1)),
				0,
			)

			self.tboard_gt.add_image(
				'{0}_gt'.format(fid),
				mask_obstacle[None, :, :],
				0,
			)

	def test_write_result(self, fid, dset, image, anomaly_p, labels_source, **_):
		dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')
		
		out_dir_base = self.workdir / 'out' / dk

		out_path_vis = out_dir_base / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		out_path_cmats = out_dir_base / 'cmats' / f'{fid}__demo_anomaly.hdf5'
		
		roi = labels_source > 0  # onroad!
		labels_gt = labels_source > 1
		
		anomaly_p_onroad = anomaly_p * roi
		
		cmats = binary_confusion_matrix(anomaly_p, labels_gt, roi=roi, normalize=False)['cmat']
		
		out_path_cmats.parent.mkdir(exist_ok=True, parents=True)
		with h5py.File(out_path_cmats, 'w') as file_cmat:
			file_cmat['cmats'] = cmats
		
		out_path_vis.parent.mkdir(exist_ok=True, parents=True)
		demo_img = cv.addWeighted(
			image // 2, 0.5, 
			adapt_img_data(anomaly_p_onroad), 0.5,
			0.0,
		)
		imwrite(out_path_vis, demo_img)
		
	


	def construct_default_pipeline(self, role):

		print(self.pre_merge_train)

		if role == 'test_v2':
			return Pipeline(
				# tr_input = self.tr_input_test,
				tr_input = TrsChain(
					self.tr_preproc_infer,
				),
				tr_batch_pre_merge = self.pre_merge_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_classify,
					TrKeepFields('anomaly_p'),
					TrNP(),
				),
				tr_output = TrsChain(
					self.test_write_result_v2,
					TrKeepFields('fid', 'anomaly_p', 'labels', 'labels_road_mask', 'obstacle_from_sem', 'dset_name', b_warn=False),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				# tr_preproc is applied on the dataset level
				
				#r_input = self.tr_input_train,
				tr_batch_pre_merge = self.pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					#self.tr_augmentation_gpu_batch, # this being removed to reduce randomness in val
					self.tr_net,
					self.tr_loss,
					self.tr_classify,
					TrKeepFieldsByPrefix('loss', 'anomaly_p'),
				),
				tr_output = TrsChain(
					self.tr_eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				#tr_input = TrsChain(
				# 	self.tr_input_train,
				# 	TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [384, 768]), fields = self.fields_for_training),
				# 	TrRandomlyFlipHorizontal(self.fields_for_training),
				#),
				tr_batch_pre_merge = self.pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_augmentation_gpu_batch,
					self.training_visualize_batch,
					self.training_start_batch,
					self.tr_net,
					self.tr_loss,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

	def loader_args_for_role(self, role):
		if role == 'train':
			return dict(
				sampler = self.datasets['train'].make_pytorch_sampler(
					short_epoch = self.cfg['train'].get('epoch_length_limit', None),
				), # this does not resume at proper epoch when resuming training
				batch_size = self.cfg['net']['batch_train'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = True,
				#pin_memory = True,
			)

		elif role == 'val':
			epoch_len = self.datasets['val'].__len__()
			short_epoch = self.cfg['train'].get('epoch_length_limit_val', None)
			if short_epoch is not None:
				epoch_len = min(epoch_len, short_epoch)

			return dict(
				sampler = torch.utils.data.sampler.SequentialSampler(range(epoch_len)), 
				batch_size = self.cfg['net']['batch_eval'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = False,
			)

		elif role == 'test' or role =='test_v2':
			self.sampler_test = torch.utils.data.sampler.SequentialSampler(range(100))
			return dict(
				sampler = self.sampler_test, # override the size later
				batch_size = self.cfg['net']['batch_infer'] ,
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = False,
			)
		else:
			raise NotImplementedError("role: " + role)
		

class Exp0560_SwapFgd_ImgAndLabelsVsGen_2048(
		ExperimentMixin_HalfFloatTraining,
		ExperimentMixin_ModularPixelLoss,
		ExperimentMixin_DiscrepancySubmodules,
		ExperimentMixin_DiscrepancyWriter,
		Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT,
	):
	# Method-resolution-order is left-to-right

	# TODO loader args sampler
	# TODO batch size
	# TODO num epochs

	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0560_Diff_SwapFgd_ImgAndLabelVsGen_2048',
		gen_name='051X_semGT__fakeSwapFgd__genNoSty__2048',
		gen_img_ext='.webp',

		net = dict(
			freeze_backbone = True,
			correlation_layer = True,

			batch_train = 4,
			batch_eval = 2, # eval is on full images instead of crops
			batch_infer = 1,

			comparator_net_name = 'ComparatorImageToGenAndLabels', # not used in this exp, but other modules read this
		),
		train = dict(
			num_workers = 4,
			epoch_limit = 65,

			# unify loss
			loss_name = 'cross_entropy',
			class_weights = [1.45693524, 19.18586532],
		),
		gen = dict(
			inpainting_name = 'pix2pixHD_405',
			segmentation_name = 'gluon-psp-ctc',
		),
		sys_road_area = None,
    )

	def init_transforms(self):
		super().init_transforms()

		self.init_discrepancy_submodules()
			
		self.sys_seg = SemSegSystem.get_implementation(self.cfg['gen']['segmentation_name'])
		self.sys_seg.init_storage()

	def init_default_datasets(self, b_threaded=False):
		dset_ctc_train = DatasetCityscapes(
			split='train',
			b_cache=False,
		)
		dset_ctc_val = DatasetCityscapes(
			split='val',
			b_cache=False,
		)
		dsets_ctc = [dset_ctc_train, dset_ctc_val]
		for dset in dsets_ctc:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_ctc_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_ctc_train)
		self.set_dataset('val', dset_ctc_val)


	def tr_load_predicted_labels(self, frame, **_):
		
		labels_ids = self.sys_seg.storage['sem_class_prediction'].read_value(**frame)
		labels_trainIds = self.sys_seg.labels_to_trainIds(labels_ids)

		return dict(
			labels_fakeErr_trainIds = labels_trainIds,
		)


	def construct_default_pipeline(self, role):

		if role == 'test_v2':
			# available to be changed depending on dset length
			self.sampler_test = torch.utils.data.sampler.SequentialSampler(range(100))
			
			return Pipeline(
				tr_input = TrsChain(
					self.tr_input_test,
					self.tr_load_predicted_labels,
				),
				tr_batch_pre_merge = self.pre_merge_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_classify,
					TrKeepFields('anomaly_p'),
					TrNP(),
				),
				tr_output = TrsChain(
					self.test_write_result_v2,
					TrKeepFields('fid', 'anomaly_p', 'labels'), #, 'labels_road_mask', 'obstacle_from_sem'),
				),
				loader_args = dict(
					sampler = self.sampler_test,
					batch_size = self.cfg['net']['batch_infer'] ,
					num_workers = self.cfg['train'].get('num_workers', 0),
					drop_last = False,
				)
			)
		else:
			return super().construct_default_pipeline(role)




def discrepancy_infer(mod_discrepancy, img_bhwc, reconstruction_bhwc = None, batch_size=8):

	dev = torch.device('cuda:0')
	
	anomaly_p_list = []

	#from time import perf_counter

	#nn_time = 0

	#t1_outer = perf_counter()

	with torch.no_grad():
		num_patches = img_bhwc.__len__()

		for batch_start in range(0, num_patches, batch_size):
			batch_end = min(batch_start+batch_size, num_patches)
			#print(batch_start, batch_end)
			
			#img_1hwc = img_bhwc[batch_idx][None]
			img_tr = (
				image_batch_preproc(
					np.stack(img_bhwc[batch_start:batch_end])
				).to(dev)
			)
			reconstr_tr = (
				image_batch_preproc(
					np.stack(reconstruction_bhwc[batch_start:batch_end])
				).to(dev) 
				if reconstruction_bhwc is not None else img_tr
			)

			#t1 = perf_counter()

			pred_anomaly_logits = mod_discrepancy.tr_net(image=img_tr, gen_image=reconstr_tr)['pred_anomaly_logits']
			anomaly_p = mod_discrepancy.tr_classify(pred_anomaly_logits=pred_anomaly_logits)['anomaly_p']

			anomaly_p = anomaly_p.cpu()

			# t2 = perf_counter()

			# nn_time += (t2-t1)

			# unbatch anomaly_p
			anomaly_p_list.append(anomaly_p.numpy())

			#del anomaly_p, pred_anomaly_logits

	# t2_outer =perf_counter()

	# print(f'batch {batch_size} dicrepancy_infer nn time {nn_time} / {t2_outer - t1_outer} total')

	return np.concatenate(anomaly_p_list, axis=0)


def inpaint_patches(obdet, frt, b_show=False, b_make_inpainting=True):
	# we specifically don't take sys_inp from override, sys_inp shouldn't be part of mod_discrepancy anyway
	imp = obdet.mod_discrepancy.sys_inp.imp
		
	patch_extraction_result = imp.extract_patches_given_road_mask(
		frt.image,
		frt.labels_road_mask,
		patch_size = imp.patch_size,
		context_size = imp.context_size,
		patch_overlap = imp.patch_overlap,
		context_restriction = imp.context_restriction,
	)
	frt.update(patch_extraction_result)
	
	if b_make_inpainting:
		frt.patches_reconstruction = imp.inpaint_func(
			images_hwc_list = patch_extraction_result.patches_context_bhwc, 
			masks_hw_list = patch_extraction_result.patches_mask_bhw,
		)
		
		if b_show:
			show(list(frt.patches_context_bhwc), list(frt.patches_reconstruction))
	

def infer_patch_disc_comparator(obdet, frt, b_show=True, mod_discrepancy_override=None):
	
	inpaint_patches(obdet, frt)
	
	mod_discrepancy = mod_discrepancy_override or obdet.mod_discrepancy
	
	anomaly_ps = discrepancy_infer(
		mod_discrepancy,
		img_bhwc = frt.patches_context_bhwc,
		reconstruction_bhwc = frt.patches_reconstruction,
	)
	
	anomaly_ps_norec = discrepancy_infer(
		mod_discrepancy,
		img_bhwc = frt.patches_context_bhwc,
		reconstruction_bhwc = None,
	)
	
	frt.anomaly_ps = anomaly_ps
	frt.anomaly_ps_norec = anomaly_ps_norec
	
	if b_show:
		show(list(frt.patches_context_bhwc), list(frt.patches_reconstruction), list(frt.anomaly_ps), list(frt.anomaly_ps_norec))


	
def infer_patch_disc_vsself(obdet, frt, mod_discrepancy_override, b_show=True):
	
	inpaint_patches(obdet, frt)
	
	mod_discrepancy = mod_discrepancy_override or obdet.mod_discrepancy
	
	anomaly_ps = discrepancy_infer(
		mod_discrepancy,
		img_bhwc = frt.patches_context_bhwc,
		reconstruction_bhwc = None,
	)
	
	anomaly_ps_rec = discrepancy_infer(
		mod_discrepancy,
		img_bhwc = frt.patches_reconstruction,
		reconstruction_bhwc = None,
	)
	
	frt.anomaly_ps = anomaly_ps
	frt.anomaly_ps_rec = anomaly_ps_rec
	
	if b_show:
		show(list(frt.patches_context_bhwc), list(frt.patches_reconstruction), list(frt.anomaly_ps), list(frt.anomaly_ps_rec))


def infer_patch_fuse_disc(obdet, frt, b_make_inpainting=True, fusion_type='max', b_show=False):
	"""
	@param: fusion_type: max or weighted
	"""
	
	#from time import perf_counter

	#t1 = perf_counter()

	inpaint_patches(obdet, frt, b_make_inpainting=b_make_inpainting)
	
	#t2 = perf_counter()

	mod_discrepancy = obdet.mod_discrepancy
	
	anomaly_ps = discrepancy_infer(
		mod_discrepancy,
		img_bhwc = frt.patches_context_bhwc,
		reconstruction_bhwc = frt.patches_reconstruction if b_make_inpainting else frt.patches_context_bhwc,
	)

	#t3 = perf_counter()

	#print(f'Time: inpaint {t2-t1} | discrepancy {t3-t2}')

	frt.anomaly_ps = anomaly_ps
	
	if b_show:
		show(list(frt.patches_context_bhwc), list(frt.patches_reconstruction), list(frt.anomaly_ps), list(frt.anomaly_ps_norec))

	imp = obdet.mod_discrepancy.sys_inp.imp
	h, w = frt.image.shape[:2]
		
	if fusion_type == 'weighted':
		fusion_result = imp.fuse_patches(
			image_background = np.zeros((h, w, 1), dtype=np.float32),
			patch_grid = frt.patch_grid,
			context_grid = frt.context_grid,
			patches_reconstruction = [p[:, :, None] for p in frt.anomaly_ps],
			patches_mask_bhw = frt.patches_mask_bhw,
			out_dtype = np.float32,
			**imp,
		)

	elif fusion_type == 'max':
		fusion_result = imp.fuse_patches_max(
			image_background = np.zeros((h, w, 1), dtype=np.float32),
			patch_grid = frt.patch_grid,
			context_grid = frt.context_grid,
			patches_value_bhwc = [p[:, :, None] for p in frt.anomaly_ps],
			patches_mask_bhw = frt.patches_mask_bhw,
			**imp,
		)

	anomaly_p_fused = fusion_result.image_fused[:, :, 0]

	return dict(
		anomaly_p = anomaly_p_fused,
		fusion_weight_map = fusion_result.fusion_weight_map,
	)


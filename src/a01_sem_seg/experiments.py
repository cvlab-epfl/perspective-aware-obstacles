
import logging
log = logging.getLogger('exp')
import numpy as np
import torch
# import imageio
from ..pipeline.experiment import ExperimentBase, add_experiment
from .networks import *
from .transforms import *
from ..pipeline.config import CONFIG_PSP
from ..paths import DIR_DATA
from ..datasets.dataset import ChannelLoaderImage, ChannelResultImage, ImageBackgroundService
from ..datasets.cityscapes import DatasetCityscapesSmall, CityscapesLabelInfo
from ..datasets.bdd100k import DatasetBDD_Segmentation, BDDLabelInfo
from ..pytorch_semantic_segmentation import models as ptseg_archs

from .class_statistics import find_frames_containing_classes


def tr_batch_softmax(pred_logits, **_):
	return dict(
		pred_prob = torch.nn.functional.softmax(pred_logits),
	)

def tr_batch_argmax(pred_logits, **_):
	return dict(
		pred_labels = pred_logits.argmax(dim=1, keepdim=False).byte(),
	)

class ExperimentSemSeg(ExperimentBase):

	def init_transforms(self):
		super().init_transforms()

		#self.tr_input.insert(0, TrZeroCenterImgs())

		# TODO organize standard transform-chains

		#self.tr_output_pre_unbatch += [
			#self.class_mod,
			#TrRemoveFields(['image', 'pred_prob_pre_softmax']),
			#TrNP(),
		#]

		#self.tr_output_human_readable.append(SemSegProbabilityToHumanReadable())

		self.tr_input = TrsChain()

		self.tr_colorimg = SemSegLabelsToColorImg(
			colors_by_classid=CityscapesLabelInfo.colors_by_trainId, 
		)

		self.tr_postprocess_log = TrsChain(
			TrNP(),
			self.tr_colorimg,
		)

		self.tr_prepare_batch_test = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image')
		)

		self.tr_prepare_batch_train = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image', 'labels'),
		)

		self.tr_augmentation_crop_and_flip = TrsChain(
			TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [540, 960]), fields = ['image', 'labels']),
			TrRandomlyFlipHorizontal(['image', 'labels']),
		)

	def setup_dset(self, dset):
		pass

	def tr_apply_net(self, **kwargs):
		return self.net_mod(**kwargs)

	def init_loss(self):
		# TODO by class name from cfg
		self.loss_mod = LossCrossEntropy2d()
		self.cuda_modules(['loss_mod'])

	def init_log(self, frames_to_log=None):
		super().init_log()

		frames_to_log = frames_to_log or self.frames_to_log
		self.frames_to_log = set(frames_to_log)

		# Write the ground-truth for comparison
		for fid in self.frames_to_log:
			fid_no_slash = str(fid).replace('/', '__')
			fr = self.datasets['val'].original_dataset().get_frame_by_fid(fid)
			#fr.apply(self.tr_colorimg)

			labels_colorimg = self.tr_colorimg.forward('', fr.labels)

			ImageBackgroundService.imwrite(self.train_out_dir / f'gt_image_{fid_no_slash}.webp', fr.image)
			ImageBackgroundService.imwrite(self.train_out_dir / f'gt_labels_{fid_no_slash}.png', labels_colorimg)
			
			self.tboard_gt.add_image(
				'{0}_img'.format(fid_no_slash),
				fr.image.transpose((2, 0, 1)),
				0,
			)

			self.tboard_gt.add_image(
				'{0}_class'.format(fid_no_slash),
				labels_colorimg.transpose((2, 0, 1)),
				0,
			)

	def eval_batch_log(self, frame, fid, pred_prob, **_):
		if fid in self.frames_to_log:
			frame.apply(self.tr_postprocess_log)

			fid_no_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			ImageBackgroundService.imwrite(self.train_out_dir / f'e{epoch:03d}_labels_{fid_no_slash}.png', frame.pred_labels_colorimg)

			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.pred_labels_colorimg.transpose((2, 0, 1)),
				epoch,
			)

	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),

		if role == 'test':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_apply_net,
					tr_batch_softmax,
					tr_batch_argmax,
					TrKeepFields('pred_prob', 'pred_labels'),
					TrNP(),
				),
				tr_output = TrsChain(
					TrColorimg('pred_labels'),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_prepare_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.tr_apply_net,
					self.loss_mod,
					tr_batch_argmax,
					TrKeepFieldsByPrefix('loss', 'pred_labels'),
				),
				tr_output = TrsChain(
					self.eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = TrsChain(
					self.tr_input,
					self.tr_augmentation_crop_and_flip,
				),
				tr_batch_pre_merge = self.tr_prepare_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.training_start_batch,
					self.tr_apply_net,
					self.loss_mod,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)


class ExpSemSegPSP(ExperimentSemSeg):

	cfg = add_experiment(CONFIG_PSP,
		name='semseg_psp_01_ctc_try2',
		net = dict (
			batch_train = 3,
			batch_eval = 6,
		),
		train = dict (
			crop_size = [384, 768],
			epoch_limit = 50,
		),
	)

	def init_loss(self):
		# TODO by class name from cfg
		self.loss_mod = LossPSP()
		self.cuda_modules(['loss_mod'])

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		self.net_mod = ptseg_archs.PSPNet(
			num_classes = self.cfg['net']['num_classes'], 
			pretrained = self.cfg['net'].get('backbone_pretrained', True), 
			use_aux = self.cfg['net'].get('use_aux', True), 
		)

		if self.cfg['net'].get('backbone_freeze', False):
			log.info('Freeze backbone')
			for i in range(5):
				backbone_mod = getattr(self.net_mod, f'layer{i}')
				for param in backbone_mod.parameters():
					param.requires_grad = False


		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.load_checkpoint_to_net(self.net_mod, chk)
	
		self.cuda_modules(['net_mod'])

	def tr_apply_net(self, image, **_):
		return dict(
			pred_logits = self.net_mod(image),
		)

	def setup_dset(self, dset):
		dset.discover()
		dset.load_class_statistics()

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetCityscapesSmall(split='train', b_cache=b_threaded)
		dset_val = DatasetCityscapesSmall(split='val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_val.frames[i].fid for i in [2, 3, 4, 5, 6, 8, 9]])
		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)


class ExpSemSegPSP02(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='0501_PSP_CTC_02',
	)


class ExpSemSegPSP_BDD(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='semseg_psp_01_bdd',
		net = dict(
			apex_mode = False,
		),
	)

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetBDD_Segmentation(split='train', b_cache=b_threaded)
		dset_val = DatasetBDD_Segmentation(split='val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_val.frames[i].fid for i in [2, 3, 4, 5, 6, 8, 9]])
		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)


# class DatasetCityscapesSmall_PredictedSemantics(DatasetCityscapesSmall):
# 	def __init__(self, *args, dir_semantics=DIR_DATA / 'sem_seg/psp_cityscapes', **kwargs):
# 		super().__init__(*args, **kwargs)

# 		self.dir_semantics = dir_semantics

# 		self.add_channels(
# 			pred_labels=ChannelLoaderImage(
# 				img_ext='.png',
# 				file_path_tmpl='{dset.dir_semantics}/{dset.split}/{fid}_gtFine_labelIds{channel.img_ext}',
# 			),
# 		)



# class DatasetLostAndFoundSmall_PredictedSemantics()

#add_experiment(name = 'semseg_psp_baseline_01',
	#base=CONFIG_PSP,
	#net = dict(
	#),
#)





channel_pred_labels_bus = ChannelResultImage('sem_seg/pred_bus', suffix='_trainIds', img_ext='.png')
channel_pred_labels_bus_colorimg = ChannelResultImage('sem_seg/pred_bus', suffix='_colorimg', img_ext='.png')


BUS_EXCLUDED_CLASSES = [CityscapesLabelInfo.name2id[n] for n in ['bus', 'truck', 'train']]

def dset_exclude_classes(dset, classes):
	idx_present, idx_not_present = find_frames_containing_classes(dset.class_statistics, classes)

	log.info('Excluding {ne}/{na} frames (classes {cs})'.format(
		ne = idx_present.__len__(), na = dset.frames.__len__(), cs=classes),
	)

	dset.frames = [dset.frames[i] for i in idx_not_present]


class CtcNoBusMixin:
	def setup_dset(self, dset):
		super().setup_dset(dset)

		dset.discover()
		dset.load_class_statistics()

		dset.add_channels(
			pred_labels=channel_pred_labels_bus,
			pred_labels_colorimg=channel_pred_labels_bus_colorimg,
		)
		dset.channel_disable('pred_labels', 'pred_labels_colorimg')

	def init_default_datasets(self, b_threaded=False, b_exclude=True):
		dset_train = DatasetCityscapesSmall(split='train', b_cache=b_threaded)
		dset_val = DatasetCityscapesSmall(split='val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

			if b_exclude:
				dset_exclude_classes(dset, BUS_EXCLUDED_CLASSES)

		self.frames_to_log = set([dset_val.frames[i].fid for i in [2, 3, 4, 5, 6, 8, 9]])

		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)


class ExpSemSegPSP_NoBus(CtcNoBusMixin, ExpSemSegPSP):

	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='0503_psp_noBus'
	)

	# def init_default_datasets(self, b_threaded=False, b_remove_bus=True):
	# 	# Cityscapes with prediction channel
	# 	#dset_train = DatasetCityscapesSmall(split='train', b_cache=b_threaded)
	# 	# mask the buses!
	#
	# 	dset_val = DatasetCityscapesSmall(split='val', b_cache=b_threaded)
	# 	dset_test = DatasetCityscapesSmall(split='val', b_cache=b_threaded)
	#
	# 	dsets = [dset_val] #[dset_train, dset_val]
	# 	for dset in dsets:
	#
	# 		dset.add_channels(
	# 			pred_labels = channel_pred_labels_bus,
	# 			pred_labels_colorimg = channel_pred_labels_bus_colorimg,
	# 		)
	#
	# 		dset.channel_disable('pred_labels', 'pred_labels_colorimg')
	#
	# 		dset.discover()
	#
	# 	#self.frames_to_log = set([dset_val[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])
	#
	# 	#self.set_dataset('train', dset_train)
	# 	self.set_dataset('val', dset_val)


class ExpSemSegBayes(ExperimentSemSeg):

	cfg = add_experiment(CONFIG_PSP,
		name='0503_BayesSemSeg',
		net = dict(
			batch_eval = 6,
			batch_train = 4,
		),
		train = dict (
			crop_size = [384, 768],
		),
	)

	uncertainty_field_name = 'pred_var_dropout'

	def setup_dset(self, dset):
		dset.discover()

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		self.net_mod = BayesianSegNet(self.cfg['net']['num_classes'], pretrained=True)

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def tr_net(self, image, **_):
		result = self.net_mod(image)

		return dict(
			pred_logits = result,
		)

	def tr_net_with_uncertainty(self, image, **_):
		result = self.net_mod.forward_multisample(image)

		return dict(
			pred_prob = result['mean'],
			pred_labels = result['mean'].argmax(dim=1).byte(),
			pred_var_dropout = torch.sum(result['var'], 1),
		)

	def construct_default_pipeline(self, role):

		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net_with_uncertainty,
					TrKeepFields('pred_labels', 'pred_var_dropout'),
					TrNP(),
				),
				tr_output = TrsChain(
					TrColorimg('pred_labels'),
				),
				loader_args = self.loader_args_for_role(role),
			)
		else:
			return super().construct_default_pipeline(role)
		
		# elif role == 'val':
		# 	return Pipeline(
		# 		tr_input = TrsChain(
		# 		),
		# 		tr_batch_pre_merge = self.tr_prepare_batch_train,
		# 		tr_batch = TrsChain(
		# 			TrAsType({'labels': torch.LongTensor}), # long tensor error
		# 			TrCUDA(),
		# 			self.tr_net,
		# 			self.loss_mod,
		# 			self.class_softmax,
		# 			TrKeepFieldsByPrefix('loss', 'pred_prob'),
		# 		),
		# 		tr_output = TrsChain(
		# 			self.eval_batch_log,
		# 			TrKeepFieldsByPrefix('loss'),
		# 			TrNP(),
		# 		),
		# 		loader_args = self.loader_args_for_role(role),
		# 	)

		# elif role == 'train':
		# 	return Pipeline(
		# 		tr_input = TrsChain(
		# 			TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [540, 960]), fields = ['image', 'labels']),
		# 			TrRandomlyFlipHorizontal(['image', 'labels']),
		# 		),
		# 		tr_batch_pre_merge = self.tr_prepare_batch_train,
		# 		tr_batch = TrsChain(
		# 			TrAsType({'labels': torch.LongTensor}), # long tensor error
		# 			TrCUDA(),
		# 			self.training_start_batch,
		# 			self.tr_net,
		# 			self.loss_mod,
		# 			self.training_backpropagate,
		# 			TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
		# 		),
		# 		tr_output = TrsChain(
		# 			TrKeepFieldsByPrefix('loss'),
		# 			TrNP(),
		# 		),
		# 		loader_args = self.loader_args_for_role(role),
		# 	)



class ExpSemSegBayes_NoBus(CtcNoBusMixin, ExpSemSegBayes):
	cfg = add_experiment(ExpSemSegBayes.cfg,
		name='0503_BayesSegNet_noBus',
	)

class ExpSemSegBayes_BDD(ExpSemSegBayes):
	cfg = add_experiment(ExpSemSegBayes.cfg,
		name='0120_BayesSegNet_BDD',
	)

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetBDD_Segmentation(split='train', b_cache=b_threaded)
		dset_val = DatasetBDD_Segmentation(split='val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_val.frames[i].fid for i in [2, 3, 4, 5, 6, 8, 9]])
		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)


class ExpSemSegPSP_Ensemble(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='0501_PSP_CTC',
	)

	uncertainty_field_name = 'pred_var_ensemble'

	@classmethod
	def make_sub_exp(cls, exp_id):
		cfg = add_experiment(cls.cfg,
			name = "{orig_name}_{i:02d}".format(
				orig_name=cls.cfg['name'], 
				i=exp_id,
			),
		)
		return cls(cfg)

	def load_subexps(self, additional_subexp_names=[]):
		self.sub_exps = []
		name = self.cfg['name']

		workdir_base = Path(self.workdir).parent
		sub_exp_dirs = list(workdir_base.glob(self.cfg['name'] + '_*'))

		sub_exp_dirs += [workdir_base / a for a in additional_subexp_names]

		for se_dir in sub_exp_dirs:
			se_cfg = json.loads((se_dir / 'config.json').read_text())
			sexp = self.__class__(se_cfg)

			self.sub_exps.append(sexp)

		log.info('Found sub-exps: {ses}'.format(ses=', '.join(se.cfg['name'] for se in self.sub_exps)))

	def init_net(self, role):
		if role == 'master_eval':
			for exp in self.sub_exps:
				exp.init_net('eval')

			#self.net_mods = [exp.net_mod for exp in self.sub_exps]
		else:
			super().init_net(role)

	def tr_ensemble(self, image, **_):

		results = []

		# for net in self.net_mods:
		# 	results.append(
		# 		torch.nn.functional.softmax(
		# 			net(image=image)['pred_logits'],
		# 			dim=1,
		# 		),
		# 	)
		for sub_exp in self.sub_exps:
			results.append(
				torch.nn.functional.softmax(
					sub_exp.tr_apply_net(image=image)['pred_logits'],
					dim=1,
				),
			)


		results = torch.stack(results)
		avg = torch.mean(results, 0)
		var = torch.sum(torch.var(results, 0), 1)

		return dict(
			pred_prob = avg,
			pred_labels = avg.argmax(dim=1).byte(),
			pred_var_ensemble = var,
		)

	def construct_default_pipeline(self, role):
		if role == 'test':
			return Pipeline(
				tr_input=TrsChain(
				),
				tr_batch_pre_merge=TrsChain(
					TrZeroCenterImgs(),
					tr_torch_images,
					TrKeepFields('image')
				),
				tr_batch=TrsChain(
					TrCUDA(),
					self.tr_ensemble,
					TrKeepFields('pred_labels', 'pred_var_ensemble'),
					TrNP(),
				),
				tr_output=TrsChain(
					TrColorimg('pred_labels')
				),
				loader_args=self.loader_args_for_role(role),
			)
		else:
			return super().construct_default_pipeline(role)


class ExpSemSegPSP_Ensemble_BDD(ExpSemSegPSP_Ensemble):
	cfg = add_experiment(ExpSemSegPSP_NoBus.cfg,
		name='0121_PSPEns_BDD',
		epoch_limit=20, # bdd something has 12 epochs
	)

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetBDD_Segmentation(split='train', b_cache=b_threaded)
		dset_val = DatasetBDD_Segmentation(split='val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_val.frames[i].fid for i in [2, 3, 4, 5, 6, 8, 9]])
		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)



### end of eval ###


class ExpSemSegPSPnoBus_Ensemble(CtcNoBusMixin, ExpSemSegPSP_Ensemble):
	cfg = add_experiment(ExpSemSegPSP_NoBus.cfg,
		name='0104_PSP_CTCnoBus',
	)


class ExpSemSegProbout(ExpSemSegPSP):

	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='0106_ProboutPSP',
		net = dict(
			batch_eval = 4,
			batch_train = 2,
			small_variance_weights = True,
		),
		train = dict (
			crop_size = [384, 768],
			optimizer = dict(
				weight_decay = 1e-6,
			),
		),
	)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		self.net_mod = PerspectiveSceneParsingNet_ProbOut(
			self.cfg['net']['num_classes'],
			pretrained=True,
			small_variance_weights=self.cfg['net']['small_variance_weights'],
		)

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def init_transforms(self):
		super().init_transforms()

		self.tr_postprocess_log = TrsChain(
			TrNP(),
			tr_class_argmax,
			SemSegLabelsToColorImg(
				colors_by_classid=CityscapesLabelInfo.colors_by_trainId,
				fields=[('pred_labels', 'pred_labels_colorimg')],
			),
			tr_probout_sum_unc,
		)

	def eval_batch_log(self, frame, fid, pred_prob, **_):
		if fid in self.frames_to_log:
			frame.apply(self.tr_postprocess_log)
			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.pred_labels_colorimg.transpose((2, 0, 1)),
				self.state['epoch_idx'],
			)
			self.tboard.add_image(
				'{0}_var'.format(fid),
				frame.pred_var_sum,
				self.state['epoch_idx'],
			)

	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),

		pre_batch_train = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image', 'labels'),
		)

		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = TrsChain(
					TrZeroCenterImgs(),
					tr_torch_images,
					TrKeepFields('image')
				),
				tr_batch = TrsChain(
					TrCUDA(),
					self.net_mod,
					tr_batch_softmax,
					TrKeepFields('pred_prob', 'pred_var_ens'),
					TrNP(),
				),
				tr_output = TrsChain(
					tr_class_argmax,
					tr_probout_sum_unc,
					# Convert
					TrAsType({'pred_labels': np.uint8}),
					#lambda pred_labels, **_: dict(pred_labels=pred_labels.astype(np.uint8)),
					SemSegLabelsToColorImg(colors_by_classid=CityscapesLabelInfo.colors_by_trainId),
					#SemSegProbabilityToEntropy()
				),
				loader_args = self.loader_args_for_role(role),
			)
		elif role == 'val':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = pre_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.net_mod,
					self.net_mod.prob_loss,
					self.class_softmax,
					TrKeepFieldsByPrefix('loss', 'pred'),
				),
				tr_output = TrsChain(
					self.eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = TrsChain(
					TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [540, 960]), fields = ['image', 'labels']),
					TrRandomlyFlipHorizontal(['image', 'labels']),
				),
				tr_batch_pre_merge = pre_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.training_start_batch,
					self.net_mod,
					self.net_mod.prob_loss,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

class ExpSemSegProbout_P2(ExpSemSegProbout):
	cfg = add_experiment(ExpSemSegProbout.cfg,
		name='0106_ProboutPSP_noSmVar',
		net=dict(
			small_variance_weights=False,
		),
	)

class ExpSemSegProbout_NoBus(CtcNoBusMixin, ExpSemSegProbout):
	cfg = add_experiment(ExpSemSegProbout.cfg,
		name='0106_ProboutPSP_noBus',
	)

	# def setup_dset(self, dset):
	# 	super().setup_dset(dset)
	# 	dset.load_class_statistics()
	# 	dset_exclude_classes(dset, BUS_EXCLUDED_CLASSES)

def DenseNet_build_optimizer(self, role, chk_optimizer=None):
	log.info('Building optimizer')
	# self.optimizer = self.protesis.build_optimizer(self.cfg['train']['optimizer'], self.net_mod)

	cfg_opt = self.cfg['train']['optimizer']

	network = self.net_mod
	self.optimizer = torch.optim.RMSprop(
		[p for p in network.parameters() if p.requires_grad],
		lr=cfg_opt['learn_rate'],
		weight_decay=cfg_opt.get('weight_decay', 0),
	)
	self.learn_rate_scheduler = ReduceLROnPlateau(
		self.optimizer,
		patience=cfg_opt['lr_patience'],
		min_lr=cfg_opt['lr_min'],
	)

	if chk_optimizer is not None:
		self.optimizer.load_state_dict(chk_optimizer['optimizer'])

class ExpSemSegDenseNet(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
	name='0108_DenseNet_std',
		net=dict(
			batch_train=2,
			batch_eval=3,
			dropout = 0.2,
		),
		# We train with RMS-Prop with a constant learning rate of 0.001 and weight decay 10−4.
		train=dict(
			crop_size=[364, 364],
			optimizer = dict(
				learn_rate = 0.001,
				weight_decay = 1e-4,
			),
		),
	)

	def init_loss(self):
		# TODO by class name from cfg
		self.loss_mod = LossCrossEntropy2d()

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		cfg_net = self.cfg['net']

		self.net_mod = FCDenseNet103(out_channels=cfg_net['num_classes'], dropout=cfg_net['dropout'])

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def build_optimizer(self, role, chk_optimizer=None):
		DenseNet_build_optimizer(self, role, chk_optimizer)

	def tr_apply_net(self, image, **_):
		return dict(
			pred_logits = self.net_mod(image),
		)

class ExpSemSegDenseNet_NoBus(CtcNoBusMixin, ExpSemSegDenseNet):
	cfg = add_experiment(ExpSemSegDenseNet.cfg,
		name='0108_DenseNet_noBus',
	)


class ExpSemSegEpistemic(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
		name='0109_EpistemicPSP',
		net=dict(
			batch_train=3,
			batch_eval=6,
		),
		train=dict(
			crop_size=[384, 768],
		),
	)

	def init_loss(self):
		# TODO by class name from cfg
		self.loss_mod = LossCrossEntropy2d()

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		cfg_net = self.cfg['net']

		self.net_mod = PerspectiveSceneParsingNet_Epistemic(num_classes = cfg_net['num_classes'])

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def eval_batch_log(self, frame, fid, pred_prob, pred_prob_var, **_):
		if fid in self.frames_to_log:
			frame.apply(self.tr_postprocess_log)
			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.pred_labels_colorimg.transpose((2, 0, 1)),
				self.state['epoch_idx'],
			)
			self.tboard.add_image(
				'{0}_var'.format(fid),
				pred_prob_var,
				self.state['epoch_idx'],
			)

	def transforms_for_pipeline(self):
		self.tr_net_test = TrsChain(
			self.net_mod,
			self.net_mod.epistemic_softmax.forward_mean_and_var,
		)

		self.tr_net_val = TrsChain(
			self.net_mod,
			self.net_mod.epistemic_softmax.forward_mean_and_var,
			self.net_mod.epistemic_softmax.loss,
		)

		self.tr_net_train = TrsChain(
			self.net_mod,
			self.net_mod.epistemic_softmax.forward_mean,
			self.net_mod.epistemic_softmax.loss,
		)

	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),
		self.transforms_for_pipeline()

		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net_test,
					TrKeepFields('pred_prob', 'pred_prob_var'),
					TrNP(),
				),
				tr_output = TrsChain(
					tr_class_argmax,
					TrAsType({'pred_labels': np.uint8}),
					SemSegLabelsToColorImg(colors_by_classid=CityscapesLabelInfo.colors_by_trainId),
				),
				loader_args = self.loader_args_for_role(role),
			)
		elif role == 'test_dropout':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.net_mod.forward_multisample,
					TrKeepFieldsByPrefix('pred'),
					TrNP(),
				),
				tr_output = TrsChain(
					tr_class_argmax,
					TrAsType({'pred_labels': np.uint8}),
					SemSegLabelsToColorImg(colors_by_classid=CityscapesLabelInfo.colors_by_trainId),
				),
				loader_args = self.loader_args_for_role('test'),
			)


		elif role == 'val':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.tr_net_val,
					TrKeepFieldsByPrefix('loss', 'pred_prob'),
				),
				tr_output = TrsChain(
					self.eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = self.tr_augmentation_crop_and_flip,
				tr_batch_pre_merge = self.tr_prepare_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.LongTensor}), # long tensor error
					TrCUDA(),
					self.training_start_batch,
					self.tr_net_train,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

class ExpSemSegEpistemic_NoBus(CtcNoBusMixin, ExpSemSegEpistemic):
	cfg = add_experiment(ExpSemSegEpistemic.cfg,
		name=ExpSemSegEpistemic.cfg['name'] + '_noBus',
	)

class ExpSemSegEpistemicDenseNet(ExpSemSegEpistemic):
	cfg = add_experiment(ExpSemSegDenseNet.cfg, # cfg from Dense
		name='0110_DenseNetEpistemic_std',
	)

	def init_loss(self):
		self.loss_mod = LossCrossEntropy2d()

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		cfg_net = self.cfg['net']

		self.net_mod = DenseNet_Epistemic(cfg_net['num_classes'], dropout=cfg_net['dropout'])

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def build_optimizer(self, role, chk_optimizer=None):
		DenseNet_build_optimizer(self, role, chk_optimizer)


class ExpSemSegEpistemicDenseNet_NoBus(CtcNoBusMixin, ExpSemSegEpistemicDenseNet):
	cfg = add_experiment(ExpSemSegEpistemicDenseNet.cfg,
		name=ExpSemSegEpistemicDenseNet.cfg['name'] + '_NoBus',
	)
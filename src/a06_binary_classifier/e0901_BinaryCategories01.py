
import logging
log = logging.getLogger('exp')
import numpy as np
import torch
import einops
import imageio

from ..pipeline.frame import Frame
from ..pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix
from ..pipeline.experiment import ExperimentBase
from ..pipeline.pipeline import Pipeline
from ..pipeline.config import add_experiment
from ..pipeline.transforms_imgproc import TrZeroCenterImgs, TrRandomCrop, TrRandomlyFlipHorizontal
from ..pipeline.transforms_pytorch import tr_torch_images, TrCUDA, TrNP
from ..datasets.generic_sem_seg import class_weights_from_class_distrib
from ..a01_sem_seg.transforms import TrColorimg
from ..a01_sem_seg.exp0130_half_precision import ExpSemSegPSP_Apex

from ..datasets.cityscapes import DatasetCityscapesSmall
from ..pytorch_semantic_segmentation import models as ptseg_archs
from ..common.jupyter_show import adapt_img_data

from torch.nn.functional import binary_cross_entropy_with_logits


def label_divisions_random(num_classes, num_bits):
	
	is_included_template = np.arange(num_bits) < num_bits // 2
	
	class_included_table = np.tile(is_included_template[None, :], reps=(num_classes, 1))
	
	for c in range (num_classes):
		np.random.shuffle(class_included_table[c])

	return class_included_table


def class_weights_from_dset(dset):
	""" Creates class weights by trainId """
	label_info = dset.label_info
	# full ids of classes which have a trainId
	class_ids = label_info.table_trainId_to_label[:label_info.num_trainIds]
	# distribution of those classes
	trainids_areas = dset.class_statistics['class_area_total'][class_ids]

	return class_weights_from_class_distrib(trainids_areas)


class Exp0901_BinaryCategories01(ExpSemSegPSP_Apex):

	cfg = add_experiment(
		name='0901_BinaryCategories01',
		net = dict (
			batch_train = 12,
			batch_eval = 8, # full images
			backbone_freeze = True,
			loss_partition_positive_weights = True,
			loss_class_weights = False,
		),
		train = dict (
			crop_size = [384, 768],
			checkpoint_interval = 1,
			optimizer = dict(
				type = 'adam',
				learn_rate = 1e-4,
				lr_patience = 5,
				lr_min = 1e-8,
				weight_decay = 0,
			),
			epoch_limit = 50,
		),
		binary_category_table = [
			[ True,  True, False, False],
			[False,  True, False,  True],
			[ True, False, False,  True],
			[False,  True, False,  True],
			[False,  True, False,  True],
			[ True, False, False,  True],
			[False,  True, False,  True],
			[False,  True, False,  True],
			[False,  True,  True, False],
			[False, False,  True,  True],
			[False, False,  True,  True],
			[False, False,  True,  True],
			[False, False,  True,  True],
			[ True,  True, False, False],
			[ True, False, False,  True],
			[ True, False, False,  True],
			[False,  True,  True, False],
			[False,  True,  True, False],
			[False,  True,  True, False],
		],
	)

	def init_transforms(self):
		super().init_transforms()

		self.binary_category_table = np.array(self.cfg['binary_category_table'])
		self.tr_labels_to_categories = TrLabelsToBinaryDivisions(class_inclusion_table=self.binary_category_table)


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


	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		num_classes, num_cats = self.binary_category_table.shape

		self.net_mod = ptseg_archs.PSPNet(
			num_classes = num_cats, # output for each class
			pretrained = True,
			use_aux = False,
		)

		if self.cfg['net'].get('backbone_freeze', False):
			log.info('Freeze backbone')
			for i in range(5):
				backbone_mod = getattr(self.net_mod, f'layer{i}')
				for param in backbone_mod.parameters():
					param.requires_grad = False


		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])
	
		self.cuda_modules(['net_mod'])

	def tr_apply_net(self, image, **_):
		return dict(
			pred_category_logits = self.net_mod(image),
		) 

	def init_loss(self):
		# class weights from loaded training dataset

		if self.cfg['net']['loss_category_weights']:
			cat_stats = self.tr_labels_to_categories.calc_loss_weights_dset(dset = self.datasets['train'])
			self.partition_positive_weights = cat_stats['cat_positive_weights']
			# normalize so that they sum to 1
			self.partition_positive_weights /= (1+self.partition_positive_weights)
			log.debug(f"Partition positive fractions {cat_stats['cat_positive_fractions']}")
		else:
			log.debug(f"Partition positive weights disabled")
			self.partition_positive_weights = None

		self.loss_mod = TrMultiBinaryCrossEntropyLoss(
			positive_weights = self.partition_positive_weights,
		)

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

			category_gt_masks = self.tr_labels_to_categories(fr)['labels_binary_categories']

			gt_image = np.concatenate([
				np.concatenate(category_gt_masks[i::2], axis=0)
				for i in range(2)
			], axis=1).astype(np.uint8) * 255

			imageio.imwrite(self.train_out_dir / f'gt_image_{fid_no_slash}.webp', fr.image)
			imageio.imwrite(self.train_out_dir / f'gt_labels_{fid_no_slash}.png', gt_image)
			
			# self.log_gt.add_image(
			# 	'{0}_img'.format(fid_no_slash),
			# 	fr.image.transpose((2, 0, 1)),
			# 	0,
			# )

			# self.log_gt.add_image(
			# 	'{0}_class'.format(fid_no_slash),
			# 	labels_colorimg.transpose((2, 0, 1)),
			# 	0,
			# )

	def eval_batch_log(self, frame, fid, pred_category_logits, **_):
		if fid in self.frames_to_log:
			# frame.apply(self.tr_postprocess_log)

			fid_no_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			probs = torch.sigmoid(pred_category_logits.detach()).cpu().numpy()

			pred_image = adapt_img_data(np.concatenate([
				np.concatenate(probs[i::2], axis=0)
				for i in range(2)
			], axis=1))
			
			imageio.imwrite(self.train_out_dir / f'e{epoch:03d}_pred_{fid_no_slash}.webp', pred_image)

			# self.log.add_image(
			# 	'{0}_class'.format(fid),
			# 	frame.pred_labels_colorimg.transpose((2, 0, 1)),
			# 	epoch,
			# )

	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),

		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_apply_net,
					TrByField([('pred_category_logits', 'pred_category_probs')], operation=torch.sigmoid),
					TrCategoryLogitsToClassProbs(self.binary_category_table),
					# TrKeepFields('pred_category_logits'),
					TrKeepFields('pred_class_trainId', 'pred_class_prob', 'pred_category_probs'),
					TrNP(),
				),
				tr_output = TrsChain(
					TrCategoryLogitsToBinaryClasses(self.binary_category_table),
					TrColorimg('pred_class_trainId'),
					TrColorimg('pred_class_by_bitstring_trainId'),

					# tr_class_argmax,
					# TrAsType({'pred_labels': np.uint8}),
					# SemSegLabelsToColorImg(colors_by_classid=CityscapesLabelInfo.colors_by_trainId),
				),
				loader_args = self.loader_args_for_role(role),
			)
			return None

		elif role == 'val':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels': torch.ByteTensor}),
					TrCUDA(),
					self.tr_labels_to_categories,
					self.tr_apply_net,
					self.loss_mod,
					TrKeepFieldsByPrefix('loss', 'pred_category_logits'),
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
					TrAsType({'labels': torch.ByteTensor}),
					TrCUDA(),
					self.tr_labels_to_categories,
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


class TrLabelsToBinaryDivisions(TrByField):
	def __init__(self, class_inclusion_table, fields=[('labels', 'labels_binary_categories')]):
		super().__init__(fields)
		self.class_inclusion_table = class_inclusion_table
		
		num_cats = self.class_inclusion_table.shape[1]
		self.classes_in_category = [
			np.where(self.class_inclusion_table[:, cat])[0]
			for cat in range(num_cats)
		]
		# print(self.classes_in_category)
		
		
	def forward(self, field_name, labels):
		num_classes, num_cats = self.class_inclusion_table.shape
		
		with torch.no_grad():
			class_masks = [
				labels == c
				for c in range(num_classes)
			]

			s = labels.shape
			
			if isinstance(labels, np.ndarray):
	# 			category_masks = np.zeros(labels.shape + (num_cats,), dtype=bool)
				category_masks = np.zeros(s[:-2] + (num_cats,) + s[-2:], dtype=bool)

			elif isinstance(labels, torch.Tensor):
	# 			category_masks = torch.zeros(labels.shape + (num_cats,), dtype=torch.uint8, device=labels.device)
				category_masks = torch.zeros(s[:-2] + (num_cats,) + s[-2:], dtype=torch.uint8, device=labels.device)
				# https://github.com/pytorch/pytorch/issues/22992
				class_masks = [x.byte() for x in class_masks]

			for cat, cat_cl_list in enumerate(self.classes_in_category):
				for cl in cat_cl_list:
					category_masks[..., cat, :, :] |= class_masks[cl]
				
		return category_masks
	
	
	def calc_loss_weights(self, class_areas):
		num_classes, num_cats = self.class_inclusion_table.shape
		
		area_sum = np.sum(class_areas)
		
		fractions = np.zeros(num_cats, dtype=np.float64)
		pos_weights = np.zeros(num_cats, dtype=np.float64)
		
		for cat in range(num_cats):
			positive_fraction = np.sum(class_areas[self.class_inclusion_table[:, cat]]) / area_sum
			
			class_distrib = np.array([1-positive_fraction, positive_fraction], dtype=np.float64)
			# class_weights[0] = neg weight, class_weights[1] = pos weight
			class_weights = class_weights_from_class_distrib(class_distrib)
			
			positive_weight = class_weights[1] / class_weights[0]
		
			fractions[cat] = positive_fraction
			pos_weights[cat] = positive_weight
			
		return dict(
			cat_positive_fractions = fractions,
			cat_positive_weights = pos_weights,
		)
	
	def calc_loss_weights_dset(self, dset):
		label_info = dset.label_info
	
		dset.load_class_statistics()	
		class_areas = dset.class_statistics['class_area_total']
		class_areas_trainIds = class_areas[label_info.table_trainId_to_label[:label_info.num_trainIds]]
		
		return self.calc_loss_weights(class_areas_trainIds)


class TrMultiBinaryCrossEntropyLoss(TrBase):
	def __init__(self, positive_weights=None):
		self.positive_weights = torch.Tensor(positive_weights) if positive_weights is not None else None		
		log.debug(f'BCE Loss with positive weights {self.positive_weights}')

	def __call__(self, labels_binary_categories, pred_category_logits, class_weight_perpix=None, **_):
		"""
		@param pred_category_logits: [B C H W] float32, C is category 
		@param labels_binary_categories: [B C H W] uint8, binary mask GT, C is category
		@param class_weight_perpix: [B H W] float32
		"""
		
		reshape = 'B C H W -> (B H W) C'.lower()

		bce_loss = binary_cross_entropy_with_logits(
			input = einops.rearrange(pred_category_logits, reshape),
			target = einops.rearrange(labels_binary_categories, reshape).float(),
			weight = einops.rearrange(class_weight_perpix, 'B H W -> (B H W) ()'.lower()) 
				if class_weight_perpix is not None else None,
			pos_weight = self.positive_weights,
			reduction = 'mean',
		)
	
		return dict(
			loss = bce_loss,
		)
	
	def cuda(self):
		if self.positive_weights is not None:
			self.positive_weights = self.positive_weights.cuda()
		return self

class TrMultiBinaryCrossEntropyLoss_ClassWeighted(TrMultiBinaryCrossEntropyLoss):
	pass


class TrPixelClassWeights(TrByField):
	def __init__(self, class_weights, fields=[('labels', 'class_weight_perpix')]):
		super().__init__(fields=fields)
		
		self.class_weights = class_weights
		
		# fill in to 256 so that the invalid label 255 gets a 0 weight
		# self.class_weights_255 = np.zeros(256, dtype=self.class_weights.dtype)
		self.class_weights_255 = np.zeros(256, dtype=np.float32)
		self.class_weights_255[:self.class_weights.shape[0]] = self.class_weights
		
	def forward(self, field_name, labels, **_):
		return self.class_weights_255[labels.reshape(-1)].reshape(labels.shape)


class TrCategoryLogitsToClassProbs(TrBase):
	def __init__(self, class_inclusion_table):
		self.class_inclusion_table = class_inclusion_table
		
		num_cats = self.class_inclusion_table.shape[1]
		self.classes_in_category = [
			np.where(self.class_inclusion_table[:, cat])[0]
			for cat in range(num_cats)
		]
		
	def __call__(self, pred_category_probs, **_):
		num_classes, num_cats = self.class_inclusion_table.shape
		
		B, C, H, W = pred_category_probs.shape
		class_probs = torch.ones((B, num_classes, H, W), device=pred_category_probs.device)
		
		for cat in range(num_cats):
			included_classes = torch.from_numpy(self.class_inclusion_table[:, cat])
# 			print(included_classes.shape)
			excluded_classes = torch.from_numpy(~self.class_inclusion_table[:, cat])
			
# 			print(class_probs[:, included_classes].shape, pred_category_probs[:, cat:cat+1].shape)
# 			print(class_probs[:, excluded_classes].shape, pred_category_probs[:, cat:cat+1].shape)
			
			
			class_probs[:, included_classes] *= pred_category_probs[:, cat:cat+1]
			class_probs[:, excluded_classes] *= ( 1. - pred_category_probs[:, cat:cat+1] )
			
		
		class_probs_np = class_probs.detach().cpu().numpy()
		class_choice = np.argmax(class_probs_np, axis=1)
		
		class_best_prob = np.amax(class_probs_np, axis=1)

		return dict(
			pred_class_trainId = class_choice,
			pred_class_prob = class_best_prob,
		)
		
# 		tr_cimg = TrColorimg()
		
# 		show(*[tr_cimg.forward(None, cc) for cc in class_choice])
# 		print(class_choice.dtype)
		
# 		class_best_prob = class_probs_np[class_choice]


class TrCategoryLogitsToBinaryClasses(TrBase):

	@staticmethod
	def partition_table_to_ints(partition_table):	
		num_cls, num_bits = partition_table.shape
		
		to_pad = 8 - num_bits
		
		# invert channel order to have least significant bit on the right
		partition_table = partition_table[:, ::-1]

		if to_pad < 0:
			raise ValueError(f"Using np.packbits we can't deal with more than 8 categories yet, num bits {num_bits}")

		elif to_pad > 0:
			padding_shape = (num_cls, to_pad)
					
			partition_table = np.concatenate([
				np.zeros(padding_shape, dtype=bool),
				partition_table,
			], axis=1)
		
		class_bitstring_ints = np.packbits(partition_table, axis=1)
		
		
		bitstring_int_to_class = np.full(255, fill_value=255, dtype=np.uint8)
		bitstring_int_to_class[class_bitstring_ints] = np.arange(num_cls)[:, None]
		
		return dict(
			class_bitstring_ints = class_bitstring_ints,
			bitstring_int_to_class = bitstring_int_to_class,
		)

	def __init__(self, class_partition_table):
		self.class_partition_table = class_partition_table

		v = self.partition_table_to_ints(self.class_partition_table)
		self.class_bitstring_ints = v['class_bitstring_ints']
		self.bitstring_int_to_class = v['bitstring_int_to_class']


	def __call__(self, pred_category_bits, **_):
		"""
		@param pred_category_probs: numpy [C, H, W]
		"""
		# num_classes, num_cats = self.class_inclusion_table.shape
	
		C, H, W = pred_category_bits.shape

		# invert channel order to have least significant bit on the right
		pred_category_bitstring = pred_category_bits[::-1]
		# pad with zero to fill the 8 bit number
		to_pad = 8 - pred_category_bitstring.shape[0]
		if to_pad < 0:
			raise ValueError(f"Using np.packbits we can't deal with more than 8 categories yet, prob shape {pred_category_bitstring.shape}")
		# elif to_pad > 0:
		# 	pred_category_bitstring = np.concatenate([
		# 		np.zeros((to_pad,) + pred_category_bitstring.shape[1:], dtype=bool),
		# 		pred_category_bitstring,
		# 	])
		
		# [0] to get rid of a 1-sized dimension, it results from packbits?
		pred_category_int = np.packbits(pred_category_bitstring, axis=0)[0] >> to_pad

		pred_class = self.bitstring_int_to_class[pred_category_int.reshape(-1)].reshape(pred_category_int.shape)

		return dict(
			pred_category_int = pred_category_int,
			pred_class_by_bitstring_trainId = pred_class,
		)


class Exp0902_BinaryCategories02(Exp0901_BinaryCategories01):
	"""
	Use 6 bits instead of 4
	"""
	cfg = add_experiment(Exp0901_BinaryCategories01.cfg,
		name='0902_BinaryCategories02',
		binary_category_table = [
			[True, False, False,  True, False,  True],
			[ True, False,  True, False,  True, False],
			[False,  True,  True, False, False,  True],
			[False,  True, False,  True,  True, False],
			[False,  True,  True,  True, False, False],
			[False, False,  True, False,  True,  True],
			[False, False,  True, False,  True,  True],
			[False, False,  True, False,  True,  True],
			[ True, False,  True,  True, False, False],
			[False, False,  True, False,  True,  True],
			[ True,  True, False, False,  True, False],
			[ True,  True,  True, False, False, False],
			[False, False, False,  True,  True,  True],
			[False,  True,  True, False, False,  True],
			[ True, False,  True, False, False,  True],
			[ True, False, False,  True, False,  True],
			[False,  True, False,  True,  True, False],
			[False, False,  True,  True,  True, False],
			[ True,  True,  True, False, False, False]
		],
	)

class Exp0903_BinaryCategories_ClsWeight(Exp0902_BinaryCategories02):
	"""
	Weight loss by pixel's original class rarity
	"""
	cfg = add_experiment(Exp0902_BinaryCategories02.cfg,
		name='0903_BinaryCategories_ClsWeight',
	)

	def init_loss(self):
		self.loss_mod = TrMultiBinaryCrossEntropyLoss_ClassWeighted()

	def construct_default_pipeline(self, role):
		self.tr_prepare_batch_train = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrPixelClassWeights(
				class_weights = class_weights_from_dset(self.datasets['train']),
			),
			TrKeepFields('image', 'labels', 'class_weight_perpix'),
		)

		return super().construct_default_pipeline(role)


class Exp0904_BinaryCategories_ClsWeight_OtherSplit(Exp0903_BinaryCategories_ClsWeight):
	cfg = add_experiment(Exp0903_BinaryCategories_ClsWeight.cfg,
		name='0904_BinaryCategories_ClsWeight_OtherSplit',

		binary_category_table = [
			[0, 0, 0, 1, 1, 1],
			[0, 1, 0, 1, 0, 1],
			[0, 1, 1, 0, 1, 0],
			[0, 0, 1, 0, 1, 1],
			[1, 1, 0, 1, 0, 0],
			[1, 0, 0, 1, 1, 0],
			[1, 0, 0, 1, 1, 0],
			[0, 1, 1, 1, 0, 0],
			[0, 1, 0, 1, 0, 1],
			[0, 1, 1, 0, 0, 1],
			[0, 1, 1, 0, 0, 1],
			[0, 1, 1, 0, 0, 1],
			[1, 1, 0, 0, 0, 1],
			[1, 0, 0, 1, 1, 0],
			[1, 0, 1, 1, 0, 0],
			[1, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 1, 0],
			[1, 1, 0, 1, 0, 0],
			[0, 1, 1, 0, 1, 0],
		],
	)

from ..a01_sem_seg.experiments import ExpSemSegPSP

class Exp0905_PSP_Frozen(ExpSemSegPSP):
	cfg = add_experiment(ExpSemSegPSP.cfg,
		name = '0905_PSP_Frozen',
		net = dict(
			backbone_freeze = True,
		)
	)


class Exp0906_BinaryCategories_NoFreeze(Exp0902_BinaryCategories02):
	"""
	Use 6 bits instead of 4
	"""
	cfg = add_experiment(Exp0902_BinaryCategories02.cfg,
		name='0906_BinaryCategories_NoFreeze',
		net = dict(
			batch_train = 5,
			batch_eval = 6, # full imgs not crops
			backbone_freeze = False,
		),
	)

class Exp0907_BinaryCategories_ClsWeight_NoFreeze(Exp0903_BinaryCategories_ClsWeight):
	cfg = add_experiment(Exp0903_BinaryCategories_ClsWeight.cfg,
		name='0907_BinaryCategories_ClsWeight_NoFreeze',
		net = dict(
			batch_train = 5,
			batch_eval = 6,
			backbone_freeze = False,
		),
	)

CODES_B6_M3_H2 = [
	[False, False, False,  True,  True,  True],
	[False, False,  True, False,  True,  True],
	[False, False,  True,  True, False,  True],
	[False, False,  True,  True,  True, False],
	[False,  True, False, False,  True,  True],
	[False,  True, False,  True, False,  True],
	[False,  True, False,  True,  True, False],
	[False,  True,  True, False, False,  True],
	[False,  True,  True, False,  True, False],
	[False,  True,  True,  True, False, False],
	[ True, False, False, False,  True,  True],
	[ True, False, False,  True, False,  True],
	[ True, False, False,  True,  True, False],
	[ True, False,  True, False, False,  True],
	[ True, False,  True, False,  True, False],
	[ True, False,  True,  True, False, False],
	[ True,  True, False, False, False,  True],
	[ True,  True, False, False,  True, False],
	[ True,  True, False,  True, False, False],
]


class Exp0908_BinPart_Code632_NoClsW(Exp0906_BinaryCategories_NoFreeze):
	cfg = add_experiment(Exp0906_BinaryCategories_NoFreeze.cfg,
		name='0908_BinPart_Code632_NoClsW',
		binary_category_table = CODES_B6_M3_H2[:19],
	)


class Exp0909_BinPart_Multihead_Code632_NoClsW(Exp0908_BinPart_Code632_NoClsW):
	cfg = add_experiment(Exp0908_BinPart_Code632_NoClsW.cfg,
		name='0909_BinPart_Multihead_Code632_NoClsW',
		net = dict(
			batch_train = 4, # instead of 5 compensate mem usage to fit half gpu
		),
	)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		num_classes, num_cats = self.binary_category_table.shape

		self.net_mod = ptseg_archs.PSPNet_Multihead(
			num_heads = num_cats,
			num_classes = 1, # each head 1 class
			pretrained = True,
		)

		if self.cfg['net'].get('backbone_freeze', False):
			log.info('Freeze backbone')
			for param in self.net_mod.backbone.parameters():
				param.requires_grad = False
				
		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])
	
		self.cuda_modules(['net_mod'])


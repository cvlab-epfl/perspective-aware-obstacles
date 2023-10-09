
import logging
log = logging.getLogger('exp.BinPart')
import numpy as np
import torch
import einops
import imageio

from ..pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix
from ..pipeline.experiment import ExperimentBase
from ..pipeline.pipeline import Pipeline
from ..pipeline.config import add_experiment
from ..pipeline.transforms_imgproc import TrZeroCenterImgs, TrRandomCrop, TrRandomlyFlipHorizontal
from ..pipeline.transforms_pytorch import tr_torch_images, TrCUDA, TrNP, TrOnehotTorch
from ..pipeline.evaluations import Evaluation, TrChannelSave, TrChannelLoad
from ..a01_sem_seg.exp0130_half_precision import ExpSemSegPSP_Apex
from ..a01_sem_seg.transforms import TrColorimg
from ..a01_sem_seg.networks import LossCrossEntropy2d

from ..datasets.dataset import imwrite, ImageBackgroundService, ChannelLoaderImage
from ..datasets.generic_sem_seg import class_weights_from_class_distrib, TrSemSegLabelTranslation
from ..datasets.cityscapes import DatasetCityscapesSmall
from ..pytorch_semantic_segmentation import models as ptseg_archs
from ..common.jupyter_show import adapt_img_data

from torch.nn.functional import binary_cross_entropy_with_logits

from .e0901_BinaryCategories01 import CODES_B6_M3_H2, TrPixelClassWeights, TrMultiBinaryCrossEntropyLoss, TrCategoryLogitsToBinaryClasses


def prepare_unknown_class_cityscapes():
	unk_trainId = 19
	dset_ctc = DatasetCityscapesSmall(split='train')
	dset_ctc.load_class_statistics()
	label_info = dset_ctc.label_info
	class_areas_by_origId = dset_ctc.class_statistics['class_area_total']
	
	unknown_classes = set(label.id for label in label_info.labels if label.ignoreInEval)
	unknown_classes -= set([2, 3]) # rectification border, out of roi
	unknown_classes = np.array(list(sorted(unknown_classes)))
	log.info(f'Class Id counted as unknown: {unknown_classes}')
	
	known_class_ids = label_info.table_trainId_to_label[:label_info.num_trainIds]
	# distribution of those classes
	trainids_areas = class_areas_by_origId[known_class_ids]
	
	unknown_area_sum = np.sum(class_areas_by_origId[unknown_classes])
	
	trainids_areas = np.concatenate(
		[trainids_areas, [unknown_area_sum]]
	)
	
	class_weights = class_weights_from_class_distrib(trainids_areas)
	class_weights /= np.mean(class_weights)
	
	log.info(f'Class areas: {trainids_areas}\n Class weights: {class_weights}')
	
	table_label_to_trainId = label_info.table_label_to_trainId.copy()
	table_label_to_trainId[unknown_classes] = unk_trainId
	
	return dict(
		class_areas = trainids_areas,
		class_weights = class_weights,
		table_to_trainId = table_label_to_trainId,
		tr_to_trainId = TrSemSegLabelTranslation(
			table_label_to_trainId,
			fields=[('labels_source', 'labels')],
		),
	)

class TrLabelsToPartitionMasks(TrByField):
	def __init__(self, class_partition_table, fields=[('labels', 'labels_binary_categories')]):
		super().__init__(fields)
		self.class_partition_table = class_partition_table
		
		num_classes, num_cats = self.class_partition_table.shape

		self.classes_in_category = [
			np.where(self.class_partition_table[:, cat])[0]
			for cat in range(num_cats)
		]
		# print(self.classes_in_category)

		# sample_table[partition_id][label_sequence] = bool sequence for this partition
		# extended to 255 because invalid class is denoted as 255
		self.sample_table = np.zeros((num_cats, 256), dtype=bool)
		self.sample_table[:, :num_classes] = self.class_partition_table.T


	def forward(self, _, labels):
		num_classes, num_cats = self.class_partition_table.shape

		return np.stack([
			self.sample_table[part_id][labels.reshape(-1)].reshape(labels.shape)
			for part_id in range(num_cats)
		], axis=-3)
	
	def calc_loss_weights(self, class_areas):
		num_classes, num_cats = self.class_partition_table.shape
		
		area_sum = np.sum(class_areas)
		
		fractions = np.zeros(num_cats, dtype=np.float64)
		pos_weights = np.zeros(num_cats, dtype=np.float64)
		
		for cat in range(num_cats):
			positive_fraction = np.sum(class_areas[self.class_partition_table[:, cat]]) / area_sum
			
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

	@staticmethod
	def draw_partitions(partition_masks_or_probs):
		return adapt_img_data(np.concatenate([
			np.concatenate(partition_masks_or_probs[i::2], axis=0)
			for i in range(2)
		], axis=1))


# CFG_UNK_AWARE = add_experiment(ExpSemSegPSP_Apex.cfg,
# 	name='CFG_UNK_AWARE',
# 	net = dict(
# 		batch_train = 6,
# 		batch_eval = 6,
# 		num_classes = 20,
# 		apex_mode = 'O1',
# 		backbone_freeze = False,
# 		use_aux = False,
# 	),
# 	train = dict(
# 		epoch_limit = 30,
# 	)
# )

class Exp091X_BinPart_Base(ExpSemSegPSP_Apex):

	cfg = add_experiment(
		name='0901_BinaryCategories01',
		unk_class_variant = 'cityscapes',
		net = dict (
			batch_train = 6,
			batch_eval = 8, # full images
			apex_mode = 'O1',
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
			num_workers = 4,
			epoch_limit = 30,
		),
		binary_category_table = CODES_B6_M3_H2,
	)

	def init_transforms(self):
		super().init_transforms()
		
		self.tr_pre_batch_test = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image')
		)

		self.init_partitions_and_weights()

		self.tr_bitstring_classes = TrCategoryLogitsToBinaryClasses(self.binary_category_table)
		
		self.tr_for_log = TrsChain(
			TrNP(),
			self.tr_bitstring_classes,
			TrColorimg('pred_class_by_bitstring_trainId'),
		)

		self.tr_augmentation_crop_and_flip = TrsChain(
			TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [540, 960]), fields = ['image', 'labels']),
			TrRandomlyFlipHorizontal(['image', 'labels']),
		)

	def init_partitions_and_weights(self):
		
		self.binary_category_table = np.array(self.cfg['binary_category_table'], dtype=bool)
		
		self.tr_labels_to_partitions = TrLabelsToPartitionMasks(
			class_partition_table=self.binary_category_table,
		)

		tr_to_trainIds = lambda **_: {}
		cls_weights = None
		
		if self.cfg['unk_class_variant'] == 'cityscapes':
			self.unk_class_info = prepare_unknown_class_cityscapes()
			tr_to_trainIds = self.unk_class_info['tr_to_trainId']
			cls_weights = self.unk_class_info['class_weights']

		if not self.cfg['net']['loss_class_weights']:
			# equal weights for all classes
			# note that the class 19 "unknown" also gets 1
			# but the "really invalid" locations (rectification border) get 0
			cls_weights = np.ones(20) 

		log.debug(f'Class weights: {cls_weights}')

		self.tr_input = TrsChain(
			tr_to_trainIds,
		)

		self.tr_pre_batch_train = TrsChain(
			self.tr_labels_to_partitions,
			TrPixelClassWeights(class_weights = cls_weights),
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image', 'labels_binary_categories', 'class_weight_perpix'),
		)

	def setup_dset(self, dset):
		dset.discover()
		dset.load_class_statistics()

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		num_classes, num_cats = self.binary_category_table.shape

		self.net_mod = ptseg_archs.PSPNet(
			num_classes = num_cats, # output for each class
			pretrained = True,
			use_aux = False,
		)

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])
	
		self.cuda_modules(['net_mod'])

	def tr_apply_net(self, image, **_):
		logits = self.net_mod(image)
		return dict(
			pred_category_logits = logits,
			pred_category_bits = logits > 0,
			pred_category_probs = torch.sigmoid(logits),
		) 

	def init_loss(self):
		# class weights from loaded training dataset

		if self.cfg['net']['loss_partition_positive_weights']:
			cat_stats = self.tr_labels_to_partitions.calc_loss_weights_dset(dset = self.datasets['train'])
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

			category_gt_masks = self.tr_labels_to_partitions(fr)['labels_binary_categories']

			gt_image = np.concatenate([
				np.concatenate(category_gt_masks[i::2], axis=0)
				for i in range(2)
			], axis=1).astype(np.uint8) * 255

			imwrite(
				self.train_out_dir / f'gt_image_{fid_no_slash}.webp', 
				fr.image,
			)
			imwrite(
				self.train_out_dir / f'gt_labels_{fid_no_slash}.png',
				gt_image,
			)
			
	def eval_batch_log(self, frame, fid, **_):

		if fid in self.frames_to_log:
			# frame.apply(self.tr_postprocess_log)

			fid_no_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			frame = frame.copy()
			frame.apply(self.tr_for_log)
			
			imwrite(
				self.train_out_dir / f'e{epoch:03d}_pred_prob_{fid_no_slash}.webp', 
				self.tr_labels_to_partitions.draw_partitions(frame.pred_category_probs),
			)
			imwrite(
				self.train_out_dir / f'e{epoch:03d}_pred_{fid_no_slash}.png', 
				frame.pred_class_by_bitstring_trainId_colorimg,
			)
			p =self.train_out_dir / f'e{epoch:03d}_pred_prob_{fid_no_slash}.webp'

	def construct_default_pipeline(self, role):

		if role == 'test':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_pre_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_apply_net,
					# TrByField([('pred_category_logits', 'pred_category_probs')], operation=torch.sigmoid),
					#TrCategoryLogitsToClassProbs(self.binary_category_table),
					# TrKeepFields('pred_category_logits'),
					# TrKeepFields('pred_class_trainId', 'pred_class_prob', 'pred_category_probs'),
					TrKeepFields('pred_category_probs', 'pred_category_bits'),
					TrNP(),
				),
				tr_output = TrsChain(
					# self.tr_bitstring_classes,
					# TrColorimg('pred_class_by_bitstring_trainId'),
				),
				loader_args = self.loader_args_for_role(role),
			)
			return None

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_pre_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels_binary_categories': torch.ByteTensor}),
					TrCUDA(),
					self.tr_apply_net,
					self.loss_mod,
					TrKeepFieldsByPrefix('loss', 'pred_category_probs', 'pred_category_bits'),
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
					self.tr_input, # translate labels after crop
					self.tr_augmentation_crop_and_flip,
				),
				tr_batch_pre_merge = self.tr_pre_batch_train,
				tr_batch = TrsChain(
					TrAsType({'labels_binary_categories': torch.ByteTensor}),
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


class Exp0910(ExpSemSegPSP_Apex):
	"""
	"""
	cfg = add_experiment(ExpSemSegPSP_Apex.cfg,
		name='0910_PSP_with_unknown_class',
		unk_class_variant = 'cityscapes',
		net = dict(
			batch_train = 6,
			batch_eval = 6,
			num_classes = 20,
			apex_mode = 'O1',
			backbone_freeze = False,
			use_aux = False,
		),

	)
		
	def init_transforms(self):
		super().init_transforms()

		if self.cfg['unk_class_variant'] == 'cityscapes':
			self.unk_class_info = prepare_unknown_class_cityscapes()
			
		# labels must be changed before augmentation, which operates on labels
		self.tr_input.insert(0, self.unk_class_info['tr_to_trainId'])

		

	def init_loss(self):
		self.loss_mod = LossCrossEntropy2d(
			weight = torch.from_numpy(self.unk_class_info['class_weights']),
		)
		self.cuda_modules(['loss_mod'])


class Exp0911_BinPart_PartWeight(Exp091X_BinPart_Base):
	cfg = add_experiment(Exp091X_BinPart_Base.cfg,
		name='0911_BinPart_PartWeight',
		net = dict(
			loss_partition_positive_weights = True,
			loss_class_weights = False,
		)
	)

class Exp0912_BinPart_PartWeight_Perm(Exp091X_BinPart_Base):
	cfg = add_experiment(Exp091X_BinPart_Base.cfg,
		name='0912_BinPart_PartWeight_Perm',
		net = dict(
			loss_partition_positive_weights = True,
			loss_class_weights = False,
		),
		binary_category_table = [CODES_B6_M3_H2[r] for r in [11,  7, 13,  5, 17,  1,  9,  3,  0, 18,  2, 16,  6, 10, 15, 12,  4,  8, 14]],
	)

class Exp0913_BinPart_PartWeight_ClsW(Exp091X_BinPart_Base):
	cfg = add_experiment(Exp091X_BinPart_Base.cfg,
		name='0913_BinPart_PartWeight_ClsW',
		net = dict(
			loss_partition_positive_weights = True,
			loss_class_weights = True,
		)
	)



class EvaluationSemSegWithUnknown(Evaluation):

	def __init__(self, exp):
		self.exp = exp
		self.construct_persistence()

	def construct_persistence(self):
		self.persistence_base_dir = '{channel.ctx.exp.workdir}/pred/{dset.name}_{dset.split}/'

		self.chan_pred_trainId = ChannelLoaderImage(self.persistence_base_dir+'data/{fid_no_slash}_predTrainId.png')
		self.chan_pred_trainId.ctx = self
		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'demo/{fid_no_slash}_demo.webp')
		self.chan_demo.ctx = self

	def construct_tr_make_predictions(self, dset):
		self.exp.unk_class_info['tr_to_trainId']

		self.tr_colorimg = TrColorimg('pred_labels')
		self.tr_colorimg.set_override(19, [0, 0, 0])
		
		# self.tr_colorimg.set_override(19, [170, 85, 0])

		return TrsChain(
			self.tr_colorimg,
			self.tr_make_demo,
			TrChannelSave(self.chan_pred_trainId, 'pred_labels'),
			TrChannelSave(self.chan_demo, 'demo'),
		)


	def tr_make_demo(self, image, pred_labels_colorimg, labels = None, **_):
	#def out_demo(fid, dset, image, pred_class_trainId_colorimg, pred_class_by_bitstring_trainId_colorimg, pred_class_prob, labels_colorimg=None, **_):
	#fid_no_slash = fid.replace('/', '__')
	
# 	pred_class_prob_img = adapt_img_data(pred_class_prob)
# 	print(np.min(pred_class_prob), np.max(pred_class_prob))

		EMPTY_IMG = np.zeros(image.shape, dtype=np.uint8)

		labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG

		# prob_based_img = (pred_class_trainId_colorimg.astype(np.float32) * (0.25 + 0.75*pred_class_prob[:, :, None])).astype(np.uint8)
		
		out = self.img_grid_2x2([image, labels_colorimg, pred_labels_colorimg, EMPTY_IMG])
	
		return dict(
			demo = out,
		)

class EvaluationSemSegBitstrings(EvaluationSemSegWithUnknown):

	def construct_persistence(self):
		super().construct_persistence()

		self.chan_pred_bitstring = ChannelLoaderImage(self.persistence_base_dir+'data/{fid_no_slash}_predBitstring.png')
		self.chan_pred_bitstring.ctx = self

	def construct_tr_make_predictions(self, dset):

		self.tr_bitstring_classes = TrCategoryLogitsToBinaryClasses(self.exp.binary_category_table)
		self.tr_bitstring_classes.bitstring_int_to_class[0] = 19
		self.tr_bitstring_classes.bitstring_int_to_class[self.tr_bitstring_classes.bitstring_int_to_class == 255] = 20

		# self.exp.unk_class_info['tr_to_trainId']

		self.tr_colorimg = TrColorimg('pred_class_by_bitstring_trainId')
		self.tr_colorimg.set_override(19, [0, 0, 0])
		self.tr_colorimg.set_override(20, [117, 58, 0])

		return TrsChain(
			self.tr_bitstring_classes,
			self.tr_colorimg,
			self.tr_make_demo,
			TrChannelSave(self.chan_pred_bitstring, 'pred_category_int'),
			TrChannelSave(self.chan_pred_trainId, 'pred_class_by_bitstring_trainId'),
			TrChannelSave(self.chan_demo, 'demo'),
		)

	def tr_make_demo(self, image, pred_class_by_bitstring_trainId_colorimg, labels = None, **_):

		EMPTY_IMG = np.zeros(image.shape, dtype=np.uint8)

		labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG

		# prob_based_img = (pred_class_trainId_colorimg.astype(np.float32) * (0.25 + 0.75*pred_class_prob[:, :, None])).astype(np.uint8)
		
		out = self.img_grid_2x2([image, labels_colorimg, pred_class_by_bitstring_trainId_colorimg, EMPTY_IMG])
	
		return dict(
			demo = out,
		)

class EvaluationSemSegBitPerClass(EvaluationSemSegBitstrings):
	def construct_tr_make_predictions(self, dset):

		self.tr_bitstring_classes = self.exp.tr_bitstring_classes
		
		# self.exp.unk_class_info['tr_to_trainId']

		self.tr_colorimg = TrColorimg('pred_class_by_bitstring_trainId')
		self.tr_colorimg.set_override(19, [0, 0, 0])
		self.tr_colorimg.set_override(20, [117, 58, 0])

		return TrsChain(
			self.tr_bitstring_classes,
			self.tr_colorimg,
			self.tr_make_demo,
			TrChannelSave(self.chan_pred_trainId, 'pred_class_by_bitstring_trainId'),
			TrChannelSave(self.chan_demo, 'demo'),
		)

class EvalComparison(Evaluation):

	def __init__(self, out_dir, eval_a, eval_b):
		self.out_dir = out_dir
		self.eval_a = eval_a
		self.eval_b = eval_b
		self.construct_persistence()

	def construct_persistence(self):
		self.chan_demo = ChannelLoaderImage('{channel.ctx.out_dir}/data/{fid_no_slash}_demo.webp')
		self.chan_demo.ctx = self

	def construct_tr(self):
		self.tr_colorimg = TrColorimg('labels')
		self.tr_colorimg.set_override(19, [0, 0, 0])
		self.tr_colorimg.set_override(20, [117, 58, 0])

		return TrsChain(
			TrChannelLoad(self.eval_a.chan_pred_trainId, 'pred_labels_a'),
			TrChannelLoad(self.eval_b.chan_pred_trainId, 'pred_labels_b'),
			self.tr_colorimg,
			self.tr_make_demo,
			TrChannelSave(self.chan_demo, 'demo'),
		)

	def tr_make_demo(self, image, pred_labels_a, pred_labels_b, labels=None, **_):

		labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG
		a_colorimg = self.tr_colorimg.forward(None, pred_labels_a)
		b_colorimg = self.tr_colorimg.forward(None, pred_labels_b)
		out = self.img_grid_2x2([image, labels_colorimg, a_colorimg, b_colorimg])
	
		return dict(
			demo = out,
		)


def onehot_predictions_to_class(pred_category_bits, zero_class=19, conflict_class=20, **_):
	"""
	Cls, H, W = pred_category_bits.shape
	"""

	num_true_bits = np.count_nonzero(pred_category_bits, axis=0)
	
	selected_class = np.argmax(pred_category_bits, axis=0)

	selected_class[num_true_bits == 0] = zero_class
	selected_class[num_true_bits > 1] = conflict_class
	
	return dict(
		pred_class_by_bitstring_trainId = selected_class,
	)



class Exp0914_BitPerClass(Exp091X_BinPart_Base):
	"""
	## Sem seg but binary cross entropy instead of SoftMax

	Unknown class is all-0, but perpix weight is non-0.
	Invalid is all-0 but perpix weight is 0.

	Changes wrt bin-partition:
	* self.tr_labels_to_partitions - onehot instead of partition table
	* threshold in CUDA
	* self.tr_bitstring_classes - all-0 is unknown(19), more than one 1 is conflict (20)
	"""

	cfg =  add_experiment(Exp091X_BinPart_Base.cfg,
		name='0914_BitPerClass',
		net = dict(
			num_classes = 19, # in the NN
			loss_category_weights = False, # weight already expressed by classes
			loss_class_weights = True,
		),
		dir_checkpoint = '/cvlabdata1/home/lis/exp/0914_BitPerClass',
	)

	def init_loss(self):
		# class weights from loaded training dataset
		# no perpix weights needed because each class is in its own bit

		if self.cfg['net']['loss_category_weights']:
			class_weights = class_weights_from_class_distrib(self.unk_class_info['class_areas'])

			self.partition_positive_weights = class_weights
			log.debug(f"Bin cross entropy positive weights {class_weights}")
		else:
			log.debug(f"Bin cross entropy weights disabled")
			self.partition_positive_weights = None


		self.loss_mod = TrMultiBinaryCrossEntropyLoss(
			positive_weights = self.partition_positive_weights,
		)
		self.cuda_modules(['loss_mod'])


	def init_transforms(self):
		super().init_transforms()

		self.binary_category_table = np.eye(self.cfg['net']['num_classes'], dtype=bool)

		self.tr_pre_batch_train = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image', 'labels'), 
		)

		self.tr_onehot = TrOnehotTorch(
			num_channels = 19,
			dtype = torch.uint8,
			fields = [('labels', 'labels_binary_categories')],
		)

		self.tr_bitstring_classes = onehot_predictions_to_class
		
		self.tr_colorimg = TrColorimg('pred_class_by_bitstring_trainId')
		self.tr_colorimg.set_override(19, [0, 0, 0])
		self.tr_colorimg.set_override(20, [117, 58, 0])

		self.tr_for_log = TrsChain(
			TrNP(),
			self.tr_bitstring_classes,
			self.tr_colorimg,
		)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')

		num_classes, num_cats = self.binary_category_table.shape

		self.net_mod = ptseg_archs.PSPNet(
			num_classes = num_cats, # output for each class
			pretrained = True,
			use_aux = False,
		)

		if chk is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])
	
		self.cuda_modules(['net_mod'])

	def eval_batch_log(self, frame, fid, **_):

		if fid in self.frames_to_log:
			# frame.apply(self.tr_postprocess_log)

			fid_no_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			frame = frame.copy()
			frame.apply(self.tr_for_log)
			
			imwrite(
				self.train_out_dir / f'e{epoch:03d}_pred_{fid_no_slash}.png', 
				frame.pred_class_by_bitstring_trainId_colorimg,
			)

	def construct_default_pipeline(self, role):

		if role == 'test':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_pre_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_apply_net,
					TrKeepFieldsByPrefix('pred_category_bits', 'pred_category_probs'),
				),
				tr_output = TrsChain(
					TrNP(),
					# self.tr_bitstring_classes,
					TrColorimg('labels'),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_input,
				tr_batch_pre_merge = self.tr_pre_batch_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_onehot,
					self.tr_apply_net,
					self.loss_mod,
					TrKeepFieldsByPrefix('loss', 'pred_category_bits', 'pred_category_probs'),
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
				tr_batch_pre_merge = self.tr_pre_batch_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_onehot,
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

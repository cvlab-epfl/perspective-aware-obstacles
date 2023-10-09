
from .networks import *
from .transforms import *
from ..pipeline import *

add_experiment(name = 'autoenc_comp2_ch01',
	base=CONFIG_BASE,
	net = dict(
		type = 'autoenc_comp2_ch01',
		batch_eval = 4,
		batch_train = 2,
		num_classes = 19,
		num_intermediate_ch = 4,
	),
	train = dict(
		optimizer = dict(
			learn_rate = 0.0004,
		),
	),
)

add_experiment(name = 'autoenc_comp3_ch01',
	base= EXPERIMENT_CONFIGS['autoenc_comp2_ch01'],
	net = dict(
		batch_train = 2,
	)
)

add_experiment(name = 'autoenc_comp3_ch02',
	base= EXPERIMENT_CONFIGS['autoenc_comp2_ch01'],
	train = dict(
		optimizer = dict(
			learn_rate = 0.0004,
			lr_patience = 5, # high patience disables lr-decay
			lr_min = 1e-8,
		),
	),
)

class ExpAutoEncComp2(ExperimentBase):

	def init_transforms(self):
		super().init_transforms()

		self.tr_for_tensorboard = TrsChain([
			TrKeepFields('pred_image_reconstr'),
			TrNP(),
			tr_untorch_images,
			TrZeroCenterImgsUndo('pred_image_reconstr'),
			TrRename(('pred_image_reconstr', 'log_image')),
		])

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = self.cfg['net']

		self.net_mod = CompAutoEncoderMask01(
			num_intermediate_ch=cfg_net['num_intermediate_ch'],
			num_semantic_classes = cfg_net['num_classes'],
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def init_loss(self):
		# TODO by class name from cfg
		self.loss_mod = AutoEncoderLoss()
		self.cuda_modules(['loss_mod'])

	def init_log(self, fids_to_display):
		super().init_log(fids_to_display)
		
		# save the target images in the 'gt' log
		for fid in fids_to_display:
			fr = self.datasets['val'].get_frame_by_fid(fid)
			# TODO generalize with self.tr_for_tensorboard_gt
			self.log_gt.add_image(
				'img_{0}'.format(self.short_frame_names[fid]),
				fr.image,
				0,
			)
		
		# build the pipeline step for logging
		def log_selected_images(fid, frame, **_):
			if fid in self.fids_to_display:
				frc = frame.copy()
				frc.apply(self.tr_for_tensorboard)
				self.log.add_image(
					'img_{0}'.format(self.short_frame_names[fid]),
					frc.log_image,
					self.state['epoch_idx'],
				)
				
		self.log_selected_images = log_selected_images

	def calculate_fully_predicted_image(self, frame, passthrough_mask, passthrough_mask_d08, **_):
		pred_image_reconstr_01 = frame.apply(self.net_mod)['pred_image_reconstr']

		frame.passthrough_mask_d08 = 1 - frame.passthrough_mask_d08
		pred_image_reconstr_02 = frame.apply(self.net_mod)['pred_image_reconstr']

		pmask_4d = passthrough_mask[:, None, :, :]

		pred_image_reconstr_full = (
			pmask_4d * pred_image_reconstr_02
			+
			(1-pmask_4d) * pred_image_reconstr_01
		)

		return dict(
			pred_image_reconstr_01 = pred_image_reconstr_01,
			pred_image_reconstr_02 = pred_image_reconstr_02,
			pred_image_reconstr_full = pred_image_reconstr_full,
		)

	#def calculate_fully_predicted_image_blatent(self, frame, passthrough_mask, passthrough_mask_d08, **_):

		#pred_image_reconstr_01 = frame.apply(self.net_mod.blackout_latent)['pred_image_reconstr']

		#frame.passthrough_mask_d08 = 1 - frame.passthrough_mask_d08
		#pred_image_reconstr_02 = frame.apply(self.net_mod.blackout_latent)['pred_image_reconstr']

		#pmask_4d = passthrough_mask[:, None, :, :]

		#pred_image_reconstr_full = (
			#pmask_4d * pred_image_reconstr_02
			#+
			#(1-pmask_4d) * pred_image_reconstr_01
		#)

		#return dict(
			#pred_image_reconstr_01 = pred_image_reconstr_01,
			#pred_image_reconstr_02 = pred_image_reconstr_02,
			#pred_image_reconstr_full = pred_image_reconstr_full,
		#)

	def construct_default_pipeline(self, role, b_test_semantics_only=False):
		if role in {'train', 'val'}:
			tr_input = TrsChain([
				TransformAddChessBoard(),
			])
			if role == 'train':
				tr_input.append(TrRandomlyFlipHorizontal(['image', 'labels']))

			tr_batch_pre_merge = TrsChain([
				TrZeroCenterImgs(),
				tr_torch_images,
				TransformLabelsToOnehot(),
				chessboard_downsample_08_and_torch,
				TrKeepFields('image', 'labels_onehot', 'passthrough_mask_d08'),
			])

			pipeline = super().construct_default_pipeline(role)
			pipeline.tr_input = tr_input
			pipeline.tr_batch_pre_merge = tr_batch_pre_merge
			return pipeline

		if role.startswith('test'):
			if not b_test_semantics_only:
				tr_input = TrsChain(
					TransformAddChessBoard(),
				)
			else:
				tr_input = TrsChain(
					tr_add_blask_mask,
				)


			tr_batch_pre_merge = TrsChain(
				TrZeroCenterImgs(),
				tr_torch_images,
				TransformLabelsToOnehot(),
				chessboard_downsample_08_and_torch,
				# TrKeepFields('image', 'labels_onehot', 'passthrough_mask_d08'),
			)

			if role == 'test':
				tr_batch_pre_merge.append(
					TrKeepFields('image', 'labels_onehot', 'passthrough_mask_d08'),
				)

				tr_batch = TrsChain(
					TrCUDA(),
					self.net_mod,
					TrKeepFields('pred_image_reconstr'),
					TrNP(),
				)
				tr_output = TrsChain(
					tr_untorch_images,
					TrZeroCenterImgsUndo('pred_image_reconstr'),
				)


			elif role == 'test_fullreconstr':
				tr_batch_pre_merge.append(
					TrKeepFields('image', 'labels_onehot', 'passthrough_mask', 'passthrough_mask_d08'),
				)
				tr_batch = TrsChain(
					TrCUDA(),
					self.calculate_fully_predicted_image,
					TrKeepFields('pred_image_reconstr_01', 'pred_image_reconstr_02', 'pred_image_reconstr_full'),
					TrNP(),
				)
				tr_output = TrsChain(
					tr_untorch_images,
					TrZeroCenterImgsUndo(['pred_image_reconstr_01', 'pred_image_reconstr_02', 'pred_image_reconstr_full']),
				)

			return Pipeline(
				tr_input = tr_input,
				tr_batch_pre_merge = tr_batch_pre_merge,
				tr_batch = tr_batch,
				tr_output = tr_output
			)

		elif role == 'test_noiselatent':
			return Pipeline(
				tr_input = TrsChain(
					TransformAddChessBoard(),
				),
				tr_batch_pre_merge = TrsChain(
					TrZeroCenterImgs(),
					tr_torch_images,
					TransformLabelsToOnehot(),
					chessboard_downsample_08_and_torch,
					TrKeepFields('image', 'labels_onehot', 'passthrough_mask', 'passthrough_mask_d08'),
				),
				tr_batch = TrsChain(
					TrCUDA(),
					self.net_mod.try_noise,
					TrKeepFields('pred_image_reconstr'),
					TrNP(),
				),
				tr_output = TrsChain(
					tr_untorch_images,
					TrZeroCenterImgsUndo('pred_image_reconstr'),
				),
			)

#add_experiment(name = 'autoenc_unet',
	#base=CONFIG_PSP,
 	#dataset = dict(
		#load_labels = False,
	#),

	#net = dict(
		#type = 'autoenc_unet',
		#loss = 'autoenc',
		#classfunc = 'autoenc',
		#batch_train = 5,
		#batch_eval = 10,
		#channels = [3, 16, 24, 48, 96, 192, 384],
	#),

	#train = dict(
		#optimizer = dict(
			#learn_rate = 0.001,
			#learn_rate = 0.25*1e-4,
		#),
	#),
#)

#add_experiment(name = 'autoenc_unet_squeeze_1',
	#base=EXPERIMENT_CONFIGS['autoenc_unet'],
 	#dataset = dict(
		#load_labels = False,
	#),

	#net = dict(
		#type = 'autoenc_unet',
		#loss = 'autoenc',
		#classfunc = 'autoenc',
		#batch_train = 5,
		#batch_eval = 10,
		#channels = [3, 24, 32, 64, 96, 192, 96],
	#),

	#train = dict(
		#optimizer = dict(
			#learn_rate = 0.001,
			#learn_rate = 0.25*1e-4,
		#),
	#),
#)

#add_experiment(name = 'autoenc_comp_noseg_4',
	#base=EXPERIMENT_CONFIGS['autoenc_unet'],
 	#dataset = dict(
		#load_labels = False,
	#),

	#net = dict(
		#type = 'autoenc_comp_noseg',
		#loss = 'autoenc',
		#classfunc = 'autoenc',
		#batch_train = 2,
		#batch_eval = 4,
		#num_intermediate_ch = 4,
	#),

	#train = dict(
		#optimizer = dict(
			#learn_rate = 0.0002,
			#learn_rate = 0.25*1e-4,
		#),
	#),
#)

#add_experiment(name = 'autoenc_comp_l2only',
	#base=EXPERIMENT_CONFIGS['autoenc_comp_noseg_4'],
	#net = dict(
		#loss = 'autoenc_l2only',
	#),
#)

#add_experiment(name = 'autoenc_comp2_noseg_4',
	#base=EXPERIMENT_CONFIGS['autoenc_comp_noseg_4'],
	#net = dict(
		#type = 'autoenc_comp2_noseg',
	#),
	#train = dict(
		#optimizer = dict(
			#learn_rate = 0.0004,
		#),
	#),
#)

add_experiment(name = 'autoenc_blur_photo01',
	base= EXPERIMENT_CONFIGS['autoenc_comp2_ch01'],
	net = dict(
		blur_ksize = 61,
		blur_std = 21.5,
		batch_eval = 2,
		batch_train = 1,
	),
	train = dict(
		optimizer = dict(
			learn_rate = 0.0004,
			lr_patience = 5, # high patience disables lr-decay
			lr_min = 1e-8,
		),
	),
)

class ExpAutoEncBlurPhotographic(ExpAutoEncComp2):

	def init_transforms(self):
		super().init_transforms()

		cfg_net = self.cfg['net']
		self.blur = GaussianBlur(cfg_net['blur_ksize'], cfg_net['blur_std'])
		self.cuda_modules(['blur'])

		self.tr_blur = TrByField([('image', 'image_blurred')], operation=self.blur)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		cfg_net = self.cfg['net']

		self.net_mod = PhotographicGenerator(
			num_ch_input = 3+cfg_net['num_classes'],
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	#def init_loss(self):
		#self.loss_mod = AutoEncoderLoss()
		#self.cuda_modules(['loss_mod'])

	def init_log(self, fids_to_display):
		super().init_log(fids_to_display)

		# save the target images in the 'gt' log
		for fid in fids_to_display:
			fr = self.datasets['val'].get_frame_by_fid(fid)
			# TODO generalize with self.tr_for_tensorboard_gt
			self.log_gt.add_image(
				'img_{0}'.format(self.short_frame_names[fid]),
				fr.image,
				0,
			)

		# build the pipeline step for logging
		def log_selected_images(fid, frame, **_):
			if fid in self.fids_to_display:
				frc = frame.copy()
				frc.apply(self.tr_for_tensorboard)
				self.log.add_image(
					'img_{0}'.format(self.short_frame_names[fid]),
					frc.log_image,
					self.state['epoch_idx'],
				)

	def construct_default_pipeline(self, role):
		if role in {'train', 'val'}:
			tr_input = TrsChain()

			if role == 'train':
				tr_input.append(TrRandomlyFlipHorizontal(['image', 'labels']))

			tr_batch_pre_merge = TrsChain(
				TrZeroCenterImgs(),
				tr_torch_images,
				TransformLabelsToOnehot(),
				TrKeepFields('image', 'labels_onehot'),
			)

			pipeline = super().construct_default_pipeline(role)
			pipeline.tr_input = tr_input
			pipeline.tr_batch_pre_merge = tr_batch_pre_merge
			pipeline.tr_batch.insert(1, self.tr_blur), # blur before reconstruction
			#pipeline.tr_batch.insert(1, lambda image, **_: dict(image_blurred=image)), # blur before reconstruction
			pipeline.tr_batch.insert(2, TrPytorchNoGrad(['image', 'image_blurred', 'labels_onehot']))
			#pipeline.tr_batch.insert(0, tr_print)

			return pipeline

		if role == 'test':
			tr_input = TrsChain()

			tr_batch_pre_merge = TrsChain(
				TrZeroCenterImgs(),
				tr_torch_images,
				TransformLabelsToOnehot(),
				TrKeepFields('image', 'labels_onehot'),
			)

			tr_batch = TrsChain(
				TrCUDA(),
				self.tr_blur,
				self.net_mod,
				TrKeepFields('pred_image_reconstr'),
				TrNP(),
			)

			tr_output = TrsChain(
				tr_untorch_images,
				TrZeroCenterImgsUndo('pred_image_reconstr'),
			)

			return Pipeline(
				tr_input = tr_input,
				tr_batch_pre_merge = tr_batch_pre_merge,
				tr_batch = tr_batch,
				tr_output = tr_output
			)

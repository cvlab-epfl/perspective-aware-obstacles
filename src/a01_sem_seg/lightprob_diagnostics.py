
from .networks import *
from .experiments import *

class ExpSemSegProboutDiagnostic(ExpSemSegProbout):

	cfg = add_experiment(ExpSemSegProbout.cfg,
		name='0504_ProboutPSP_diagnostic',
	)

	class LossMonitor(TrBase):
		def __init__(self, interval_batches = 10, tensorboard=None):
			self.interval = interval_batches
			self.tensorboard = tensorboard

			self.batch_counter = 0

		def print_diagnostics(self, name, value):

			value = value.detach()

			if value.shape.__len__() > 1:

				range = [torch.min(value), torch.max(value)]
				mean_abs = torch.mean(torch.abs(value))

				print('	{name}	{min:.3g} {avg:.3g} {max:.3g}'.format(
					name=name, min=float(range[0]), avg=float(mean_abs), max=float(range[1]),
				))

			else:
				print('	{name}	{val:.3g}'.format(
					name=name, val=float(value),
				))

		def __call__(self, **fields):

			names = ['pred_prob_pre_softmax', 'pred_prob_var', 'loss', 'loss_xe', 'loss_alpha', 'loss_s']

			if self.batch_counter % self.interval == 0:

				print('-- B {n:04d} -- '.format(n = self.batch_counter))
				for name in names:
					value = fields.get(name, None)
					if value is not None:
						self.print_diagnostics(name, value)

			self.batch_counter += 1

	def init_transforms(self):
		super().init_transforms()

		self.tr_monitor = self.LossMonitor()

	def eval_batch_log(self, frame, fid, pred_prob, **_):
		if fid in self.frames_to_log:
			frame.apply(self.tr_postprocess_log)
			self.log.add_image(
				'{0}_class'.format(fid),
				frame.pred_labels_colorimg.transpose((2, 0, 1)),
				self.state['epoch_idx'],
			)
			self.log.add_image(
				'{0}_var'.format(fid),
				frame.pred_uncertainty_sum,
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
					self.class_softmax,
					TrKeepFields('pred_prob', 'pred_prob_var'),
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
					self.tr_monitor,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

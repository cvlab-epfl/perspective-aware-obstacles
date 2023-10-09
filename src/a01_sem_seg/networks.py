
import torch
import numpy as np
from torch import nn
from torch.nn import functional
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain
from ..pytorch_semantic_segmentation import utils as ptseg_utils
from ..pytorch_semantic_segmentation import models as ptseg_models
from ..common.util_networks import Padder

class LossCrossEntropy2d(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_sfm = nn.LogSoftmax(dim=1) # along channels
		self.nll_loss = nn.NLLLoss(
			weight,
			reduction='mean' if size_average else 'none',
			ignore_index=ignore_index,
		)

	def forward(self, pred_logits, labels, **other):
		return dict(loss = self.nll_loss(
			self.log_sfm(pred_logits),
			labels,
		))


class ClassifierSoftmax(nn.Module):
	def __init__(self):
		super().__init__()
		self.softmax = torch.nn.Softmax2d()

	#def __call__(self, pf, net_output):
		#pred_softmax = self.softmax(net_output[None, :, :, :])
		#pf.pred_prob = pred_softmax[0, :, :, :].data.cpu().numpy()
		#classify(pf)

	def forward(self, pred_logits, **_):
		if pred_logits.shape.__len__() == 4:
			pred_softmax = self.softmax(pred_logits)
		else:
			# single sample, with no batch dimension
			pred_softmax = self.softmax(pred_logits[None])[0]

		return dict(
			pred_prob = pred_softmax
		)

class PerspectiveSceneParsingNet(ptseg_models.PSPNet):
	def forward(self, image, **_):
		pred_raw = super().forward(image)
		return dict(
			pred_logits = pred_raw,
		)

class LossPSP(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_sfm = nn.LogSoftmax(dim=1) # along channels
		self.nll_loss = nn.NLLLoss(weight, reduction='mean' if size_average else 'none', ignore_index=ignore_index)
		self.cel = LossCrossEntropy2d(weight, size_average, ignore_index)

	def forward(self, pred_logits, labels, **other):
		if isinstance(pred_logits, tuple):
			pred_raw_main, pred_raw_aux = pred_logits

			loss_main = self.nll_loss(self.log_sfm(pred_raw_main), labels)
			loss_aux = self.nll_loss(self.log_sfm(pred_raw_aux), labels)

			return dict(
				loss = loss_main * (1.0/1.4) + loss_aux * (0.4/1.4),
				loss_main = loss_main,
				loss_aux = loss_aux,
			)
		else:
			return self.cel(pred_logits, labels, **other)


class BayesianSegNet(ptseg_models.SegNetBayes):

	def forward(self, img):

		# the network fails if the image size is not divisible by 32
		# pad to 32  -> run net -> remove padding

		padder = Padder(img.shape, 32)
		img = padder.pad(img)

		result = super().forward(img)

		return padder.unpad(result)

		# sz = np.array(img.shape[2:])
		# sz_deficit = sz - 32 * (sz // 32)

		# if np.any(sz_deficit != 0):
		# 	img_padded = functional.pad(img, (0, sz_deficit[1], 0, sz_deficit[0]), 'reflect')
		# 	#print('deficit', sz_deficit, 'padded', img_padded.shape)

		# 	result = super().forward(img_padded)
		# 	result = result[:, :, :sz[0], :sz[1]]
		# 	return result

		# else:
		# 	return super().forward(img)

	# def __call__(self, image, **_):
	# 	result= self.forward_multisample(image)
	#
	# 	return dict(
	# 		pred_logits = result['mean'],
	# 		pred_uncertainty = result['var'],
	# 	)

from .lightprobnets.losses.probabilistic_classification_losses import DirichletProbOutLoss

class ProbabilisticSemSegLoss(DirichletProbOutLoss):
	def __init__(self, label_smoothing=0.01, random_off_targets=False, roi=None):
		super().__init__(Frame(model='?'), label_smoothing=label_smoothing, random_off_targets=random_off_targets, mult=False)
		self.roi = roi

	def cuda(self, **kwargs):
		self._smoothed_onehot.cuda()
		
		return super().cuda(**kwargs)

	def forward(self, pred_logits, pred_var, labels, **_):
		batch, num_cl, h, w = pred_logits.shape

		# print(pred_logits.shape, pred_prob_var.shape, labels.shape)
		# test_probs.reshape([batch, num_cl, -1]).transpose(0, 1).shape

		mean_spread = pred_logits.transpose(0, 1).reshape([num_cl, -1]).transpose(0, 1)
		var_spread = pred_var.transpose(0, 1).reshape([num_cl, -1]).transpose(0, 1)
		labels_spread = labels.reshape(-1)

		mask = (labels_spread != 255)

		# print('mask', mask.shape)
		# print(mean_spread.shape, var_spread.shape, labels_spread.shape)

		mean_spread = mean_spread[mask,:]
		var_spread = var_spread[mask, :]
		labels_spread = labels_spread[mask]

		res = super().forward(
			dict(
				mean1=mean_spread,
				variance1=var_spread,
			),
			dict(
				target1=labels_spread,
			),
		)

		out = dict(
			loss=res['total_loss'],
			loss_xe=res['xe'],
		)

		alpha = res.get('alpha', None)
		if alpha is not None:
			out['loss_alpha'] = alpha

		s = res.get('s', None)
		if alpha is not None:
			out['loss_s'] = s

		return out


class PerspectiveSceneParsingNet_ProbOut(ptseg_models.PSPNet):
	def __init__(self, num_classes, pretrained=True, small_variance_weights=True):
		# 2 times more output channels to make variance
		super().__init__(num_classes*2, pretrained=pretrained, use_aux=False)

		if small_variance_weights:
			self.adjust_initial_variance_weights()

		self.prob_loss = ProbabilisticSemSegLoss()

	def cuda(self, **kwargs):
		self.prob_loss._smoothed_onehot.cuda()

		return super().cuda(**kwargs)

	def adjust_initial_variance_weights(self):
		final_conv = self.final[4]
		num_outputs = final_conv.weight.shape[0]
		final_conv.weight.data[num_outputs // 2:] *= 1e-3
		final_conv.bias.data *= 0

	def forward(self, image, **_):
		pred_raw = super().forward(image)

		#print('prr', float(torch.min(image)), float(torch.max(image)), float(torch.min(pred_raw)), float(torch.max(pred_raw)))
		#print(pred_raw[0, 0, 200, 200])

		mean, variance = pred_raw.chunk(chunks=2, dim=1)

		#print('varb', float(torch.min(variance)), float(torch.max(variance)))

		variance = functional.softplus(variance)

		#print('vara', float(torch.min(variance)), float(torch.max(variance)))

		return dict(
			pred_logits = mean,
			pred_var = variance,
			loss_var_avg=torch.mean(variance),
			loss_var_max=torch.max(variance),
		)


def print_mam(name, value):
	print('{0}:	{1:.3f}	{2:.3f}	{3:.3f}'.format(name, float(torch.min(value)), float(torch.mean(value)), float(torch.max(value))))

class EpistemicSoftmax(nn.Module):
	def __init__(self, num_samples=8, ignore_index = 255):
		super().__init__()

		self.num_samples = num_samples
		self.distrib_normal = torch.distributions.normal.Normal(0, 1.)
		self.softmax = torch.nn.Softmax2d()
		self.neg_log_likelihood = torch.nn.NLLLoss(ignore_index = ignore_index, reduction='mean')

		if torch.cuda.is_available():
			self.cuda()

	def cuda(self, **kwargs):
		self.distrib_normal.loc = self.distrib_normal.loc.cuda()
		self.distrib_normal.scale = self.distrib_normal.scale.cuda()
		return super().cuda(**kwargs)

	def forward_mean(self, pred_logits, pred_logits_var, **_):

		#print('-')
		#print_mam('	logits', pred_logits)
		#print_mam('	var lgit', pred_logits_var)

		avg_softmax = torch.zeros_like(pred_logits)
		for sample_idx in range(self.num_samples):
			avg_softmax += self.softmax(
				pred_logits
				+
				self.distrib_normal.sample(pred_logits.shape) * pred_logits_var
			).clamp(min=1e-14)
		avg_softmax *= (1 / self.num_samples)

		return dict(
			pred_prob=avg_softmax,
		)

	def forward_mean_and_var(self, pred_logits, pred_logits_var, **_):
		softmax_samples = torch.stack([
			self.softmax(
				pred_logits
				+
				self.distrib_normal.sample(pred_logits.shape) * pred_logits_var
			).clamp(min=1e-14)
			for sample_idx in range(self.num_samples)
		])
		softmax_avg = torch.mean(softmax_samples, 0)
		softmax_var = torch.sum(torch.var(softmax_samples, 0), 1)

		return dict(
			pred_prob=softmax_avg,
			pred_prob_var=softmax_var,
		)

	def loss(self, pred_prob, labels, **_):
		# prob_log = torch.log(pred_prob).clamp(min=-50) # -inf could happen here!

		prob_log = torch.log(pred_prob) #.clamp(min=-50) # -inf could happen here!

		#print_mam('	pred_prob', pred_prob)
		#print_mam('	log(pred)', prob_log)

		return dict(
			# NLLLoss takes log-softmax!
			loss = self.neg_log_likelihood(prob_log, labels),
		)

	def forward(self, pred_logits, pred_logits_var, **_):
		"""
		:param pred_logits: logits [B x num_class x W x H]
		:param pred_logits_var: var [B x 1 x W x H]
		:return: mean ( softmax ( logit tables ) ), variance ( softmax ( logit samples ) )
		"""
		if self.training:
			return self.forward_loss(pred_logits, pred_logits_var)
		else:
			return self.forward_mean_and_var(pred_logits, pred_logits_var)


class PerspectiveSceneParsingNet_Epistemic(ptseg_models.PSPNet):
	def __init__(self, num_classes, pretrained=True, small_variance_weights=True):
		# 2 times more output channels to make variance
		super().__init__(num_classes+1, pretrained=pretrained, use_aux=False)

		self.num_classes = num_classes
		self.epistemic_softmax = EpistemicSoftmax()

	def forward(self, image, **_):
		pred_raw = super().forward(image)

		pred_logits = pred_raw[:, :self.num_classes]
		pred_var = torch.abs(pred_raw[:, self.num_classes:]) # last dimension

		return dict(
			pred_logits = pred_logits,
			pred_logits_var = pred_var,
			loss_var_avg = torch.mean(pred_var),
			loss_var_max = torch.max(pred_var),
		)

# from dense import FCDenseNet103

# class DenseNet_Epistemic(FCDenseNet103):
# 	def __init__(self, num_classes, dropout, num_samples=16):
# 		super().__init__(out_channels=num_classes+1, dropout=dropout)
# 		self.num_classes = num_classes
# 		self.epistemic_softmax = EpistemicSoftmax()
# 		self.num_samples = num_samples

# 	def forward(self, image, **_):
# 		pred_raw = super().forward(image)

# 		pred_logits = pred_raw[:, :self.num_classes]
# 		pred_var = torch.abs(pred_raw[:, self.num_classes:]) # last dimension

# 		return dict(
# 			pred_logits = pred_logits,
# 			pred_logits_var = pred_var,
# 			loss_var_avg=torch.mean(pred_var),
# 			loss_var_max=torch.max(pred_var),
# 		)

# 	def forward_multisample(self, image, num_samples=None, **_):
# 		# dropout must be active
# 		backup_train_mode = self.training
# 		self.train()

# 		softmax = torch.nn.Softmax2d()
# 		num_samples = num_samples if num_samples else self.num_samples

# 		preds_probs = []
# 		preds_alea = []
# 		#preds_alea_raw = []

# 		self.epistemic_softmax.num_samples = self.num_samples

# 		tr_fwd = TrsChain(
# 			self.forward,
# 			self.epistemic_softmax.forward_mean_and_var,
# 		)

# 		for idx in range(num_samples):
# 			fr = Frame(image=image)
# 			fr.apply(tr_fwd)
# 			preds_probs.append(softmax(fr['pred_logits']).cpu()) # NP?
# 			preds_alea.append(fr['pred_prob_var'].cpu())
# 			#preds_alea_raw.append(fr['pred_logits_var'].cpu())
# 			del fr

# 		preds_alea = torch.mean(torch.stack(preds_alea).cuda(), 0)
# 		#preds_alea_raw = torch.mean(torch.stack(preds_alea_raw), 0)[:, 0]

# 		preds_probs = torch.stack(preds_probs).cuda()
# 		preds_avg = torch.mean(preds_probs, 0)
# 		preds_var = torch.sum(torch.var(preds_probs, 0), 1)

# 		# restore mode
# 		self.train(backup_train_mode)

# 		return dict(
# 			pred_prob = preds_avg,
# 			pred_var_alea = preds_alea,
# 			pred_var_dropout = preds_var,
# 			#pred_var_alea_raw = preds_alea_raw,
# 		)


import logging
log = logging.getLogger('exp')
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..pipeline.config import add_experiment
from .experiments import ExpSemSegPSP


CFG_PSP_MODERN = add_experiment(ExpSemSegPSP.cfg,
	name = 'PSP_MODERN',
	net = dict(
		batch_train = 5,
		batch_eval = 6, # full imgs not crops
		backbone_pretrained = True,
		apex_mode = 'O1',
		use_aux = False,
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
)

class Exp0130(ExpSemSegPSP):
	"""
	Fp32 baseline, but with FusedAdam optimizer
	"""

	cfg = add_experiment(CFG_PSP_MODERN,
		name = '0130_PSP_FP32_Baseline',
	)

	# FusedAdam is now in experiment.py
	# def build_optimizer(self, role, chk_optimizer=None):
	# 	log.info('Building optimizer')

	# 	cfg_opt = self.cfg['train']['optimizer']

	# 	network = self.net_mod
	# 	self.optimizer = FusedAdam(
	# 		[p for p in network.parameters() if p.requires_grad],
	# 		lr=cfg_opt['learn_rate'],
	# 		weight_decay=cfg_opt.get('weight_decay', 0),
	# 	)
	# 	self.learn_rate_scheduler = ReduceLROnPlateau(
	# 		self.optimizer,
	# 		patience=cfg_opt['lr_patience'],
	# 		min_lr = cfg_opt['lr_min'],
	# 	)

	# 	if chk_optimizer is not None:
	# 		self.optimizer.load_state_dict(chk_optimizer['optimizer'])

class ExpSemSegPSP_Apex(ExpSemSegPSP):
	cfg = CFG_PSP_MODERN
	
	def init_net(self, role):
		""" Role: val or train - determines which checkpoint is loaded"""
		
		super().init_net(role)

		apex_mode = self.cfg['net'].get('apex_mode', None)
		if apex_mode:
			import apex.amp
			log.info(f'Initializing APEX.AMP mode {apex_mode}')
			if role == 'train':
				self.net_mod, self.optimizer = apex.amp.initialize(self.net_mod, self.optimizer, opt_level=apex_mode)
			elif role == 'eval':
				self.net_mod = apex.amp.initialize(self.net_mod, opt_level=apex_mode)
			else:
				raise NotImplementedError(f'role={role}')

	def training_backpropagate(self, loss, **_):
		# log.debug(f'Backpropagate: loss is {loss.dtype} {loss}')
		loss = loss.half()
		with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
			scaled_loss.backward()
		self.optimizer.step()

	def run_evaluation(self, eval_obj, dset=None, b_one_batch=False):
		tr_test = self.construct_default_pipeline('test')
		dset = dset or eval_obj.get_dset()
		tr_test.tr_output.append(eval_obj.construct_tr_make_predictions(dset))

		log.info(f'Test pipeline: {tr_test}')
		tr_test.execute(dset, b_accumulate=False, b_one_batch=b_one_batch)

class Exp0131(ExpSemSegPSP_Apex):
	cfg = add_experiment(CFG_PSP_MODERN,
		name = '0131_PSP_HalfPrecision_ApexO1',
		net = dict(
			apex_mode = 'O1',
		)
	)

class Exp0132(ExpSemSegPSP_Apex):
	cfg = add_experiment(CFG_PSP_MODERN,
		name = '0132_PSP_HalfPrecision_ApexO2',
		net = dict(
			apex_mode = 'O2',
		)
	)


# class Exp0133(ExpSemSegPSP):
# 	cfg = add_experiment(CFG_PSP_MODERN,
# 		name = '0133_PSP_HalfPrecision_Explicit',
# 	)

# 	def init_transforms(self):
# 		super().init_transforms()
# 		tr_img_to_half = TrAsType({'image': torch.HalfTensor})
# 		self.tr_prepare_batch_train.append(tr_img_to_half)
# 		self.tr_prepare_batch_test.append(tr_img_to_half)

# 	def build_net(self, role, chk=None, chk_optimizer=None):
# 		""" Build net and optimizer (if we train) """
# 		super().build_net(self, role, chk=chk, chk_optimizer=chk_optimizer)
# 		self.net_mod.half() # half precision!


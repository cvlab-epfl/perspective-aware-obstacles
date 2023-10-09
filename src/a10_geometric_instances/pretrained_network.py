import torch
import numpy as np
from pathlib import Path
import json
from ..paths import DIR_EXP

SIGMA_FACTOR_ADD = float(0.5 * np.log(np.log(2)))

class SpatialEmbeddings_network_wrapper:
	
	def get_pretrained_network(self, weight_file):
		from .SpatialEmbeddings.src.models import get_model

		weight_file = Path(weight_file)
		run_dir = weight_file.parent
		cfg = json.loads((run_dir / 'config.json').read_text())

		# net = get_model('branched_erfnet', {'num_classes': [3, 1]})
		net = get_model(cfg['model']['name'], cfg['model']['kwargs'])
		
		checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))

		self.n_sigma_d = cfg['loss_opts']['n_sigma']

		# wrap to preface all weights with module.
		if False:
			wrap_mod = torch.nn.Module()
			wrap_mod.module = net
			wrap_mod.load_state_dict(checkpoint['model_state_dict'])
		else:
			net.load_state_dict(checkpoint['model_state_dict'])

		if torch.cuda.is_available():
			net.cuda()
		net.eval()
		return net

	def run_net(self, image_tr, net_module, **_):
		with torch.no_grad():
			is_batched = image_tr.shape.__len__() >= 4

			# add batch dimension if not present
			if not is_batched:
				image_tr = image_tr[None]

			# run network
			net_out = net_module(image_tr.cuda())

			spatial_offset = torch.tanh(net_out[:, 0:2])
			spatial_sigma = net_out[:, 2:2+self.n_sigma_d]
			# print('pred sigms shape', spatial_sigma.shape, self.n_sigma_d)
			spatial_vote_radius = torch.exp(-5 * spatial_sigma + SIGMA_FACTOR_ADD)
			# print('pred sigms shape radius', spatial_vote_radius.shape, self.n_sigma_d)

			# separate channels
			result = dict(
				# apply tanh to offset, see my_loss.py in SpatialEmbeddings
				centerpoint_offset = spatial_offset,
				# spatial_sigma = net_out[:, 2],
				centerpoint_vote_radius = spatial_vote_radius,
				centerpoint_seed = net_out[:, -1],
			) 

			# if input was not batched, drop the batch dimension
			if not is_batched:
				result = {k: v[0] for k, v in result.items()}
		
			return result

	def load_weights(self, weight_file):
		self.net_mod = self.get_pretrained_network(weight_file)
	
	def __call__(self, image_tr, **_):
		return self.run_net(image_tr=image_tr, net_module=self.net_mod)

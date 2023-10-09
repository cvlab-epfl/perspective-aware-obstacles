import torch
import numpy as np
from pathlib import Path
import json
from ..paths import DIR_EXP2

SIGMA_FACTOR_ADD = float(0.5 * np.log(np.log(2)))

class SpatialEmbeddingsNetwork:

	def __init__(self, checkpoint_dir=None):
		self.checkpoint_dir = Path(checkpoint_dir)

	def load_from_checkpoint(self, checkpoint_dir=None):
		# TODO: separate 
		from .SpatialEmbeddings.src.models import get_model

		self.checkpoint_dir = Path(checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir) 
		path_cfg = self.checkpoint_dir / 'config.json'
		path_checkpoint = self.checkpoint_dir / 'best_iou_model.pth'

		cfg = json.loads(path_cfg.read_text())

		self.n_sigma_d = cfg['loss_opts']['n_sigma']
		self.n_seed_d = cfg['loss_opts'].get('n_seed', 1)

		checkpoint_data = torch.load(path_checkpoint, map_location=torch.device('cpu'))

		net = get_model(cfg['model']['name'], cfg['model']['kwargs'])
		net.load_state_dict(checkpoint_data['model_state_dict'])

		# wrap to preface all weights with module: 
		# needed for the pretrained weights, which use DataParllel
			# wrap_mod = torch.nn.Module()
			# wrap_mod.module = net
			# wrap_mod.load_state_dict(checkpoint['model_state_dict'])

		if torch.cuda.is_available():
			net.cuda()

		net.eval()

		self.net = net

		return net

	@staticmethod
	def spatial_embeddings_parse_net_output(net_out, n_sigma_dims, n_seed_dims):
		spatial_offset = torch.tanh(net_out[:, 0:2])
		spatial_sigma = net_out[:, 2:2+n_sigma_dims]
		spatial_seed = net_out[:, 2+n_sigma_dims:2+n_sigma_dims+n_seed_dims]
		spatial_vote_radius = torch.exp(-5 * spatial_sigma + SIGMA_FACTOR_ADD)

		# separate channels
		return dict(
			# apply tanh to offset, see my_loss.py in SpatialEmbeddings
			centerpoint_offset = spatial_offset,
			# spatial_sigma = net_out[:, 2],
			vote_radius = spatial_vote_radius,
			seed = spatial_seed,
		) 

	def predict(self, image, **_):
		with torch.no_grad():
			is_batched = image.shape.__len__() >= 4

			# add batch dimension if not present
			if not is_batched:
				image = image[None]

			# run network
			net_out = self.net(image.cuda())
			result = self.spatial_embeddings_parse_net_output(
				net_out, 
				n_sigma_dims=self.n_sigma_d,
				n_seed_dims=self.n_seed_d,
			)
			
			# if input was not batched, drop the batch dimension
			if not is_batched:
				result = {k: v[0] for k, v in result.items()}
		
			return result

	def __call__(self, image, **_):
		return self.predict(image=image)


class SpatialEmbeddingsInference:
	# may be a special case of a more general inference class

	def __init__(self, out_dir, network=None, storage={}):

		self.out_dir = out_dir

		# net submodule
		# the submodule should be overridable
		self.network = network or SpatialEmbeddingsNetwork()

		# storage submodules

		# right now the dset is stored in frame, but not in channel
		# are channels ever resused between datasets?
			# rather no, so we should put necessary info (root_dir) into channel instances
			# this way they can survive serialization to other processes

		# the storage should be overrideable by the module constructor?
		# out dir should be overridable config

		self.storage = dict(
			centerpoint_offset = ChannelLoaderHDF5_NotShared(out_dir / '{fid_no_slash}_spatial_offset.hdf5', 'centerpoint_offset'),
			centerpoint_seed = ChannelLoaderHDF5_NotShared(out_dir / '{fid_no_slash}_spatial_seed.hdf5', 'centerpoint_seed'),
			centerpoint_vote_radius = ChannelLoaderHDF5_NotShared(out_dir / '{fid_no_slash}_spatial_vote_radius.hdf5', 'centerpoint_vote_radius'),
			# demo_netout = ChannelLoaderImage(out_dir / '{fid_no_slash}_vis.webp'),
			# demo_naive = ChannelLoaderImage(out_dir / 'demo_cluster_naive' / '{fid_no_slash}_cluster_native.webp'),
			# demo_cluster_blur = ChannelLoaderImage(out_dir / 'demo_cluster_blur' / '{fid_no_slash}_cluster_blur.webp'),
			# demo_templ = ChannelLoaderImage(out_dir / 'demo_templ' / '{fid_no_slash}_templ.webp'),
		)
		# TODO merge with override

		for c in self.storage.values(): c.ctx = self

		# dset submodules?
			# its tempting to put the preprocess (like crop) in the Dataset object itself
				# DatasetX(..., preprocess = tr(...))
				# but shoudln't the dataset be about the storage only?
			# Dataset describes the space of inputs. If they are significantly altered in preprocessing, we should call it a different Dataset (as evaluation result will be different too)
				# if a preprocessing procedure is in the method itself, we will not put it on Dataset
				# but data-related choices (Which ROI) should be part of the dataset config
			# If we really need a purer data class, it could be FileSet, explicitly about the storage
				# Is the above storage list a FileSet?
					# no list of files, as these are determined by the dset
				# or a StorageGroup module?



		# postprocess / signal proc submodules
			# we will usually run them from saved files... 
				# thats because of gpu vs cpu machines
			# a joint-pipeline should always be available
			# the switch between files caching the net output, and a joint pipeline running the net should be smooth




	# steps
		# load image
		# torch and cuda
		# batch (optional)
		# postprocessing: visualization, anything else?
		# show or save

	# convenient running 
		# run for a group of frames, show results
		# run for a group of frames (dset), save results
		
		# autobatching, but modular/optional: 

























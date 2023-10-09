
import torch
import numpy as np
from pathlib import Path
from functools import partial
from easydict import EasyDict
from tqdm import tqdm

from ..pipeline.config import add_experiment
from ..pipeline.transforms import TrsChain, tr_print
from ..pipeline.transforms_imgproc import TrShow, TrImgGrid, image_grid
from ..pipeline.transforms_pytorch import tr_torch_images, TrNP
from ..datasets.dataset import ChannelLoaderImage, ChannelLoaderHDF5_NotShared
from ..pipeline.evaluations import TrChannelLoad, TrChannelSave
from ..pipeline.pipeline import Pipeline
from ..pipeline.frame import Frame
from ..pipesys.bind import bind
from ..paths import DIR_EXP

from .pretrained_network import SpatialEmbeddings_network_wrapper
from .cluster_geometry import tr_vis_centerpoint_offset, tr_centerpoint_calc_centers, tr_centerpoint_resolve_sigma, tr_plot_votes, tr_find_clusters__naive, draw_cluster_map, tr_discontinuity_blur

class Exp1001_InstanceSegmentationSpatialEmbeddings:
	cfg = add_experiment({},
		#TODO don't repeat the names
		name = '1001_InstanceSegmentationSpatialEmbeddings',
	)

	def __init__(self):
		self.workdir = DIR_EXP / self.cfg['name']

		self.construct_persistence()

	def construct_persistence(self):
		out_dir = Path('{channel.ctx.workdir}/{dset.name}_{dset.split}')

		out_hdf_file = out_dir / 'spatial_pretrained/{fid_no_slash}_spatial.hdf5'

		self.storage = dict(
			centerpoint_offset = ChannelLoaderHDF5_NotShared(out_dir / 'spatial_pretrained/{fid_no_slash}_spatial_offset.hdf5', 'spatial_offset'),
			centerpoint_seed = ChannelLoaderHDF5_NotShared(out_dir / 'spatial_pretrained/{fid_no_slash}_spatial_seed.hdf5', 'spatial_seed'),
			centerpoint_sigma = ChannelLoaderHDF5_NotShared(out_dir / 'spatial_pretrained/{fid_no_slash}_spatial_sigma.hdf5', 'spatial_sigma'),
			demo_naive = ChannelLoaderImage(out_dir / 'demo_cluster_naive' / '{fid_no_slash}_cluster_native.webp'),
			demo_cluster_blur = ChannelLoaderImage(out_dir / 'demo_cluster_blur' / '{fid_no_slash}_cluster_blur.webp'),
			demo_templ = ChannelLoaderImage(out_dir / 'demo_templ' / '{fid_no_slash}_templ.webp'),
		)
		for c in self.storage.values(): c.ctx = self

	def init_default_datasets(self):

		from ..datasets.lost_and_found import DatasetLostAndFound
		from ..datasets.CAOS_StreetHazards import DatasetCaosStreetHazards

		self.dsets = EasyDict(
			laf = DatasetLostAndFound(),
			synth = DatasetCaosStreetHazards(),
		)

		for dset in self.dsets.values():
			dset.discover()

		self.dsets.synth.frames = self.dsets.synth.frames[:150]

	def run_pretrained_net__init(self):
		weight_file_location = self.workdir / 'weights' / 'cars_pretrained_model.pth'

		self.net_SpatialEmbeddings = SpatialEmbeddings_network_wrapper()
		self.net_SpatialEmbeddings.load_weights(weight_file_location)

	@staticmethod
	def torch_image(image_np, **_):
		return torch.from_numpy(image_np.transpose(2, 0, 1)).float() * (1./255.)

	@staticmethod
	def to_np(x):
		return x.cpu().numpy()

	@staticmethod
	def tr_demo(image_np, cp_offset, cp_seed, cp_sigma, **_):
		dirimg = (0.5 - cp_offset)
		dirimg = np.concatenate([dirimg, np.zeros((1,)+dirimg.shape[1:], dtype=dirimg.dtype)], axis=0)
		dirimg = (dirimg*255).astype(np.uint8).transpose(1, 2, 0)
		
		demo = image_grid([image_np, dirimg, cp_seed, cp_sigma], num_cols = 2, downsample = 2)
		
		
		return dict(demo=demo)

	def run_pretrained_net__pipe(self, dset):
		# return Pipeline(
		# 	tr_batch = TrsChain(

		# 	)
		# )

		self.tr_input = TrChannelLoad(dset.channels['image'], 'image_np')

		fields = ['centerpoint_offset', 'centerpoint_seed', 'centerpoint_sigma']

		self.tr_run_net = TrsChain(
			bind(self.torch_image, 'image_np').outs('image_tr'),
			self.net_SpatialEmbeddings,
			TrNP(fields=fields),
		)

		self.tr_save_results = TrsChain(*[
			TrChannelSave(self.storage[chn], chn)
			for chn in fields
		])

		self.tr_run_and_show = TrsChain(
			self.tr_input, 
			self.tr_run_net,
			bind(self.tr_demo, image_np='image_np', cp_offset='centerpoint_offset', cp_seed='centerpoint_seed', cp_sigma='centerpoint_sigma'),
			TrShow('demo'),
		)
		
		self.tr_run_and_save = TrsChain(
			self.tr_input, 
			self.tr_run_net,
			self.tr_save_results,
		)

	@staticmethod
	def addroi(centerpoint_offset, ROI, **_):
		return dict(
			centerpoint_offset = centerpoint_offset * ROI[None],
		)
	
	def visualize_offset__pipe(self, dset):
		self.tr_input = TrsChain(
			TrChannelLoad(dset.channels['image'], 'image_np'),
			TrChannelLoad(self.storage['centerpoint_offset'], 'centerpoint_offset'),
		)
		
		roi = getattr(dset, 'roi', None)
		roi_fr = dict(roi = roi)
		if roi is not None:
			self.tr_input.append(partial(self.addroi, ROI=roi))
		
		self.tr_vis_offset = TrsChain(
			self.tr_input,
			tr_vis_centerpoint_offset,
			TrShow(['image_np', 'vis_offset']),
		)
	

	def visualize_votes_naive__pipe(self, dset):
		self.tr_vis_votes = TrsChain(
			self.tr_input,
			tr_vis_centerpoint_offset,
			tr_centerpoint_calc_centers,
			tr_plot_votes,
			tr_find_clusters__naive,
			draw_cluster_map,
	# 		TrShow(['image_np', 'vis_offset'], ['vote_heatmap', 'cluster_overlay']),
			TrImgGrid(
				['image_np', 'vis_offset', 'vote_heatmap', 'cluster_overlay'],
				num_cols = 2,
				out_name = 'demo',
			),		
		)

		self.tr_vis_votes_and_show = TrsChain(
			self.tr_vis_votes,
			TrShow('demo'),
		)

		self.tr_vis_votes_and_save = TrsChain(
			self.tr_vis_votes,
			TrChannelSave(self.storage['demo_naive'], 'demo'),
		)

	def visualize_votes_clusterblur__pipe(self, dset, cluster_func):
		"""
		cluster_func 
			in: centerpoint_target, centerpoint_sigma etc
			out: 
				cluster_centers[N x 2]
				cluster_idx_map[H x W]
		"""
		self.tr_vis_cluster = TrsChain(
			self.tr_input,
			TrChannelLoad(self.storage['centerpoint_sigma'], 'centerpoint_sigma'),
			tr_vis_centerpoint_offset,
			tr_centerpoint_calc_centers,
			tr_plot_votes,
			cluster_func,
			tr_discontinuity_blur,
			draw_cluster_map,
	# 		TrShow(['image_np', 'vis_offset'], ['vote_heatmap', 'cluster_overlay']),
			TrImgGrid(
				['image_np', 'vis_offset', 'vote_heatmap_with_clusters_img', 'vote_heatmap', 'cluster_overlay', 'discontinuity'],
				num_cols = 2,
				out_name = 'demo',
			),		
		)

		self.tr_vis_cluster_and_show = TrsChain(
			self.tr_vis_cluster,
			TrShow('demo'),
		)

		self.tr_vis_cluster_and_save = TrsChain(
			self.tr_vis_cluster,
			TrChannelSave(self.storage['demo_cluster_blur'], 'demo'),
		)





from matplotlib import cm


class Exp1003_CenterpointAgnosticGCen:
	cfg = add_experiment({},
		#TODO don't repeat the names
		name = '1003_CenterpointAgnosticGCen',
		net_checkpoint = 'agnostic_gcen/run_2020-01-15_16:29:08/best_iou_model.pth',
		shape_dim = 1,
	)

	def __init__(self):
		self.workdir = DIR_EXP / self.cfg['name']

		self.construct_persistence()

	def construct_persistence(self):
		out_dir = Path('{channel.ctx.workdir}/{dset.name}_{dset.split}')

		out_hdf_file = out_dir / 'net_output/{fid_no_slash}_spatial.hdf5'

		self.storage = dict(
			centerpoint_offset = ChannelLoaderHDF5_NotShared(out_dir / 'net_output/{fid_no_slash}_spatial_offset.hdf5', 'centerpoint_offset'),
			centerpoint_seed = ChannelLoaderHDF5_NotShared(out_dir / 'net_output/{fid_no_slash}_spatial_seed.hdf5', 'centerpoint_seed'),
			centerpoint_vote_radius = ChannelLoaderHDF5_NotShared(out_dir / 'net_output/{fid_no_slash}_spatial_vote_radius.hdf5', 'centerpoint_vote_radius'),
			demo_netout = ChannelLoaderImage(out_dir / 'net_output' / '{fid_no_slash}_vis.webp'),
			demo_naive = ChannelLoaderImage(out_dir / 'demo_cluster_naive' / '{fid_no_slash}_cluster_native.webp'),
			demo_cluster_blur = ChannelLoaderImage(out_dir / 'demo_cluster_blur' / '{fid_no_slash}_cluster_blur.webp'),
			demo_templ = ChannelLoaderImage(out_dir / 'demo_templ' / '{fid_no_slash}_templ.webp'),
		)
		for c in self.storage.values(): c.ctx = self

	def init_default_datasets(self):

		from ..datasets.lost_and_found import DatasetLostAndFound
		from ..datasets.road_anomaly import DatasetRoadAnomaly
		from ..datasets.fishyscapes import DatasetFishyscapes
		from ..datasets.CAOS_StreetHazards import DatasetCaosStreetHazards

		self.dsets = EasyDict(
			laf = DatasetLostAndFound(),
			# synth = DatasetCaosStreetHazards(),
			ranomaly = DatasetRoadAnomaly(),
			fishy = DatasetFishyscapes(),
		)

		for dset in self.dsets.values():
			dset.discover()

		# self.dsets.synth.frames = self.dsets.synth.frames[:150]

	def run_pretrained_net__init(self):
		self.net_SpatialEmbeddings = SpatialEmbeddings_network_wrapper()
		self.net_SpatialEmbeddings.load_weights(self.workdir / self.cfg['net_checkpoint'])

	@staticmethod
	def torch_image(image_np, **_):
		return torch.from_numpy(image_np.transpose(2, 0, 1)).float() * (1./255.)

	@staticmethod
	def to_np(x):
		return x.cpu().numpy()

	@staticmethod
	def tr_demo(image_np, cp_offset, cp_seed, cp_sigma, **_):

		dirimg = tr_vis_centerpoint_offset(centerpoint_offset = cp_offset)['vis_offset']

		# dirimg = (0.5 - cp_offset)
		# dirimg = np.concatenate([dirimg, np.zeros((1,)+dirimg.shape[1:], dtype=dirimg.dtype)], axis=0)
		# dirimg = (dirimg*255).astype(np.uint8).transpose(1, 2, 0)
		
		if cp_sigma.shape[0] == 2:
			sigma_ratio = np.arctan(cp_sigma[1] / cp_sigma[0]) * (2 / np.pi)
			# print(np.min(sigma_ratio), np.max(sigma_ratio))
			sigma_img = cm.get_cmap('PuOr')(sigma_ratio, bytes=True)[:, :, :3]
		else:
			sigma_img = cp_sigma[0]


		demo = image_grid([image_np, dirimg, cp_seed, sigma_img], num_cols = 2, downsample = 2)
		
		return dict(demo=demo)

	def run_pretrained_net__pipe(self, dset):
		# return Pipeline(
		# 	tr_batch = TrsChain(

		# 	)
		# )

		if hasattr(dset, 'channels') and 'image' in dset.channels:
			self.tr_input = TrChannelLoad(dset.channels['image'], 'image_np')
		else:
			self.tr_input = TrsChain()

		fields = ['centerpoint_offset', 'centerpoint_seed', 'centerpoint_vote_radius',]

		self.tr_run_net = TrsChain(
			bind(self.torch_image, 'image_np').outs('image_tr'),
			self.net_SpatialEmbeddings,
			TrNP(fields=fields),
		)

		self.tr_save_results = TrsChain(*[
			TrChannelSave(self.storage[chn], chn)
			for chn in fields
		])
		self.tr_save_results.append(
			TrChannelSave(self.storage['demo_netout'], 'demo')
		)

		tr_make_demo = bind(self.tr_demo, image_np='image_np', cp_offset='centerpoint_offset', cp_seed='centerpoint_seed', cp_sigma='centerpoint_vote_radius')

		self.tr_run_and_show = TrsChain(
			self.tr_input, 
			self.tr_run_net,
			tr_print,
			tr_make_demo,
			TrShow('demo'),
		)
		
		self.tr_run_and_save = TrsChain(
			self.tr_input, 
			self.tr_run_net,
			tr_make_demo,
			self.tr_save_results,
		)

	@staticmethod
	def addroi(centerpoint_offset, ROI, **_):
		return dict(
			centerpoint_offset = centerpoint_offset * ROI[None],
		)

	def run_tr_on_dset(self, tr, ds, frames=None):
		frames = frames or ds.frames
		for fr in tqdm(frames):
			fr.copy().apply(tr)
			

	def run_net_on_dset(self, ds, frames=None):
		self.run_pretrained_net__pipe(ds)
		frames = frames or ds.frames
		for fr in tqdm(frames):
			fr.copy().apply(self.tr_run_and_save)
	
	def visualize_offset__pipe(self, dset):
		self.tr_input = TrsChain(
			TrChannelLoad(dset.channels['image'], 'image_np'),
			TrChannelLoad(self.storage['centerpoint_offset'], 'centerpoint_offset'),
		)
		
		roi = getattr(dset, 'roi', None)
		roi_fr = dict(roi = roi)
		if roi is not None:
			self.tr_input.append(partial(self.addroi, ROI=roi))
		
		self.tr_vis_offset = TrsChain(
			self.tr_input,
			tr_vis_centerpoint_offset,
			TrShow(['image_np', 'vis_offset']),
		)
	

	def visualize_votes_naive__pipe(self, dset):
		self.tr_vis_votes = TrsChain(
			self.tr_input,
			tr_vis_centerpoint_offset,
			tr_centerpoint_calc_centers,
			tr_plot_votes,
			tr_find_clusters__naive,
			draw_cluster_map,
	# 		TrShow(['image_np', 'vis_offset'], ['vote_heatmap', 'cluster_overlay']),
			TrImgGrid(
				['image_np', 'vis_offset', 'vote_heatmap', 'cluster_overlay'],
				num_cols = 2,
				out_name = 'demo',
			),		
		)

		self.tr_vis_votes_and_show = TrsChain(
			self.tr_vis_votes,
			TrShow('demo'),
		)

		self.tr_vis_votes_and_save = TrsChain(
			self.tr_vis_votes,
			TrChannelSave(self.storage['demo_naive'], 'demo'),
		)

	def visualize_votes_clusterblur__pipe(self, dset, cluster_func):
		"""
		cluster_func 
			in: centerpoint_target, centerpoint_sigma etc
			out: 
				cluster_centers[N x 2]
				cluster_idx_map[H x W]
		"""
		self.tr_vis_cluster = TrsChain(
			self.tr_input,
			TrChannelLoad(self.storage['centerpoint_vote_radius'], 'centerpoint_sigma'),
			tr_vis_centerpoint_offset,
			tr_centerpoint_calc_centers,
			tr_plot_votes,
			cluster_func,
			tr_discontinuity_blur,
			draw_cluster_map,
	# 		TrShow(['image_np', 'vis_offset'], ['vote_heatmap', 'cluster_overlay']),
			TrImgGrid(
				['image_np', 'vis_offset', 'vote_heatmap_with_clusters_img', 'vote_heatmap', 'cluster_overlay', 'discontinuity'],
				num_cols = 2,
				out_name = 'demo',
			),		
		)

		self.tr_vis_cluster_and_show = TrsChain(
			self.tr_vis_cluster,
			TrShow('demo'),
		)

		self.tr_vis_cluster_and_save = TrsChain(
			self.tr_vis_cluster,
			TrChannelSave(self.storage['demo_cluster_blur'], 'demo'),
		)


class ChannelLoaderImageCropped(ChannelLoaderImage):
	def __init__(*args, crop_x=[0, 1], crop_y=[0, 1], **kwargs):
		super().__init__(*args, **kwargs)

		self.crop_x = np.array(crop_x)
		self.crop_y = np.array(crop_y)

	def read_file(self, path):
		img = super().read_file(path)

		h, w = img.shape[:2]
		rx = self.crop_x * w
		ry = self.crop_y * h

		return img[ry[0]:ry[1], rx[0]:rx[1]]


# def crop_dset
# 	dset.roi = None
# 	img_ch = dset.channels['image']
# 	dset.channels['image'] = img_ch

class Exp1008_CenterpointAgnosticGCen_2d_extra(Exp1003_CenterpointAgnosticGCen):
	cfg = add_experiment({},
		#TODO don't repeat the names
		name = '1008_CenterpointAgnosticGCen_2d_extracl',
		net_checkpoint = 'run_2020-02-17_20:00:22/best_iou_model.pth',
		shape_dim = 2,
	)

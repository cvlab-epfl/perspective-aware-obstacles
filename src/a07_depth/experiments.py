
import logging
from pathlib import Path
from functools import lru_cache
import torch
import numpy as np
from easydict import EasyDict
import cv2 as cv
from ..paths import DIR_EXP
from ..pipesys.bind import bind
from ..pipeline.transforms import TrsChain, TrBase
from ..pipeline.transforms_imgproc import TrShow, TrImgGrid
from ..pipeline.evaluations import TrChannelLoad, TrChannelSave
from ..datasets.dataset import ChannelLoaderImage, ChannelLoaderHDF5_NotShared
from ..datasets.cityscapes import CityscapesLabelInfo
from ..a01_sem_seg.transforms import SemSegLabelsToColorImg
from ..pytorch_semantic_segmentation import models as ptseg_archs

from .MiDaS import monodepth_net, utils as midas_utils

log = logging.getLogger('exp')

def torch_image(image_np, **_):
	return torch.from_numpy(image_np.transpose(2, 0, 1).astype(np.float32)) * (1./255.)


class TrColorimgNew(TrBase):
	def __init__(self, colors_by_classid=CityscapesLabelInfo.colors_by_trainId):
		"""
		colors_by_classid : [num_classes x 3] uint8
		"""
		self.set_class_colors(colors_by_classid)

	def set_class_colors(self, colors_by_classid):
		self.colors_by_classid = colors_by_classid
		# extend to 255 to allow the "unlabeled" areas
		# if we don't specify the dtype, it will be float64 and tensorboard will display it wrongly
		self.colors_by_classid_ext = np.zeros((256, 3), dtype=self.colors_by_classid.dtype)
		self.colors_by_classid_ext[:self.colors_by_classid.shape[0]] = self.colors_by_classid

	def set_override(self, class_id, color):
		self.colors_by_classid_ext[class_id] = color

	def __call__(self, labels, **_):
		fr_sh = labels.shape
		return self.colors_by_classid_ext[labels.reshape(-1)].reshape((fr_sh[0], fr_sh[1], 3))


class SemSegSimple:
	cfg = EasyDict(
		name = '0121_PSPEns_BDD_00',	
		net = dict(
			batch_train = 5,
			batch_eval = 6, # full imgs not crops
			backbone_pretrained = True,
			apex_mode = 'O1',
			use_aux = False,

			num_classes = 19,
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

	IMG_MEAN_DEFAULT = [0.485, 0.456, 0.406]
	IMG_STD_DEFAULT = [0.229, 0.224, 0.225]
	tr_colorimg = TrColorimgNew()

	@classmethod
	@lru_cache(maxsize=1)
	def get_mean_std_torch(cls):
		def totr(x):
			return torch.FloatTensor(x)[None, :, None, None].cuda()

		return EasyDict(
			mean = totr(cls.IMG_MEAN_DEFAULT),
			std_inv = totr([1/x for x in cls.IMG_STD_DEFAULT]),
		)

	def __init__(self, workdir, cfg=None):
		self.workdir = workdir
		if cfg is not None:
			self.cfg = cfg

	def construct_persistence(self):
		out_dir = Path('{channel.ctx.workdir}/{dset.name}_{dset.split}')

		self.storage = dict(
			pred_labels_trainIds = ChannelLoaderImage(out_dir / 'semantic/{fid_no_slash}_predTrainIds.png'),
			pred_labels_colorimg = ChannelLoaderImage(out_dir / 'semantic/{fid_no_slash}_predColorImg.png'),
		)
		for c in self.storage.values(): c.ctx = self

	def load(self):
		self.net_mod = ptseg_archs.PSPNet(
			num_classes = self.cfg['net']['num_classes'], 
			pretrained = self.cfg['net'].get('backbone_pretrained', True), 
			use_aux = self.cfg['net'].get('use_aux', True), 
		)

		try:
			self.net_mod.load_state_dict(torch.load(DIR_EXP / self.cfg.name / 'chk_best.pth')['weights'])
		except RuntimeError as e:
			log.warning(f'Mismatched keys {e}')
		
		self.net_mod.eval()
		self.net_mod.cuda()

	def predict_semseg_simple(self, image_tr, **_):

		norm = self.get_mean_std_torch()

		with torch.no_grad():
			# add batch dim
			img = image_tr[None].cuda() 
			# normalization expected by the backbone
			img = (img - norm.mean)*norm.std_inv
			# predict logits
			logits = self.net_mod(img)
			# class = argmax of logits (1 is channel dimension)
			class_trainId = torch.argmax(logits, dim=1)
			# unbatch
			class_trainId = class_trainId[0]

		class_trainId_np = class_trainId.cpu().numpy().astype(np.uint8)

		return dict(
			semantic_trainId = class_trainId_np,
		)

	def init_pipelines(self, dset):
		
		self.pipeline_predict = TrsChain(
			bind(torch_image, 'image_np').outs('image_tr'),
			self.predict_semseg_simple,
			bind(self.tr_colorimg, labels = 'semantic_trainId').outs('semantic_colorimg'),
		)

		self.pipeline_input = TrChannelLoad(dset.channels['image'], 'image_np')

		self.pipeline_demo_show = TrsChain(
			self.pipeline_input,
			self.pipeline_predict,
			TrShow(['image_np', 'semantic_colorimg']),
		)


		self.pipeline_run_and_save = TrsChain(
			self.pipeline_input,
			self.pipeline_predict,
			TrChannelSave(self.storage['pred_labels_trainIds'], 'semantic_trainId'),
			TrChannelSave(self.storage['pred_labels_colorimg'], 'semantic_colorimg'),
		)


class MiDaS_Monodepth:
	MiDaS_PRETRAINED_PATH = Path('/cvlabsrc1/cvlab/pytorch_model_zoo/MiDaS/MiDaS_pretrained.pth')

	def __init__(self, workdir = DIR_EXP / '0701_Monodepth_MiDaS'):
		self.workdir = workdir

	def load(self):
		log.info(f'MiDaS loading from {self.MiDaS_PRETRAINED_PATH}')
		self.midas_net = monodepth_net.MonoDepthNet(self.MiDaS_PRETRAINED_PATH)
		self.midas_net.eval().cuda()

	def predict_monodepth_simple(self, image_np, **_):
		h, w = image_np.shape[:2]
		img_torch_small = midas_utils.resize_image(image_np.astype(np.float32)*(1/255))
		
		with torch.no_grad():
			depth_inv_small = self.midas_net(img_torch_small.cuda())
		
		depth_inv_np = midas_utils.resize_depth(depth_inv_small, width=w, height=h)

		return dict(
			depth_inv = depth_inv_np,
		)

	@staticmethod
	def midas_depth_inv_to_depth(depth_inv):
		return 1./(1+depth_inv)

	def construct_persistence(self):
		out_dir = Path('{channel.ctx.workdir}/{dset.name}_{dset.split}')

		self.storage = dict(
			depth_inv = ChannelLoaderHDF5_NotShared(out_dir / 'depth_inv/{fid_no_slash}_depth_inv.hdf5', 'depth_inv'),
			demo = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_depth_demo.webp'),
		)
		for c in self.storage.values(): c.ctx = self

	def init_pipelines(self, dset):
		self.pipeline_make_demo = TrsChain(
			bind(self.predict_monodepth_simple, 'image_np').outs(depth_inv='depth_inv'),
			bind(MiDaS_Monodepth.midas_depth_inv_to_depth, 'depth_inv').outs('depth'),
			bind(RGBD_OpenCV_Exploration.opencv_rgbd_normals, 'depth'),
			bind(RGBD_OpenCV_Exploration.depth_gradients, 'depth').outs(depth_grad='depth_grad'),
			TrImgGrid([
					'image_np', 'depth',
					'depth_grad', 'depth_normals_for_display',
				],
				num_cols = 2,
				out_name = 'demo',
			),
		)

		self.pipeline_input = TrChannelLoad(dset.channels['image'], 'image_np')

		self.pipeline_demo_show = TrsChain(
			self.pipeline_input,
			self.pipeline_make_demo,
			TrShow('demo'),
		)

		self.pipeline_run_and_save = TrsChain(
			self.pipeline_input,
			self.pipeline_make_demo,
			bind(np.float16, 'depth_inv').outs('depth_inv'),
			TrChannelSave(self.storage['demo'], 'demo'),
			TrChannelSave(self.storage['depth_inv'], 'depth_inv'),
		)


class RGBD_OpenCV_Exploration:
	LAF_intrinsics = {
		"fx": 2268.36, 
		"fy": 2312.0, 
		"u0": 1048.64, 
		"v0": 519.27
	}
	LAF_K = np.array([
		[LAF_intrinsics['fx'], 0, LAF_intrinsics['u0']],
		[0, LAF_intrinsics['fy'], LAF_intrinsics['v0']],
		[0, 0, 1],
	])
	LAF_K[:2] *= 0.5 # we downsample 2x

	@staticmethod
	def cv_rgbd_normals(depth_3d, K):
		nr = cv.rgbd_RgbdNormals.create(depth_3d.shape[0], depth_3d.shape[1], depth=cv.CV_32F, K=K)
		return nr.apply(depth_3d)

	# def cv_rgbd_planes(depth_3d, normals, )

	@staticmethod
	def opencv_rgbd_to_3d(depth, camera_K = LAF_K, **_):
		return dict(
			depth_pts3d = cv.rgbd.depthTo3d(depth=depth, K=camera_K),
		)

	@classmethod
	def opencv_rgbd_normals(cls, depth, camera_K = LAF_K, **_):
		depth_pts3d = cv.rgbd.depthTo3d(depth=depth, K=camera_K)
		
		depth_normals = cls.cv_rgbd_normals(depth_pts3d, K=camera_K)
		
	# 	pl = cv.rgbd_RgbdPlane()
	# 	res = pl.apply(depth_pts3d, depth_normals)
	# 	print(res)

		return dict(
			depth_pts3d = depth_pts3d,
			depth_normals = depth_normals,
			depth_normals_for_display = (depth_normals+1)*0.5,
		)

	@staticmethod
	def depth_gradients(depth, **_):
		dx = cv.Scharr(depth, cv.CV_32F, 1, 0)
		dy = cv.Scharr(depth, cv.CV_32F, 0, 1)
		
		return dict(
			depth_grad = np.abs(dx) + np.abs(dy),
			depth_grad_x = dx,
			depth_grad_y = dy,
		)
		
class Exp0702_DepthPlanarity:
	def __init__(self, workdir=None):
		self.exp_depth = MiDaS_Monodepth()
		self.workdir = self.exp_depth.workdir
		self.exp_semantic = SemSegSimple(workdir=self.workdir)

		self.exp_depth.construct_persistence()
		self.exp_semantic.construct_persistence()

	def construct_persistence(self):
		pass

	def init_pipelines(self, dset):
		pass

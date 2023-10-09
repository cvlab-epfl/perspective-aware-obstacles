
from pathlib import Path
from math import ceil
from easydict import EasyDict

import torch, kornia, einops
import numpy as np

from ..datasets.dataset import imwrite
from ..paths import DIR_EXP2, DIR_DATA
from ..common.jupyter_show_image import show
from .bbox_transform import draw_bboxes, draw_bboxes_with_scores

from .YOLO.models import Darknet
from .YOLO.utils.utils import non_max_suppression

from ..a10_geometric_instances.centerpoint_network import SpatialEmbeddingsNetwork
from ..a10_geometric_instances.centerpoint_network_vis import vis_centerpoint_offset


def get_num_patch_and_offset(image_dim, patch_size, min_overlap):
	"""
	min_overlap is a fraction of patch_size
	
	Want to calculate offset btw patches.
	
		gw patches along image width W
	
	W = (gw - 1)*offset + patch_size
	offset <= patch_size * (1 - min_overlap)
	
	gw = (W - patch_size) / offset + 1
	
	We get minimal gw by using maximal offset, that is patch_size * (1 - min_overlap).
	
		gw = ceil( (W - patch_size) / ( (1-min_overlap)*patch_size) )
		
	Knowing gw, we get offset:
	
		offset = (W - patch_size) / (gw-1)
	
	"""

	gw = ceil( (image_dim - patch_size) / ((1-min_overlap) * patch_size) + 1)
	
	step = (image_dim - patch_size) / (gw-1)
	step = int(step)
	
	return EasyDict(
		num_patches = gw,
		step = step,
		offsets = torch.arange(gw, dtype=torch.int32) * step,
	)
	
	
def img_to_patches(img_tr, patch_size, min_overlap):
	_, h, w = img_tr.shape
	
	steps_y = get_num_patch_and_offset(image_dim=h, patch_size=patch_size, min_overlap=min_overlap)
	steps_x = get_num_patch_and_offset(image_dim=w, patch_size=patch_size, min_overlap=min_overlap)
	
	num_patches = steps_y.num_patches * steps_x.num_patches
	
	batch = []
	offsets = []
	
	for tl_y in steps_y.offsets:
		for tl_x in steps_x.offsets:
			offsets.append([tl_x, tl_y])	
			batch.append(img_tr[:, tl_y:tl_y+patch_size, tl_x:tl_x+patch_size])

	offsets = torch.tensor(offsets)
	batch = torch.stack(batch)
	
	return EasyDict(
		image_batch = batch,
		offsets = offsets,
	)
	


# 	patches = img_tr.unfold(dimension=1, size=512, step=400).unfold(dimension=2, size=512, step=400)
# 	print(patches.shape)
# 	patches = einops.rearrange(patches, 'c gw gh pw ph -> (gh gw) c pw ph')
# 	print(img_tr.shape, patches.shape)
# 	show(kornia.utils.tensor_to_image(patches[0]), kornia.utils.tensor_to_image(patches[7]))


def test_img_to_patches():
	img_np = fr0.image
	img_tr = kornia.utils.image_to_tensor(img_np) * (1/255)
	
	imgs_as_patches = img_to_patches(img_tr, patch_size=512, min_overlap=0.2)

	batch = imgs_as_patches.image_batch
	offsets = imgs_as_patches.offsets
	
	print(batch.shape, offsets.shape)
	to_show = [0, 1, 6]
	print(offsets[to_show])
	show([kornia.utils.tensor_to_image(batch[i]) for i in to_show])


# test_img_to_patches()
# img_tr = kornia.utils.image_to_tensor(img_np) * (1/255)

DIR_DATA = Path('/mnt/data-research/data')

class ObjdetectInference:

	cfg_yolo_rgb = dict(
		name = 'yolo-rgb',
		checkpoint = DIR_EXP2/ '110_Yolo/01_test/run2/checkpoints/yolov3_ckpt_latest.pth',
		arch_file = Path('src/a11_instance_detector/YOLO/config/yolov3_1class.cfg'),
		patch_size = 512,
		patch_min_overlap = 0.2,
	)

	cfg_infer_rgb = EasyDict(
		yolo = cfg_yolo_rgb,
	)
	
	
	def __init__(self, cfg):
		self.cfg = cfg
	
	def load_networks(self):
		self.net_yolo = Darknet(self.cfg.yolo.arch_file)
		self.net_yolo.load_state_dict(torch.load(self.cfg.yolo.checkpoint)) 
		# TODO switch to run2
		self.net_yolo.cuda().eval()
		
	@staticmethod
	def exec_net_by_piece(net, batch, piece_size):
		n_b = batch.shape[0]
		results = []
		
		for start in range(0, n_b, piece_size):
			results.append(net(batch[start:min(start+piece_size, n_b)]))

		return torch.cat(results, dim=0)

	
	def infer_image(self, img_np):
		img_tr = kornia.utils.image_to_tensor(img_np) * (1/255)
		
		img_tr_gpu = img_tr.cuda()
		
		with torch.no_grad():
			b = img_to_patches(img_tr_gpu, patch_size=self.cfg.yolo.patch_size, min_overlap=0.2)
			image_batch = b.image_batch
			patch_offsets = b.offsets

			detections_g = self.exec_net_by_piece(self.net_yolo, image_batch, piece_size=4)
				
		detections = detections_g.cpu()

		# add patch offsets to bbox centers
		detections[:, :, :2] += patch_offsets[:, None, :]
		
		detections = einops.rearrange(detections, 'b n c -> (b n) c')
		detections_filtered = non_max_suppression(detections[None], conf_thres=0.7, nms_thres=0.4)[0]

		
		bb = detections_filtered.numpy()[:, :4]
		out_img = draw_bboxes_with_scores(img_np, bb, detections_filtered[:, 4])
		# TODO colors
		#show(out_img)
			
		return EasyDict(
			dets = detections_filtered,
			offsets = patch_offsets,
			boxes = bb,
			demo_img = out_img,
		)
	

	def process_frame(self, dset, idx):
		fr = dset[idx]
		out = self.infer_image(fr.image)


		fid = fr.fid.replace('/', '__') 

		variant_name = self.cfg.yolo.name

		if 'centerpoint' in self.cfg:
			variant_name = f'{self.cfg.yolo.name}_{self.cfg.centerpoint.name}'

		out_path = DIR_DATA / 'yolocp' / f'{dset.name}_{dset.split}' / f'{fid}--{variant_name}.webp'

		imwrite(out_path, out['demo_img'])



class ObjdetectInferenceCps(ObjdetectInference):


	cfg_yolo_crisplit = dict(
		name = 'yolo-crisplit',
		checkpoint = DIR_EXP2/ '110_Yolo/02_cp/run2/checkpoints/yolov3_ckpt_latest.pth',
		arch_file = Path('src/a11_instance_detector/YOLO/config/yolov3_1class_4in.cfg'),
		patch_size = 512,
		patch_min_overlap = 0.2,
	)

	cfg_yolo_cpall = dict(
		name = 'yolo-all',
		checkpoint = DIR_EXP2/ '110_Yolo/03_cp_all/run1/checkpoints/yolov3_ckpt_latest.pth',
		arch_file = Path('src/a11_instance_detector/YOLO/config/yolov3_1class_4in.cfg'),
		patch_size = 512,
		patch_min_overlap = 0.2,
	)

	cfg_infer_cp = EasyDict(
		yolo = cfg_yolo_crisplit,
		centerpoint = dict(
			name = 'cp3-all',
			checkpoint_dir = DIR_EXP2 / '101_SpatialEmbeddingsNet/101.06.road_offset_zero/run_2020-03-11_19:39:52',
		),
	)

	def load_networks(self):
		super().load_networks() # yolo

		self.net_cp = SpatialEmbeddingsNetwork(checkpoint_dir = self.cfg.centerpoint.checkpoint_dir)
		self.net_cp.load_from_checkpoint()
		

	def infer_image(self, img_np):
		img_tr = kornia.utils.image_to_tensor(img_np) * (1/255)
		
		img_tr_gpu = img_tr.cuda()
		
		with torch.no_grad():
			cp_res = self.net_cp(image=img_tr_gpu[None])

			centerpoint_offset = cp_res['centerpoint_offset']
			centerpoint_radius = cp_res['vote_radius']
			inputs = torch.cat([
				centerpoint_offset,
				centerpoint_radius,
			], dim=1)

			cp_visualization = vis_centerpoint_offset(centerpoint_offset[0])['vis_centerpoint_offset']


			b = img_to_patches(inputs[0], patch_size=self.cfg.yolo.patch_size, min_overlap=0.2)
			input_batch = b.image_batch
			patch_offsets = b.offsets

			detections_g = self.exec_net_by_piece(self.net_yolo, input_batch, piece_size=4)
				
		detections = detections_g.cpu()

		# add patch offsets to bbox centers
		detections[:, :, :2] += patch_offsets[:, None, :]
		
		detections = einops.rearrange(detections, 'b n c -> (b n) c')
		detections_filtered = non_max_suppression(detections[None], conf_thres=0.7, nms_thres=0.4)[0]

		bb = detections_filtered.numpy()[:, :4]
		out_img = draw_bboxes_with_scores(img_np, bb, detections_filtered[:, 4])
		
		out_img_vectors = draw_bboxes(cp_visualization, bb, color_default=(0, 0, 0))

		# show([out_img, cp_visualization])

		demo_img = np.concatenate([
			out_img[::2, ::2],
			out_img_vectors[::2, ::2],
		], axis=0)

		

		return EasyDict(
			dets = detections_filtered,
			offsets = patch_offsets,
			boxes = bb,
			demo_img = demo_img,
		)

# 	def infer_image(self, image):
		
# exp = ObjdetectInference(ObjdetectInference.cfg_yolo_rgb)
# exp.load_networks()









# def nms_pytorch(detections, conf_thres, nms_thres):
	
# 	dets_above_thr = detections[:, 4] >= conf_thres
	
# 	dets_filtered = detections[dets_above_thr]
	
# 	dets_nms = nms(
# 		boxes = dets_filtered[:, :4], 
# 		scores = dets_filtered[:, 4]*dets_filtered[:, 5], 
# 		iou_threshold = nms_thres,
# 	)
		
# 	return dets_filtered[dets_nms]

	


# dets = infer_full_image(exp, dset_ctc[71].image)

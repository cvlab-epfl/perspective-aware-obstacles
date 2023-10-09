
import numpy as np
import torch
import cv2 as cv

from kornia.color.hls import rgb_to_hls, hls_to_rgb
from kornia.utils import image_to_tensor

from .synth_obstacle_dset2 import fuse_blur, DsetSynthObstacles_FusionFlat

from ..common.jupyter_show_image import imwrite
from ..a11_instance_detector.bbox_transform import labels_to_bbox
from ..a12_inpainting.vis_imgproc import image_montage_same_shape
from ..a13_context.pytorch_load_gluon import load_gluon_weight_file, load_weights_into_pytorch_deeplab
from ..gluoncvth.utils.preset import input_transform


def obstacle_color_adjust_hls(inj_tr, m_weights):
	inj_hls = rgb_to_hls(inj_tr)
	inj_hls = inj_hls * m_weights[None, :, None, None]
	inj_back = hls_to_rgb(inj_hls)
	return inj_back


def adjust_obstacle(fr, image_with_injection, bb, obid, tr_dev, net_seg):	
	h, w = fr.obstacle_instance_map.shape
	
	bb = bb.astype(np.int32) # to prevent negatives underflowing
	tl = np.maximum([0, 0], bb[0:2] - 128)
	br = np.minimum([w, h], bb[2:4] + 128)
	img_crop = fr.image[tl[1]:br[1], tl[0]:br[0]]
	inj_crop = fr.image_injections_with_margin[tl[1]:br[1], tl[0]:br[0]]
	mask_crop = fr.obstacle_instance_map[tl[1]:br[1], tl[0]:br[0]] == obid
	blur_ksize = 5
	mask_smooth = cv.GaussianBlur(mask_crop.astype(np.float32), ksize=(blur_ksize, blur_ksize), sigmaX=0)
	mask_area = float(np.sum(mask_smooth))
	mask_area_fraction = float(mask_area / np.prod(mask_smooth.shape))
		
	
	bg_tr = image_to_tensor(img_crop)[None].float().to(tr_dev) * (1./255.)
	inj_tr = image_to_tensor(inj_crop)[None].float().to(tr_dev) * (1./255.)
	mask_tr = torch.from_numpy(mask_smooth).float().to(tr_dev)
	
	w_m = torch.nn.Parameter(torch.ones((3,)).to(tr_dev))
	w_m.requires_grad = True
	optimizer = torch.optim.Adam([w_m], lr=0.1)
	
	images = []
	captions = []
	
	# w_m_limit = torch.tensor([2*np.pi, 1., 1.]).to(tr_dev)

	for i in range(16):
		optimizer.zero_grad()
		
		with torch.set_grad_enabled(True):
			inj_tr_opt = obstacle_color_adjust_hls(inj_tr, w_m).clamp(min=0., max=1.)
			fuse_tr = bg_tr * (1.-mask_tr) + inj_tr_opt * mask_tr
			fuse_tr_preproc = input_transform.transforms[1](fuse_tr[0])[None]
			
			logits = net_seg(fuse_tr_preproc)[0]
			class_p = torch.softmax(logits, 1)
			
			# print(logits.device, fuse_tr.device, mask_tr.device)
			
			road_p_at_obstacle = torch.sum(mask_tr * class_p[:, 0]) / mask_area
			(-road_p_at_obstacle).backward() # negate when using optimizer
			
			# class_road_np = class_p.data[0].cpu().numpy()
			
			# images += [
			# 	np.transpose((fuse_tr * 255.).cpu().byte().numpy()[0], [1, 2, 0]), 
			# 	class_road_np[0],
			# ]
			
			# print('p_road at obstacle', float(road_p_at_obstacle), 'w_m', w_m.data.cpu())
			
			desc_w = 'W_hls ' + ' '.join(f'{float(ze):.03f}' for ze in w_m.data.cpu())
			desc_rp = f'P_road {float(road_p_at_obstacle):.03f}'
			
			# print(i, desc_w, desc_rp)
			
			captions += [desc_w, desc_rp]
				
		optimizer.step()
		w_m.data = w_m.data.clamp(min=0.)
		# w_m.data = torch.minimum(w_m.data, w_m_limit)
		
		# w_m += w_m.grad * 0.1
		

	inj_final = np.transpose(inj_tr_opt.data.cpu().numpy()[0], [1, 2, 0])

	image_with_injection[tl[1]:br[1], tl[0]:br[0]] = (
		image_with_injection[tl[1]:br[1], tl[0]:br[0]] * (1.-mask_smooth[:, :, None])
		+
		inj_final * mask_smooth[:, :, None]
	)
	
	return image_with_injection
	
	
	
	
def segment_image(image, tr_dev, net_seg):
	with torch.no_grad():
		image_tr = input_transform(image)[None].to(tr_dev)
		logits = net_seg(image_tr)[0] # undoes aux_loss
		p_class = torch.softmax(logits, axis=1)
		# classes = torch.argmax(logits, axis=1)
		# classes_np = classes[0].cpu().numpy()
		return p_class.cpu().numpy()[0]
	
	
def opt_process_frame(fr, tr_dev, net_seg):
	# initial segmentation
	img_fused_orig = fuse_blur(
		image_bg = fr.image,
		image_objects = fr.image_injections_with_margin,
		mask = fr.obstacle_instance_map > 0,
		blur_ksize = 5,
	)
	seg_init = segment_image(img_fused_orig, tr_dev=tr_dev, net_seg=net_seg)
	
	# optimize obstacles	
	h, w = fr.obstacle_instance_map.shape
	_, bboxes = labels_to_bbox(fr.obstacle_instance_map)

	
	image_with_injection = fr.image.astype(np.float32) / 255.
	
	for obs_id in range(1, fr.obstacle_classes.shape[0]):
		bb = bboxes[obs_id-1]
		image_with_injection = adjust_obstacle(fr, image_with_injection, bb, obs_id, tr_dev=tr_dev, net_seg=net_seg)
		
	# final segmentation
	seg_final = segment_image(image_with_injection, tr_dev=tr_dev, net_seg=net_seg)
	
	image_with_injection = (image_with_injection*255.).astype(np.uint8)
	
	img_demo = image_montage_same_shape([
		img_fused_orig, image_with_injection,
		seg_init[0], seg_final[0],
	], num_cols=2, border=4)
	
	return dict(
		img_fused = image_with_injection,
		img_demo = img_demo,
	)
	

class DsetSynthObstacles_FusionOpt(DsetSynthObstacles_FusionFlat):
	def gen__process_frame(self, fr):
		res = opt_process_frame(fr, tr_dev=self.tr_dev, net_seg=self.net_seg)

		imwrite(self.img_path('image_fused', fr.frame_id), res['img_fused'])
		imwrite(self.img_path('demo', fr.frame_id, type='demo'), res['img_demo'])
		fr['image_fused'] = res['img_fused']
		return fr
	
	def gen__init(self):
		super().gen__init()

		self.tr_dev = torch.device('cuda')

		self.net_seg = load_weights_into_pytorch_deeplab(load_gluon_weight_file())
		self.net_seg.eval()
		self.net_seg = self.net_seg.to(self.tr_dev)


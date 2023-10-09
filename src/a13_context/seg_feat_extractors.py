

import numpy as np
import torch
from kornia.utils import image_to_tensor

from easydict import EasyDict


from road_anomaly_benchmark.datasets.dataset_registry import Registry

from ..datasets.cityscapes import CityscapesLabelInfo
from ..paths import DIR_EXP

from ..a01_sem_seg.networks import PerspectiveSceneParsingNet

SegFeatExtractorRegistry = Registry()

IMG_MEAN_DEFAULT = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMG_STD_DEFAULT = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]

def img_input_normalization(img_tr):
	img_tr *= (1./255.)
	img_tr -= IMG_STD_DEFAULT
	img_tr *= (1./IMG_MEAN_DEFAULT)
	return img_tr

def PSP_feat_forward(self, image, **_):
	in_size = image.size()
	val = image

	for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
		val = layer(val)
	feat_ppm = self.ppm(val)
	del val

	feat_final = self.final[:3](feat_ppm)
	classification = self.final[3:](feat_final)
	
	
	return EasyDict(
		feat_ppm = feat_ppm,
		feat_final = feat_final,
		pred_logits = torch.nn.functional.interpolate(classification, in_size[2:], mode='bilinear'),
	)





@SegFeatExtractorRegistry.register_class()
class FeatExtractorOldPSP:
	
	configs = [EasyDict(name = 'PspOld')]

	def __init__(self, _):
		self.load()
		
	def load(self):
		try:
			weight_path = DIR_EXP / '0120_PSPEns_BDD_00' / 'chk_last.pth'
			weight_checkpoint = torch.load(weight_path, map_location='cpu')
		except:
			weight_path = DIR_EXP / '0121_PSPEns_BDD_00' / 'chk_last.pth'
			weight_checkpoint = torch.load(weight_path, map_location='cpu')
		
		net_psp = PerspectiveSceneParsingNet(num_classes=19)
		net_psp.load_state_dict(weight_checkpoint['weights'])
		net_psp.eval()
		self.net_psp = net_psp.cuda()

	def predict(self, fr):
		with torch.no_grad():
			img_tr = image_to_tensor(fr.image[::2, ::2])
			img_tr = img_tr.float()[None]
			img_input_normalization(img_tr)

			feat_out = PSP_feat_forward(self.net_psp, img_tr.cuda())

			feat_out = EasyDict({
				k: v.cpu().numpy()[0] for k, v in feat_out.items()
			})

		cls_result = np.argmax(feat_out.pred_logits, axis=0)
		cls_color = CityscapesLabelInfo.convert_trainIds_to_colors(cls_result)

		feat_out.cls_pred_trainId = cls_result
		feat_out.cls_pred_color = cls_color

		feat_out.update(fr)
		return feat_out

		
	def extract(self, image):
		res = self.predict(EasyDict(image=image))
		
		return EasyDict({
			'PspOld_final': res['feat_final'],
			'PspOld_ppm': res['feat_ppm'],
			'PspOld_logits': res['pred_logits'],
		})
		

@SegFeatExtractorRegistry.register_class()
class FeatExtractorGluon:
	
	configs = [EasyDict(
		name = 'deeplab_v3b_plus_wideresnet_citys',
		), EasyDict(
		name = 'deeplab_resnet50_citys',
		),
	]

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.load()
		
	def load(self):
		import mxnet
		from gluoncv.model_zoo import get_model
		from gluoncv.data.transforms.presets.segmentation import test_transform

		self.mxnet = mxnet
		self.mxnet_context = mxnet.gpu()
		self.test_transform = test_transform

		gluon_net_name = self.cfg.name #'deeplab_v3b_plus_wideresnet_citys'
		self.net_semseg = get_model(gluon_net_name, pretrained=True, ctx = self.mxnet_context)	
		

	def extract(self, image):
		predict_func = getattr(self, f'predict__{self.cfg.name}')
		
		img_mxnet = self.mxnet.nd.array(image)
		img_preproc = self.test_transform(img_mxnet, self.mxnet_context)
		
		net_out_feats = predict_func(self.net_semseg, img_preproc)
		return net_out_feats
	
	def predict(self, fr):
		feat_out = self.extract(fr.image)
		logits = feat_out[f'{self.cfg.name}.logits']
		cls_result = np.argmax(logits, axis=0)
		cls_color = CityscapesLabelInfo.convert_trainIds_to_colors(cls_result)

		feat_out.cls_pred_trainId = cls_result
		feat_out.cls_pred_color = cls_color
		feat_out.update(fr)

		return feat_out

	@staticmethod
	def predict__deeplab_v3b_plus_wideresnet_citys(self, x):
		"""
		# original func
		def predict(self, x):
			h, w = x.shape[2:]
			self._up_kwargs['height'] = h
			self._up_kwargs['width'] = w
			x = self.mod1(x)
			m2 = self.mod2(self.pool2(x))
			x = self.mod3(self.pool3(m2))
			x = self.mod4(x)
			x = self.mod5(x)
			x = self.mod6(x)
			x = self.mod7(x)
			x = self.head.demo(x, m2)
			import mxnet.ndarray as F
			x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
			return x
		"""

		h, w = x.shape[2:]

		feats = {}

		x = self.mod1(x)
		m2 = self.mod2(self.pool2(x))
		feats['mod2'] = m2.asnumpy()[0]
	# 	print('m2', m2.shape)
		x = self.mod3(self.pool3(m2))
		feats['mod3'] = x.asnumpy()[0]
	# 	print('mod3', x.shape)
		x = self.mod4(x)
		feats['mod4'] = x.asnumpy()[0]
	# 	print('mod4', x.shape)
		x = self.mod5(x)
	# 	print('mod5', x.shape)
		feats['mod5'] = x.asnumpy()[0]
		x = self.mod6(x)
		feats['mod6'] = x.asnumpy()[0]
	# 	print('mod6', x.shape)
		x = self.mod7(x)
	# 	print('mod7', x.shape)
		feats['mod7'] = x.asnumpy()[0]
		x = self.head.demo(x, m2)


		from mxnet.ndarray.contrib import BilinearResize2D
		x = BilinearResize2D(x, height=h, width=w)
		feats['logits'] = x.asnumpy()[0]

		return EasyDict({
			f'deeplab_v3b_plus_wideresnet_citys.{ch_name}': ch_val
			for ch_name, ch_val in feats.items()
		})

	@staticmethod
	def predict__deeplab_resnet50_citys(self, x):
		"""
		# deeplab_resnet50_citys
		def base_forward(self, x):
			"forwarding pre-trained network"
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.maxpool(x)
			x = self.layer1(x)
			x = self.layer2(x)
			c3 = self.layer3(x)
			c4 = self.layer4(c3)
			return c3, c4

		def predict(self, x):
			h, w = x.shape[2:]
			self._up_kwargs['height'] = h
			self._up_kwargs['width'] = w
			c3, c4 = self.base_forward(x)
			x = self.head.demo(c4)
			import mxnet.ndarray as F
			pred = F.contrib.BilinearResize2D(x, **self._up_kwargs)
			return pred
		"""

		feats = {}
		h, w = x.shape[2:]

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		feats['layer1'] = x.asnumpy()[0]
		x = self.layer2(x)
		feats['layer2'] = x.asnumpy()[0]
		c3 = self.layer3(x)
		feats['c3'] = c3.asnumpy()[0]
		c4 = self.layer4(c3)
		feats['c4'] = c4.asnumpy()[0]
		
		x = self.head.demo(c4)
		from mxnet.ndarray.contrib import BilinearResize2D
		x = BilinearResize2D(x, height=h, width=w)
		
		feats['logits'] = x.asnumpy()[0]

		return EasyDict({
			f'deeplab_resnet50_citys.{ch_name}': ch_val
			for ch_name, ch_val in feats.items()
		})


"""
# danet_resnet101_citys
def base_forward(self, x):
	"forwarding pre-trained network"
	x = self.conv1(x)
	x = self.bn1(x)
	x = self.relu(x)
	x = self.maxpool(x)
	x = self.layer1(x)
	x = self.layer2(x)
	c3 = self.layer3(x)
	c4 = self.layer4(c3)
	return c3, c4

def hybrid_forward(self, F, x):
	c3, c4 = self.base_forward(x)

	x = self.head(c4)
	x = list(x)
	x[0] = F.contrib.BilinearResize2D(x[0], **self._up_kwargs)
	x[1] = F.contrib.BilinearResize2D(x[1], **self._up_kwargs)
	x[2] = F.contrib.BilinearResize2D(x[2], **self._up_kwargs)

	outputs = [x[0]]
	outputs.append(x[1])
	outputs.append(x[2])

	return tuple(outputs)

def predict(self, x):
	h, w = x.shape[2:]
	self._up_kwargs['height'] = h
	self._up_kwargs['width'] = w
	pred = self.forward(x)
	if self.aux:
		pred = pred[0]
	return pred
"""

def print_layer_shapes(r):
	for k, v in r.items():
		if isinstance(v, np.ndarray):
			print(f'	{k}: {v.shape}')

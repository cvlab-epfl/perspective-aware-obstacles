
from pathlib import Path
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
import click
import torch

from ..paths import DIR_DATA

from ..datasets.dataset import ChannelLoaderImage
from ..datasets.cityscapes import CityscapesLabelInfo

from .vis_imgproc import image_montage_same_shape
from ..common.registry import ModuleRegistry

class SemSegSystem:

	def __init__(self, cfg):
		self.cfg = cfg

	def init_storage(self):
		out_dir = DIR_DATA / '1206sem-{channel.ctx.cfg.name}' / '{dset.name}-{dset.split}' 
	
		self.storage = dict(
			sem_class_prediction = ChannelLoaderImage(out_dir / 'labels' / '{fid_no_slash}_labelId.png'),
			sem_class_prediction_color = ChannelLoaderImage(out_dir / 'labels' / '{fid_no_slash}_colorImg.png'),
			demo = ChannelLoaderImage(out_dir / 'demo' / '{fid_no_slash}.webp'),
		)
		for c in self.storage.values(): c.ctx = self

	def load_values(self, frame):
		k = 'sem_class_prediction'
		return {
			k: self.storage[k].read_value(**frame)
		}

	def load_into_frame(self, frame):
		channels = self.load_values(frame)
		frame.update(channels)
		return channels

	def tr_load(self, **fr):
		return self.load_into_frame(fr)

	def load(self):
		...

	def predict(self, image_np):
		"""
		@return Class IDs following Cityscapes notation, as uint8
		"""
		raise NotImplementedError()

	def predict_with_colors(self, image_np):
		sem_class_prediction = self.predict(image_np=image_np)

		return EasyDict(
			sem_class_prediction = sem_class_prediction,
			sem_class_prediction_color = self.labels_to_colors(sem_class_prediction),
		)

	@classmethod
	def get_implementation(cls, name):
		return ModuleRegistry.get(cls, name)

	@staticmethod
	def make_demo(image_np, sem_class_prediction_color):
		return EasyDict(
			demo = image_montage_same_shape([image_np, sem_class_prediction_color], downsample=2, border=8),
		)

	def labels_to_colors(self, class_map):
		return CityscapesLabelInfo.convert_ids_to_colors(class_map)

	def labels_to_trainIds(self, class_map):
		return CityscapesLabelInfo.convert_ids_to_trainIds(class_map)

	def process_and_save(self, frame):
		frame.update(self.predict_with_colors(image_np=frame.image))
		frame.update(self.make_demo(image_np=frame.image, sem_class_prediction_color = frame.sem_class_prediction_color))

		for k in ['sem_class_prediction', 'sem_class_prediction_color', 'demo']:
			self.storage[k].write_value(frame[k], **frame)


	def process_dset(self, dset):
		for fr in tqdm(dset):
			self.process_and_save(fr)


@ModuleRegistry(SemSegSystem, 'dummy')
class SemSeg_Dummy(SemSegSystem):
	default_cfg = EasyDict(
		name = 'dummy',
	)

	def predict(self, image_np):
		img_h, img_w = image_np.shape[:2]

		labels = np.zeros((img_h, img_w), dtype=np.uint8)

		labels[:img_h//2] = CityscapesLabelInfo.name2id['sky']
		labels[img_h//2:] = CityscapesLabelInfo.name2id['road']

		return labels


@ModuleRegistry(SemSegSystem, 'gluon-psp-ctc')
class SemSeg_Gluon(SemSegSystem):
	
	configs = [
		EasyDict(
			name = 'gluon-psp-ctc',
			gluon_net_name = 'psp_resnet101_citys',
		),
		EasyDict(
			name = 'gluon-fastscnn-ctc',
			gluon_net_name = 'fastscnn_citys',
		),
		EasyDict(
			name = 'gluon-deeeplab101-ctc',
			gluon_net_name = 'deeplab_resnet101_citys',
		),
		EasyDict(
			name = 'gluon-deeplabW-ctc',
			gluon_net_name = 'deeplab_v3b_plus_wideresnet_citys',
		)
	]

	def load(self):
		# only load mxnet if we are running the network
		import mxnet
		from gluoncv.model_zoo import get_model
		from gluoncv.data.transforms.presets.segmentation import test_transform

		self.mxnet = mxnet
		self.mxnet_context = mxnet.gpu()
		self.test_transform = test_transform

		self.net_semseg = get_model(self.cfg.gluon_net_name, pretrained=True, ctx = self.mxnet_context)

	def predict(self, image_np):
		pred_trainIds = self.gluon_semseg_predict(
			image_np = image_np, 
			network = self.net_semseg, 
			mxnet_context = self.mxnet_context,
		)

		return CityscapesLabelInfo.convert_trainIds_to_ids(pred_trainIds)

	def gluon_semseg_predict(self, image_np, network, mxnet_context):
		#img_mxnet = mxnet.nd.array(image_np)
		img_mxnet = self.mxnet.nd.array(image_np)
		img_preproc = self.test_transform(img_mxnet, mxnet_context)

		net_out_logits = network.predict(img_preproc)
		# argmax of logits to get predicted class
		net_out_class = self.mxnet.nd.argmax(net_out_logits, 1)
		return net_out_class[0].asnumpy().astype(np.uint8)


@ModuleRegistry(SemSegSystem, 'DeepLab3W-ctc')
class SemSeg_DeepLab3W(SemSegSystem):
	
	default_cfg = EasyDict(
		name = 'DeepLab3W-ctc',
		#gluon_net_name = 'psp_resnet101_citys',
	)

	def load(self):
		from .deeplab_wide.deepv3 import DeepWV3Plus

		# download file
		checkpoint_file = Path(torch.hub.get_dir()).absolute().parent / 'DeepLab3W' / 'DeepLab3W_cityscapes.pth'
		if not checkpoint_file.is_file():
			checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
			import gdown
			gdown.download(
				url = 'https://drive.google.com/uc?export=download&confirm=maad&id=1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl',
				output = str(checkpoint_file),
			)
		#print(checkpoint_file.read_text())

		# load checkpoint from file
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		# remove the "module." prefix from parameter names, as it does not match the architecture
		prefix_len_to_remove = len('module.')
		weights = {k[prefix_len_to_remove:]: v for k, v in checkpoint['state_dict'].items()}

		# enter the weights into the architecture
		net = DeepWV3Plus(num_classes=19)
		net.eval()
		net.load_state_dict(weights, strict=False)

		self.dev = torch.device('cuda:0')
		self.net = net.to(self.dev)

		self.preproc_mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to(self.dev)
		self.preproc_std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to(self.dev)

	def predict(self, image_np):

		with torch.no_grad():
			img_tr = torch.from_numpy(image_np).permute(2, 0, 1)[None]
			img_tr = img_tr.float().to(self.dev)
			img_tr *= (1./255.)
			img_tr -= self.preproc_mean
			img_tr /= self.preproc_std

			logits = self.net(img_tr)
			pred_trainIds = torch.argmax(logits, axis=1)
			pred_trainIds = pred_trainIds[0].cpu().numpy().astype(np.uint8)

		return CityscapesLabelInfo.convert_trainIds_to_ids(pred_trainIds)


import click
from .demo_case_selection import DatasetRegistry

@click.command()
@click.argument('sem_seg_name')
@click.argument('dset_name')
def main(sem_seg_name, dset_name):
	sys_seg = SemSegSystem.get_implementation(sem_seg_name)
	sys_seg.init_storage()
	sys_seg.load()

	for dsn in dset_name.split(','):
		print(f'Processing: {sem_seg_name} - {dsn}')
		dset = DatasetRegistry.get_implementation(dsn)
		sys_seg.process_dset(dset)

if __name__ == '__main__':
	main()

# # try if the pipeline holds
# python -m src.a12_inpainting.sys_segmentation dummy demo-selection-v1

# # demo set with gluon
# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc demo-selection-v1

# # 
# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc LostAndFound-val

# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc LostAndFound-train

# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc RoadAnomaly-test

# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc FishyLAF-val

# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc 1204-SynthObstacleDset-v1-Ctc-val
# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc 1204-SynthObstacleDset-v1-Ctc-train

# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc RoadAnomaly2-sample1
# python -m src.a12_inpainting.sys_road_area semcontour-roadwalk-v1 RoadAnomaly2-sample1
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 RoadAnomaly2-sample1
# python -m src.a12_inpainting.discrepancy_experiments ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy  RoadAnomaly2-sample1
# python -m src.a12_inpainting.demo_imgs_2 RoadAnomaly2-sample1


# python -m src.a12_inpainting.sys_segmentation gluon-psp-ctc 1204-SynthObstacleDset-v1-Ctc-train
# python -m src.a12_inpainting.sys_reconstruction pix2pixHD_405 1204-SynthObstacleDset-v1-Ctc-train


# python -m src.a12_inpainting.sys_segmentation DeepLab3W-ctc ObstacleTrack-all,LostAndFound-train,LostAndFound-test


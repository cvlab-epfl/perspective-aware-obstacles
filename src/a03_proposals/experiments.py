
from mrcnn import utils
from mrcnn import visualize

from ..pipeline import TrsChain
from ..datasets.dataset import ChannelResultImage, ChannelLoaderHDF5
from functools import partial



COCO_class_names = [
	'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
	'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	'teddy bear', 'hair drier', 'toothbrush',
]

def create_rcnn_module():
	import mrcnn.model as modellib
	from mrcnn.config import Config

	class CocoConfig(Config):
		"""Configuration for training on MS COCO.
		Derives from the base Config class and overrides values specific
		to the COCO dataset.
		"""
		# Give the configuration a recognizable name
		NAME = "coco"

		# We use a GPU with 12GB memory, which can fit two images.
		# Adjust down if you use a smaller GPU.
		IMAGES_PER_GPU = 2

		# Uncomment to train on 8 GPUs (default is 1)
		# GPU_COUNT = 8

		# Number of classes (including background)
		NUM_CLASSES = 1 + 80  # COCO has 80 classes

	class InferenceConfig(CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	cfg = InferenceConfig()

	module = modellib.MaskRCNN(
		mode="inference",
		model_dir="/cvlabdata2/home/lis/pytorch_pretrained/SharpMask_TF",
		config = cfg,
	)
	module.load_weights("/cvlabdata2/home/lis/pytorch_pretrained/MaskRCNN_TF/mask_rcnn_coco.h5", by_name=True)

	return module


def tr_rcnn_detect_(image, rcnn_module, **_):
	results = rcnn_module.detect([image])

	return {
		'mrcnn_' + k: value
		for (k, value) in results[0].items()
	}

def tr_rcnn_display(image, mrcnn_rois, mrcnn_class_ids, mrcnn_scores, mrcnn_masks, **_):
	visualize.display_instances(
		image, mrcnn_rois, mrcnn_masks, mrcnn_class_ids,
		COCO_class_names, mrcnn_scores,
	)

def get_detect_transform(rcnn_mod):
	return partial(tr_rcnn_detect_, rcnn_module=rcnn_mod)

def get_demo_transform(rcnn_mod):
	return TrsChain(
		get_detect_transform(rcnn_mod),
		tr_rcnn_display,
	)

def init_mrcnn_channels():
	path_pred_hdf5 = '{dset.dir_out}/mrcnn/mrcnn_predictions_{dset.split}.hdf5'
	make_ch = lambda fn, c=None: ChannelLoaderHDF5(
			file_path_tmpl=path_pred_hdf5,
			var_name_tmpl='{fid}/'+fn,
			compression=c,
		)

	return  dict(
		mrcnn_scores = make_ch('mrcnn_scores'),
		mrcnn_class_ids = make_ch('mrcnn_class_ids'),
		mrcnn_rois = make_ch('mrcnn_rois'),
		mrcnn_masks = make_ch('mrcnn_masks', c=7),
	)

channel_mrcnn_vis = ChannelResultImage('mrcnn/vis', suffix='_vis')
channels_mrcnn_predictions = init_mrcnn_channels()

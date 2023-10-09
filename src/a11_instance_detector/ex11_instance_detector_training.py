
import torch
from src.paths import DIR_EXP2
from easydict import EasyDict

from ..a10_geometric_instances.centerpoint_network import SpatialEmbeddingsNetwork
from ..paths import DIR_EXP2

from .dataset_Cityscapes_crops import YoloObjDset
from .YOLO.train_ex import YoloTraining, TRAIN_CFG_image


class YoloTrainingCenterpoints(YoloTraining):
	
	def __init__(self, cfg):
		super().__init__(cfg)
		
		net_dirs = cfg.cp_net_dirs
		self.cp_nets = [SpatialEmbeddingsNetwork(checkpoint_dir = d) for d in net_dirs]
		for n in self.cp_nets: n.load_from_checkpoint()
			
	def prepare_training_batch(self, batch):
		img_batch = batch['image'].cuda()
		targets_batch = batch['yolo_targets']
		
		b_size_orig = img_batch.shape[0]
		
		with torch.no_grad():
			
			inputs = []
			targets = []
			
			for i, net in enumerate(self.cp_nets):
				cp_res = net(image=img_batch)
				centerpoint_offset = cp_res['centerpoint_offset']
				centerpoint_radius = cp_res['vote_radius']
				
				inputs.append(torch.cat([
					centerpoint_offset,
					centerpoint_radius,
				], dim=1))
				
				tb = targets_batch.clone()
				tb[:, 0] += i*b_size_orig
				
				targets.append(tb)
				
			inputs = torch.cat(inputs, dim=0)
			targets = torch.cat(targets, dim=0)
		
		# print(f'i sh {inputs.shape} tsh {targets.shape}')
		
		return dict(
			inputs = inputs,
			targets = targets,
		)
	

def default_dsets():
	dset_tr = YoloObjDset(split='train', subsample=3000)

	dset_val = YoloObjDset(split='val')
	# there are 10k crops in val, subsample deterministically to ~500
	dset_val.dset.frames = dset_val.dset.frames[::20]
	# limit because this is slow
	dset_val.dset.frames = dset_val.dset.frames[:50]

	return dict(
		train = dset_tr,
		val = dset_val,
	)


def train_yolo_1101():
	dsets = default_dsets()

	cfg = EasyDict(TRAIN_CFG_image)
	cfg.batch_size = 32
	# cfg.batch_size = 1

	cfg.dir_out = DIR_EXP2 / '110_Yolo' / '01_test' / 'run2'
	
	ytr = YoloTraining(cfg)
	ytr.datasets = dsets
	ytr.yolo_training()

def train_yolo_1102():
	dsets = default_dsets()

	cfg = EasyDict(TRAIN_CFG_image)
	cfg.batch_size = 16

	cfg.cp_net_dirs = [
		DIR_EXP2 / '101_SpatialEmbeddingsNet' / '101.09A.cri_split_A' / 'run_current',
		DIR_EXP2 / '101_SpatialEmbeddingsNet' / '101.09B.cri_split_B' / 'run_current',
	]

	
	cfg.dir_out = DIR_EXP2 / '110_Yolo' / '02_cp' / 'run2'
	cfg.net_cfg_path = TRAIN_CFG_image.net_cfg_path.parent / 'yolov3_1class_4in.cfg'

	# cfg.batch_size = 1

	ytr = YoloTrainingCenterpoints(cfg)
	ytr.datasets = dsets
	ytr.yolo_training()

def train_yolo_1103():
	dsets = default_dsets()

	cfg = EasyDict(TRAIN_CFG_image)
	cfg.batch_size = 32
	
	cfg.cp_net_dirs = [
		DIR_EXP2 / '101_SpatialEmbeddingsNet' / '101.06.road_offset_zero' / 'run_2020-03-11_19:39:52',
	]

	cfg.dir_out = DIR_EXP2 / '110_Yolo' / '03_cp_all' / 'run2'
	cfg.net_cfg_path = TRAIN_CFG_image.net_cfg_path.parent / 'yolov3_1class_4in.cfg'

	cfg.batch_size = 1

	ytr = YoloTrainingCenterpoints(cfg)
	ytr.datasets = dsets
	ytr.yolo_training()


if __name__ == '__main__':
	import sys
	exps = {
		'1101': train_yolo_1101,
		'1102': train_yolo_1102,
		'1103': train_yolo_1103,
	}

	expname = sys.argv[1]
	exp_f = exps[expname]
	exp_f()

	# print('No exp', expname)

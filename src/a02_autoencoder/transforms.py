
import numpy as np
import torch
from ..pipeline.transforms import TrBase
from ..datasets.cityscapes import cityscapes_num_trainids
from ..common.geometry import rot_around_z

class TransformLabelsToOnehot(TrBase):
	def __init__(self, num_classes = cityscapes_num_trainids):
		self.num_classes = num_classes
	
	def __call__(self, labels, **_):
		sh = labels.shape

		if isinstance(labels, np.ndarray):
			idx = torch.from_numpy(labels.astype(np.int))
		else:
			idx = labels.clone()

		one_hot = torch.FloatTensor(self.num_classes, sh[0], sh[1]).zero_()
		idx.requires_grad = False
		one_hot.requires_grad = False

		ignore_mask = (idx == 255)
		idx[ignore_mask] = 0
		
		one_hot = one_hot.scatter_(0, idx.unsqueeze(0).type(torch.LongTensor), 1.)
		one_hot[0][ignore_mask] = 0

		return dict(
			labels_onehot = one_hot,
		)

def chessboard_mask(size_xy, angle, width_xy, phase_xy, fraction_black):
	grid = (np.stack(np.meshgrid(range(size_xy[0]), range(size_xy[1])), axis=2)
		.transpose([1, 0, 2])
		.reshape(-1, 2)
		.astype(np.float32))

	R = rot_around_z(angle)[:2, :2]

	R_grid = grid @ R.astype(np.float32)

	width_xy = np.array(width_xy, dtype=np.float32)
	phase_xy = np.array(phase_xy, dtype=np.float32)

	remainder = np.remainder((R_grid - phase_xy[None, :]) / width_xy[None, :], 1)

	b_or_w = np.logical_xor(remainder[:, 0] > fraction_black, remainder[:, 1] > fraction_black)

	mask = b_or_w.reshape(size_xy).transpose()

	return mask

class TransformAddChessBoard(TrBase):
	def __init__(self, width_mean_std=(220, 30), fraction_mean_std=(0.5, 0.05), angle=None):
		self.width_mean_std = width_mean_std
		self.fraction_mean_std = fraction_mean_std
		self.angle = angle
	
	def __call__(self, image, **_):
		size_xy = image.shape[:2][::-1]
		width_xy = np.random.normal(self.width_mean_std[0], self.width_mean_std[1], size=2)
		fraction = np.random.normal(self.fraction_mean_std[0], self.fraction_mean_std[1])
		phase = np.random.uniform(0, self.width_mean_std[0], size=2)
		angle = np.random.uniform(0, 2*np.pi) if self.angle is None else self.angle
				
		mask = chessboard_mask(size_xy, angle, width_xy, phase, fraction).astype(np.float32)
		
		return dict(
			passthrough_mask = mask,
		)

def tr_add_blask_mask(image, **_):
	size_yx = image.shape[:2]
	return dict(
		passthrough_mask = np.zeros(size_yx, dtype=np.float32)
	)

def chessboard_downsample_16_and_torch(passthrough_mask, **_):
	return dict(
		passthrough_mask_d16 = torch.from_numpy(passthrough_mask[::16, ::16].astype(np.float32))
	)

def chessboard_downsample_08_and_torch(passthrough_mask, **_):
	return dict(
		passthrough_mask_d08 = torch.from_numpy(passthrough_mask[::8, ::8].astype(np.float32))
	)


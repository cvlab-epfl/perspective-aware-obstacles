from functools import lru_cache
import numpy as np 

@lru_cache(4)
def get_yx_maps(height_width):
	H, W = height_width
	pos_h = np.arange(H, dtype=np.float32)
	pos_w = np.arange(W, dtype=np.float32)
	pos_h = np.tile(pos_h[:, None], [1, W])
	pos_w = np.tile(pos_w[None, :], [H, 1])
	return np.stack([pos_h, pos_w], axis=0)


import numpy as np
import cv2 as cv
import math

from ..common.jupyter_show_image import adapt_img_data
#from ..a10_geometric_instances.visualization_utils import allocate_n_colors, color_index_map

def image_montage_same_shape(imgs, num_cols=2, downsample=1, border=0, border_color=(0, 0, 0), captions=None, caption_color=(200, 128, 0), caption_size=3):
	"""
	example:
	`image_montage_same_shape(
		imgs = [
			A, B,
			C, D,
		],
		num_cols=2,
		border = 4,
		border_color = (128, 128, 128),
		captions = [
			'A', 'B',
			'C', 'D',
		],
	)`
	"""


	num_imgs = imgs.__len__()
	num_rows = int(np.ceil(num_imgs / num_cols))
	
	row_cols = np.array([num_rows, num_cols])

	img_sizes = np.array([
		img.shape[:2] for img in imgs
	], dtype=np.int32) // downsample
	
	img_size_biggest = np.max(img_sizes, axis=0) 
	img_size_with_border = img_size_biggest + border

	full_size = (num_rows * img_size_biggest[0] + (num_rows-1)*border, num_cols * img_size_biggest[1] + (num_cols-1)*border, 3)

	out = np.full(full_size, fill_value=border_color, dtype=np.uint8)

	row_col_pos = np.array([0, 0])

	for idx, img in enumerate(imgs):
		# none means black section
		if img is not None:
			#img = ensure_numpy_image(img)
			if downsample != 1:
				img = img[::downsample, ::downsample]
			
			img = adapt_img_data(img)
			if img.shape.__len__() == 2:
				img = np.tile(img[:, :, None], (1, 1, 3))

			if captions is not None and captions[idx]:
				# print('img', img.shape, img.dtype)
				caption = captions[idx]
				img = img.copy()
				# shadow
				cv.putText(img, caption, (18, 50), cv.FONT_HERSHEY_DUPLEX, caption_size / downsample, color=(0, 0, 0), thickness=math.ceil(4/downsample))
				# foreground
				cv.putText(img, caption, (18, 50), cv.FONT_HERSHEY_DUPLEX, caption_size / downsample, color=caption_color, thickness=math.ceil(2/downsample))

			img_sz = img_sizes[idx]
			tl = img_size_with_border * row_col_pos
			br = tl + img_sz

			out[tl[0]:br[0], tl[1]:br[1]] = img[:img_sz[0], :img_sz[1]]
		
		row_col_pos[1] += 1
		if row_col_pos[1] >= num_cols:
			row_col_pos[0] += 1
			row_col_pos[1] = 0

	return out
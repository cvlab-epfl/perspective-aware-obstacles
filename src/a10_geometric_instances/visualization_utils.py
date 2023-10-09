
import numpy as np
import cv2 as cv

def allocate_n_colors(n, hsv_saturation=0.5, hsv_value=1., zero_black=False, hsv=False, random=False):
	"""
	Returns uint8[n+1 x 3]
		colors[0] is black
		colors[1...n] is a selected color
	"""

	if not zero_black:
		colors_with_zero = allocate_n_colors(n=n, hsv_saturation=hsv_saturation, hsv_value=hsv_value, zero_black=True, random=random)
		return colors_with_zero[1:]
	
	else:
		hsv_saturation = np.uint8(hsv_saturation*255)
		hsv_value = np.uint8(hsv_value*255)

		if random:
			hues = np.random.uniform(0, 1, size=n)
		else: # sequential in hue
			hues = np.linspace(0, 0.8, n)

		colors = np.zeros((n+1, 3), dtype=np.uint8)		
		colors[1:, 0] = hues*255
		colors[1:, 1] = hsv_saturation
		colors[1:, 2] = hsv_value
		
		if not hsv:
			colors = cv.cvtColor(colors[:, None, :], cv.COLOR_HSV2RGB)[:, 0]

	return colors


def color_index_map(idx_map, colors):
	if isinstance(colors, int):
		colors = allocate_n_colors(colors, zero_black=True, random=True)

	idx_flat = idx_map.reshape(-1)
	is_not_zero = idx_flat != 0
	idx_color = idx_flat % (colors.shape[0]-1) + is_not_zero

	return colors[idx_color].reshape(idx_map.shape + (colors.shape[1],))

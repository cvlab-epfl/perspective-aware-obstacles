
import numpy as np
import cv2 as cv

def allocate_n_colors(n, hsv_saturation=0.5, hsv_value=1., zero_black=False, hsv=False):
	"""
	Returns uint8[n+1 x 3]
		colors[0] is black
		colors[1...n] is a selected color
	"""
	hsv_saturation = np.uint8(hsv_saturation*255)
	hsv_value = np.uint8(hsv_value*255)

	if zero_black:
		colors = np.zeros((n+1, 3), dtype=np.uint8)
		colors[1:, 0] = np.linspace(0, 0.8, n)*255
		colors[1:, 1] = hsv_saturation
		colors[1:, 2] = hsv_value
	else:
		colors = np.zeros((n, 3), dtype=np.uint8)
		colors[:, 0] = np.linspace(0, 0.8, n)*255
		colors[:, 1] = hsv_saturation
		colors[:, 2] = hsv_value

	if not hsv:
		colors = cv.cvtColor(colors[:, None, :], cv.COLOR_HSV2RGB)[:, 0]

	return colors


def color_index_map(idx_map, colors=10):
	if isinstance(colors, int):
		colors = allocate_n_colors(colors, zero_black=False)

	return colors[idx_map.reshape(-1)].reshape(idx_map.shape + (colors.shape[1],))

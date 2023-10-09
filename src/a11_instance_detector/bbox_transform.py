
import numba
import numpy as np
import cv2 as cv
from easydict import EasyDict

@numba.njit
def labels_to_bbox(labels, ignore_zero=True):
	"""
	@return
		labels_unique: label for each bbox, sorted
		bboxes: in [tl_x, tl_y, br_x, br_y] format
	"""
	h, w, = labels.shape
	
	labels_unique = np.unique(labels)
	n_labels = len(labels_unique)
	top_label = int(np.max(labels_unique))

	label_reindex = np.zeros(top_label, dtype=np.uint16)
	for idx, label in enumerate(labels_unique):
		label_reindex[label] = idx
	
	bboxes = np.zeros((n_labels, 4), dtype=np.uint16)
	bboxes[:, 0] = w
	bboxes[:, 1] = h
	
	for row in range(h):
		for col in range(w):
			label_here = labels[row, col]

			if not (label_here == 0 and ignore_zero):
				bbox_idx = label_reindex[label_here]
				
				if col < bboxes[bbox_idx, 0]:
					bboxes[bbox_idx, 0] = col
				
				if col > bboxes[bbox_idx, 2]:
					bboxes[bbox_idx, 2] = col
					
				if row < bboxes[bbox_idx, 1]:
					bboxes[bbox_idx, 1] = row
				
				if row > bboxes[bbox_idx, 3]:
					bboxes[bbox_idx, 3] = row
	
	if ignore_zero and labels_unique[0] == 0:
		return labels_unique[1:], bboxes[1:]
	else:
		return labels_unique, bboxes
	
# 	return {
# 		'labels': labels_unique,
# 		'bboxes': bboxes,
# 	}
	
# 	return dict(
# 		labels = labels_unique,
# 		bboxes = bboxes,
# 	)
	
# labels_to_bbox(dset[0].instances)

def draw_bboxes(image, bboxes, colors=None, color_default=None):
	canvas = image.copy()

	color = tuple(color_default) if color_default is not None else (255, 128, 0,)
	for idx, bbox in enumerate(bboxes.astype(np.int)):
		tl = tuple(bbox[:2])
		br = tuple(bbox[2:])

		if colors is not None:
			color = tuple(colors[idx])
		# print(color)

		cv.rectangle(canvas, tl, br, tuple(map(int, color)), thickness=2)
		
	return canvas

from matplotlib import cm
scalar_to_color = cm.get_cmap('magma')

def scores_to_colors(scores):
	return scalar_to_color(scores[:, None], bytes=True)[:, 0, :3]

# def draw_bboxes(image, bboxes):
# 	canvas = image.copy()
	
# 	for bbox in bboxes.astype(np.int):
# 		tl = tuple(bbox[:2])
# 		br = tuple(bbox[2:])
# 		cv.rectangle(canvas, tl, br, (255, 0, 0,))
		
# 	return canvas

def draw_bboxes_with_scores(image, bboxes, scores):
	colors = scores_to_colors(scores)
	return draw_bboxes(image, bboxes, colors=colors)


def demo_bboxes(image, instances, **_):
	instances = instances.copy()
	instances[instances < 1000] = 0
	
	labels_unique, bboxes = labels_to_bbox(instances)
	
	vis = draw_bboxes(image, bboxes)
	
	show(vis)
	
# dset[0].apply(demo_bboxes)


def bbox_calc_dimensions(bboxes):
	tl_x = bboxes[:, 0]
	tl_y = bboxes[:, 1]
	br_x = bboxes[:, 2] 
	br_y = bboxes[:, 3]
	width = br_x - tl_x
	height =  br_y - tl_y
	area = width * height
	#center = 0.5 * (tl_x + br_x)

	return EasyDict(
		width = width,
		height = height,
		area = area,
	)


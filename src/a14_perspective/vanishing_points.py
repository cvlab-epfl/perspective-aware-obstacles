
import numpy as np
import cv2 as cv
from easydict import EasyDict
from ..common.jupyter_show_image import show
from . import auto_rectification

#from .common.util import *
import warnings


def detect_lines(img, display=False):
	#img = cv.resize(img, None, fx=0.5, fy=0.5)

	# alg = cv.ximgproc.createFastLineDetector(
	# 	# TODO should be relative to image
	# 	21, # length threshold,
	# 	2.0, # distance threshold, big because the tile lines are thick
	# 	50, # canny threshold
	# 	50,
	# 	3,
	# 	True, # merge
	# )

	alg = cv.ximgproc.createFastLineDetector(
		10,
		1.414,
		50,
		80,
		7,
	)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	lines = alg.detect(img_gray)

	if display:
		highlight = alg.drawSegments(np.zeros(img_gray.shape, dtype=np.uint8), lines)
		highlight = highlight[:, :, 2]
		show(highlight)

		#img_edges = cv.Canny(img_gray, 50, 150,apertureSize = 3)
		#show(highlight, img_edges)

	return lines

def detect_lines_hough(img, thr = 500):
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img_edges = cv.Canny(img_gray, 50, 150,apertureSize = 3)

	show(255-img_edges)

	lines_h = cv.HoughLines(img_edges, 1, np.pi/180, thr)

	return lines_h


def segments_to_edgelets(lines):
	"""
		Convert lines in format Nx4 [x1, y1, x2, y2]
		to (positions, directions, strengths) where
			positions : Nx2 = centers of lines
			directions : Nx2 = length-1 vector along line
			strengths : N = lengths
	"""

	#(x1, y1, x2, y2)
	lines = lines.reshape(-1, 2, 2)

	positions = (lines[:, 0, :] + lines[:, 1, :]).astype(np.float64)*0.5

	line_vectors = lines[:, 1, :] - lines[:, 0, :]
	lengths = np.linalg.norm(line_vectors, axis=1)

	directions = line_vectors / lengths.reshape(-1, 1)
	directions *= -1

	return (positions.astype(np.float64), directions.astype(np.float64), lengths.astype(np.float64))

# def find_vanishing_points(img, display=True, display_lines=False):
# 	#edgelets1 = rectification.compute_edgelets(img)
# 	#print([e.shape for e in edgelets1])
# 	#rectification.vis_edgelets(img, edgelets1)

# 	RANSAC_ITERS = 1500
# 	RANSAC_THRESHOLD_DEG = 3

# 	segments = fast_detect(img, display=display_lines)
# 	edgelets1 = segments_to_edgelets(segments)

# 	vp1 = rectification.ransac_vanishing_point(
# 		edgelets1,
# 		num_ransac_iter=RANSAC_ITERS,
# 		threshold_inlier=RANSAC_THRESHOLD_DEG,
# 	)
# 	vp1 = rectification.reestimate_model(vp1, edgelets1, threshold_reestimate=5)

# 	if display:
# 		rectification.vis_model(img, vp1) # Visualize the vanishing point model

# 	edgelets2 = rectification.remove_inliers(vp1, edgelets1, 10)
# 	vp2 = rectification.ransac_vanishing_point(
# 		edgelets2,
# 		num_ransac_iter=RANSAC_ITERS,
# 		threshold_inlier=RANSAC_THRESHOLD_DEG,
# 	)
# 	vp2 = rectification.reestimate_model(vp2, edgelets2, threshold_reestimate=5)

# 	if display:
# 		rectification.vis_model(img, vp2) # Visualize the vanishing point model

# 	edgelets3 = rectification.remove_inliers(vp2, edgelets2, 10)
# 	vp3 = rectification.ransac_vanishing_point(
# 		edgelets3,
# 		num_ransac_iter=RANSAC_ITERS,
# 		threshold_inlier=RANSAC_THRESHOLD_DEG,
# 	)
# 	vp3 = rectification.reestimate_model(vp3, edgelets3, threshold_reestimate=5)

# 	if display:
# 		rectification.vis_model(img, vp3) # Visualize the vanishing point model

# 	return np.stack([v[:2] for v in vpts])



# def find_vanishing_points_street(img, segments=None, display=True, save=None, display_lines=False, RANSAC_ITERS=100, RANSAC_THRESHOLD_DEG=3):
# 	RANSAC_ITERS = 1500
# 	RANSAC_THRESHOLD_DEG = 4.5


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


def draw_vp_lines(image, vps, edgelets, show=True, stride=4, save=None, angle_thr=7.5, highlight_mask = None):
	
	num_vps = vps.__len__()
	canvas = image.copy()
	colors = allocate_n_colors(num_vps)
		
	highlight_mask = highlight_mask if highlight_mask is not None else np.zeros(num_vps, dtype=bool)

	locations, directions, strengths = edgelets

	for vp, color, hg in zip(vps, colors, highlight_mask):
		color = tuple(map(int, color))
		thickness = 3 if hg else 1
		stride_here = max(1, stride // 2) if hg else stride

		inliers = auto_rectification.compute_votes(edgelets, vp, angle_thr) > 0
		pts = locations[inliers]
		pts = pts[::stride_here]

		vp_cv = to_cv_pt(vp)
		
		for pt in pts:
			canvas = cv.line(canvas, to_cv_pt(pt), vp_cv, color, thickness)
		
	for vp, color in zip(vps, colors):
		color = tuple(map(int, color))
		canvas = cv.circle(canvas, to_cv_pt(vp), 8, color, -1)
		canvas = cv.circle(canvas, to_cv_pt(vp), 8, (0, 0, 0), 2)
		
	return canvas



def attractor_choose(img_shape):
	h, w = img_shape[:2]

	return np.array([
		w * 1/2, h * (1/4),
	], dtype=np.float32)


def attractor_weighted_distance(attractor, vp):
	weight = np.array([0.15, 1], dtype=np.float32)
	dist = np.linalg.norm((attractor-vp[:2])*weight)
	return dist


def find_vanishing_points(segments=None, RANSAC_ITERS=100, RANSAC_THRESHOLD_DEG=3, max_num_vps=3, display=True, save=None, img=None, attr_distance_thr = 512):

	if segments is None:
		segments = detect_lines(img, display=display)

	edgelets_all = segments_to_edgelets(segments)
	edgelets = edgelets_all

	vps = []
	votes = []
	num_inliers = []
	attractor_distances = []

	attractor = attractor_choose(img.shape)

	with warnings.catch_warnings():
	# 	warnings.simplefilter('ignore')

		while vps.__len__() < max_num_vps and edgelets[0].__len__() > 1:
			vp = auto_rectification.ransac_vanishing_point(
				edgelets = edgelets,
				num_ransac_iter=RANSAC_ITERS,
				threshold_inlier=RANSAC_THRESHOLD_DEG,
				b_small_sample = edgelets[0].__len__() < 10,
			)

			vp = auto_rectification.reestimate_model(vp, edgelets, threshold_reestimate=RANSAC_THRESHOLD_DEG)

			inliers = np.where(auto_rectification.compute_votes(edgelets, vp, RANSAC_THRESHOLD_DEG) > 0)[0]

			edgelets = auto_rectification.remove_inliers(vp, edgelets, RANSAC_THRESHOLD_DEG)

			vps.append(vp)
			num_inliers.append(inliers.shape[0])
			votes.append(inliers)

			attr_dist = attractor_weighted_distance(attractor, vp)
			attractor_distances.append(attr_dist)

			# if vp[1] < img.shape[0]: # above bottom of img
			# 	vps_good.append(vp)
			# 	to_find -= 1

			# else:
			# 	vps_bad.append(vp)

	num_inliers = np.array(num_inliers, dtype=np.uint32)

	desc_order = np.argsort(num_inliers)[::-1]
	
	vps = np.array(vps)[desc_order]
	num_inliers = num_inliers[desc_order]
	attractor_distances = np.array(attractor_distances)[desc_order]

	#highlight_mask = attractor_distances < attr_distance_thr
	best_idx = np.argmin(attractor_distances)
	highlight_mask = np.zeros(vps.__len__(), dtype=bool)
	
	print(f"Central Vp at {num_inliers[best_idx]} votes")

	if attractor_distances[best_idx] < attr_distance_thr:
		highlight_mask[best_idx] = True

	if (display or save) and img is not None:
		demo_img = draw_vp_lines(
			img, vps, edgelets_all, angle_thr = RANSAC_THRESHOLD_DEG,
			highlight_mask = highlight_mask,
		)
		show(demo_img)

		# auto_rectification.vis_model_multi(
		# 	img,
		# 	vps,
		# 	stride=1,
		# 	edgelets = edgelets_all,
		# 	show = display,
		# 	save = save, #save + '_{n}.jpg'.format(n=vp_num) if save else None
		# )

	

	return EasyDict(
		vps = vps, 
		num_inliers = num_inliers,
		best_idx = best_idx,
		# votes = votes,
		angle_thr = RANSAC_THRESHOLD_DEG,
		attractor_distances = attractor_distances,
	)







def find_vanishing_points_street(img, segments=None, display=True, save=None, display_lines=False):
	RANSAC_ITERS = 1500
	RANSAC_THRESHOLD_DEG = 4.5
	tries = 8
	to_find = 2

	if segments is None:
		segments = detect_lines(img, display=display_lines)

	edgelets_all = segments_to_edgelets(segments)
	edgelets = edgelets_all

	vps_good = []
	vps_bad = []
	vp_num = 0

	# with warnings.catch_warnings():
	# 	warnings.simplefilter('ignore')

	while to_find > 0 and tries > 0:
		vp = ransac_vanishing_point(
			edgelets,
			num_ransac_iter=RANSAC_ITERS,
			threshold_inlier=RANSAC_THRESHOLD_DEG,
		)
		vp = reestimate_model(vp, edgelets, threshold_reestimate=RANSAC_THRESHOLD_DEG)

		if vp[1] < img.shape[0]: # above bottom of img
			vps_good.append(vp)
			to_find -= 1

		else:
			vps_bad.append(vp)

		edgelets = remove_inliers(vp, edgelets, 10)
		tries -= 1
		vp_num += 1

	if display or save:
		vis_model_multi(
			img,
			vps_good,
			edgelets = edgelets_all,
			show = display,
			save = save, #save + '_{n}.jpg'.format(n=vp_num) if save else None
			stride = 1,
		)

	if display:
		print(f'Vanishing points: {len(vps_good)} good, {len(vps_bad)} bad')

	return np.stack([v[:2] for v in vps_good])


def to_cv_pt(pt):
	return (int(pt[0]), int(pt[1]))

def vanishing_points_to_scale_gradient(vpts, display_img=None, display=False):
	# find vanishing poitns of the street -> those above y=bottom line
	# we choose lowest y
	ysort = np.argsort(vpts[:, 1])
	vpts_street = vpts[ysort[:2]]

	# horizon line
	horizon_direction = vpts_street[1, :] - vpts_street[0, :]
	horizon_direction /= np.linalg.norm(horizon_direction)

	horizon_normal = np.array([-horizon_direction[1], horizon_direction[0]])

	# intercept w0 for linear model:
	# 	s = w0 + w1 * horizon_normal dot [x, y]
	# we know that s(vanishing point) = 0
	# 	0 = w0 + w1 * horizon_normal dot [x, y]
	# 	0 = w1 * (w0/w1 + 1 * horizon_normal dot [x, y])

	# print('vpts_street', vpts_street)

	# w0/w1 = - horizon_normal dot [x, y]
	intercept_relative = - horizon_normal @ vpts_street[0, :].reshape(-1, 1)

	if display_img is not None:
		# to_cv_pt(vpts_street[0, :] + horizon_direction * 1000),
		# to_cv_pt(vpts_street[0, :] - horizon_direction * 1000),
		cv.line(
			display_img,
			to_cv_pt(vpts_street[0, :]),
			to_cv_pt(vpts_street[1, :]),
			(255, 255, 0),
			2,
		)

		if display:
			show(display_img[:, :, ::-1])

	return horizon_normal, intercept_relative

def draw_hough_lines(img, lines):

	for line in lines:
		(radius, angle) = line[0]

		line_normal = np.array([np.cos(angle), np.sin(angle)])
		line_direction = np.array([-line_normal[1], line_normal[0]])
		line_center = line_normal * radius
		pt1 = line_center - line_direction * 1000
		pt2 = line_center + line_direction * 1000

		img = cv.line(img, to_cv_pt(pt1), to_cv_pt(pt2), (0, 255, 255), 2)

	return img





import numpy as np
from ..common.jupyter_show_image import show, adapt_img_data
from functools import lru_cache, partial
import numba, einops
from scipy.spatial import cKDTree
import cv2 as cv
from .visualization_utils import allocate_n_colors, color_index_map

@lru_cache()
def coord_map(shape, shorter_edge=None):
	h, w = shape
	shorter_edge = shorter_edge or min(h, w)
	
	axes = [
		np.linspace(0, w / shorter_edge, w)[None, :],
		np.linspace(0, h / shorter_edge, h)[:, None],
	]
	bc = np.broadcast_arrays(*axes)
	return np.stack(bc, axis=0)

@lru_cache()
def coord_map_torch(shape, shorter_edge=None):
	return torch.from_numpy(coord_map(shape, shorter_edge=shorter_edge)).cuda()

def coord_map_test():
	cmap = coord_map((128, 256))
	print(cmap.shape)
	show([cmap[0], cmap[1]])


from matplotlib.colors import hsv_to_rgb

def tr_vis_centerpoint_offset(centerpoint_offset, MAX_LENGTH = 0.2, **_):
	
	_, h, w = centerpoint_offset.shape
	
	vis_hsv = np.ones((h, w, 3), dtype=np.float32)
	
	offset_length = np.sqrt(np.sum(centerpoint_offset * centerpoint_offset, axis=0))	
	offset_angle = np.arctan2(centerpoint_offset[1], centerpoint_offset[0])
	
	vis_hsv[:, :, 0] = offset_angle * (0.5/np.pi) + 0.5
	vis_hsv[:, :, 1] = np.minimum(offset_length * (1/MAX_LENGTH), 1.)
	
	vis_rgb = hsv_to_rgb(vis_hsv)
	
	vis_rgb = (vis_rgb * 255).astype(np.uint8)
	
	return dict(
		vis_offset = vis_rgb,
	)
	
def tr_centerpoint_calc_centers(centerpoint_offset, **_):
	_, h, w = centerpoint_offset.shape
	
	if isinstance(centerpoint_offset, np.ndarray):
		centerpoint_target = centerpoint_offset + coord_map((h, w))
	else:
		centerpoint_target = centerpoint_offset + coord_map_torch((h, w))
		
	# centerpoint_target = centerpoint_offset + coord_map((h, w), shorter_edge=1024)
	
	return dict(
		centerpoint_target = centerpoint_target,
	)

SIGMA_FACTOR_ADD = float(0.5 * np.log(np.log(2)))
def sigma_to_radius(sigma):
    return np.exp(-5 * sigma + SIGMA_FACTOR_ADD)

def tr_centerpoint_resolve_sigma(centerpoint_sigma, **_):
	return dict(
		centerpoint_cluster_radius = sigma_to_radius(centerpoint_sigma),
	)

def tr_centerpoint_resolve_sigma_torch(centerpoint_sigma, **_):
	return dict(
		centerpoint_cluster_radius = torch.exp(-5 * centerpoint_sigma + SIGMA_FACTOR_ADD),
	)



@numba.njit(nogil=True)
def num__count_votes(vote_map_int, counts):
# 	_, h, w = vote_map_int.shape
	h, w = counts.shape
	
	h = np.uint32(h)
	w = np.uint32(w)
	
	vote_map_int = vote_map_int.reshape(2, -1).astype(np.uint32)
	
	for iv in range(vote_map_int.shape[1]):
		vote_x = vote_map_int[0, iv]
		vote_y = vote_map_int[1, iv]
		
		if 0 <= vote_x and vote_x < w and 0 <= vote_y and vote_y < h:
			counts[vote_y, vote_x] += 1


def tr_plot_votes(centerpoint_target, **_):
	_, h, w = centerpoint_target.shape
	
	vote_counts = np.zeros((h, w), dtype=np.uint32)
	vote_counts_quarter = np.zeros((h//4, w//4), dtype=np.uint32)
	
	centerpoint_target = np.rint(centerpoint_target * min(h, w)).astype(np.uint32)
	
	num__count_votes(centerpoint_target, vote_counts)
	
	vote_counts_quarter = np.zeros((h//4, w//4), dtype=np.uint32)
	num__count_votes(centerpoint_target // 4, vote_counts_quarter)
	
	return dict(
		vote_counts_per_pix = vote_counts,
		vote_counts_per_pix_q = vote_counts_quarter,
		vote_heatmap = 1 - np.minimum(vote_counts, 100) * (1/100),
	)


def tr_find_clusters__naive(vote_counts_per_pix, centerpoint_target, **_):
	h, w = vote_counts_per_pix.shape
	xy = coord_map(centerpoint_target.shape[1:]).reshape(2, -1).transpose()
	vote_counts_flat = vote_counts_per_pix.reshape(-1)
	vote_index_sorted = np.argsort(vote_counts_flat)[::-1]
	
# 	print(vote_index_sorted.shape, vote_index_sorted)
# 	print('best votes', vote_counts_flat[vote_index_sorted[:25]])
# 	print('best votes', xy[vote_index_sorted[:25]])

	to_flat = partial(einops.rearrange, pattern='dim h w -> (h w) dim')
	target_points = to_flat(centerpoint_target)
	target_points_tree = cKDTree(target_points)
	
	cluster_idx_current = 1
	cluster_idx_map = np.zeros_like(vote_counts_flat)
	cluster_centers = []
	
	NUM_CLUSTERS = 10
	RADIUS = 0.1
	
	for vote_winner_idx in vote_index_sorted:
		vote_winner_position = np.array(xy[vote_winner_idx])
	
		if cluster_idx_map[vote_winner_idx] == 0:


			winner_followers = target_points_tree.query_ball_point(vote_winner_position, RADIUS, n_jobs=4)
			winner_followers = np.array(winner_followers)

			cluster_idx_map[winner_followers] = cluster_idx_current
			cluster_idx_current += 1
			cluster_center = np.mean(target_points[winner_followers], axis=0)
			cluster_centers.append(cluster_center)
# 			print(f'cluster sized {winner_followers.shape[0]} center {cluster_center}')
		
			
# 			print(f'{vote_winner_position} already has cluster {cluster_idx_map[vote_winner_idx]}')
	
			if cluster_idx_current > NUM_CLUSTERS:
				break

	cluster_idx_map_2d = cluster_idx_map.reshape((h, w))
				
# 	show(cluster_idx_map_2d)
	return dict(
		cluster_centers = np.array(cluster_centers),
		cluster_idx_map = cluster_idx_map_2d,
	)
	

def fix_umat(a):
	return a.get() if isinstance(a, cv.UMat) else a
	

"""
exp(- dx**2 / 2 sigma ** 2) > 0.5
- dx**2 / 2 sigma ** 2 > ln(0.5)
dx**2 / 2 sigma**2 < -ln(0.5)
dx**2 < - 2 sigma**2 ln(0.5)
|dx| < sigma sqrt(-2 ln(0.5))
"""
SIGMA_TO_RADIUS = np.sqrt(-2*np.log(0.5))


def tr_find_clusters__blur1(vote_counts_per_pix, centerpoint_target, centerpoint_sigma, vote_heatmap, **_):
	NUM_CLUSTERS = 35
	RADIUS = 0.025
	# MIN_CLUSTER_VOTE_COUNT = 1000
	MIN_QUALITY = 62
	BLUR_KERNEL_SIZE = 15

	h, w = vote_counts_per_pix.shape
	side = min(h, w)
	xy = coord_map(centerpoint_target.shape[1:]).reshape(2, -1).transpose()
# 	vote_index_sorted = np.argsort(vote_counts_flat)[::-1]
	
# 	print(vote_index_sorted.shape, vote_index_sorted)
# 	print('best votes', vote_counts_flat[vote_index_sorted[:25]])
# 	print('best votes', xy[vote_index_sorted[:25]])



	
	vote_counts_per_pix_float = vote_counts_per_pix.astype(np.float32)
	ks = BLUR_KERNEL_SIZE
	votes_blurred = cv.GaussianBlur(vote_counts_per_pix_float, (ks, ks), ks // 3)
	
	vote_counts_flat = votes_blurred.reshape(-1)
	vote_index_sorted = np.argsort(vote_counts_flat)[::-1]
	
	target_points_flat = einops.rearrange(centerpoint_target, 'dim h w -> (h w) dim')
	target_points_tree = cKDTree(target_points_flat)
	
	# sigmas_flat = einops.rearrange(centerpoint_sigma, 'h w -> (h w)')
	
	cluster_idx_current = 1
	cluster_idx_map = np.zeros_like(vote_counts_flat, dtype=np.int)
	cluster_centers = []
	cluster_r = []

	def refine_cluster(cluster_center_position, radius):

		# find followers
		winner_followers = target_points_tree.query_ball_point(cluster_center_position, radius, n_jobs=8)
		winner_followers = np.array(winner_followers)

		num_followers = winner_followers.__len__()

		# find center and size
		follower_votes = target_points_flat[winner_followers]
		
		new_center = np.mean(target_points_flat[winner_followers], axis=0)

		incoherence = np.linalg.norm(follower_votes - new_center[None, :], axis=0)
		incoherence_sum = np.sum(incoherence)

		quality = np.sqrt(num_followers)/incoherence_sum
		# quality = 1/np.mean(incoherence)

		# new_sigma = np.mean(sigmas_flat[winner_followers], axis=0)
		# new_radius = SIGMA_TO_RADIUS * new_sigma
		
		new_radius = RADIUS
		# new_radius = RADIUS

		# print(f'num {num_followers}, quality {quality}')

		return (new_center, new_radius, winner_followers, quality)

	
	cluster_drawing = np.zeros(vote_counts_per_pix.shape + (3,), dtype=np.uint8)
	
	def draw_cluster(center, radius, color=(0, 255, 0), th=2):
		scale = 1024
		center_cv = tuple((scale*center).astype(np.int))
		r_cv = int(radius*scale)
		cv.circle(cluster_drawing, center=center_cv, radius=r_cv, color=color, thickness=th)
		
	num_checked = 0
	for vote_winner_idx in vote_index_sorted:

		vote_winner_position = np.array(xy[vote_winner_idx])
	
		if cluster_idx_map[vote_winner_idx] == 0:
			
			center = vote_winner_position
			radius = RADIUS
			# print(f'c {vote_winner_position}')
			center, radius, followers, quality = refine_cluster(center, radius)
			center, radius, followers, quality = refine_cluster(center, radius)
			
# 			draw_cluster(center, radius)
# 			draw_cluster(center, RADIUS, color=(0, 100, 0))
			
			followers = followers[cluster_idx_map[followers] == 0]
			
			num_followers = followers.__len__()
# 			print(num_followers, radius, center)
			
			if quality > MIN_QUALITY:
	# 			print(f'blur value {vote_counts_flat[vote_winner_idx]}, followers from kdtree {winner_followers.__len__()}')

				cluster_idx_map[followers] = cluster_idx_current
				cluster_idx_current += 1
	# 			cluster_center = np.mean(target_points[winner_followers], axis=0)
				cluster_centers.append(center)
				cluster_r.append(radius)
	# 			print(f'cluster sized {winner_followers.shape[0]} center {cluster_center}')
			else:
				cluster_idx_map[followers] = -1
				draw_cluster(center, radius, color=(100, 100, 100))
			
			num_checked += 1
		
	
			
			if num_checked > NUM_CLUSTERS:
				break

	cluster_centers = np.array(cluster_centers)



	colors = allocate_n_colors(cluster_idx_current)
	
	for cl_center, cl_radius, color in zip(cluster_centers, cluster_r, colors):
		color_cv = tuple(map(int, color))
		draw_cluster(cl_center, radius, color=color_cv, th=4)
	
	
	vthr = 100
	vote_heatmap_img = adapt_img_data( 1-np.clip(votes_blurred * (1/vthr), 0, 1) )
	mask = cluster_drawing > 10
	vote_heatmap_img[mask] = cluster_drawing[mask]
	# vote_heatmap_with_clusters_img = np.maximum(cluster_drawing, vote_heatmap_img)
	vote_heatmap_with_clusters_img = vote_heatmap_img

	# vote_heatmap_with_clusters_img = 0.5*vote_heatmap_img + 0.5*


	cluster_idx_map[cluster_idx_map < 0] = 0
	

	cluster_idx_map_2d = einops.rearrange(cluster_idx_map, '(h w) -> h w', h=centerpoint_target.shape[1])
	
	return dict(
		cluster_centers = cluster_centers,
		cluster_idx_map = cluster_idx_map_2d,
		vote_heatmap_with_clusters_img = vote_heatmap_with_clusters_img,
	)

def tr_discontinuity_blur(centerpoint_offset, **_):
	
	BLUR_KERNEL_SIZE = 31
	
	coff_x = centerpoint_offset[0]
	coff_y = centerpoint_offset[1]
	
	ks = BLUR_KERNEL_SIZE
	coff_x_blurred, coff_y_blurred = (cv.GaussianBlur(coff, (ks, ks), ks // 3) for coff in (coff_x, coff_y))

	coff_x_abs_blurred, coff_y_abs_blurred = (cv.GaussianBlur(np.abs(coff), (ks, ks), ks // 3) for coff in (coff_x, coff_y))

	def off_len(x, y):
		return np.sqrt(x*x + y*y)
	
	coff_blurred_len = off_len(coff_x_blurred, coff_y_blurred)
	coff_abs_blurred_len = off_len(coff_x_abs_blurred, coff_y_abs_blurred)
	
	
	# coherence = 

# 	show([coff_blurred_len, coff_abs_blurred_len, coff_abs_blurred_len - coff_blurred_len])
	
	return dict(
		discontinuity = coff_abs_blurred_len - coff_blurred_len,
	)
	


def draw_cluster_map(image_np, cluster_centers, cluster_idx_map, **_):
	h, w = cluster_idx_map.shape
	num_clusters = cluster_centers.shape[0]

	colors_hsv = allocate_n_colors(num_clusters, zero_black=True, hsv=True)

	cluster_img_hsv = np.zeros((h, w, 3), dtype=np.uint8)

	hues = np.linspace(0, 0.8, cluster_centers.shape[0])

	cluster_img_hsv = color_index_map(cluster_idx_map, colors=colors_hsv)
	
	for center, color in zip(cluster_centers, colors_hsv[1:]):
		cpix = tuple(np.rint(center * min(h, w)).astype(np.int))
		hue = color[0]
		cv.drawMarker(cluster_img_hsv, cpix, (int(hue), 255, 255), cv.MARKER_CROSS, 30, 2)
					  

	cluster_img_rgb = cv.cvtColor(cluster_img_hsv, cv.COLOR_HSV2RGB)
	cluster_overlay = cv.addWeighted(image_np, 0.5, cluster_img_rgb, 0.5, 0)
	
	return dict(
		cluster_img_rgb = cluster_img_rgb,
		cluster_overlay = cluster_overlay,
	)

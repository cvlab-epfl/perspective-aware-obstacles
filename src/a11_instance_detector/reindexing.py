
import numpy as np
import cv2 as cv

def reindex(idx_map):
	idx_unique = np.unique(idx_map)
	idx_max = idx_unique[-1]
	num_unique = idx_unique.__len__()

	table = np.zeros(idx_max+1, dtype=np.uint16)
	table[idx_unique] = np.arange(num_unique) + 1
		
	return table[idx_map.reshape(-1)].reshape(idx_map.shape)

def instances_from_connected_components(inst_map, class_id_list):
	
	inst_map_out = inst_map.copy()
	
	for id in class_id_list:
		mask = inst_map == id
	
		num_inst, labels, stats, centroids = cv.connectedComponentsWithStats(
			mask.astype(np.uint8),
			connectivity = 4,
		)
		
		inst_map_out[mask] = labels[mask].astype(np.uint32) + (1000*id)
	
	return inst_map_out


def classid_list_to_ranges(classes_selected):
	"""
	Combine selected classes into consecutive ranges
	[24, 25, 26, 27, 28, 31, 32, 33] --> [(24, 29), (31, 34)]
	[19, 20, 24, 25, 26, 27, 28, 31, 32, 33] --> [(19, 21), (24, 29), (31, 34)]
	"""
	classes_selected = list(classes_selected)
	classes_selected.sort()
	ranges = []
	range_start = classes_selected[0]
	prev_value = classes_selected[0]
	
	for cid in classes_selected[1:]:
		if cid == prev_value + 1:
			pass # its continuing the range
		else:
			ranges.append((range_start, prev_value+1))
			range_start = cid
		prev_value = cid
	
	ranges.append((range_start, cid+1))
	return ranges


def classmap_test_in_ranges(class_map, ranges):
	rl, rh = ranges[0]
	mask = (rl <= class_map) & (class_map < rh)
	for rl, rh in ranges[1:]:
		mask |= (rl <= class_map) & (class_map < rh)

	return mask

def reindex_instances(inst_map, class_ids_base, class_ids_extra=[]):
	
	if class_ids_extra.__len__():
		inst_map = instances_from_connected_components(inst_map, class_ids_extra)
	
	inst_class = inst_map // 1000

	bg_mask = inst_class == 0
	class_map = inst_class.copy()
	class_map[bg_mask] = inst_map[bg_mask]
	

	# combine selected classes into consecutive ranges
	# [24, 25, 26, 27, 28, 31, 32, 33] --> [(24, 29), (31, 34)]
	# [19, 20, 24, 25, 26, 27, 28, 31, 32, 33] --> [(19, 21), (24, 29), (31, 34)]
	ranges = classid_list_to_ranges(list(class_ids_base) + list(class_ids_extra))

# 	print(classes_selected, ranges)
	
	mask = classmap_test_in_ranges(inst_class, ranges)
	inst_map_reindex = np.zeros_like(inst_map, dtype=np.uint8)

	if np.count_nonzero(mask):
		inst_map_masked = inst_map[mask]
		inst_map_masked_reindex = reindex(inst_map_masked)

		inst_map_reindex[mask] = inst_map_masked_reindex
	
	# label_map = inst_class.astype(np.uint8) * mask
	# print('ranges', ranges, 'area', np.count_nonzero(mask), '\nunique inst map', np.unique(inst_map_reindex), 'class', np.unique(class_map), 'masked', np.unique(inst_map[mask]))

	return dict(
		instance_map = inst_map_reindex,
		class_map = class_map,
	)

# reindex_instances(tfr.instances, (24, 25, 26, 27, 28, 31, 32, 33), (19, 20))
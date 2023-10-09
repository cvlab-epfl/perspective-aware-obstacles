from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

from ..paths import DIR_DATA
from ..common.jupyter_show_image import show



def average_distance(disparity_raw, focal=1., baseline=1.):
	# To obtain the disparity values, compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256., while a value p = 0 is an invalid measurement. 

	# filter out invalid
	disparity = disparity_raw[disparity_raw > 0]
	
	# d = ( float(p) - 1. ) / 256.
	disparity = (disparity - 1.) * (1./256.)
	
	distance = (baseline*focal) / disparity
	
	return np.mean(distance)
	

	
def instance_distance_dev(tfr):
	
	instances_ids = np.unique(tfr.instances[tfr.instances >= 2000])
	
	obstacle_avg_distance = []
	obstacle_area = []
	obstacle_id = []
		
	for inst_id in instances_ids:
		mask = tfr.instances == inst_id
		disparity_raw_crop = tfr.disparity[mask]
		avg_distance = average_distance(disparity_raw_crop, focal = 2268.36, baseline = 0.222126)
	
		obstacle_id.append(inst_id)
		obstacle_avg_distance.append(avg_distance)
		obstacle_area.append(np.count_nonzero(mask))
	
	return obstacle_id, obstacle_avg_distance, obstacle_area

	
def instance_distance_calc_and_save(dset):
	obstacle_avg_distance = []
	obstacle_id = []
	obstacle_fr = []
	obstacle_area = []

	out_path = DIR_DATA / 'obstacle_avg_distance' / f'{dset.name}_{dset.split}.hdf5'
	print(out_path)
	
	
	dset.set_channels_enabled('instances', 'disparity')
	
	for fri in tqdm(range(dset.__len__())):
		fr = dset[fri]
		obs_id, obs_d, obs_area = instance_distance_dev(fr)
		obstacle_fr += [fri] * obs_id.__len__()
		obstacle_id += obs_id
		obstacle_avg_distance += obs_d
		obstacle_area += obs_area
		
	out_path.parent.mkdir(exist_ok=True, parents=True)
	
	with h5py.File(out_path, 'w') as fout:
		fout['obstacle_frame_number'] =  np.array(obstacle_fr, dtype=np.uint16)
		fout['obstacle_id'] = np.array(obstacle_id, dtype=np.uint32)
		fout['obstacle_average_distance'] = np.array(obstacle_avg_distance, dtype=np.float32)
		fout['obstacle_area'] = np.array(obstacle_area, dtype=np.uint32)
	

	
def instance_distance_load(dset):
	out_path = DIR_DATA / 'obstacle_avg_distance' / f'{dset.name}_{dset.split}.hdf5'
	
	with h5py.File(out_path, 'r') as f_in:
		res = {
			name: value[:]
			for name, value
			in f_in.items()
		}
		
	ad = res['obstacle_average_distance']
	ad[ad > 200] = 100
	return res


def main():
	from .lost_and_found import DatasetLostAndFound

	for split in ['test', 'train']:
		dset_laf = DatasetLostAndFound(split=split, only_interesting=False)
		dset_laf.discover()
		instance_distance_calc_and_save(dset_laf)
	
if __name__ == '__main__':
	main()

# obs_dist_info = instance_distance_load(dset_laf)
	
# plt.hist(obs_dist_info['obstacle_average_distance'])
	

from pathlib import Path

from PIL import Image

def imread(path):
	return np.asarray(Image.open(path))

IMWRITE_OPTS = dict(
	webp = dict(quality = 90),
)

def imwrite(path, data, create_parent_dir=True, opts=None):
	# TODO option to write in background thread

	path = Path(path)
	if create_parent_dir:
		path.parent.mkdir(exist_ok=True, parents=True)
	
	# log.info(f'write {path}')

	try:
		opts_effective = IMWRITE_OPTS.get(path.suffix.lower()[1:], {})
		if opts is not None:
			opts_effective.update(opts)

		Image.fromarray(data).save(
			path, 
			**opts_effective,
		)
	except Exception as e:
		log.exception(f'Saving {path}')


import h5py
import numpy as np
from easydict import EasyDict

def hdf5_write_hierarchy_to_group(group, hierarchy):
	for name, value in hierarchy.items():
		# sub-dict
		if isinstance(value, dict):
			hdf5_write_hierarchy_to_group(
				group = group.create_group(name), 
				hierarchy = value
			)
		# label or single value
		elif isinstance(value, (str, bytes, float, int)):
			group.attrs[name] = value
		# ndarray
		elif isinstance(value, np.ndarray):
			group[name] = value
		else:
			raise TypeError(f'Failed to write type {type(value)} to hdf: {name}={value}')
			
def hdf5_write_hierarchy_to_file(path, hierarchy, create_parent_dir=True):
	if create_parent_dir:
		path = Path(path)
		path.parent.mkdir(exist_ok=True, parents=True)

	with h5py.File(path, 'w') as f:
		hdf5_write_hierarchy_to_group(f, hierarchy)
	
def hdf5_read_hierarchy_from_group(group):
	return EasyDict(
		# label or single value
		**group.attrs,
		# numeric arrays
		**{
			name: hdf5_read_hierarchy_from_group(value) 
			if isinstance(value, h5py.Group) else value[()]
			for name, value in group.items()
		}
	)
	
def hdf5_read_hierarchy_from_file(path):
	with h5py.File(path, 'r') as f:
		return hdf5_read_hierarchy_from_group(f)
		
		
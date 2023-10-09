
from pathlib import Path
import multiprocessing
from functools import partial

import json
import logging
import click
import numpy as np
from tqdm import tqdm
from PIL import Image as PIL_Image

log = logging.getLogger('reindex_labels')
log.setLevel(logging.DEBUG)

def imread(path):
	return np.asarray(PIL_Image.open(path))

def imwrite(path, data):
	path = Path(path)
	path.parent.mkdir(exist_ok=True, parents=True)
	PIL_Image.fromarray(data).save(path)

def mapping_to_table(map_json):
	
	map_dict = json.loads(map_json)

	table = np.arange(0, 256, dtype=np.uint8)

	for key, val in map_dict.items():
		table[int(key)] = int(val)

	return table

def worker(table, path_src_and_path_dest):
	path_src, path_dest = path_src_and_path_dest
	labels_src = imread(path_src)

	labels_reindex = table[labels_src.reshape(-1)].reshape(labels_src.shape)

	imwrite(path_dest, labels_reindex)



@click.command()
@click.argument('src_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('dest_dir', type=click.Path())
@click.option('--map', type=str, help='JSON mapping {"0":255,"1":0,"2":1}')
def main(src_dir, dest_dir, map):
	src_dir = Path(src_dir)
	dest_dir = Path(dest_dir)

	table = mapping_to_table(map)
	worker_func = partial(worker, table)

	paths_src = list(src_dir.glob('*_labels_semantic.png'))
	paths_out = [dest_dir / p.name for p in paths_src]
	tasks = list(zip(paths_src, paths_out))

	with multiprocessing.Pool() as pool:
		it = pool.imap_unordered(worker_func, tasks, chunksize=8)
		for _ in tqdm(it, total=paths_src.__len__()):
			...


if __name__ == '__main__':
	main()

# python reindex_labels.py labels_masks_oldid labels_masks_newid
# python ~/dev/phd/unknown-dangers/src/datasets/conversion_tools/RoadObstacle2048/reindex_labels.py labels_masks_oldid labels_masks_newid --map {"0":255,"1":0,"2":1}

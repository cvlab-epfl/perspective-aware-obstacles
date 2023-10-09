#!/usr/bin/env python

import click
import numpy as np
import h5py
import os

def opt_mask_file(fin, fout):
	keys = list(fin.keys())
	if 'masks' in keys:
		fout.create_dataset(
			'masks',
			compression="gzip", compression_opts=7,
			data = fin['masks'][:].astype(bool),
		)
		fout['scores'] = fin['scores'][:]
	else:
		for k in keys:
			opt_mask_file(fin[k], fout.create_group(k))

@click.command()
@click.argument('src_file', type=click.Path())
@click.argument('dest_file', type=click.Path())
def main(src_file, dest_file):
	with h5py.File(src_file, 'r') as fin:

		try:
			os.remove(dest_file)
		except OSError as e:
			print(e)

		with h5py.File(dest_file, 'w') as fout:
			opt_mask_file(fin, fout)

main()

# default:
# python src/a03_proposals/sharpmask_optimize_output_file.py /cvlabdata1/home/lis/data/laf_sharpmasks.hdf5 /cvlabdata1/home/lis/data/laf_sharpmasks_opt.hdf5

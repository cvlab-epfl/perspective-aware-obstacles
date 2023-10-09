
from pathlib import Path
import click
import h5py



@click.command()
@click.argument('src_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('dest_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=False))
def main(src_path, dest_path, out_path):
	src_path, dest_path, out_path = (Path(p) for p in (src_path, dest_path, out_path))
	
	if out_path.is_file():
		raise FileExistsError(out_path)

	with h5py.File(src_path, 'r') as src_file:
		with h5py.File(dest_path, 'r') as dest_file:
			with h5py.File(out_path, 'w') as out_file:
				
				for frame_name, group_dest in dest_file.items():

					group_src = src_file[frame_name]

					out_file.copy(group_dest, out_file)
					group_out = out_file[frame_name]
					group_out.copy(group_src['perspective'], group_out)


if __name__ == '__main__':
	main()


"""
DIR_DATA=...

DIR_SRC=$DIR_DATA/1230_SynthObstacle/placement_v3persp3D_cityscapes-train
DIR_DS=$DIR_DATA/1230_SynthObstacle/placement_v2b_cityscapes-train
mv $DIR_DS/labels.hdf5 $DIR_DS/labels-nopers.hdf5
python src/a12_inpainting/synth_dset_transfuse_perspective_info.py $DIR_SRC/labels.hdf5 $DIR_DS/labels-nopers.hdf5 $DIR_DS/labels.hdf5 

DIR_SRC=$DIR_DATA/1230_SynthObstacle/placement_v3persp3D_cityscapes-val
DIR_DS=$DIR_DATA/1230_SynthObstacle/placement_v2b_cityscapes-val
mv $DIR_DS/labels.hdf5 $DIR_DS/labels-nopers.hdf5
python src/a12_inpainting/synth_dset_transfuse_perspective_info.py $DIR_SRC/labels.hdf5 $DIR_DS/labels-nopers.hdf5 $DIR_DS/labels.hdf5 
"""



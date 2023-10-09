
"""

The older loader produced fids
	05_Schafgasse_1__05_Schafgasse_1_000011_000130
whereas the new loader does
	05_Schafgasse_1_000011_000130

This program will link from the old ones to the new ones
	05_Schafgasse_1__05_Schafgasse_1_000011_000130_colorImg.png
	->
	05_Schafgasse_1_000011_000130_colorImg.png

"""

from pathlib import Path
import os

from tqdm import tqdm
import click

def name_transform(name):
	if not ('__' in name):
		return False

	scene_name, rest = name.split('__')

	return rest



@click.group()
def main():
	...

def remove_previous_symlink(p):
	if p.exists():
		if p.is_symlink():
			p.unlink()
		else:
			raise FileExistsError(f'File at {p} is not a symlink')


@main.command()
@click.argument('src_dir', type=click.Path(dir_okay=True, file_okay=False, exists=True))
def rename_dir(src_dir):
	src_dir = Path(src_dir)
	num = 0

	for f in src_dir.iterdir():
		if f.is_file():
			new_name = name_transform(f.name)
			
			if new_name:
				new_path = f.with_name(new_name)
				remove_previous_symlink(new_path)
				
				#print(f.name, '->', new_path)
				# symlink to just f.name, because its in the target directory already
				os.symlink(f.name, new_path)
				num += 1
		
		elif f.is_dir():
			scene_dir = f
			scene_name = scene_dir.name

			for c in scene_dir.iterdir():
				link_path = src_dir / c.name
				remove_previous_symlink(link_path)
				
				os.symlink(f'{scene_name}/{c.name}', link_path)
				num += 1

	return num


@main.command()
@click.argument('data_dir', type=click.Path(dir_okay=True, file_okay=False, exists=True))
def rename_all_data(data_dir):
	data_dir = Path(data_dir)

	for split in ['test', 'train']:
		dsname = f'LostAndFound-{split}'

		dirs = [
			f'1206sem-gluon-psp-ctc/{dsname}/labels',
			f'1207roadar-semcontour-roadwalk-v1/{dsname}/labels',
			f'1208inp-sliding-deepfill-v1/{dsname}/labels',
			f'1208inp-sliding-deepfill-v1/{dsname}/images',
			f'1208inp-pix2pixHD_405/{dsname}/images',
		]

		for dname in dirs:
			dir = data_dir / dname
			result = 'DOES NOT EXIST'
			if dir.exists():
				result = rename_dir.callback(dir)

			print(dir, '   |   ', result, flush=True)


if __name__ == '__main__':
	main()


# 1206sem-gluon-psp-ctc/LostAndFound-test/labels
# 1206sem-gluon-psp-ctc/LostAndFound-train/labels
# 1207roadar-semcontour-roadwalk-v1/LostAndFound-test/labels
# 1207roadar-semcontour-roadwalk-v1/LostAndFound-train/labels
# 1208inp-sliding-deepfill-v1/LostAndFound-test/images
# 1208inp-sliding-deepfill-v1/LostAndFound-test/labels
# 1208inp-sliding-deepfill-v1/LostAndFound-train/images
# 1208inp-sliding-deepfill-v1/LostAndFound-train/labels


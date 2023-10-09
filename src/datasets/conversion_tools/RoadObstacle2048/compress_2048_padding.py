
from pathlib import Path
import numpy as np
from PIL import Image
import cv2 as cv
import click

def imread(path):
	return np.asarray(Image.open(path))

IMWRITE_OPTS = dict(
	webp = dict(quality = 95),
)

def imwrite(path, data, create_parent_dir=True):
	# TODO option to write in background thread

	path = Path(path)
	if create_parent_dir:
		path.parent.mkdir(exist_ok=True, parents=True)

	Image.fromarray(data).save(
		path, 
		**IMWRITE_OPTS.get(path.suffix.lower()[1:], {}),
	)

import subprocess

@click.command()
@click.argument('src_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('dest_file', type=click.Path())
@click.option('--label/--no-label', default=False)
def main(src_file, dest_file, label):


	if label:
		# magick gives us cleaner label resizing
		cmd = """
			convert {src} -filter point -resize 2000x1500 {dest}
		""".format(src=src_file, dest=dest_file).strip().split()
		
		subprocess.run(cmd, check=True)
		# img_2000 = imread(dest_file)

		if Path(dest_file).stem.endswith('color'):
			# PIL reads those color images saved by magick as single-channel even if they have 3 channels.
			# perhaps its color-LUT going wrong
			img_2000 = cv.imread(dest_file)[:, :, [2, 1, 0]]
		else:
			img_2000 = imread(dest_file)

		# print(img_2000.shape, img_2000.dtype, np.unique(img_2000))

		# return

		# print(np.unique(imread(src_file)))
		# print()
		
		sh = list(img_2000.shape)
		sh[0] = 1024
		sh[1] = 2048
		sh = tuple(sh)
		
		img_2048 = np.zeros(sh, dtype=np.uint8)
		img_2048[:, :2000] = img_2000[1500-1024:, :]

		# remove area near the border
		img_2048[:, 2000-12:] = 255

	else:
		img_data = imread(src_file)

		img_2000 = cv.pyrDown(img_data)

		img_2048 = np.zeros((1024, 2048, 3), dtype=np.uint8)

		img_2048[:, :2000] = img_2000[1500-1024:, :]

		# continuation padding
		img_2048[:, 2000:] = img_2048[:, 1999:2000]

	
	imwrite(dest_file, img_2048)


if __name__ == '__main__':
	main()

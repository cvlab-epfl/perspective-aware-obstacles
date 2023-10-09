
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
	

@click.command()
@click.argument('src_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('dest_file', type=click.Path())
@click.option('--label/--no-label', default=False)
@click.option('--blur', type=int, default=5, help="blur kernel size")
def main(src_file, dest_file, label, blur):

	img_data = imread(src_file)

	# crop to aspect ratio 2x1, cutting off the top
	img_data_proc = img_data[1000:, :]

	print(img_data_proc.shape)

	if label:
		img_data_proc = cv.resize(img_data_proc, (2048, 1024), interpolation=cv.INTER_NEAREST)

	else:
		# blur to prevent downsampling artifacts
		img_data_proc = cv.GaussianBlur(img_data_proc, (blur, blur), 0)

		img_data_proc = cv.resize(img_data_proc, (2048, 1024), interpolation=cv.INTER_AREA)
	
	imwrite(dest_file, img_data_proc)


if __name__ == '__main__':
	main()

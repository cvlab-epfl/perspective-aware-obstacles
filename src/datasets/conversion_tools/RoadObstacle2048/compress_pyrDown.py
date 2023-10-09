
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
def main(src_file, dest_file):

	img_data = imread(src_file)

	img_data_resized = cv.pyrDown(img_data)

	imwrite(dest_file, img_data_resized)

if __name__ == '__main__':
	main()

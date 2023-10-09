
from pathlib import Path
import json
from operator import itemgetter

import numpy as np
import cv2 as cv
import click
from PIL import Image as PIL_Image
from tqdm import tqdm

def imwrite(path, data):
	path = Path(path)
	path.parent.mkdir(exist_ok=True, parents=True)
	PIL_Image.fromarray(data).save(path)


def render_masks(polygon_path, dir_out):

	with polygon_path.open('r') as fin:
		polygon_obj = json.load(fin)

	w = polygon_obj["imageWidth"]
	h = polygon_obj["imageHeight"]

	canvas_labels = np.full((h, w), 255, dtype=np.uint8)
	canvas_colors = np.zeros((h, w, 3), dtype=np.uint8)
	canvas_instances = np.zeros((h, w), dtype=np.uint8)


	# sort by label name, descending, so that road is first and obstacle_* is later
	shapes = list(polygon_obj["shapes"])
	shapes.sort(key = itemgetter('label'), reverse=True)

	inst_id = 1

	for shape in shapes:
		label_name = shape["label"]

		if label_name.startswith('road'):
			label = 0
			color = (255, 255, 255)
		elif label_name.startswith('obstacle'):
			label = 1
			color = (255, 125, 0)
		else:
			raise ValueError(f'Label {label_name} is neither road nor obstacle')

		points = np.array([shape['points']])
		points = np.rint(points).astype(np.int32)

		cv.fillPoly(canvas_labels, points, label)
		cv.fillPoly(canvas_colors, points, color)

		if label == 1:
			cv.fillPoly(canvas_instances, points, inst_id)
			inst_id += 1
	
	imwrite(dir_out / f'{polygon_path.stem}_labels_semantic.png', canvas_labels)
	imwrite(dir_out / f'{polygon_path.stem}_labels_semantic_color.png', canvas_colors)
	imwrite(dir_out / f'{polygon_path.stem}_instances.png', canvas_instances)

	del polygon_obj["imageData"]

	with (dir_out / polygon_path.name).open('w') as fout:
		json.dump(polygon_obj, fout, indent='	')



@click.command()
@click.argument('dir_src', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument('dir_out', type=click.Path(dir_okay=True, file_okay=False))
def main(dir_src, dir_out):
	dir_src = Path(dir_src)
	dir_out = Path(dir_out)

	dir_out.mkdir(exist_ok=True, parents=True)

	tasks = list(dir_src.glob('*.json'))

	for p in tqdm(tasks):
		render_masks(
			p,
			dir_out,
		)


if __name__ == '__main__':
	main()



# Demo Harness
#
# Generate images for visual comparison on CV results. Inputs:
# - a sequence of example frames, each stored as a dict
# - a set of methods to compare
# For a given image, the harness runs each method and generates a side-by-side comparison labeled by method name.
# The resulting images are saved in a directory determined by comparison name.

# Methods
# A method should be a callable which takes the frame dict as an argument and outputs a dict of new channels.
# The output dict is merged with the frame dict.
# The image is expected to be at `output["demo_image"]`
# it is saved to `out_dir / frame["demo_image"]`


from functools import partial
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Pool_thread

import numpy as np
import cv2 as cv
from tqdm import tqdm

from .vis_imgproc import image_montage_same_shape

from ..datasets.dataset import imwrite


def image_add_caption(caption_text, image_data, text_color = (200, 150, 30), bg_darken = True):

	text_shared_params = dict(
		text = caption_text,
		fontFace = cv.FONT_HERSHEY_SIMPLEX,
		fontScale = 1.5,
		thickness = 2,
	)

	text_size, _ = cv.getTextSize(**text_shared_params)
	text_tl = (8, 8)


	if bg_darken:
		image_data[text_tl[1]-2:text_tl[1]+text_size[1]+2, text_tl[0]-2:text_tl[0]+text_size[0]+2] //= 2

	text_origin = (text_tl[0], text_tl[1] + text_size[1])

	out_img = cv.putText(img = image_data, org = text_origin, color=text_color, **text_shared_params)

	return out_img





from dataclasses import dataclass
from typing import Mapping, Callable


@dataclass
class DemoHarnessWorker:
	methods_by_name: Mapping[str, Callable]
	out_dir: Path
	num_cols: int
	outputs_to_save: Mapping[str, str] # key = output channel, value = suffix to save
		# = {'demo_image': ''} # mutable default not allowed

	def make_demo_image_for_method(self, method_func, method_name, frame):
		result = method_func(**frame)

		# if isinstance(result, dict):
		# 	out_image = result['demo_image']
		# else:
		# 	out_image = result

		return {
			suffix: image_add_caption(caption_text = method_name, image_data = result[channel])
			for (channel, suffix) in self.outputs_to_save.items()
		}

	def __call__(self, frame):
		frame_name = frame['name']
		input_img = frame.get('image_data')

		# each element is a dict(suffix -> image)
		imgs_per_method = [
			self.make_demo_image_for_method(method_func = method_func, method_name = method_name, frame = frame)
			for (method_name, method_func) in self.methods_by_name.items()
		]

		for suffix in imgs_per_method[0].keys():
			imgs_for_grid = [suffix_to_img[suffix] for suffix_to_img in imgs_per_method]

			fused_image = image_montage_same_shape(imgs = imgs_for_grid, num_cols = self.num_cols, border=16)

			s = f'--{suffix}' if suffix else ''

			out_path = self.out_dir / f'{frame_name}{s}.webp'

			imwrite(out_path, fused_image)


class FakePool:
	def __enter__(self):
		pass

	def __exit__(self):
		pass

	def imap(self, func, iterable, chunksize=None):
		for sample in iterable:
			yield func(sample)


def demo_harness(methods_by_name : dict, samples : list, out_dir : Path, execution = 'thread', outputs_to_save = None, cols=None):
	"""
	execution: 'process', 'thread', None
	"""

	worker_func = DemoHarnessWorker(
		methods_by_name = methods_by_name,
		out_dir = out_dir,
		outputs_to_save = outputs_to_save or {'demo_image': ''},
		num_cols = cols or int(np.floor(np.sqrt(samples.__len__()))),
	)
	
	out_dir.mkdir(exist_ok=True, parents=True)

	pool_class = {
		'process': Pool,
		'thread': Pool_thread,
	}.get(execution, FakePool)

	with pool_class() as pool:
		for _ in tqdm(pool.imap(worker_func, samples), total = samples.__len__()):
			...
			

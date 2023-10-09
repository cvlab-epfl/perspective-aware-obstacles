
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tqdm import tqdm
import click

CROP_SIZE = 512
CLASS_RANGE = [24, 33]

# CITYSCAPES_DIR = Path('/cvlabsrc1/cvlab/dataset_Cityscapes/2048x1024')
# CITYSCAPES_CROPS_DIR = Path('/cvlabdata2/home/lis/data/Cityscapes2048__SpatialEmbeddings_train_crops')
# OUT_IMAGE_DIR = CITYSCAPES_CROPS_DIR / 'images'
# OUT_INSTANCE_DIR = CITYSCAPES_CROPS_DIR / 'instances'

from ..a11_instance_detector.reindexing import classid_list_to_ranges, classmap_test_in_ranges
from ..a11_instance_detector.bbox_transform import labels_to_bbox

def process_frame(task_spec, selected_classes, num_all_classes, crop_size=512):
	"""
	@param task_spec
		in_image_path
		in_instance_path
		fid_tmpl - format(cropid=n)
		out_image_tmpl - format(fid=fid)
		out_instance_tmpl - format(fid=fid)

	@return list of crops, each is
		fid
		centered_class_id
		bool array of classes present

	Difference from SpatialEmbeddings: using bboxes instead of center-of-mass
	"""

	# load data
	image_handle = Image.open(task_spec.in_image_path)
	instances_handle = Image.open(task_spec.in_instance_path)
	instance_map = np.array(instances_handle, copy=False)

	h, w = instance_map.shape
	crop_size_half = crop_size // 2

	#instance_classes = instance_map // 1000 # classes of the instanced objects
	#selected_object_mask = classmap_test_in_ranges(instance_classes, selected_class_ranges)

	bbox_inst_ids, bboxes = labels_to_bbox(instance_map)

	# box midpoints
	bbox_mids = 0.5 * (bboxes[:, :2] + bboxes[:, 2:])

	# top-left corners of the crop
	crop_x1s = np.clip(bbox_mids[:, 0:1] - crop_size_half, 0, w - crop_size).round().astype(np.int)
	crop_y1s = np.clip(bbox_mids[:, 1:2] - crop_size_half, 0, h - crop_size).round().astype(np.int)

	# boxes for crops
	crop_boxes = np.concatenate([
		crop_x1s,
		crop_y1s,
		crop_x1s + crop_size,
		crop_y1s + crop_size,
	], axis=1).astype(np.int)


	# filter instances
	indices_to_crop = [i for i, inst_id in enumerate(bbox_inst_ids) if inst_id // 1000 in selected_classes]

	# perform crops
	crop_index = 0
	crop_infos = []

	def save_component(path_tmpl, fid, handle):
		out_path = Path(path_tmpl.format(fid=fid))
		out_path.parent.mkdir(exist_ok=True, parents=True)
		handle.save(out_path)

	for inst_id, crop_bbox in zip(bbox_inst_ids, crop_boxes):
		class_id = inst_id // 1000

		if class_id in selected_classes:
			image_crop_handle = image_handle.crop(crop_bbox)
			instance_crop_handle = instances_handle.crop(crop_bbox)

			instance_crop_data = np.array(instance_crop_handle, copy=False)
			instance_crop_histogram = np.bincount(instance_crop_data.reshape(-1) // 1000, minlength=num_all_classes).astype(bool)

			fid = task_spec.fid_tmpl.format(cropid=crop_index)

			save_component(task_spec.out_image_tmpl, fid, image_crop_handle)
			save_component(task_spec.out_instance_tmpl, fid, instance_crop_handle)

			crop_infos.append(dict(
				fid = fid,
				object_class = int(class_id),
				classes_present = instance_crop_histogram,
			))

			crop_index += 1

	return crop_infos


@click.command()
@click.option('--classes', default='all_objects')
@click.option('--dir_in', type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option('--dir_out', type=click.Path(file_okay=False))
@click.option('--ext_in', default='.webp')
@click.option('--split', default='train')
def main(classes, dir_in, dir_out, ext_in, split):

	# class ranges
	if classes == 'all_objects':
		classes = range(24, 33+1)
	else:
		classes = list(map(int, classes.split(',')))
	# elif classes == 'cri_split_A':
	#     classes = [24, 27, 28, 29, 31, 32]
	# elif classes == 'cri_split_B':

	# ranges = classid_list_to_ranges(classes)
	# print(f'Filtering class ranges: {ranges}')
	print(f'Cropping classes: {classes}')

	# setup paths
	dir_in = Path(dir_in)
	dir_out = Path(dir_out)

	image_dir = dir_in / 'images' / 'leftImg8bit' / split
	instance_dir = dir_in / 'gtFine' / split
	
	name_tmpl_img_out = f'images/{split}/{{fid}}_crop.webp'
	name_tmpl_inst_out = f'instances/{split}/{{fid}}_inst_crop.png'

	tmpl_img_out = str(dir_out / name_tmpl_img_out)
	tmpl_inst_out = str(dir_out / name_tmpl_inst_out)

	# discover files
	images = list(image_dir.glob(f'*/*{ext_in}'))
	images.sort()

	#re_get_fid = re.compile(r'([\w/]+)_leftImage8bit' + ext_in + '$')
	ending_length = '_leftImg8bit'.__len__() + ext_in.__len__()

	tasks = []

	for img_path in images:
		fid = str(img_path.relative_to(image_dir))[:-ending_length]

		tasks.append(SimpleNamespace(
			fid_tmpl = f'{fid}-{{cropid:02d}}',
			out_image_tmpl = tmpl_img_out,
			out_instance_tmpl = tmpl_inst_out,
			in_image_path = img_path,
			in_instance_path = instance_dir / f'{fid}_gtFine_instanceIds.png',
		))


	worker = partial(process_frame, selected_classes = classes, num_all_classes = 34, crop_size=512)

	with Pool(20) as p:
		r = list(tqdm(
			p.imap(worker, tasks), 
			total=len(tasks),
		))

	print('num frames processed', r.__len__())


	frame_list = []

	for task, crops_from_one_frame in zip(tasks, r):
		for result in crops_from_one_frame:
			fid = result['fid']

			frame_list.append(dict(
				fid = fid,
				path_img = name_tmpl_img_out.format(fid=fid),
				path_inst = name_tmpl_inst_out.format(fid=fid),
				object_class = int(result['object_class']),
			))
	

	with (dir_out / f'frames_{split}.json').open('w') as file:
		json.dump(frame_list, file, indent='	')

	#with h5py.File(dir_out / 'split' / 'frames.hdf5', 'w') as file:
		


if __name__ == '__main__':
	main()

	# cityscapes dataset
	# CITYSCAPES_DIR=os.environ.get('CITYSCAPES_DIR')
	
	# IMAGE_DIR = CITYSCAPES_DIR / 'images/leftImg8bit' / 'train'
	# INSTANCE_DIR = CITYSCAPES_DIR / 'gtFine' / 'train'
	
	# # load images/instances
	# images = list(IMAGE_DIR.glob(f'*/*.webp'))
	# # glob.glob(os.path.join(IMAGE_DIR, 'train', '*/*.png'))
	# # images = [p.relative_to(IMAGE_DIR) for p in images]
	# images.sort()

	# instances = list(INSTANCE_DIR.glob(f'*/*instanceIds.png'))
	# # instances = glob.glob(os.path.join(INSTANCE_DIR, 'train', '*/*instanceIds.png'))
	# # instances = [p.relative_to(INSTANCE_DIR) for p in instances]
	# instances.sort()

	# num_imgs = images.__len__()
	# num_inst = instances.__len__()
	# if num_imgs != num_inst:
	#     raise RuntimeError(f'num images {num_imgs} != num inst {num_inst}')

	# with Pool(16) as p:
	#     r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))

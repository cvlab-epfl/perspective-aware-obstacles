
import logging
from pathlib import Path
import os, re, json
from functools import partial
from multiprocessing import Pool

import numpy as np
import click
from tqdm import tqdm

import torch
from kornia.utils import image_to_tensor


from .reindexing import classmap_test_in_ranges, classid_list_to_ranges
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain
from ..datasets.dataset import DatasetBase, ChannelLoaderImage
from ..datasets.cityscapes import DatasetCityscapes, DIR_CITYSCAPES
from ..pipeline.evaluations import TrChannelLoad, TrChannelSave

# from .bbox_transform import labels_to_bbox

# from .generic_sem_seg import DatasetLabelInfo
from ..paths import DIR_DSETS, DIR_DATA

# Labels as defined by the dataset

log = logging.getLogger('exp')

DIR_CITYSCAPES_CROPS = Path(os.environ.get('DIR_CITYSCAPES_CROPS', DIR_DATA / 'dset_cityscapes_obj-crops-512' ))

class DatasetCityscapesCrops(DatasetBase):
	name = 'cityscapes_crops'
	label_info = DatasetCityscapes.label_info

	def __init__(self, dir_root=DIR_CITYSCAPES_CROPS, split='train'):
		super().__init__()
		self.dir_root = Path(dir_root)
		self.split = split

		# self.add_channels(
		# 	image = ChannelLoaderImage(
		# 		img_ext = '.webp',
		# 		file_path_tmpl = '{dset.dir_root}/images/leftImg8bit/{dset.split}/{frame.fid_scene}_leftImg8bit_{frame.fid_crop:03d}{channel.img_ext}',
		# 	),
		# 	instances = ChannelLoaderImage(
		# 		img_ext = '.png',
		# 		file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{frame.fid_scene}_gtFine_instanceIds_{frame.fid_crop:03d}{channel.img_ext}',
		# 	),
		# )
	
		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.webp',
				file_path_tmpl = '{dset.dir_root}/{dset.split}/images/{frame.fid}_img{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/{dset.split}/instances/{frame.fid}_inst{channel.img_ext}',
			),
		)
	

	def discover_from_files(self):
		# for img_ext in self.IMG_FORMAT_TO_CHECK:
		dir_imgs = (self.dir_root / 'images' / 'leftImg8bit' / self.split)
		img_files = dir_imgs.glob('*/*.webp')

		# img_ext = self.channels['image'].img_ext
		regexp_img_name = re.compile(r'(?P<fid_scene>[\w/]+)_leftImg8bit_(?P<fid_crop>\d+)\.webp')

		self.frames = []

		for img_path in img_files:
			img_path_rel = img_path.relative_to(dir_imgs)

			m = regexp_img_name.match(str(img_path_rel))
			if m:
				fid_parts = m.groupdict()
				fid_scene = fid_parts['fid_scene']
				fid_crop = int(fid_parts['fid_crop'])
				self.frames.append(Frame(
					fid = f"{fid_scene}__{fid_crop:03d}",
					fid_scene = fid_scene,
					fid_crop = fid_crop,
				))
			else:
				log.warning(f'Found file which does not match the expected image dir pattern: {img_path_rel}')

		super().discover()

	def index_file_path(self):
		return self.dir_root / self.split / 'index.json'

	def discover(self):
		with self.index_file_path().open('r') as file_index:
			self.dset_index = json.load(file_index)

		self.frames = [Frame(
			dset=self,
			**f
		) for f in self.dset_index]

		super().discover()

	@staticmethod
	def extract_crops__process_frame(frame_idx, dset_base, dset_out, class_ranges):
		CROP_SIZE = 512

		frame_in = dset_base[frame_idx]

		image = frame_in.image
		instances = frame_in.instances
		
		h, w, _ = image.shape

		object_classes = instances // 1000
		object_mask = classmap_test_in_ranges(object_classes, class_ranges)

		object_instance_ids = np.unique(instances[object_mask])
		object_instance_ids = object_instance_ids[object_instance_ids != 0]

		out_db_rows = []

		# loop over instances
		for index, obj_inst_id in enumerate(object_instance_ids):
			
			y, x = np.where(instances == obj_inst_id)

			# center
			ym, xm = np.mean(y), np.mean(x)
			
			tl_y = int(np.clip(ym-CROP_SIZE/2, 0, h-CROP_SIZE))
			tl_x = int(np.clip(xm-CROP_SIZE/2, 0, w-CROP_SIZE))

			img_crop, inst_crop, mask_crop = [
				a[tl_y:tl_y+CROP_SIZE, tl_x:tl_x+CROP_SIZE]
				for a in [image, instances, object_mask]
			]

			frame_ident = dict(
				fid = f'{frame_in.fid}-{index:02d}',
				fid_scene = frame_in.fid,
				fid_crop = index,
			)

			frame_out = Frame(
				dset = dset_out,
				image = img_crop,
				instances = inst_crop,
				**frame_ident,
			)

			# write crop files
			for chname in ('image', 'instances'):
				ch = dset_out.channels[chname]
				ch.save(dset_out, frame_out, chname)


			objects_present = np.unique(inst_crop[mask_crop] // 1000 ) 
			if objects_present[0] == 0:
				objects_present = objects_present[1:]

			out_db_rows.append(dict(
				focused_object_class = int(obj_inst_id // 1000),
				object_classes_present = list(map(int, objects_present)),
				**frame_ident,
			))

			# im_crop.save(out_image_path.parent / (out_image_path.stem + f"_{j:03d}.webp"))
			# instance_crop.save(out_instance_path.parent / (out_instance_path.stem + f"_{j:03d}.png"))

		return out_db_rows

	def extract_crops(self, dset_base, classes=range(24, 33+1), debug_limit_frames=0, num_workers=16):
		ranges = classid_list_to_ranges(classes)
		print(f'Filtering class ranges: {ranges}')

		num_frames = dset_base.frames.__len__()
		dset_base.set_channels_enabled('image', 'instances')

		if debug_limit_frames:
			num_frames = min(num_frames, debug_limit_frames)

		task_func = partial(self.extract_crops__process_frame,
			dset_base = dset_base,
			dset_out = self,
			class_ranges = ranges,
		)

		crop_db = []

		with Pool(num_workers) as p:
			for crop_db_sublist in tqdm(p.imap(task_func, range(num_frames)), total=num_frames):
				crop_db += crop_db_sublist

		crop_db.sort(key=lambda f: f['fid'])

		log.info(f'Processed {num_frames} frames and extracted {crop_db.__len__()} crops')

		with self.index_file_path().open('w') as file_index:
			json.dump(crop_db, file_index, indent='	')

		log.info(f'Index file written to {self.index_file_path()}')




from .bbox_transform import labels_to_bbox, draw_bboxes
from .reindexing import reindex_instances
import numpy as np

reindex_inst_class_ids = (24, 25, 26, 27, 28, 31, 32, 33)
reindex_gen_inst = (19, 20)




class YoloObjDset:
	def __init__(self, split='train', subsample=None):
		self.dset = DatasetCityscapesCrops(split=split)
		self.dset.discover()

		self.random_subsample = subsample

		#self.shuffled_sample_indices = None

	# def __len__(self):
	# 	if self.shuffled_sample_indices is None:
	# 		return self.dset.__len__()
	# 	else:
	# 		return self.shuffled_sample_indices.__len__()
	
	def __len__(self):
		if self.random_subsample is None:
			return self.dset.__len__()
		else:
			return self.random_subsample
	
	
	def __getitem__(self, idx):
		if self.random_subsample is not None:
			idx = np.random.randint(self.dset.__len__()-1)

		fr = self.dset[idx]
		fr.apply(self.training_frame_bboxes)
		
		return dict(
			image = image_to_tensor(fr.image) * (1./255.),
			yolo_targets = fr.yolo_targets,
		)

	# def shuffle(self, size=3000):
	# 	ds_size = self.dset.__len__()
	# 	self.shuffled_sample_indices = np.random.permutation(ds_size)[:size]

	@staticmethod
	def training_frame_bboxes(instances, **_):
		re_result = reindex_instances(
			instances,
			class_ids_base = reindex_inst_class_ids,
			class_ids_extra = reindex_gen_inst,
		)
		
		inst_reindeed = re_result['instance_map']
		
		bbox_labels, bboxes = labels_to_bbox(inst_reindeed)
		
		# our bboxes are in [tl_x, tl_y, br_x, br_y]
		
		# yolo needs (x, y, w, h)
		
		bb_tl = bboxes[:, :2]
		bb_br = bboxes[:, 2:]
		bb_center = (bb_tl + bb_br) * 0.5
		bb_wh = bb_br - bb_tl
			
		return dict(
			bboxes = bboxes,
			bboxes_center = bb_center,
			bboxes_wh = bb_wh,
			
			yolo_targets = torch.from_numpy(np.concatenate([
				np.zeros((bboxes.__len__(), 2), dtype=np.float32),
				bb_center.astype(np.float32),
				bb_wh.astype(np.float32),
			], axis=1)) / 512,
		)
		
	@staticmethod
	def collate_fn(batch):
		img_batch = torch.stack([
			fr['image'] for fr in batch
		])


		bbox_targets = []

		for bi, fr in enumerate(batch):

			targets = fr['yolo_targets']
			targets[:, 0] = bi

			bbox_targets.append(targets)

		targets_batch = torch.cat(bbox_targets, dim=0)

		
		return dict(
			image = img_batch,
			yolo_targets = targets_batch,
		)


@click.command()
@click.option('--classes', default='all_objects')
@click.option('--dir_in', type=click.Path(dir_okay=True, file_okay=False, exists=True), default=DIR_CITYSCAPES)
@click.option('--split', default='train')
@click.option('--dir_out', type=click.Path(file_okay=False), default=DIR_CITYSCAPES_CROPS)
@click.option('--debug-limit-frames', default=0)
@click.option('--num_workers', default=16)
def main(classes, dir_in, dir_out, split, debug_limit_frames, num_workers):

	# class ranges
	if classes == 'all_objects':
		classes = range(24, 33+1)
	else:
		classes = list(map(int, classes.split(',')))

	dset_in = DatasetCityscapes(dir_root = dir_in, split=split, b_cache=False)
	dset_in.discover()
	dset_in.tr_post_load_pre_cache = TrsChain()

	dset_out = DatasetCityscapesCrops(dir_root = dir_out, split=split)

	dset_out.extract_crops(dset_base=dset_in, classes=classes, debug_limit_frames=debug_limit_frames, num_workers=num_workers)


if __name__ == '__main__':
	main()
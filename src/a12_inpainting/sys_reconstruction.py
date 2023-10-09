
from pathlib import Path
from functools import partial
from easydict import EasyDict
from tqdm import tqdm
import cv2 as cv

from ..pipeline.transforms import TrsChain

from ..datasets.dataset import imread, ChannelLoaderImage
from ..datasets.lost_and_found import DatasetLostAndFound
from ..paths import DIR_DATA, DIR_DATA_cv1

from .patching_square import ImpatcherSlidingSquares

from ..common.name_list import name_list
from ..common.registry import ModuleRegistry
from .sys_road_area import RoadAreaSystem


def preproc_laf(labels_source, image, fid, **_):
	return dict(
		fid = fid,
		image = image,
		labels_road_mask = labels_source >= 1,
	)


def preproc_laf_pred_labels(image, fid, prediction_dir, **_):
	labels_predicted_trainIds = imread(prediction_dir / f'{fid}_predTrainIds.png')

	return dict(
		fid = fid,
		image = image,
		labels_road_mask = labels_predicted_trainIds <= 1, # road and sidewalk
	)


def laf_load_inpainted_image(fid, inp_storage_dir, **_):
	return dict(
		gen_image = imread(inp_storage_dir / 'images' / f'{fid}__inpainted.webp'),
	)


def decorate_laf_dset_to_load_inpainted_images(dset, mask_source='gt'):
	dset.tr_post_load_pre_cache = partial(
		laf_load_inpainted_image, 
		inp_storage_dir = DIR_DATA / f'1205_inputs_inpainted_{mask_source}' / f'{dset.name}-{dset.split}',
	)


def preproc_LAF_inp_for_discrepancy(split = 'train', mask_source = 'gt', dir_out = None, num_workers=3):

	preproc_func = {
		'gt': preproc_laf,
		'psp-bdd100k': partial(preproc_laf_pred_labels, prediction_dir = DIR_DATA_cv1 / 'lost_and_found' / 'eval_PSPEnsBdd' / 'labels' / split),
		'gluon-psp-ctc': ...,
	}[mask_source]
	
	dset = DatasetLostAndFound(split=split, only_interesting=False)
	dset.tr_post_load_pre_cache = preproc_func
	dset.discover()

	imp = ImpatcherSlidingSquares(
		patch_size = 200, 
		context_size = 400, 
		patch_overlap = 0.7, 
		context_restriction = False,
		inpainter_variant_name = 'deep-fill-tf2',
	)

	dir_out = dir_out or (DIR_DATA / f'1205_inputs_inpainted_{mask_source}' / f'{dset.name}-{dset.split}')

	imp.gen__run(
		in_frame_sampler = dset, 
		dir_out = dir_out,
		num_workers = num_workers,
	)
	
	print(f'Written to {dir_out}')


# @click.command()
# @click.argument('dset', type=str)
# @click.argument('split', type=str)
# @click.option('--num_workers', type=int, default=3)
# @click.option('--mask_source', default='gt')
# def main(dset, split, num_workers, mask_source):

# 	if dset == 'laf':
# 		preproc_LAF_inp_for_discrepancy(split=split, num_workers=num_workers, mask_source=mask_source)

# if __name__ == '__main__':
# 	main()

# python -m src.a12_inpainting.sys_reconstruction laf train --num_workers=3
# python -m src.a12_inpainting.sys_reconstruction laf train --num_workers=3 --mask_source=psp-bdd100k



class InpaintingSystem:

	def __init__(self, cfg):
		self.cfg = cfg

	def init_storage(self):
		out_dir = DIR_DATA / '1208inp-{channel.ctx.cfg.name}' / '{dset.name}-{dset.split}' 
	
		self.out_dir = out_dir

		self.storage = dict(
			image_inpainted = ChannelLoaderImage(out_dir / 'images' / '{fid}__inpainted.webp'),
		)
		for c in self.storage.values(): c.ctx = self

		road_area_name = self.cfg.get('road_area_name')
		if road_area_name:
			self.sys_roadarea = RoadAreaSystem.get_implementation(self.cfg.road_area_name)
			self.sys_roadarea.init_storage()
		

	def load(self):
		...

	def frame_load_roadarea(self, frame, **_):
		return self.sys_roadarea.load_into_frame(frame)

	def frame_load_inpainting(self, frame, **_):
		return dict(
			gen_image = self.storage['image_inpainted'].read_value(**frame)
		)

	def decorate_dset_to_load_inp_image(self, dset):
		dset.tr_post_load_pre_cache = TrsChain(
			self.frame_load_roadarea,
			self.frame_load_inpainting,
		)

	@classmethod
	def get_implementation(cls, name):
		return ModuleRegistry.get(cls, name)


@ModuleRegistry(InpaintingSystem)
class InpaintingSystem_SlidingWindow(InpaintingSystem):

	configs = [
		EasyDict(
			name = 'sliding-deepfill-v1',
			road_area_name = 'semcontour-roadwalk-v1',
			inpainter_variant_name = 'deep-fill-tf2',
		),
		EasyDict(
			name = 'sliding-mmdeepfillv2',
			road_area_name = 'semcontour-roadwalk-v1',
			inpainter_variant_name = 'mmagic.deepfillv2_8xb2_places-256x256',
		),
		EasyDict(
			name = 'sliding-mmaotgan',
			road_area_name = 'semcontour-roadwalk-v1',
			inpainter_variant_name = 'mmagic.aot_gan',
		),
	]

	def load(self):

		self.imp = ImpatcherSlidingSquares(
			patch_size = 200, 
			context_size = 400, 
			patch_overlap = 0.7, 
			context_restriction = False,
			inpainter_variant_name = self.cfg.inpainter_variant_name,
		)

		self.init_storage()

		dir_out = self.out_dir

	def process_dset(self, dset, num_workers=3):
		# this has no effect for Synth dset which does not conform to the transform-based dset class
		dset.tr_post_load_pre_cache = self.frame_load_roadarea
		
		self.imp.gen__run(
			in_frame_sampler = dset, 
			dir_out = Path(str(self.out_dir).replace('{channel.ctx.cfg.name}', self.cfg.name).format(dset=dset)),
			num_workers = num_workers,
		)


@ModuleRegistry(InpaintingSystem, 'sliding-deepfill-v1-gtroad')
class InpaintingSystem_SlidingWindow_gtroad(InpaintingSystem_SlidingWindow):

	default_cfg = EasyDict(
		name = 'sliding-deepfill-v1-gtroad',
		road_area_name = 'gt',
	)


@ModuleRegistry(InpaintingSystem, 'pix2pixHD_405')
class InpaintingSystem_pix2pixHD(InpaintingSystem):

	default_cfg = EasyDict(
		name = 'pix2pixHD_405',
		road_area_name = 'semcontour-roadwalk-v1',
		pix2pixHD_variant = '0405_nostyle_crop_ctc',
	)

	def load(self):
		from ..a04_reconstruction.experiments import TrPix2pixHD_Generator
		
		self.mod_generator = TrPix2pixHD_Generator(self.cfg.pix2pixHD_variant, b_postprocess=True)
		
		self.init_storage()

	def process_frame(self, fr, dset=None):
		sys_semseg = self.sys_roadarea.sys_semseg

		try:
			sem_class_prediction = sys_semseg.storage['sem_class_prediction'].read_value(**fr)
		except FileNotFoundError as e:
			sem_class_prediction = dset.placement_dset[fr.frame_id].labels_source

		gen_image = self.mod_generator.generate_auto_resolution(sem_class_prediction)['gen_image']

		self.storage['image_inpainted'].write_value(gen_image, **fr)

	def process_dset(self, dset, num_workers=None):

		for fr in tqdm(dset):
			fr.dset = dset # for saving
			self.process_frame(fr, dset=dset)
		

@ModuleRegistry(InpaintingSystem)
class InpaintingSystem_blur(InpaintingSystem):

	configs = [
		EasyDict(
			name = 'blur111',
			kernel_size = 111,

			road_area_name = 'semcontour-roadwalk-v1',
		),
		EasyDict(
			name = 'blur301',
			kernel_size = 301,

			road_area_name = 'semcontour-roadwalk-v1',
		),
		EasyDict(
			name = 'blur201',
			kernel_size = 201,

			road_area_name = 'semcontour-roadwalk-v1',
		),
		EasyDict(
			name = 'blur71',
			kernel_size = 71,

			road_area_name = 'semcontour-roadwalk-v1',
		),
	]

	def load(self):
		self.init_storage()
		ks = self.cfg.kernel_size
		self.kernel_size = (ks, ks)

	def process_frame(self, fr, dset=None):
		# self.sys_roadarea.load_into_frame(fr)
		gen_image = cv.GaussianBlur(fr.image, self.kernel_size, 0)

		# dset_name_in_frame = fr.get('dset_name')
		# if dset_name_in_frame:
		# 	dset_name, dset_split = dset_name_in_frame.split('-', maxsplit=1)
		# 	dset_entry = EasyDict(
		# 		name = dset_name,
		# 		split = dset_split,
		# 	)
		# else:
		# 	dset_entry = dset

		if 'dset' in fr:
			# frame already has dset entry to save to
			self.storage['image_inpainted'].write_value(gen_image, **fr)
		else:
			# get saving location from dset 
			self.storage['image_inpainted'].write_value(gen_image, dset=dset, **fr)


	def process_dset(self, dset, num_workers=1):

		wf = partial(self.process_frame, dset=dset)

		if num_workers > 1:
			from multiprocessing.dummy import Pool as Pool_thread

			with Pool_thread(num_workers) as pool:
				for _ in tqdm(pool.imap(wf, dset), total=len(dset)):
					...

		else:
			for fr in tqdm(dset):
				if 'dset' not in fr:
					fr.dset = dset # for saving
				wf(fr)


import click
from .demo_case_selection import DatasetRegistry

@click.command()
@click.argument('sys_name')
@click.argument('dset_name')
@click.option('--num_workers', type=int, default=3)
def main(sys_name, dset_name, num_workers):
	system = InpaintingSystem.get_implementation(sys_name)
	system.init_storage()
	system.load()

	for dset_name_single in name_list(dset_name):
		dset = DatasetRegistry.get_implementation(dset_name_single)
		system.process_dset(dset, num_workers=num_workers)

if __name__ == '__main__':
	main()


# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 LostAndFound-test
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 LostAndFound-train
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 RoadAnomaly-test
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 FishyLAF-val

# python -m src.a12_inpainting.sys_reconstruction pix2pixHD_405 FishyLAF-LafRoi
# python -m src.a12_inpainting.sys_reconstruction pix2pixHD_405 RoadAnomaly2-sample1

# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 1230_SynthObstacle_Fusion-v2blur5_cityscapes-train


# python -m src.a12_inpainting.sys_reconstruction pix2pixHD_405 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train
# python -m src.a12_inpainting.sys_reconstruction pix2pixHD_405 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val

# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 1230_SynthObstacle_Fusion_Fblur5-v3Dsz02_cityscapes-val
# python -m src.a12_inpainting.sys_reconstruction sliding-deepfill-v1 1230_SynthObstacle_Fusion_Fblur5-v3Dsz02_cityscapes-val --num_workers 1

# ds3 blur
# python -m src.a12_inpainting.sys_reconstruction blur71 --num_workers 32 1230_SynthObstacle_Fusion_Fblur5-v3Dsz02_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v3Dsz02_cityscapes-train 


# ds2b blur
# python -m src.a12_inpainting.sys_reconstruction blur71 --num_workers 32 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train 
# python -m src.a12_inpainting.sys_reconstruction blur111 --num_workers 32 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train,ObstacleTrack-all,LostAndFound-testNoKnown
# python -m src.a12_inpainting.sys_reconstruction blur301 --num_workers 32 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train,ObstacleTrack-all,LostAndFound-testNoKnown


# ds2b mminp
# python -m src.a12_inpainting.sys_reconstruction sliding-mmdeepfillv2 --num_workers 1 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train,ObstacleTrack-all,LostAndFound-testNoKnown 
# 
# python -m src.a12_inpainting.sys_reconstruction sliding-mmaotgan --num_workers 1 1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val,1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train,ObstacleTrack-all,LostAndFound-testNoKnown
# 
# 



import numpy as np
from ..paths import DIR_DATA
from ..common.jupyter_show import adapt_img_data
from .vis_imgproc import image_montage_same_shape
from .file_io import imread, imwrite

from ..datasets.cityscapes import CityscapesLabelInfo

def demo_semcont(fid, dset, image, gen_image, labels_road_mask, obstacle_from_sem, labels=None, labels_source=None, **_):
	
	if labels is None:
		labels = labels_source
	
	img_gt = np.zeros_like(image)
	img_gt[labels > 200] = (200, 0, 200)
	img_gt[labels == 1] = (255, 0, 0)
	
	img_ra = np.zeros_like(img_gt)
	img_ra[labels_road_mask] = (100, 100, 100)
	img_ra[obstacle_from_sem] = (0, 255, 0)
	
	dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')

	a = {
		'Ground Truth': img_gt,
		'Road Area + Islands': img_ra,
		'Input': image,
		'Inpainting': gen_image,
		'Image vs INPAINTING': imread(
			DIR_DATA / '1209discrep-1205_Discrepancy_ImgVsInpaiting_frozen' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
		'Image vs SELF (control)': imread(
			DIR_DATA / '1209discrep-1205f_Discrepancy_ImgVsSelf_frozen' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
	}
			
	out = image_montage_same_shape(list(a.values()), num_cols=2, downsample=1, border=6, captions=list(a.keys()), caption_color=(200, 128, 0))
# 	show(out)

	out_path = 	DIR_DATA / '1211demo' / dk / f'{fid}__demo_comp.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)


def demo_roadarea_laf(fid, dset, image, labels_road_mask, labels_source, obstacle_from_sem, sem_class_prediction, **_):
	dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')
	
	road_mask_laf = labels_source >= 1

	# img_gt = np.zeros_like(image)
	# img_gt[road_mask_laf] = (128, 128, 128)

	road_mask_pred = np.zeros_like(image)
	road_mask_pred[labels_road_mask] = (200, 200, 200)
	road_mask_pred[obstacle_from_sem] = (255, 255, 255)

	a = {
		'Image': image,
		'Road mask - LAF Free Space': road_mask_laf,
		'Predicted semantics': CityscapesLabelInfo.convert_ids_to_colors(sem_class_prediction),
		'Road mask - predicted': road_mask_pred,
	}
	
	out = image_montage_same_shape(list(a.values()), num_cols=2, downsample=2, border=6, captions=list(a.keys()), caption_color=(200, 128, 0))
	
	out_path = 	DIR_DATA / '1211demo-roadarea-laf' / dk / f'{fid}__demo_comp.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)


def demo_blur_levels(fid, dset, image, gen_image, labels_road_mask, obstacle_from_sem, labels=None, labels_source=None, **_):
	
	if labels is None:
		labels = labels_source
	
	img_gt = np.zeros_like(image)
	img_gt[labels > 200] = (200, 0, 200)
	img_gt[labels == 1] = (255, 0, 0)
	
	img_ra = np.zeros_like(img_gt)
	img_ra[labels_road_mask] = (100, 100, 100)
	img_ra[obstacle_from_sem] = (0, 255, 0)
	
	dir_discrep = DIR_DATA / '1209discrep'



	a = {
		'Inpainting': gen_image,
		'Ground Truth': img_gt,
		'': np.zeros((5, 5, 3), dtype=np.uint8),

		#'Road Area + Islands': img_ra,
		'Input': image,
		'Image vs INPAINTING': imread(
			DIR_DATA / '1209discrep-1215_ImgVsInp-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
		'Image vs SELF (control)': imread(
			DIR_DATA / '1209discrep-1205f_Discrepancy_ImgVsSelf_frozen' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		# blur 3
		'Input-blur(3)': imread(
			dir_discrep / 'ImgBlur03VsInp-archResy' /  dk / 'input_image' / f'{fid}__input_image.webp'
		),

		'Image-blur(3) vs INPAINTING': imread(
			dir_discrep / 'ImgBlur03VsInp-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		'Image-blur(3) vs SELF': imread(
			dir_discrep / 'ImgBlur03VsSelf-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		# blur 5
		'Input-blur(5)': imread(
			dir_discrep / 'ImgBlur05VsInp-archResy' /  dk / 'input_image' / f'{fid}__input_image.webp'
		),

		'Image-blur(5) vs INPAINTING': imread(
			dir_discrep / 'ImgBlur05VsInp-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		'Image-blur(5) vs SELF': imread(
			dir_discrep / 'ImgBlur05VsSelf-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

	}
			
	out = image_montage_same_shape(list(a.values()), num_cols=3, downsample=1, border=6, captions=list(a.keys()), caption_color=(200, 128, 0))
# 	show(out)

	out_path = 	DIR_DATA / '1211demo' / 'demo_blur_levels' / dk / f'{fid}__demo_comp.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)



def demo_variants1(fid, dset, image, gen_image, labels_road_mask, obstacle_from_sem, labels=None, labels_source=None, **_):
	
	if labels is None:
		labels = labels_source
	
	img_gt = np.zeros_like(image)
	img_gt[labels > 200] = (200, 0, 200)
	img_gt[labels == 1] = (255, 0, 0)
	
	img_ra = np.zeros_like(img_gt)
	img_ra[labels_road_mask] = (100, 100, 100)
	img_ra[obstacle_from_sem] = (0, 255, 0)
	
	dir_discrep = DIR_DATA / '1209discrep'
	dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')

	a = {
		'Ground Truth': img_gt,
		'Road Area + Islands': img_ra,
		'Input': image,
		'Inpainting': gen_image,

		#'Road Area + Islands': img_ra,
		'Input': image,
		'Image vs INPAINTING': imread(
			DIR_DATA / '1209discrep-1215_ImgVsInp-archResy' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		'Image vs SELF (control)': imread(
			DIR_DATA / '1209discrep-1205f_Discrepancy_ImgVsSelf_frozen' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		'Image vs INPAINTING - noise aug': imread(
			DIR_DATA / '1209discrep' / 'ImgVsInp-archResyNoiseImg' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),

		'Image vs INPAINTING - focal loss weighted': imread(
			DIR_DATA / '1209discrep' / 'ImgVsInp-archResyFocalWeighted' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
	}
			
	out = image_montage_same_shape(list(a.values()), num_cols=2, downsample=1, border=6, captions=list(a.keys()), caption_color=(200, 250, 0))
# 	show(out)

	out_path = 	DIR_DATA / '1211demo' / 'demo_variants1' / dk / f'{fid}__demo_comp.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)


def demo_variants3(fid, dset, image, gen_image, labels_road_mask, obstacle_from_sem, labels=None, labels_source=None, **_):
	
	if labels is None:
		labels = labels_source
	
	img_gt = np.zeros_like(image)
	img_gt[labels > 200] = (200, 0, 200)
	img_gt[labels == 1] = (255, 0, 0)
	
	img_ra = np.zeros_like(img_gt)
	img_ra[labels_road_mask] = (100, 100, 100)
	img_ra[obstacle_from_sem] = (0, 255, 0)
	
	dir_discrep = DIR_DATA / '1209discrep'
	dsk = f'{dset.name}-{dset.split}'

	a = {
		'Ground Truth': img_gt,
		'Road Area + Islands': img_ra,
		'Input': image,
		'Inpainting': gen_image,


		# 'Image vs INPAINTING (focal+noise)': imread(
		# 	DIR_DATA / '1209discrep' / 'ImgVsInp-archResy-NoiseAndFocalW' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		# ),
		# 'Image vs INPAINTING (focal+noise) - MAX': imread(
		# 	DIR_DATA / '1209discrep' / 'ImgVsInp-archResy-NoiseAndFocalW-PatchMax' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		# ),

		# 'Image vs SELF (focal+noise)': imread(
		# 	DIR_DATA / '1209discrep' / 'ImgVsSelf-archResy-NoiseAndFocalW' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		# ),
		# 'Image vs SELF (focal+noise) - MAX': imread(
		# 	DIR_DATA / '1209discrep' / 'ImgVsSelf-archResy-NoiseAndFocalW-PatchMax' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		# ),

		'Image (blur 3t 5inf) vs INPAINTING ': imread(
			DIR_DATA / '1209discrep' / 'ImgBlur03Inf05VsInp-archResy-NoiseImg-last' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
		# 'Image (blur 3t 5inf) vs SELF (focal+noise)': imread(
		# 	DIR_DATA / '1209discrep' / 'ImgBlur03Inf05VsSelf-archResy-NoiseImg-last' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		# ),
		'Resynthesis': imread(
			DIR_DATA / '1209discrep' / 'Resynth2048Orig-last' / dsk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp'
		),
	}
			
	out = image_montage_same_shape(
		list(a.values()), num_cols=2, downsample=1, border=6, border_color=(128, 128, 128),
		captions=list(a.keys()), caption_color=(200, 250, 0),
	)
# 	show(out)

	out_path = 	DIR_DATA / '1211demo' / 'demo_variants3' / dsk / f'{fid}__demo_comp.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)


# from src.a12_inpainting.sys_road_area import RoadAreaSystem
# ra = RoadAreaSystem.get_implementation('semcontour-roadwalk-v1')()
# ra.init_storage()

# from src.a12_inpainting.sys_reconstruction import InpaintingSystem
# inps = InpaintingSystem.get_implementation('sliding-deepfill-v1')()
# inps.init_storage()


# fr1 = dset_fl[1]
# fr1.update(ra.interpret_labels(ra.storage['road_area_label'].read_value(**fr1)))
# fr1.update(inps.frame_load_inpainting(fr1))
# demo_semcont(**fr1)

# from tqdm import tqdm
# for fr in tqdm(dset_fl):
# 	fr.update(ra.interpret_labels(ra.storage['road_area_label'].read_value(**fr)))
# 	fr.update(inps.frame_load_inpainting(fr))
# 	demo_semcont(**fr)

from .demo_case_selection import DatasetRegistry
import click

def name_list(name_list):
	return [name for name in name_list.split(',') if name]

@click.command()
@click.argument('dset_name')
@click.option('--vis_name', default='v1')
@click.option('--inpainting/--no-inpainting', default=True)
def main(dset_name, vis_name, inpainting):
	func = {
		'v1': demo_semcont,
		'blur_levels': demo_blur_levels,
		'roadarea-laf': demo_roadarea_laf,
		'variants1': demo_variants1,
		'variants3': demo_variants3,
		
	}[vis_name]

	from src.a12_inpainting.sys_road_area import RoadAreaSystem

	from src.a12_inpainting.sys_road_area import RoadAreaSystem
	ra = RoadAreaSystem.get_implementation('semcontour-roadwalk-v1')
	ra.init_storage()
 
	if inpainting:
		from src.a12_inpainting.sys_reconstruction import InpaintingSystem
		inps = InpaintingSystem.get_implementation('sliding-deepfill-v1')
		inps.init_storage()


	for dset_n in name_list(dset_name):

		dset = DatasetRegistry.get_implementation(dset_n)

		from tqdm import tqdm
		for fr in tqdm(dset):
			ra.load_info_frame(fr)
			ra.sys_semseg.load_info_frame(fr)
	
			if inpainting:
				fr.update(inps.frame_load_inpainting(fr))
			func(**fr)

if __name__ == '__main__':
	main()

#python -m src.a12_inpainting.demo_imgs_2 LostAndFound-train --vis_name roadarea-laf --no-inpainting        

#python -m src.a12_inpainting.demo_imgs_2 RoadAnomaly2-sample1,FishyLAF-LafRoi --vis_name blur_levels

#python -m src.a12_inpainting.demo_imgs_2 FishyLAF-LafRoi --vis_name variants1

#python -m src.a12_inpainting.demo_imgs_2 FishyLAF-LafRoi --vis_name variants2

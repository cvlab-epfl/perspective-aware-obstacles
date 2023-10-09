

import numpy as np
import cv2 as cv

from ..common.jupyter_show_image import show, imread, imwrite

from ..paths import DIR_DATA
from ..datasets.cityscapes import CityscapesLabelInfo
from ..datasets.cityscapes import DatasetCityscapes

SRC_GEN_NAME = '051X_semGT__fakeSwapFgd__genNoSty'
TARGET_GEN_NAME = '051X_semGT__fakeSwapFgd__genNoSty__2048'


def dev_convert_synth_frame(fid, dset, labels_source, instances, image=None, src_gen_name = SRC_GEN_NAME, target_gen_name = TARGET_GEN_NAME, b_show = False, **_):
	dir_src = DIR_DATA / 'discrepancy_dataset' / dset.name / src_gen_name
	dir_target = DIR_DATA / 'discrepancy_dataset' / dset.name / target_gen_name
	
	labels_altered_lowres_trainIds = imread(dir_src / 'labels' / dset.split / f'{fid}_fakeTrainIds.png')
	labels_source_trainIds = CityscapesLabelInfo.convert_ids_to_trainIds(labels_source)

	# which areas were altered
	alteration_mask = labels_source_trainIds[::2, ::2] != labels_altered_lowres_trainIds
	
	
	# find map: instance -> new altered label
	instances_altered_map = instances[::2, ::2][alteration_mask]
	target_labels_altered = labels_altered_lowres_trainIds[alteration_mask]
	
	alteration_pairs = np.unique(
		np.stack([instances_altered_map, target_labels_altered], axis=1),
		axis=0,
	)
	#print(alteration_pairs)

	# apply alterations in high res
	labels_altered_highres_trainIds = labels_source_trainIds.copy()
	
	
	for inst_id, target_trainId in alteration_pairs:
		labels_altered_highres_trainIds[ instances == inst_id ] = target_trainId
	
	num_errors = np.count_nonzero(labels_altered_highres_trainIds[::2, ::2] != labels_altered_lowres_trainIds)
	if num_errors > 0:
		print(f'Errors {num_errors} in {fid} -- fake labels')
	
	
	# write altered image
	imwrite(dir_target / 'labels' / dset.split / f'{fid}_fakeTrainIds.png', labels_altered_highres_trainIds)
	
	
	# error map
	discrepancy_map_lowres = imread(dir_src / 'labels' / dset.split / f'{fid}_errors.png')
	
	roi = labels_source_trainIds != 255
	discrepancy_map_highres_binary = (labels_altered_highres_trainIds != labels_source_trainIds) & roi
	discrepancy_map_highres = discrepancy_map_highres_binary.astype(np.uint8)
	discrepancy_map_highres[np.logical_not(roi)] = 255

	# errors = discrepancy_map_highres[::2, ::2] != discrepancy_map_lowres
	# num_errors = np.count_nonzero(errors)
	# if num_errors > 0:
	# 	print(f'Errors {num_errors} in {fid} -- discrepancy map')
	# 	if b_show:
	# 		show(errors)
	
	imwrite(dir_target / 'labels' / dset.split / f'{fid}_errors.png', discrepancy_map_highres)
	imwrite(dir_target / 'labels' / dset.split / f'{fid}_errors_binary.png', discrepancy_map_highres_binary)

	
	# upsample generated image
	gen_image_lowres = imread(dir_src / 'gen_image' / dset.split / f'{fid}_gen.jpg')
	gen_image_highres = cv.resize(gen_image_lowres, dsize = (2048, 1024), interpolation = cv.INTER_CUBIC)
	imwrite(dir_target / 'gen_image' / dset.split / f'{fid}_gen.webp', gen_image_highres)
	
	
	if b_show:
		labels_source_vis = CityscapesLabelInfo.convert_ids_to_colors(labels_source)
		labels_altered_vis = CityscapesLabelInfo.convert_trainIds_to_colors(labels_altered_lowres_trainIds)

		labels_altered_vis_highres = CityscapesLabelInfo.convert_trainIds_to_colors(labels_altered_highres_trainIds)

		show([image, labels_source_vis, alteration_mask], [labels_altered_vis_highres, labels_altered_vis])
		show([discrepancy_map_highres, discrepancy_map_lowres])

	#	show([image, discrepancy_map_highres], [gen_image_lowres, gen_image_highres])

#dev_convert_synth_frame(**fr1)
	
import click

@click.command()
@click.argument('split')
@click.option('--src_gen_name', default=SRC_GEN_NAME)
@click.option('--target_gen_name', default=TARGET_GEN_NAME)
def main(split, src_gen_name, target_gen_name):
	dset = DatasetCityscapes(split=split)
	dset.channel_enable('instances')
	dset.discover()

	from tqdm import tqdm

	for fr in tqdm(dset):
		dev_convert_synth_frame(
			**fr,
			src_gen_name=src_gen_name,
			target_gen_name=target_gen_name,
			b_show = False,
		)


if __name__ == '__main__':
	main()

# python -m src.a12_inpainting.convert_resynth_dset_to_2048 val 

# python -m src.a12_inpainting.convert_resynth_dset_to_2048 train

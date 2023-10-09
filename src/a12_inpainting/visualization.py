from pathlib import Path

import click
import numpy as np
import cv2 as cv
from tqdm import tqdm

from .vis_imgproc import image_montage_same_shape
from ..paths import DIR_DATA
from .dataset_modules import DatasetRegistry
from .sys_reconstruction import InpaintingSystem
from .sys_road_area import RoadAreaSystem

from ..common.jupyter_show_image import imread, imwrite, adapt_img_data

def name_list(name_list):
	return [name for name in name_list.split(',') if name]

@click.group()
def main():
	pass


@main.command("")
@click.argument('method')
@click.argument('dset')
@click.argument('out_dir')
@click.option('--fids', default='')
@click.option('--roadarea', default='semcontour-roadwalk-v1')
@click.option('--inpainter', default='sliding-deepfill-v1')
# @click.option('--tex/--no-tex', default=False)
@click.option('--tex-fig', default='')
def demo_inpainting(method, dset, out_dir, fids, roadarea, inpainter, tex_fig):
	out_dir = Path(out_dir)
	dset = DatasetRegistry.get_implementation(dset)
	inp = InpaintingSystem.get_implementation(inpainter)
	inp.init_storage()
	roadarea = RoadAreaSystem.get_implementation(roadarea)
	roadarea.init_storage()

	fids = name_list(fids)
	frame_iterable = (dset[fid] for fid in fids) if fids else dset

	if tex_fig:
		out_dir_tex = out_dir / 'tex' / 'pictures' / tex_fig

		tex_lines = [
			"\\makebox[0.32\linewidth]{Input image}",
			"\\makebox[0.32\linewidth]{Inpainting of the road area}",
			"\\makebox[0.32\linewidth]{Discrepancy score}",
		]

	for fr in tqdm(frame_iterable):
		sh = fr.image.shape[:2]
		sh_half = (sh[0] // 2, sh[1] // 2)

		path_inp = inp.storage['image_inpainted'].resolve_file_path(fr.dset, fr)
		path_score = DIR_DATA / '1209discrep' / method / fr.dset['dset_key'] / 'demo_anomaly_score' / f'{fr.fid}__demo_anomaly.webp'

		#roadarea.load_into_frame(fr)

		chans = {
			'image': fr.image,
			'inp': imread(path_inp),
			'score': imread(path_score)
		}

		#nonroad = ~fr.labels_road_mask
		nonroad = fr.label_pixel_gt > 2
		#print(fr.keys())
		#return
		chans['score'][nonroad] = fr.image[nonroad]

		chans_half = {
			k: cv.resize(img, sh_half[::-1])
			for k, img in chans.items()
		}

		img_composite = image_montage_same_shape(list(chans.values()), num_cols=3, border=4)

		imwrite(out_dir / f'{fr.fid}_steps.webp', img_composite)

		if tex_fig:

			for k, img in chans_half.items():
				fname = f'{fr.fid}__{k}.jpg'
				imwrite(out_dir_tex / fname, img)
				tex_lines.append(
					f"\\includegraphics[width=0.32\linewidth]{{pictures/{tex_fig}/{fname}}}"
				)
			
			tex_lines += [
				'',
				'\\vspace{5pt}',
				'',
			]

	if tex_fig:
		(out_dir_tex / 'fig.tex').write_text('\n'.join(tex_lines))
	
		# return
	

from multiprocessing.dummy import Pool as Pool_thread

@main.command()
@click.argument('methods')
@click.argument('dset')
@click.argument('out_dir')
@click.option('--fids', default='')
@click.option('--tex-fig', default='')
def method_comparison(methods, dset, out_dir, fids, tex_fig):
	out_dir = Path(out_dir)
	dset_name = dset
	dset = DatasetRegistry.get_implementation(dset)
	
	fids = name_list(fids) or list(range(dset.__len__()))

	from road_anomaly_benchmark.evaluation import Evaluation
	evals = [
		(name, Evaluation(name, dset_name))
		for name in name_list(methods)
	]

	def worker(key):
		fr = dset[key]
		sh = fr.image.shape[:2]
		sh_half = (sh[0] // 2, sh[1] // 2)

		mask_nonroad = fr.label_pixel_gt > 2
		mask_road = ~mask_nonroad

		image_half = cv.resize(fr.image, sh_half[::-1])
		image_dark = image_half.copy()
		mask_road_half = mask_road[::2, ::2]
		image_dark[mask_road_half] //= 2

		
		images = [image_half]
		captions = ['image']

		for method_name, ev in evals:
			anomaly_p = ev.channels['anomaly_p'].read(
				method_name = ev.method_name, 
				dset_name = fr.dset_name,
				fid = fr.fid,
			)

			# get rid of out-of-road values and negatives
			anomaly_p -= np.min(anomaly_p[mask_road])
			anomaly_p *= mask_road

			anomaly_p_heatmap = cv.resize(
				adapt_img_data(anomaly_p),
				sh_half[::-1],
			)

			demo_img = cv.addWeighted(
				image_dark, 0.5, 
				anomaly_p_heatmap, 0.5,
				0.0,
			)

			images.append(demo_img)
			captions.append(method_name)

		img_composite = image_montage_same_shape(
			imgs = images, captions = captions,
			num_cols = 3, border=4, border_color=(255, 255, 255),
		)


		if tex_fig:
			out_dir_tex = out_dir / 'tex' / 'pictures' / f'{tex_fig}__{fr.fid}'

			tex_imgs = []
			tex_captions = []
			tex_lines = []

			for img, caption in zip(images, captions):

				c = caption.replace(' ', '')
				fname = f'{fr.fid}__{c}.jpg'
				imwrite(out_dir_tex / fname, img)
				
				tex_imgs.append(
					f"\\includegraphics[width=0.32\linewidth]{{pictures/{tex_fig}/{fname}}}"
				)
				tex_captions.append(
					f"\\makebox[0.32\linewidth]{{{caption}}}",
				)


				if tex_imgs.__len__() >= 3:
					tex_lines += tex_imgs + tex_captions + [
						'',
						'\\vspace{5pt}',
						'',
					]
					tex_imgs.clear()
					tex_captions.clear()

			(out_dir_tex / 'fig.tex').write_text('\n'.join(tex_lines))

	with Pool_thread(12) as pool:
		for res in tqdm(pool.imap(worker, fids), total=fids.__len__()):
			...


if __name__ == '__main__':
	main()


#python -m src.a12_inpainting.visualization demo-inpainting 1412-4-Resnet101NoBlur Erasing-21 /home/adynathos/dev/phd/cvlab-articles/pictures/out_inp
#python -m src.a12_inpainting.visualization demo-inpainting 1412-4-Resnet101NoBlur FishyLAFObstacle-val /home/adynathos/dev/phd/cvlab-articles/pictures/out_inp

#python -m src.a12_inpainting.visualization demo-inpainting 1412-4-Resnet101NoBlur Erasing-21 /home/adynathos/dev/phd/cvlab-articles/pictures/out_inp --fids darkasphalt2_dog_1,snowstorm1_00_07_43.930,darkasphalt_bottles_1,paving_watercanS_2 --tex-fig 5_Qexamples_1

#python -m src.a12_inpainting.visualization demo-inpainting 1412-4-Resnet101NoBlur FishyLAFObstacle-val /home/adynathos/dev/phd/cvlab-articles/pictures/out_inp --fids 04_Maurener_Weg_8_000000_000150,13_Elly_Beinhorn_Str_000000_000260 --tex-fig 5_Qexamples_1





#python -m src.a12_inpainting.visualization method-comparison 1412-4-Resnet101NoBlur,Entropy_max,Resynthesis Erasing-21 /home/adynathos/dev/phd/cvlab-articles/pictures/out_comparison

#python -m src.a12_inpainting.visualization method-comparison 1409-1-NoCorr,Entropy_max,Resynthesis Erasing-21 /home/adynathos/dev/phd/cvlab-articles/pictures/out_comparison





from pathlib import Path
import numpy as np
from ..paths import DIR_DATA
from ..common.jupyter_show import adapt_img_data
from .vis_imgproc import image_montage_same_shape
from .file_io import imread, imwrite

from ..datasets.cityscapes import CityscapesLabelInfo

from .demo_case_selection import DatasetRegistry
import click
import re
import cv2 as cv

def write_image(base_path, value, out_resize=0, is_vector=False):
	if is_vector:
		imwrite(base_path.parent / (base_path.name + '.png'), value)
	else:

		imwrite(base_path.parent / (base_path.name + '.webp'), value)

		if out_resize:
			h, w = value.shape[:2]
			out_resize_h = round(h * out_resize / w)

			value_res = cv.resize(value, (out_resize, out_resize_h), interpolation=cv.INTER_AREA )
		else:
			value_res = value

		imwrite(base_path.parent / (base_path.name + '_s.jpg'), value_res, opts={'quality': 87})



def demo_crops(frame, dir_out, out_name, crop=None, out_resize=0):

	image = frame.image
	labels = frame.labels

	h, w = image.shape[:2]

	if crop is None:
		crop = (0, 0), (w, h)
	
	roi = labels < 255


	score_as_color = adapt_img_data(frame.anomaly_p, value_range=[0, 1])

	score_overlay = cv.addWeighted(
		image // 2, 0.4, 
		score_as_color, 0.6,
		0.0,
	)

	canvas = image.copy()
	canvas[roi] = score_overlay[roi]


	# drivable_space = frame.labels_road_mask
	# drivable_space_inclusions = frame.obstacle_from_sem


	channels = {
		'image': image,
		'score': canvas,
		'gen_image': frame.get('gen_image'),
		# 'semantic': CityscapesLabelInfo.convert_ids_to_colors(frame.sem_class_prediction),
		# 'drivable': drivable_space,
	}
	channels_vector = {'semantic', 'drivable'}

	dir_out.parent.mkdir(exist_ok=True, parents=True)

	crop_tl, crop_wh = crop
	crop_tlx, crop_tly = crop_tl
	crop_w, crop_h = crop_wh


	for ch_name, ch_value in channels.items():
		ch_value = ch_value[crop_tly:crop_tly+crop_h, crop_tlx:crop_tlx+crop_w]

		write_image(
			dir_out / f'{out_name}__{ch_name}', 
			value = ch_value, 
			out_resize = out_resize,
			is_vector = ch_name in channels_vector,
		)



def parse_crop(_, crop_str):
	if crop_str is None:
		return None

	match = re.match(r'(\d+)x(\d+)\+(\d+)\+(\d+)', crop_str.strip())
	if match:
		w, h, tlx, tly = map(int, match.groups())
		return (tlx, tly), (w, h)
	else:
		raise ValueError(f'Crop str does not match WxH+x+y pattern: {crop_str}')


def read_image_with_any_encoding(path, formats=('png', 'webp')):
	path = Path(path)

	for fmt in formats:
		p = path.with_name(f'{path.name}.{fmt}')
		if p.is_file():
			return imread(p)

	raise FileNotFoundError(path)


def load_demo_frame(dset, fid, method_name, mod_ra, mod_inp):
	fr = dset[fid]
	dset_name = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')

	# mod_ra.load_into_frame(fr)
	# mod_ra.sys_semseg.load_into_frame(fr)
	if mod_inp:
		fr.update(mod_inp.frame_load_inpainting(fr))

	score = read_image_with_any_encoding(
		DIR_DATA / '1209discrep' / method_name / dset_name / 'score_as_image' / f'{fid}__score_as_image'
	)
	if score.shape.__len__() > 2:
		score[:, :, 2]
	fr.anomaly_p = score.astype(np.float32) * (1/255)

	fr.anomaly_p_demoimg = read_image_with_any_encoding(
		DIR_DATA / '1209discrep' / method_name / dset_name / 'demo_anomaly_score' / f'{fid}__demo_anomaly'
	)
	
	return fr


@click.command()
@click.argument('method_name')
@click.argument('dset_name')
@click.argument('fid') # frame id
@click.option('--crop', callback=parse_crop, help="WxH+x+y", default=None)
@click.option('--inpainting', default='sliding-deepfill-v1', help='Pass `none` to disable')
@click.option('--demoname', default='crop')
@click.option('--resize', default=0, type=int)
@click.option('--name-tmpl', default='{dset_name}_{fid}_{method_name}_demo-{demoname}', type=str)
@click.option('--print-tex/--no-print-tex', default=False)
def main(method_name, dset_name, fid, name_tmpl, crop=None, inpainting=None, demoname='crop', resize=0, print_tex=False):

	from src.a12_inpainting.sys_road_area import RoadAreaSystem
	mod_ra = RoadAreaSystem.get_implementation('semcontour-roadwalk-v1')()
	mod_ra.init_storage()
 
	b_inpainting = inpainting and inpainting.lower() != 'none'
	if b_inpainting:
		from src.a12_inpainting.sys_reconstruction import InpaintingSystem
		mod_inp = InpaintingSystem.get_implementation(inpainting)()
		mod_inp.init_storage()
	else:
		mod_inp = None

	dset = DatasetRegistry.get_implementation(dset_name)
	dir_out = DIR_DATA / '1211demo' / demoname

	fids = fid.strip().split(',')

	for fid in fids:
		fr = load_demo_frame(dset=dset, fid=fid, method_name=method_name, mod_ra = mod_ra, mod_inp=mod_inp)

		out_name = name_tmpl.format(
			dset_name = dset_name,
			fid = fid,
			method_name = method_name,
			demoname = demoname,
		)

		demo_crops(frame=fr, dir_out=dir_out, out_name=out_name, crop=crop, out_resize=resize)

		if print_tex:
			br, brc = '{}'
			for ch in ['image', 'gen_image', 'score']:
				print(f'\\includegraphics[width=0.32\\linewidth]{br}pictures/Q/{out_name}__{ch}_s.jpg{brc}')

			print()

if __name__ == '__main__':
	main()


# python -m src.a12_inpainting.demo_imgs_crops $methods_ours RoadObstacles2048p-full darkasphalt2_dog_4 --crop 384x320+835+379 
# python -m src.a12_inpainting.demo_imgs_crops $methods_ours RoadObstacles2048p-full darkasphalt2_dog_4 --crop 276x286+885+392

# python -m src.a12_inpainting.demo_imgs_crops $methods_ours RoadObstacles-v003 greyasphalt_axestump_3 --crop 1473x739+274+364 --resize 480
# python -m src.a12_inpainting.demo_imgs_crops $methods_ours RoadObstacles2048p-full paving_bottles_2 --crop 1098x549+577+173 --resize 480


# python -m src.a12_inpainting.demo_imgs_crops $methods_ours --demoname qn RoadObstacles2048p-full paving_bottles_2 --crop 2000x1024+0+0 --resize 480


# python -m src.a12_inpainting.demo_imgs_crops ImgBlur05VsInp-ds2-NoiseImg ObstacleTrack-all snowstorm1_00_16_00.827 --resize 480 --crop 1920x960+0+64 --name-tmpl "{fid}"    


"""
function quant () {  
	python -m src.a12_inpainting.demo_imgs_crops ImgBlur05VsInp-ds2b-NoiseImg ObstacleTrack-all $1 --resize 640 --name-tmpl "{fid}"  --demoname quantitative 
}
quant snowstorm1_00_00_59.226
quant snowstorm1_00_11_33.009
quant snowstorm1_00_13_18.748
quant snowstorm1_00_14_40.363
quant snowstorm1_00_15_24.040
quant snowstorm1_00_10_52.886

quant darkasphalt_bottles_1
quant darkasphalt2_dog_1
quant gravel_watercanS_2
quant greyasphalt_canister_3
quant motorway_cans_1
quant paving_wood_3

function quantlaf () {  
	python -m src.a12_inpainting.demo_imgs_crops ImgBlur05VsInp-ds2b-NoiseImg LostAndFound-$1 $2 --resize 512 --name-tmpl "{fid}"  --demoname quantitative 
}


quantlaf 01_Hanns_Klemm_Str_45_000000_000200
quantlaf 01_Hanns_Klemm_Str_45_000003_000210
quantlaf 01_Hanns_Klemm_Str_45_000009_000210
quantlaf test 04_Maurener_Weg_8_000000_000150
quantlaf test 04_Maurener_Weg_8_000002_000120
quantlaf test 04_Maurener_Weg_8_000007_000080
quantlaf 13_Elly_Beinhorn_Str_000000_000260
"""

# rsync -avP  lis@pcvcpu23:/cvlabdata2/home/lis/data/1211demo/quantitative/ /mnt/data-research/data/1211demo/quantitative/

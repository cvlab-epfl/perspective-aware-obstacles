import h5py
import cv2 as cv
import numpy as np
from src.paths import DIR_DATA_cv1, DIR_DATA
from src.common.jupyter_show_image import show
from src.datasets.dataset import imread, imwrite
from src.common.jupyter_show import adapt_img_data
from src.a12_inpainting.vis_imgproc import image_montage_same_shape
from src.a12_inpainting.discrepancy_experiments import Exp1205_Discrepancy_ImgVsInpaiting_frozen, Exp1205d_Discrepancy_ImgVsEmpty

exp1 = Exp1205_Discrepancy_ImgVsInpaiting_frozen()
exp2 = Exp1205d_Discrepancy_ImgVsEmpty()
exp1.init_test_datasets()

exp_dirs = [exp1.workdir, exp2.workdir]


def compile_demo_img(fid, dset, labels_source, image, **_):

	p = DIR_DATA_cv1 / 'lost_and_found' / 'eval_PSPEnsBdd' / 'anomalyScore' / dset.split / 'anomalyScore_lag_swap_gt.hdf5'

	
	dk = getattr(dset, 'dset_key', f'{dset.name}-{dset.split}')


	anomaly_p_imgs = [
		imread(exp_workdir / 'out' / dk / 'demo_anomaly_score' / f'{fid}__demo_anomaly.webp')
		for exp_workdir in exp_dirs
	]


	roi = labels_source > 0
	gt = labels_source > 1

	gt_img = np.zeros_like(image)
	gt_img[roi] = 128
	gt_img[gt] = 255

	
	with h5py.File(p, 'r') as f:
		anomaly_p_resynthesis = f[fid][:]
		#print(anomaly_p_resynthesis)

	v = adapt_img_data(anomaly_p_resynthesis)

	demo_img_resynthesis = cv.addWeighted(
		image // 2, 0.5, 
		cv.resize(v, (2048, 1024))  * roi[:, :, None], 0.5,
		0.0,
	)

	
	images = [
		image,
		gt_img,
		demo_img_resynthesis,
		anomaly_p_imgs[0],
		anomaly_p_imgs[1]
	]

	image_captions = [
		'input',
		'gt',
		'resynthesis',
		'image vs inpainting',
		'image only'
	]
	
	out = image_montage_same_shape(images, num_cols=2, downsample=1, border=6, captions=image_captions, caption_color=(200, 128, 0))
# 	show(out)

	fidns = fid.replace('/', '-')
	out_path = DIR_DATA / '1205_demo' / 'laf' / f'{fidns}_demo.webp'
	out_path.parent.mkdir(exist_ok=True, parents=True)
	imwrite(out_path, out)
	

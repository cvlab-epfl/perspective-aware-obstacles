from .dataset import DatasetBase, imwrite, ChannelLoaderImage
from ..pipeline.frame import Frame

from .lost_and_found import DatasetLostAndFound, DIR_LOST_AND_FOUND
from ..a12_inpainting.vis_imgproc import image_montage_same_shape

from pathlib import Path
import json, os
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from ..paths import DIR_DSETS, DIR_DATA


DIR_FISHY_ORIGINAL = Path(os.environ.get('DIR_FISHY_ORIGINAL', DIR_DSETS / 'dataset_Fishyscapes'))
DIR_FISHY_LAF = Path(os.environ.get('DIR_FISHY_LAF', DIR_DSETS / 'dataset_Fishyscapes' / 'FishyLAF'))
DIR_FISHY_STATIC =  Path(os.environ.get('DIR_FISHY_STATIC', DIR_DSETS / 'dataset_Fishyscapes' / 'FishyStatic'))

class DatasetFishyscapesBDL(DatasetBase):
	name = 'fishyscapes'
	split = 'validation'
	
	def __init__(self, variant='Static', storage='/cvlabsrc1/cvlab/dataset_Fishyscapes/download'):
		super().__init__(b_cache=True)
		self.name = f'fishyscapes_{variant}'
		self.variant = variant
		self.storage = storage
	
	def discover(self):
		from bdlb.fishyscapes.fishyscapes_tfds import Fishyscapes

		data_builder = Fishyscapes(data_dir=self.storage, config=self.variant)
		data_builder.download_and_prepare()
		data = data_builder.as_dataset(split='validation')
		
		self.frames = list(map(self.fishyscapes_frame_convert, data))
		
		super().discover()
		
	@staticmethod
	def fishyscapes_frame_convert(frame):
		mask = frame['mask'].numpy()
		return Frame(
			fid = frame['basedata_id'].numpy().decode(),
			image_np = frame['image_left'].numpy(),
			anomaly_label = mask == 1,
			roi = mask == 255,
		)


class DatasetFishyscapesConvertedLAF(DatasetBase):
	name = 'FishyLAF'
	bdlib_name = 'LostAndFound'
	bdlib_split = 'validation'

	LAF_SPLIT_TEST = ['02_Hanns_Klemm_Str_44', '04_Maurener_Weg_8', '05_Schafgasse_1', '07_Festplatz_Flugfeld', '15_Rechbergstr_Deckenpfronn']
	LAF_SPLIT_TRAIN = ['01_Hanns_Klemm_Str_45', '03_Hanns_Klemm_Str_19', '06_Galgenbergstr_40', '10_Schlossberg_9', '11_Parkplatz_Flugfeld', '12_Umberto_Nobile_Str', '13_Elly_Beinhorn_Str', '14_Otto_Lilienthal_Str_24']

	ADDRESS_TO_LAF_SPLIT = dict(**{
		addr: 'test' for addr in LAF_SPLIT_TEST
	}, **{
		addr: 'train' for addr in LAF_SPLIT_TRAIN
	})

	def __init__(self, dir_root = DIR_FISHY_LAF, dir_laf = DIR_LOST_AND_FOUND, split='val'):
		super().__init__(b_cache=False)

		self.split = split
		self.dir_root = Path(dir_root)
		self.dir_laf = Path(dir_laf)

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/images/{fid}_leftImg8bit.webp',
			),
			labels = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/labels/{fid}_fishy-labels.png',
			),
		)

		if split == 'LafRoi':
			# In the LafRoi split we use LAF's free-space label (including the anomaly area) as the ROI
			# This free-space label is a convervative coarse road area

			self.add_channels(
				labels_laf = ChannelLoaderImage(
					file_path_tmpl = '{dset.dir_laf}/gtCoarse/{frame.split_laf}/{frame.fid_laf}_gtCoarse_labelIds.png',
				),
				instances_laf = ChannelLoaderImage(
					file_path_tmpl = '{dset.dir_laf}/gtCoarse/{frame.split_laf}/{frame.fid_laf}_gtCoarse_instanceIds.png',
				),
			)
			self.channel_disable('labels')
			self.channel_disable('instances_laf')

			self.tr_output = self.convert_LAF_roi

	@staticmethod
	def convert_LAF_roi(labels_laf, **_):
		"""
		In the LafRoi split we use LAF's free-space label (including the anomaly area) as the ROI
		This free-space label is a convervative coarse road area

		We convert:
		LAF labels 1=road, 2+=obstacle
		to 
		Fishy labels 1=obstacle 255=out of roi
		"""

		labels_fishy = np.full_like(labels_laf, fill_value=255, dtype=np.uint8)

		labels_fishy[labels_laf > 0] = 0 # road area
		labels_fishy[labels_laf > 1] = 1 # obstacle

		return dict(
			labels = labels_fishy,
		)

	def discover(self):
		with (self.dir_root / 'index.json').open('r') as index_file:
			index = json.load(index_file)

		self.frames = [Frame(fr) for fr in index]

		if self.split == 'LafRoi':
			for fr in self.frames:
				fr.split_laf = self.ADDRESS_TO_LAF_SPLIT[fr.fid_laf.split('/')[0]]

		super().discover()

	@classmethod
	def convert_fishyLAF_to_files(cls, dir_out = DIR_FISHY_LAF, dset_object=None):
		"""
		pip install --upgrade git+https://github.com/adynathos/bdl-benchmark.git

		from src.datasets.fishyscapes import DatasetFishyscapesConvertedLAF

		DatasetFishyscapesConvertedLAF.convert_fishyLAF_to_files()
		"""
		dir_out = Path(dir_out)

		for subdir in ['images', 'labels', 'demo']:
			(dir_out / subdir).mkdir(exist_ok=True, parents=True)


		if dset_object is None:
			from bdlb.fishyscapes.fishyscapes_tfds import Fishyscapes
			data_builder = Fishyscapes(data_dir='/cvlabsrc1/cvlab/dataset_Fishyscapes', config=cls.bdlib_name)
			data_builder.download_and_prepare()
			dset = data_builder.as_dataset(split=cls.bdlib_split)
		else:
			dset = dset_object

		#itr = dset.__iter__()
		#fr = itr.next()
		frame_index = []

		for fr in tqdm(dset):

			fid = str(fr['basedata_id'].numpy(), encoding='ascii')
			fishy_fid = str(fr['image_id'].numpy(), encoding='ascii')
			labels = fr['mask'].numpy()[:, :, 0]
			image_data = fr['image_left'].numpy()

			label_colors = np.zeros_like(image_data)
			label_colors[labels==0] = (50, 50, 50)
			label_colors[labels==1] = (255, 0,0)
			label_colors[labels==255] = (200, 0, 200)                                   

			demo_img = image_montage_same_shape([image_data, label_colors], num_cols=1, downsample=2, border=8)

			imwrite(dir_out / 'demo' / f'{fid}_demolabels.webp', demo_img)
			imwrite(dir_out / 'images' / f'{fid}_leftImg8bit.webp', image_data)
			imwrite(dir_out / 'labels' / f'{fid}_fishy-labels.png', labels)

			frame_index.append(dict(
				fid = fid,
				fid_laf = f'{fid[:-14]}/{fid}',
				fid_fishy = fishy_fid,
			))

		frame_index.sort(key=itemgetter('fid_fishy'))

		with (dir_out / 'index.json').open('w') as json_out:
			json.dump(frame_index, json_out, indent='	', sort_keys=True)


class DatasetFishyscapesConvertedStatic(DatasetFishyscapesConvertedLAF):
	name = 'FishyStatic3'
	split = 'val'
	bdlib_name = 'Static'

	def __init__(self, dir_root = DIR_FISHY_STATIC):
		super().__init__(dir_root=dir_root)


import click

@click.command()
@click.argument('dset_to_convert')
def main(dset_to_convert):
	dsets_by_name = {
		f'{ds.name}-{ds.split}': ds
		for ds in [
			DatasetFishyscapesConvertedLAF(),
			DatasetFishyscapesConvertedStatic(),	
		]
	}

	print('Fishy dsets to convert:', list(dsets_by_name.keys()))

	ds = dsets_by_name[dset_to_convert]
	ds.convert_fishyLAF_to_files(dir_out = ds.dir_root)

if __name__ == '__main__':
	main()

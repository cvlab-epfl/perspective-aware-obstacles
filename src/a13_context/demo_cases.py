from pathlib import Path

import numpy as np

from ..paths import DIR_DATA, DIR_DSETS
from ..pipeline.frame import Frame
from ..datasets.dataset import imread, DatasetBase
from ..a12_inpainting.dataset_modules import DatasetRegistry


class DsetSelectionV2(DatasetBase):
	"""

	LAF white sticks, missed by most methods
		01_Hanns_Klemm_Str_45_000005_000140
		01_Hanns_Klemm_Str_45_000005_000200

		13_Elly_Beinhorn_Str_000002_000120

	LAF box whose internal texture is similar to road
		04_Maurener_Weg_8_000000_000180


	RO dog
		darkasphalt2_dog_4 - big

	RO textured road
		darkasphalt_boot_3

		gravel_stump_1 - big stump, false positives

		greyasphalt_cansB_2 - small objects
		motorway_boot_6_and_bird - small objects

		paving_wood_3

	RO2

	"""
	name = 'demo-selection-2'
	split = 'all'

	def __init__(self, dir_root = DIR_DATA / '1301_demo-selection-2'):
		super().__init__()
		self.dir_root = dir_root

	def discover(self):
		self.frames = []

		for img_path in self.dir_root.glob('*.webp'):
			img_data = imread(img_path)

			self.frames.append(Frame(
				fid = img_path.stem,
				image = img_data,
			))

		super().discover()



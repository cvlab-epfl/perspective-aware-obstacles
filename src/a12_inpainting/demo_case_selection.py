 
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from ..paths import DIR_DATA, DIR_DSETS
from ..pipeline.frame import Frame
from ..datasets.dataset import imread, DatasetBase

from easydict import EasyDict

from .dataset_modules import DatasetRegistry

DIR_INP_DEMO_CASES = DIR_DATA / 'inpainting_fuse_cases'
DIR_INP_DEMO_CASES_IN = DIR_INP_DEMO_CASES / 'in'
DIR_INP_DEMO_CASES_OUT = DIR_INP_DEMO_CASES / 'out_v1'



demo_case_selection = [
	SimpleNamespace(
		img = '01_Hanns_Klemm_Str_45_000000_000200_leftImg8bit.webp',
		pos = (954, 457),
		size = (187, 272),
	),
	SimpleNamespace(
		img = '01_Hanns_Klemm_Str_45_000000_000270_leftImg8bit.webp',
		pos = (348, 600),
		size = (1133, 264),
	),
	SimpleNamespace(
		img = '14_Otto_Lilienthal_Str_24_000003_000120_leftImg8bit.webp',
		pos = (829, 460),
		size = (413, 363),
	),
	SimpleNamespace(
		img = '01_Hanns_Klemm_Str_45_000006_000270_leftImg8bit.webp',
		pos = (753, 497),
		size = (649, 343),
	),
	SimpleNamespace(
		img = '03_Hanns_Klemm_Str_19_000003_000130_leftImg8bit.webp',
		pos = (840, 484),
		size = (671, 260),
	),
	SimpleNamespace(
		img = 'cones02_Broadmoor_Manhole_Cones.webp',
		pos = (260, 266),
		size = (905, 368),
	),
	SimpleNamespace(
		img = 'vehicle02_Broadmoor_Cones_Skidloader.webp',
		pos = (47, 430),
		size = (934, 170),
	),
	SimpleNamespace(
		img = 'animals29_Zebra_Crossing_Abbey_Road_Style.webp',
		pos = (81, 310),
		size = (1152, 310),
	),
	SimpleNamespace(
		img = 'zurich_000098_000019_leftImg8bit.webp',
		pos = (783, 456),
		size = (236, 327),
	),
	SimpleNamespace(
		img = 'zurich_000052_000019_leftImg8bit.webp',
		pos = (0, 550),
		size = (2048, 474),
	),
	SimpleNamespace(
		img = 'hanover_000000_025437_leftImg8bit.webp',
		pos = (350, 460),
		size = (1151, 330),
	),
	SimpleNamespace(
		img = 'zurich_000072_000019_leftImg8bit.webp',
		pos = (618, 533),
		size = (968, 413),
	),
	SimpleNamespace(
		img = 'stuttgart_000043_000019_leftImg8bit.webp',
		pos = (0, 346),
		size = (2048, 565),
	),
	SimpleNamespace(
		img = 'krefeld_000000_000316_leftImg8bit.webp',
		pos = (627, 530),
		size = (975, 302),
	),
	SimpleNamespace(
		img = 'krefeld_000000_000316_leftImg8bit.webp',
		pos = (686, 518),
		size = (749, 324),
	),
]

def load_demo_cases(defs = demo_case_selection):
	return [EasyDict(
		image_data = imread(DIR_INP_DEMO_CASES_IN / fdef.img),
		name = Path(fdef.img).stem.replace('_leftImg8bit', ''),
		area_tl_xy = fdef.pos,
		area_size_xy = fdef.size,
	) for fdef in defs]


def load_cases_with_gt_labels(cases=None):
	from ..datasets.cityscapes import DatasetCityscapes
	from ..datasets.lost_and_found import DatasetLostAndFound
	
	if cases is None:
		cases = load_demo_cases()

	# map frame ids to dset,frame
	dsets = [
		DatasetCityscapes(split='train'),
		DatasetCityscapes(split='val'),
		DatasetLostAndFound(split='train', only_interesting=False),
	]

	def convert_fid(fid):
		return fid.split('/')[-1]

	fid_map = {}

	for ds in dsets:
		ds.b_cache = False
		ds.discover()
		fid_map.update({
			convert_fid(fr.fid): fr for fr in ds.frames
		})

	print(fid_map.__len__(), list(fid_map.keys())[:10])

	# load labels
	for cs in cases:
		matching_frame = fid_map.get(cs.name)

		if matching_frame is not None:
			ds = matching_frame.dset
			fr_data = ds[matching_frame.fid]

			cs.has_labels = True
			cs.dataset_name = ds.name

			if ds.name == 'cityscapes':
				lb = fr_data.labels_source
				cs.labels_cityscapes = lb
				cs.labels_road_mask = lb == 7
				cs.labels_valid_area = lb > 3

			elif ds.name == 'lost_and_found':
				lb = fr_data.labels_source
				cs.labels_laf = lb
				cs.labels_road_mask = lb >= 1
				cs.labels_valid_area = np.ones_like(lb, dtype=bool)
			else:
				raise NotImplementedError(f'Dataset {ds.name}')

		else:
			cs.has_labels = False

	return [cs for cs in cases if cs.has_labels]


class DsetSelectionV1(DatasetBase):
	name = 'demo-selection'

	def __init__(self, only_with_labels=False):
		if only_with_labels:
			self.split = 'v1-labels'
		else:
			self.split = 'v1'

		super().__init__()

	def discover(self):
		if 'labels' in self.split:
			cases = load_cases_with_gt_labels()
		else:
			cases = load_demo_cases()

		for fr in cases:
			fr.image = fr.image_data
			del fr['image_data']
			fr.fid = fr.name

		self.frames = [Frame(f) for f in cases]
		super().discover()




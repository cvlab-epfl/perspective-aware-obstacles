

from pathlib import Path
from typing import List
from operator import attrgetter

import numpy as np
from matplotlib import pyplot
from easydict import EasyDict

external_method_name_mapping = {
	'dropout | entropy': 'Bayesian DeepLab -- entropy',
	'dropout | mutual_info': 'Bayesian DeepLab -- mutual information',
	'deeplab entropy | 12ONfO6WIS16xkfu6ucHEy4_5Tre0yxC5_anomaly': 'DeepLab Softmax -- entropy',
	'deeplab | entropy': 'DeepLab Softmax -- entropy',
	'learned density ensemble | min_nll': 'Learned Embedding Density -- minimum NLL',
	'density min | min_nll': 'Learned Embedding Density -- minimum NLL',
	'learned density ensemble | regression_score': 'Learned Embedding Density -- logistic regression',
	'density ensemble | regression_score': 'Learned Embedding Density -- logistic regression',
	'learned density | negative_loglikelihood_full': 'Learned Embedding Density -- single-layer NLL',
	'void uncertainty | void_prob': 'OoD training -- void classifier',
	'void prob | void_prob': 'OoD training -- void classifier',
	'void max_entropy | entropy': 'OoD training -- max-entropy',
	'void max entropy | max_entropy': 'OoD training -- max-entropy',
	'dirichlet | dirichlet_entropy': 'Dirichlet DeepLab -- prior entropy',
	'knn | density': 'kNN Embedding -- density',
	'knn | class_density': 'kNN Embedding -- relative class density',
	'outlier head random combined | 19lXPaqNIZEj9GS0NcP6CP_tFWc52LQM8_anomaly': 'Discriminative Outlier Detection Head -- combined probability',
	'outlier head combined | outlier_head_combined': 'Discriminative Outlier Detection Head -- combined probability',
	'outlier head random patches | 1PhBnaOGUsZF-HjBwYLEt1bQRDHGSNpFf_anomaly': 'Discriminative Outlier Detection Head -- random sized patches',
	'outlier head random | outlier_head_random': 'Discriminative Outlier Detection Head -- random sized patches',
	'outlier head fixed patches | 1ZmjnA99mGj6Wv47bSDDeGZ724OQmJgm6_anomaly': 'Discriminative Outlier Detection Head -- fixed sized patches',
	'outlier head fixed | outlier_head_fixed': 'Discriminative Outlier Detection Head -- fixed sized patches',
	'epfl resynthesis | Resynthesis_anomaly': 'Resynthesis (BDL) -- discrepancy',
}


method_name_mapping = {
	'ImgVsInp-archResy-NoiseImg-last': 'Ours -- discrepancy',
	'ImgVsInp-archResy': 'Ours -- (test)',
	'ImgVsSelf-archResy': 'Ours (test2)',
	'Resynth2048Orig-last': 'Resynthesis -- discrepancy',
	'ImgVsInp-archResy-NoiseImg-last-SynthPix405': 'Ours -- Resynthesis instead of inpainting',
	'ImgVsSelf-archResy-NoiseImg-last': 'Ours -- Single image only',
	'ImgVsSelf-archResy-last': 'Ours -- Single image only, no noise',
	'ImgVsInp-archResy-last': 'Ours -- No noise augmentation',
	'PixelDistance-L1-Inp': 'Pixel comparison to inpainting -- L1',

	# 'ImgBlur03VsInp-archResy-NoiseImg-last': 'Ours t3i3 -- discrepancy',
	# 'ImgBlur03VsInp-archResy': 'Ours t3i3 -- (test)',
	# 'ImgBlur03VsSelf-archResy': 'Ours t3i3 -- (test2)',
	# 'ImgBlur03VsInp-archResy-NoiseImg-last-SynthPix405': 'Ours t3i3 -- Resynthesis instead of inpainting',
	# 'ImgBlur03VsSelf-archResy-NoiseImg-last': 'Ours t3i3 -- Single image only',
	# 'ImgBlur03VsSelf-archResy-last': 'Ours t3i3 -- Single image only, no noise',
	# 'ImgBlur03VsInp-archResy-last': 'Ours t3i3 -- No noise augmentation',

	'ImgBlur03Inf05VsInp-archResy-NoiseImg-last': 'Ours -- discrepancy',
	'ImgBlur03Inf05VsInp-archResy-NoiseImg-train2BigObj-last': 'Ours -- big object training',
	'ImgBlur03Inf05VsSelf-archResy-NoiseImg-train2BigObj-last': 'Ours -- single image, big object training',

	'ImgBlur03Inf05VsInp-archResy': 'Ours -- (test)',
	'ImgBlur03Inf05VsSelf-archResy': 'Ours (test2)',
	'ImgBlur03Inf05VsInp-archResy-NoiseImg-last-SynthPix405': 'Ours -- Resynthesis instead of inpainting',
	'ImgBlur03Inf05VsSelf-archResy-NoiseImg-last': 'Ours -- Single image only',
	'ImgBlur03Inf05VsSelf-archResy-last': 'Ours -- Single image only, no noise',
	'ImgBlur03Inf05VsInp-archResy-last': 'Ours -- No noise augmentation',
}


line_color_mapping = {
	'Ours': (0, 0, 0),
	'Resynthesis - Discrepancy': (93, 0, 175),
	'Dirichlet': (232, 54, 0),
	'Softmax': (255, 170, 0),
	'Discriminative Outlier Detection Head': (121, 202, 0),
	'Learned Embedding Density': (0, 121, 202),
	'Bayesian': (202, 42, 194),
	'OoD training': (55, 82, 90),
	'kNN': (98, 68, 34),
	'Pixel comparison': (170, 170, 255),

	'single image': (232, 54, 0),
	'no noise': (121, 202, 0),
	'resynthesis instead': (0, 121, 202),
	'image': (5, 20, 30),
}

exclude_from_curve = [
	'Resynthesis (BDL) -- discrepancy',
]

line_style_cycle = [
	'-',
	'--',
	':',
	'-.',
]

dataset_name_mapping = {
	'FishyLAF-LafRoi': "Fishyscapes: Lost and Found",
	'RoadObstacles-v003': "Road Obstacles 2000",
	'RoadObstacles2048b3-full': "Road Obstacles B3",
	'RoadObstacles2048-full': "Road Obstacles B5",
	'RoadObstacles2048b7-full': "Road Obstacles B7",
	'RoadObstacles2048p-full': "Road Obstacles",
}

class LineStyleRepository:

	def __init__(self):
		self.groups = list(line_color_mapping.keys())

		self.lines_in_group = {g: 0 for g in self.groups}

		self.cache = {} # method string -> color, linestyle


	def choose_color_group(self, method_name):
		for g in self.groups:
			if g.lower() in method_name.lower():
				return g

		raise KeyError(method_name)

	def allocate_color_style(self, method_name):
		cached = self.cache.get(method_name.lower())
		if cached:
			return cached

		# which color group?
		color_group = self.choose_color_group(method_name)

		color = tuple(c / 255 for c in line_color_mapping[color_group])

		color_idx = self.lines_in_group[color_group]
		style = line_style_cycle[color_idx]
		self.lines_in_group[color_group] = (color_idx+1) % line_style_cycle.__len__()

		result = (color, style)
		self.cache[method_name.lower()] = result
		return result



def plot_prc_pretty(curve_infos : List['CurveInfoClassification'], dset_name : str):
	
	# table_scores = DataFrame(data = [
	# 	Series({
	# 		'AveragePrecision': crv.area_average_precision, 
	# 		'AUROC': crv.area_roc, 
	# 		'FPR-at-95-TPR:': crv.fpr_at_95_tpr,
	# 		'IOU': crv.IOU_at_05,
	# 		'PDR': crv.PDR_at_05,
	# 		},
	# 		name=crv.display_name,
	# 	)
	# 	for crv in curve_infos
	# ])
	# table_scores = table_scores.sort_values('AveragePrecision', ascending=False)


	# sort descending by AP
	curve_infos_sorted = list(curve_infos)
	curve_infos_sorted.sort(key=attrgetter('area_average_precision'), reverse=True)

	style_repo = LineStyleRepository()


	fig = pyplot.figure(figsize=(8, 11))
	plot_prc = fig.subplots(1, 1)
	
	plot_prc.set_xlabel('recall')
	plot_prc.set_ylabel('precision')
	plot_prc.set_xlim([0, 1])
	plot_prc.set_ylim([0, 1])
	plot_prc.grid(True)

	for crv in curve_infos_sorted:
		if crv.display_name not in exclude_from_curve:
			print(crv.method_name, crv.display_name)
			disp_name = method_name_mapping.get(crv.method_name, crv.display_name).replace('--', '-')

			color, style = style_repo.allocate_color_style(disp_name)

			plot_prc.plot(
				crv.curve_recall, crv.curve_precision,
				label = f'{100*crv.area_average_precision:.01f} {disp_name}',
				color = color,
				linestyle = style,
			)

	fig.tight_layout()

	box = plot_prc.get_position()
	# plot_prc.set_position([box.x0, box.y0, box.width, box.height * 0.6])
	fraction = 0.35
	plot_prc.set_position([box.x0, box.y0 + box.height * fraction, box.width, box.height * (1-fraction)])
	title=f'Average Precision - {dataset_name_mapping.get(dset_name, dset_name)}'
	fig.legend(*plot_prc.get_legend_handles_labels(), title=title, loc='lower center')# bbox_to_anchor=(1, 1))



	# fig = pyplot.figure(figsize=(18, 8))
	# plot_roc = fig.subplots(1, 1)
	# plot_roc.set_xlabel('false positive rate')
	# plot_roc.set_ylabel('true positive rate')
	# plot_roc.set_xlim([0, 0.05])
	# plot_roc.set_ylim([0, 1])



	return EasyDict(
		plot_figure = fig,
		# score_table = table_scores,
	)


import pandas

def multi_dset_table(curve_infos):
	
	category_best_ap = {}
	ci_by_dset = {}

	for ci in curve_infos:
		ci_by_dset.setdefault(ci.dataset_name, []).append(ci)
		
		disp_name = method_name_mapping.get(ci.method_name, ci.display_name)

		try: 
			method_category, method_variant = [s.strip() for s in disp_name.split('--', maxsplit=1)]

			print(ci.method_name, disp_name, '->', method_category, '|', method_variant)
		except ValueError:
			method_category = disp_name
			method_variant = ''
		#	print([s.strip() for s in ci.display_name.split('--')])

		ci.method_category = method_category
		ci.method_variant = method_variant
		# ci.group_id = ci.display_name.split('-')[0].strip().lower()

		category_best_ap[ci.method_category] = max(
			category_best_ap.get(ci.method_category, 0), 
			ci.area_average_precision,
		)
	
	dsets = set(ci_by_dset.keys())


	table_columns = {}

	for ds, cis in ci_by_dset.items():
		ds_name_pretty = dataset_name_mapping.get(ds, ds)

		rows = [(ci.method_category, ci.method_variant) for ci in cis]

		# for ci in cis:
		# 	print((ci.method_category, ci.method_variant))

		# print(ds, cis.__len__())

		# print('ROWS' + '\n	'.join([] + list(map(str, rows))))

		#print('ROWS' + '\n	'.join([] + list(map(str, rows))))

		rows_to_print = [
			f'{a} -- {b}' for a, b in rows
		]
		rows_to_print.sort()
		print('ROWS\n	'+'\n	'.join(rows_to_print))


		index = pandas.MultiIndex.from_tuples(
			rows,
			names = ['Method', 'score'],
		)

		# index = [
		# 	[ci.method_category for ci in cis],
		# 	[ci.method_variant for ci in cis],
		# ]

		

		table_columns[(ds_name_pretty, 'Average Precision')] = pandas.Series(
			[ci.area_average_precision for ci in cis], 
			index=index,
		)
		table_columns[(ds_name_pretty, 'FPR95')] = pandas.Series(
			[ci.fpr_at_95_tpr for ci in cis], 
			index=index,
		)

	# method_names_all = list(set((ci.group_id, ci.display_name) for ci in curve_infos))

	

	index_tuples = list(set((ci.method_category, ci.method_variant) for ci in curve_infos))
	index = list(zip(*index_tuples))
	index = list(map(list, index))
	
	print('ITPL', index_tuples.__len__(), '\nIDD', index[0].__len__(), index[1].__len__())

	print('\n'.join(map(str, index_tuples)))

	table_columns[('Any', 'category_best_ap')] = pandas.Series(
		[category_best_ap[cat] for (cat, var) in index_tuples],
		index = index
	)
	print('========')

	for tck, tcv in table_columns.items():
		print(tck, tcv.__len__())

	print('========')

	index_cols = pandas.MultiIndex.from_tuples(
		list(table_columns.keys()),
		names = ['Dataset', 'metric'],
	)


	print(table_columns)

	table_pd = pandas.DataFrame(data = table_columns, columns = index_cols, index=index)

	ds_first_name = ci_by_dset.keys().__iter__().__next__()
	ds_first_name = dataset_name_mapping.get(ds_first_name, ds_first_name)
	table_pd = table_pd.sort_values([('Any', 'category_best_ap'), (ds_first_name, 'Average Precision')], ascending=False)

	float_format = lambda f: '-' if np.isnan(f) else f'{100*f:.01f}'

	table_tex = table_pd.to_latex(
		float_format = float_format,
		columns = [(ds, metric) for (ds, metric) in table_columns.keys() if ds != 'Any']
	)

	table_html = table_pd.to_html(
		float_format = float_format,
	)



	return EasyDict(
		table_pd = table_pd,
		table_html = table_html,
		table_tex = table_tex,
	)
	





# d = {'one' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd']), 
#       'two' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd'])} 


	# table_scores = DataFrame(data = [
	# 	Series({
	# 		'AveragePrecision': crv.area_average_precision, 
	# 		'AUROC': crv.area_roc, 
	# 		'FPR-at-95-TPR:': crv.fpr_at_95_tpr,
	# 		},
	# 		name=crv.display_name,
	# 	)
	# 	for crv in curve_infos
	# ])
	# table_scores = table_scores.sort_values('AveragePrecision', ascending=False)
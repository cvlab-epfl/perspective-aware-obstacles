

from typing import List
import dataclasses
from pathlib import Path
from operator import attrgetter

import numpy as np
from matplotlib import pyplot
from pandas import DataFrame, Series
from easydict import EasyDict

from .file_io import hdf5_read_hierarchy_from_file, hdf5_write_hierarchy_to_file, hdf5_write_hierarchy_to_group
from .baseline_names import external_method_name_mapping, plot_prc_pretty, multi_dset_table


@dataclasses.dataclass
class CurveInfoClassification:
	method_name : str
	dataset_name : str
	display_name : str

	area_average_precision : float
	curve_recall : np.ndarray
	curve_precision : np.ndarray

	area_roc : float
	curve_tpr : np.ndarray
	curve_fpr : np.ndarray

	fpr_at_95_tpr : float = -1
	threshold_at_95_tpr : float = -1

	IOU_at_05: float = float('nan')
	PDR_at_05: float = float('nan')

	def __iter__(self):
		return dataclasses.asdict(self).items()

	def save(self, path):
		hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

	@classmethod
	def from_file(cls, path):
		return cls(**hdf5_read_hierarchy_from_file(path))


def select_points_for_curve(x, y, num_points, value_range=(0, 1)):
	"""
	x is ascending
	"""

	range_start, range_end = value_range
	thresholds = np.linspace(range_start, range_end, num=num_points-2, dtype=np.float32)

	indices = []

	# points spaced equally in x space
	idx = 0
	for thr in thresholds:
		# binary search for the next threshold
		idx += np.searchsorted(x[idx:], thr)
		indices.append(idx)


	# points spaced equally as percentiles
	if x.size > num_points:
		indices += list(range(0, x.size, x.size // num_points))
	else:
		indices = list(range(x.size))

	# first and last point is always included
	indices.append(0)
	indices.append(x.size-1)

	# sort and remove duplicated
	indices = np.unique(indices)

	return dict(
		indices = indices,
		curve_x = x[indices],
		curve_y = y[indices],
	)

def calc_iou_and_pdr_at_50(uncertainty, gt):
	detection_half = uncertainty > 0.5
	tp = np.count_nonzero(detection_half & gt)
	fp = np.count_nonzero(detection_half & (~gt))
	fn = np.count_nonzero((~detection_half) & gt)
	IOU05 = tp / (tp+fp+fn)
	PDR = tp / (tp+fn)

	return EasyDict(
		IOU_at_05 = IOU05,
		PDR_at_05 = PDR,
	)

def bdl_calculate_curve(labels : List[np.ndarray], uncertainties : List[np.ndarray], method_name=None, dataset_name=None, display_name='no-name', num_points=128, b_histogram=False):
	"""
	Adapted from https://github.com/hermannsblum/bdl-benchmark/blob/master/bdlb/fishyscapes/benchmark.py#L93

	The implementation is based on sklearn ranking metrics:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/ranking.py

	Args:
	@param labels: list of gt label arrays
		0 = background
		1 = obstacle
		everything else = out of roi
	@param uncertainties:
		float anomaly score per pixel
	@param num_points: (optional) number of points to save for PR curves	
	"""
	# if dataset is None:
	# 	dataset = cls.get_dataset()

	# predict uncertainties over the dataset
	# labels = []
	# uncertainties = []
	# for batch in tqdm(dataset):
	# 	labels.append(batch['mask'].numpy())
	# 	uncertainties.append(estimator(batch['image_left']).numpy())

	# concatenate lists for labels and uncertainties together
	if (labels[0].shape[-1] > 1 and np.ndim(labels[0]) > 2) or \
			(labels[0].shape[-1] == 1 and np.ndim(labels[0]) > 3):
		# data is already in batches
		labels = np.concatenate(labels)
		uncertainties = np.concatenate(uncertainties)
	else:
		labels = np.stack(labels)
		uncertainties = np.stack(uncertainties)
	labels = labels.squeeze()
	uncertainties = uncertainties.squeeze()

	# NOW CALCULATE METRICS
	pos = labels == 1
	valid = np.logical_or(labels == 1, labels == 0)  # filter out void
	gt = pos[valid]
	uncertainty = uncertainties[valid].reshape(-1).astype(np.float32, copy=False)
	del pos, uncertainties, valid

	# IOU at 0.5
	iou_pdr = calc_iou_and_pdr_at_50(uncertainty, gt)


	# Sort the classifier scores (uncertainties)
	sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
	uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
	del sorted_indices

	# Remove duplicates along the curve
	distinct_value_indices = np.where(np.diff(uncertainty))[0]
	threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
	del distinct_value_indices#, uncertainty

	# Accumulate TPs and FPs
	tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
	fps = 1 + threshold_idxs - tps

	uncertainty_thresholds = uncertainty[threshold_idxs] # for FPR95 level

	del threshold_idxs

	# Compute Precision and Recall
	precision = tps / (tps + fps)
	precision[np.isnan(precision)] = 0
	recall = tps / tps[-1]
	# stop when full recall attained and reverse the outputs so recall is decreasing
	sl = slice(tps.searchsorted(tps[-1]), None, -1)
	precision = np.r_[precision[sl], 1]
	recall = np.r_[recall[sl], 0]
	average_precision = -np.sum(np.diff(recall) * precision[:-1])

	# select num_points values for a plotted curve
	# interval = 1.0 / num_points
	# curve_precision = [precision[-1]]
	# curve_recall = [recall[-1]]
	# idx = recall.size - 1
	# for p in range(1, num_points):
	# 	while recall[idx] < p * interval:
	# 		idx -= 1
	# 	curve_precision.append(precision[idx])
	# 	curve_recall.append(recall[idx])
	# curve_precision.append(precision[0])
	# curve_recall.append(recall[0])

	# print('rc', recall.shape, recall.dtype)
	# print('prec', precision.shape, precision.dtype)

	curves = select_points_for_curve(1-recall, precision, num_points=num_points)
	curve_recall = 1-curves['curve_x']
	curve_precision = curves['curve_y']

	del precision, recall

	if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
		# Add an extra threshold position if necessary
		# to make sure that the curve starts at (0, 0)
		tps = np.r_[0., tps]
		fps = np.r_[0., fps]
		uncertainty_thresholds = np.r_[0., uncertainty_thresholds]

	# Compute TPR and FPR
	tpr = tps / tps[-1]
	del tps
	fpr = fps / fps[-1]
	del fps

	


	# Compute AUROC
	auroc = np.trapz(tpr, fpr)

	curves = select_points_for_curve(fpr, tpr, num_points=num_points)
	curve_fpr = curves['curve_x']
	curve_tpr = curves['curve_y']

	# Compute FPR@95%TPR
	fpr_95_index = np.searchsorted(tpr, 0.95)
	fpr_tpr95 = fpr[fpr_95_index]
	fpr_95_threshold = float(uncertainty_thresholds[fpr_95_index])

	print(f'FPR95 index {fpr_95_index}/{tpr.__len__()} threshold {fpr_95_threshold}')

	#raise NotImplementedError('Do not overwrite saves')

	curve_info = CurveInfoClassification(
		method_name = method_name,
		dataset_name = dataset_name,
		display_name = display_name,
		# precision-recall curve
		area_average_precision = average_precision,
		curve_recall = curve_recall,
		curve_precision = curve_precision,
		# fpr(tpr) curve
		area_roc = auroc,
		curve_tpr = curve_tpr,
		curve_fpr = curve_fpr,
		fpr_at_95_tpr = fpr_tpr95,
		threshold_at_95_tpr = fpr_95_threshold,
		**iou_pdr,
	)

	if b_histogram:
		curve_info.score_histogram, curve_info.score_histogram_bin_edges = np.histogram(uncertainty, bins=101, range=[0., 1.])
		
	return curve_info


from yaml import safe_load

def load_curve_infos_from_yaml(yaml_path : Path, dset_name : str):
	try:
		entries = safe_load(yaml_path.read_text())
	except Exception as e:
		print(f'YAML fail on file {yaml_path}')
		raise e

	curve_infos = []

	for entry in entries:
		name = entry['name']
		if name.startswith('road'):
			name = name[4:].strip()
		mn = name + ' | ' + entry['method']

		precision = np.array(entry['precision'])
		recall = np.array(entry['recall'])
		
		ap = entry['AP']
		fpr95 = entry['FPR@95%TPR']
		average_precision_counted = -np.sum(np.diff(recall[::-1]) * precision[::-1][:-1])

		diff = np.abs(ap - average_precision_counted)
		print('Diff', diff, external_method_name_mapping[mn])

		ci = CurveInfoClassification(
			method_name = mn,
			dataset_name = dset_name,
			display_name = external_method_name_mapping[mn],
			area_average_precision = ap,
			curve_recall = recall,
			curve_precision = precision,
			area_roc = 0,
			curve_tpr = np.array([0, 1]),
			curve_fpr = np.array([0, 1]),
			fpr_at_95_tpr = fpr95,
		)

		curve_infos.append(ci)

	return curve_infos



def plot_classification_curves_draw_entry(plot_roc : pyplot.Axes, plot_prc : pyplot.Axes, curve_info : CurveInfoClassification):

	if plot_prc is not None:
		plot_prc.plot(curve_info.curve_recall, curve_info.curve_precision,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{curve_info.display_name}  {curve_info.area_average_precision:.02f}',
			marker = '.',
		)

	if plot_roc is not None:
		plot_roc.plot(curve_info.curve_fpr, curve_info.curve_tpr,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{curve_info.display_name}  {curve_info.area_roc:.02f}',
			marker = '.',
		)


def plot_classification_curves(curve_infos : List[CurveInfoClassification]):
	
	table_scores = DataFrame(data = [
		Series({
			'AveragePrecision': crv.area_average_precision, 
			'AUROC': crv.area_roc, 
			'FPR-at-95-TPR:': crv.fpr_at_95_tpr,
			'IOU': crv.IOU_at_05,
			'PDR': crv.PDR_at_05,
			},
			name=crv.display_name,
		)
		for crv in curve_infos
	])
	table_scores = table_scores.sort_values('AveragePrecision', ascending=False)


	fig = pyplot.figure(figsize=(18, 8))
	plot_roc, plot_prc = fig.subplots(1, 2)
	
	plot_prc.set_xlabel('recall')
	plot_prc.set_ylabel('precision')
	plot_prc.set_xlim([0, 1])
	plot_prc.set_ylim([0, 1])

	
	plot_roc.set_xlabel('false positive rate')
	plot_roc.set_ylabel('true positive rate')
	plot_roc.set_xlim([0, 0.2])

	# sort descending by AP
	curve_infos_sorted = list(curve_infos)
	curve_infos_sorted.sort(key=attrgetter('area_average_precision'), reverse=True)

	for crv in curve_infos_sorted:
		plot_classification_curves_draw_entry(plot_prc=plot_prc, plot_roc=plot_roc, curve_info=crv)

	def make_legend(plot_obj, position='lower right', title=None):
		handles, labels = plot_obj.get_legend_handles_labels()
		plot_obj.legend(handles, labels, loc=position, title=title)

		plot_obj.grid(True)

	make_legend(plot_prc, 'lower left', title='Average Precision')
	make_legend(plot_roc, 'lower right', title='AUROC')

	fig.tight_layout()

	return EasyDict(
		plot_figure = fig,
		score_table = table_scores,
	)


def single_frame_inspection_metrics(labels, uncertainty, fid):
	gt_positive = labels == 1
	valid = np.logical_or(labels == 1, labels == 0)  # filter out void

	# mask with ROI
	pos_valid = gt_positive[valid]
	unc_valid = uncertainty[valid].reshape(-1).astype(np.float32, copy=False)
	
	unc_binary = unc_valid > 0.5

	#area = np.count_nonzero(valid)
	area = np.prod(labels.shape)

	#tp = np.count_nonzero(uncertainty_binary & gt_positive)
	fp = np.count_nonzero(unc_binary & np.logical_not(pos_valid))
	fn = np.count_nonzero(np.logical_not(unc_binary) & pos_valid)

	#print(fid, fp, fn, np.count_nonzero(unc_binary), np.count_nonzero(pos_valid))

	results = EasyDict(
		num_fp = fp,
		num_fn = fn,
		area = area,
	)

	try:
		curve_info = bdl_calculate_curve(labels = [labels], uncertainties=[uncertainty])
		curve_info.method_name = 'pf'
		curve_info.dataset_name = fid
		curve_info.display_name = 'pf'

		results['curve'] = curve_info
		results['ap'] =  curve_info.area_average_precision

	except Exception as e:
		print(f'Frame {fid} curve calculation failed', e)
		results['ap'] = -1
		
	return results




from ..paths import DIR_DATA
from ..common.registry import ModuleRegistry

@ModuleRegistry('EvaluatorSystem_PerPixBinaryClassification', 'perpixAP')
class EvaluatorSystem_PerPixBinaryClassification:

	default_cfg = EasyDict(
		name = 'perpixAP',
		curve_num_points = 128,
	)

	def __init__(self, cfg=default_cfg):
		self.cfg = cfg

	def get_dir_out(self, dataset_name):
		return DIR_DATA / f'1210metrics-{self.cfg.name}' / dataset_name

	def save_plot(self, fig : pyplot.Figure, save_name : str, dir_out : Path):
		dir_out.mkdir(parents=True, exist_ok=True)

		for fmt in ('png', 'svg', 'pdf'):
			fig.savefig(dir_out / f'{save_name}.{fmt}')
		
		pyplot.close(fig)

	def plot_curves(self, curve_infos : List[CurveInfoClassification], save_name : str, dir_out : Path):
		vis_res = plot_classification_curves(curve_infos)
		fig = vis_res.plot_figure
		table = vis_res.score_table
		
		self.save_plot(fig, f'PerPixCurve-Plot_{save_name}', dir_out)

		(dir_out / f'PerPixCurve-Table_{save_name}.html').write_text(table.to_html())


	def evaluate_results(self, method_name, dataset_name, display_name, label_frame_list, prediction_frame_list, b_histogram=False, perframe_id_list=None):

		dir_out = self.get_dir_out(dataset_name=dataset_name)
		dir_out.mkdir(parents=True, exist_ok=True)

		if perframe_id_list:
			perframe_metrics = []

			for fid, label, uncertainty in zip(perframe_id_list, label_frame_list, prediction_frame_list):
				# try:
				# 	curve_info = bdl_calculate_curve(labels = [label], uncertainties=[uncertainty])
				# 	curve_info.method_name = method_name
				# 	curve_info.dataset_name = f'{dataset_name}/{fid}'
				# 	curve_info.display_name = display_name

				# 	perframe_curves[fid] = curve_info
				# except Exception as e:
				# 	print(f'Frame {fid} curve calculation failed', e)

				perframe_metrics.append(single_frame_inspection_metrics(
					labels = label,
					uncertainty = uncertainty,
					fid = fid,
				))

			perframe_fp = np.array([pf.num_fp for pf in perframe_metrics])
			perframe_fn = np.array([pf.num_fn for pf in perframe_metrics])
			perframe_ap = np.array([pf.ap for pf in perframe_metrics])

			perframe_fail_factor = perframe_fn * 0.8 + perframe_fp * 0.2
			frame_order = np.argsort(perframe_fail_factor)[::-1]

			table_scores = DataFrame(data = {
				'fid': [perframe_id_list[i] for i in frame_order],
				'ap': perframe_ap[frame_order],
				'false positive': perframe_fp[frame_order],
				'false negative': perframe_fn[frame_order],
			})

			#print(table_scores)
			(dir_out / method_name).mkdir(exist_ok=True, parents=True)
			(dir_out / method_name / f'PerPixCurve-TablePerFrame_{dataset_name}_{method_name}.html').write_text(table_scores.to_html())



		curve_info = bdl_calculate_curve(
			labels = label_frame_list,
			uncertainties = prediction_frame_list,
			b_histogram = b_histogram,
		)
		curve_info.method_name = method_name
		curve_info.dataset_name = dataset_name
		curve_info.display_name = display_name
		
		# write curve to disk
		curve_info.save(dir_out / 'data' / f'PerPixCurve-Data_{method_name}.hdf5')

		# plot the curve
		self.plot_curves(
			curve_infos = [curve_info],
			save_name = f'{dataset_name}_{method_name}',
			dir_out = dir_out / method_name,
		)

		if b_histogram:
			histogram = curve_info.score_histogram
			bin_edges = curve_info.score_histogram_bin_edges

			bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
			histogram_relative = histogram / np.sum(histogram)


			fig = pyplot.figure(figsize=(10, 8))
			plot = fig.subplots(1, 1)
			plot.bar(bin_centers, histogram_relative, width=bin_edges[1] - bin_edges[0])
			plot.set_yscale('log')

			plot.set_title(f'Score histogram: {method_name}')
			fig.tight_layout()

			for ext in ['png', 'pdf', 'svg']:
				fig.savefig(dir_out / method_name / f'ScoreHistogram_{dataset_name}_{method_name}.{ext}')
			
			pyplot.close(fig)

		return curve_info


	def load_curve_infos(self, dataset_name : str, method_names : List[str]) -> List[CurveInfoClassification]:
		dir_out = self.get_dir_out(dataset_name=dataset_name)
		
		curve_infos = []

		for method_name in method_names:

			if method_name == 'baselines_fishy':
				curve_infos += load_curve_infos_from_yaml(
					dir_out / 'data' / 'p_r_curve.yaml',
					dset_name = dataset_name,
				)
			else:

				curve_infos.append(
					CurveInfoClassification.from_file(dir_out / 'data' / f'PerPixCurve-Data_{method_name}.hdf5')
				)

		return curve_infos



	def produce_comparison(self, comparison_name : str, dataset_name : str, method_names : List[str], b_prc : bool = True):
		dir_out = self.get_dir_out(dataset_name=dataset_name)

		curve_infos = self.load_curve_infos(dataset_name, method_names)

		self.plot_curves(
			curve_infos = curve_infos,
			save_name = f'{dataset_name}_{comparison_name}',
			dir_out = self.get_dir_out(dataset_name=dataset_name) / 'comparison',
		)

		if b_prc:
			res = plot_prc_pretty(curve_infos = curve_infos, dset_name = dataset_name)
			fig = res.plot_figure

			self.save_plot(
				fig = fig,
				save_name = f'PRC_{dataset_name}_{comparison_name}',
				dir_out = self.get_dir_out(dataset_name=dataset_name) / 'comparison',
			)

	def produce_table(self, comparison_name : str, dataset_names : List[str], method_names : List[str]):
		dir_out = DIR_DATA / f'1210metrics-{self.cfg.name}' / 'tables'

		curve_infos = []
		for dataset_name in dataset_names:
			for m in method_names:
				try:
					curve_infos += self.load_curve_infos(dataset_name, [m])
				except (FileNotFoundError, OSError) as e:
					print(f'Warning: no file {e}')

		table_res = multi_dset_table(curve_infos)

		dir_out.mkdir(parents=True, exist_ok=True)
		(dir_out / f'Table_{comparison_name}.html').write_text(table_res.table_html)
		(dir_out / f'Table_{comparison_name}.tex').write_text(table_res.table_tex)



	@classmethod
	def get_implementation(cls, name) -> 'EvaluatorSystem_PerPixBinaryClassification':
		return ModuleRegistry.get(cls, name)()


import click
import gc

@click.command()
@click.argument('methods')
@click.argument('dsets')
@click.option('--comparison', default='Standard', type=str)
@click.option('--curves/--no-curves', default=True)
@click.option('--prc/--no-prc', default=False)
@click.option('--table/--no-table', default=False)
def main(methods, dsets, comparison, curves, prc, table):
	methods = methods.split(',')
	dsets = dsets.split(',')
	
	curve_module = EvaluatorSystem_PerPixBinaryClassification.get_implementation('perpixAP')

	if table:
		curve_module.produce_table(
			comparison_name = comparison,
			dataset_names = dsets,
			method_names = methods,
		)

	if curves or prc:
		for d in dsets:
			curve_module.produce_comparison(
				comparison_name = comparison,
				dataset_name = d,
				method_names = methods,
				b_prc = prc,
			)
	

		




		# if islands:
		# 	curve_module.produce_comparison(
		# 		comparison_name = f'{comparison}-vsIslands',
		# 		dataset_name = d,
		# 		method_names = methods + [f'{m}-islands' for m in methods],
		# 	)
			

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsZero-archResy,Island FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison IslandAlone1

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResy-islands,ImgVsInp-archResy-islandsOracle,Island FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison IslandAlone2



if __name__ == '__main__':
	main()



# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsSelf-archResy,ImgVsInp-archResyFocal FishyLAF-val,FishyLAF-LafRoi,RoadAnomaly2-sample1 --comparison Focal

# python -m src.a12_inpainting.metrics ImgVsInp-archResy,ImgVsInp-archResy-NoiseAndFocalW,baselines_fishy FishyLAF-LafRoi --comparison E2_VsBaselines


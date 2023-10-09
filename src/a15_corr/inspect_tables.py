from pathlib import Path
import click
import json, re
import numpy as np
from easydict import EasyDict
from ..paths import DIR_DATA
from road_anomaly_benchmark.evaluation import DIR_OUTPUTS, log
from road_anomaly_benchmark.metrics import MetricRegistry
from road_anomaly_benchmark.__main__ import wrap_html_table, DIR_OUTPUTS, name_list


main = click.Group()

def multirun_statistics(ags, metric):

	multirun_entries = [
		metric.extracts_fields_for_table(ag)
		for ag in ags
	]

	if multirun_entries.__len__() == 1:
		# there is only 1 run
		return multirun_entries[0]

	out_stats = {}

	for colname in multirun_entries[0].keys():
		vals = np.array([r[colname] for r in multirun_entries])
		out_stats[colname] = np.mean(vals)
		out_stats[f'{colname}-std'] = np.std(vals)

	return out_stats


@main.command()
@click.argument('comparison_name', type=str)
@click.option('--metric_names', type=str, default="PixBinaryClass,SegEval-ObstacleTrack")
@click.option('--method_names', type=str)
@click.option('--dataset_names', type=str, default="ObstacleTrack-test")
@click.option('--names', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def comparison(comparison_name, method_names, metric_names, dataset_names, order_by=None, names=None):
	print('exec start')
	from pandas import DataFrame, Series

	print('DIR_OUTPUTS', DIR_OUTPUTS)


	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	order_by = order_by or f'{dataset_names[0]}.{metric_names[0]}.f1_mean'

	
	rename_map = json.loads(Path(names).read_text()) if names is not None else {}
	
	rename_methods = rename_map.get('methods', {
		# '1513-4-P3D-Noise4-BaSpl-PdirectStck': 'Ours',
		# '1513-4-P3D-Noise4-BaSpl': 'No perspective channel',
		# '1518-2-D2b-Noise4-BaSpl-PdirectStck': 'No perspective aware synth',
		# '1518-1-P3D-Noise4-BaSpl-BaseNoSum': 'No perspective channel or fusion',
		# '1518-4-P3D-Noise4-BaSpl-NoSumPdirect': 'No fusion',
		'1518-4-P3D-Noise4-BaSpl-NoSumPdirect': 'Ours, p-synth',
		'1518-1-P3D-Noise4-BaSpl-BaseNoSum': 'No perspective channel, p-synth',
		'1518-5-D2b-Noise4-BaSpl-NoSumPdirect': 'Ours, uniform',
		'1526-3-D2b-Noise4-BaSpl-NoPers': 'No perspective channel, uniform',

		# '1518-2-D2b-Noise4-BaSpl-PdirectStck': 'Fusion, No perspective aware synth',
		# '1513-4-P3D-Noise4-BaSpl-PdirectStck': 'Ours + fusion',
		# '1513-4-P3D-Noise4-BaSpl': 'Fusion, No perspective channel',

		'1519-3-P3D-Noise4-BaSpl-PmapBgbone4Ch': 'P-map along RGB, p-synth',
		'1525-1-D2b-Noise4-BaSpl-PmapBgbone4Ch': 'P-map along RGB, uniform',

		'1519-4-P3D-Noise4-BaSpl-PmapBgboneBranch': 'P-map backbone branch, p-synth',
		'1525-2-D2b-Noise4-BaSpl-PmapBgboneBranch': 'P-map backbone branch, uniform',

		'1519-5-P3D-Noise4-BaSpl-Unwarp': 'Image warping v1',
		'1519-5-P3D-Noise4-BaSpl-Unwarp2': 'Image warping, p-synth',
		
		'1525-3-P3A-Noise4-BaSpl-NoSumPdirect':  'Ours, scale-synth',
	})
	rename_dsets = rename_map.get('datasets', {
		'ObstacleTrack-test': 'ObsT',
		'LostAndFount-testNoKnown': 'LAFtnk'
	})
	rename_metrics = rename_map.get('metrics', {
		'SegEval-ObstacleTrack': 'Obj',
		'PixBinaryClass': 'Pix',
		"Obj.fn_25": "",
		"Obj.fp_25": "",
		"Obj.f1_25": "",
		"Obj.fn_50": "",
		"Obj.fp_50": "",
		"Obj.f1_50": "",
		"Obj.fn_75": "",
		"Obj.fp_75": "",
		"Obj.f1_75": "",
		"Obj.best_f1": "",
	})
	plot_formats = rename_map.get('plots', {})
	order_by = rename_map.get(order_by, order_by)


	columns = {}

	def get_col(name):
		c = columns.get(name)
		if c is not None:
			return c
		else:
			return columns.setdefault(name, Series(dtype=np.float64))


	for dset in dataset_names:
		for metric_name in metric_names:
			metric = MetricRegistry.get(metric_name)
	
			ags = {}

			for method in method_names:

				# name_re_match = re.match(r'(.+)-rep\d*', method)
				name_re_match = re.match(r'(.+)-r\d*', method)


				if name_re_match:
					base_method_name = name_re_match.group(1)
				else:
					base_method_name = method

				try:
					ags.setdefault(base_method_name, []).append(
						metric.load(method_name = method, dataset_name = dset)
					)

				except FileNotFoundError:
					print('- missing', method)


			for method, ag_list in ags.items():
				print(method, ag_list.__len__())
				if ag_list:
					for f, v in multirun_statistics(ag_list, metric).items():

						ds = rename_dsets.get(dset, dset)
						
						is_std = f.endswith('-std')
						is_fraction = not (f.startswith('fn') or f.startswith('fp'))

						if is_std:
							f = f[:-4]

						metric_name = rename_metrics.get(metric_name, metric_name)
						met = f'{metric_name}.{f}'
						met = rename_metrics.get(met, met)

						#if met and is_std:
						#	met += '-std'

						if met:
							# colname = f'{dset}.{metric_name}.{f}'
							colname = f'{ds}.{met}'
							mn = rename_methods.get(method, method)

							if not is_std:
								get_col(colname)[mn] = f'{v*100:.1f}' if is_fraction else f'{v:.0f}'
							else:
								get_col(colname)[mn] += f' ± {v*100:.1f}' if is_fraction else f' ± {v:.1f}'


	table = DataFrame(data = columns)

	# if order_by in table:
	# 	table = table.sort_values(order_by, ascending=False)
	# else:
	# 	log.warn(f'Order by: no column {order_by}')

	print(table)

	str_formats = dict(
		#float_format = lambda f: f'{100*f:.01f}',
		float_format = lambda f: f'{f:.3g}',
		
		na_rep = '-',
	)
	table_tex = table.to_latex(**str_formats)
	table_html = wrap_html_table(
		table = table.to_html(
			classes = ('display', 'compact'), 
			**str_formats,
		),
		title = comparison_name,
	)

	# json dump for website
	# table['method'] = table.index
	# table_json = table.to_json(orient='records')
	# table_data = json.loads(table_json)
	# table_data = [{k.replace('.', '-'): v for k, v in r.items()} for r in table_data]
	# table_json = json.dumps(table_data)

	out_f = DIR_OUTPUTS / 'tables-custom' / comparison_name
	out_f.parent.mkdir(parents=True, exist_ok=True)
	out_f.with_suffix('.html').write_text(table_html)
	out_f.with_suffix('.tex').write_text(table_tex)
	# out_f.with_suffix('.json').write_text(table_json)




@main.command()
@click.argument('comparison_name', type=str)
@click.option('--method_names', type=str)
@click.option('--dataset_names', type=str, default="ObstacleTrack-test")
@click.option('--iou', type=float, default=0.5)
@click.option('--binmax', type=float, default=50)
def distance(comparison_name, method_names, dataset_names, iou : float, binmax : float):
	
	binwidth = 5

	import pandas, seaborn
	from matplotlib import pyplot

	method_names = name_list(method_names)
	dataset_names = name_list(dataset_names)

	
	rename_methods = {
		# '1513-4-P3D-Noise4-BaSpl-PdirectStck': 'Ours',
		# '1513-4-P3D-Noise4-BaSpl': 'No perspective channel',
		# '1518-2-D2b-Noise4-BaSpl-PdirectStck': 'No perspective aware synth',
		# '1518-1-P3D-Noise4-BaSpl-BaseNoSum': 'No perspective channel or fusion',
		# '1518-4-P3D-Noise4-BaSpl-NoSumPdirect': 'No fusion',
		'1518-4-P3D-Noise4-BaSpl-NoSumPdirect': 'Ours',
		'1518-1-P3D-Noise4-BaSpl-BaseNoSum': 'No perspective channel, p-synth',
		'1518-5-D2b-Noise4-BaSpl-NoSumPdirect': 'Ours, uniform',
		'1526-3-D2b-Noise4-BaSpl-NoPers': 'No perspective channel, uniform',

		# '1518-2-D2b-Noise4-BaSpl-PdirectStck': 'Fusion, No perspective aware synth',
		# '1513-4-P3D-Noise4-BaSpl-PdirectStck': 'Ours + fusion',
		# '1513-4-P3D-Noise4-BaSpl': 'Fusion, No perspective channel',

		'1519-3-P3D-Noise4-BaSpl-PmapBgbone4Ch': 'P-map along RGB, p-synth',
		'1525-1-D2b-Noise4-BaSpl-PmapBgbone4Ch': 'P-map along RGB, uniform',

		'1519-4-P3D-Noise4-BaSpl-PmapBgboneBranch': 'P-map backbone branch, p-synth',
		'1525-2-D2b-Noise4-BaSpl-PmapBgboneBranch': 'P-map backbone branch, uniform',

		'1519-5-P3D-Noise4-BaSpl-Unwarp': 'Image warping v1',
		'1519-5-P3D-Noise4-BaSpl-Unwarp2': 'Image warping, p-synth',
		
		'1525-3-P3A-Noise4-BaSpl-NoSumPdirect':  'Ours, scale-synth',
	}

	rename_dsets = {
		'ObstacleTrack-test': 'ObsT',
		'LostAndFount-testNoKnown': 'LAFtnk'
	}
	columns = {}

	metric = MetricRegistry.get('SegEvalDist-ObstacleTrack')

	dir_out = DIR_DATA / '1585_DistanceDetectionPlots'
	dir_out.mkdir(parents=True, exist_ok=True)

	# font options for RAL
	import matplotlib
	# select "type 1" font for RAL
	# this needs texlive-latex-base and cm-super
	matplotlib.rcParams['text.usetex'] = True 
	# matplotlib.rcParams['ps.fonttype'] = 42 # avoid "type 3" font

	for dset in dataset_names:
		print('-- ', dset)
		ags = [metric.load(method_name = method, dataset_name = dset) for method in method_names]
		
		if 'LostAndFound' in dset:
			focal = 2265
		elif 'ObstacleTrack' in dset:
			focal = 2265

		rels = []
		for method, ag in zip(method_names, ags):
			mn = rename_methods.get(method.replace('-rep1', ''), method)

			# tp_flag = ag.inst_gt_sIoU >= iou
			# fn_flag = np.logical_not(tp_flag)

			fp_flag = (ag.inst_pred_sIoU < iou) & (ag.inst_pred_pmap > 0)
			rels.append(pandas.DataFrame({
				'method': [mn] * np.count_nonzero(fp_flag),
				'distance': focal / ag.inst_pred_pmap[fp_flag],
			}))

		r = pandas.concat(rels)

		print(r)

		sfig = seaborn.histplot(
			data=r, x="distance", hue="method", multiple="dodge", shrink=.8,
			binrange = [0, binmax],
			binwidth = binwidth,
		)
		sfig.set(xlabel = 'depth [m]', ylabel = 'false positives')
		sfig.figure.tight_layout()

		for fmt in ['pdf', 'eps', 'png']:
			outpath = dir_out / f'{dset}_{comparison_name}_distanceFP.{fmt}'
			sfig.figure.savefig(outpath)
		pyplot.close(sfig.figure)


		rels = []
		for method, ag in zip(method_names, ags):
			mn = rename_methods.get(method.replace('-rep1', ''), method)

			tp_flag = (ag.inst_gt_sIoU >= iou) & (ag.inst_gt_pmap > 0)
			rels.append(pandas.DataFrame({
				'method': [mn] * np.count_nonzero(tp_flag),
				'distance': focal / ag.inst_gt_pmap[tp_flag],
			}))

		r = pandas.concat(rels)
		sfig = seaborn.histplot(data=r, 
			x="distance", hue="method", multiple="dodge", shrink=.8,
			binrange = [0, binmax],
			binwidth = binwidth,
		)
		sfig.set(xlabel = 'depth [m]', ylabel = 'true positives')
		sfig.figure.tight_layout()

		for fmt in ['pdf', 'eps', 'png']:
			outpath = dir_out / f'{dset}_{comparison_name}_distanceTP.{fmt}'
			sfig.figure.savefig(outpath)
		pyplot.close(sfig.figure)


	# json dump for website
	# table['method'] = table.index
	# table_json = table.to_json(orient='records')
	# table_data = json.loads(table_json)
	# table_data = [{k.replace('.', '-'): v for k, v in r.items()} for r in table_data]
	# table_json = json.dumps(table_data)

	# out_f = DIR_OUTPUTS / 'tables-custom' / comparison_name
	# out_f.parent.mkdir(parents=True, exist_ok=True)
	# out_f.with_suffix('.html').write_text(table_html)
	# out_f.with_suffix('.tex').write_text(table_tex)
	# out_f.with_suffix('.json').write_text(table_json)



if __name__ == '__main__':
	main()

"""

methods_test_1513b=1513-4-P3D-Noise4-BaSpl-PdirectStck,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep1,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep2,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep3,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep4,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep5,1513-4-P3D-Noise4-BaSpl,1513-4-P3D-Noise4-BaSpl-rep1,1513-4-P3D-Noise4-BaSpl-rep2,1513-4-P3D-Noise4-BaSpl-rep3,1513-4-P3D-Noise4-BaSpl-rep4,1513-4-P3D-Noise4-BaSpl-rep5

methods_test_1518a=1518-1-P3D-Noise4-BaSpl-BaseNoSum-rep1,1518-1-P3D-Noise4-BaSpl-BaseNoSum-rep2,1518-1-P3D-Noise4-BaSpl-BaseNoSum-rep3,1518-1-P3D-Noise4-BaSpl-BaseNoSum-rep4,1518-1-P3D-Noise4-BaSpl-BaseNoSum-rep5
methods_test_1518b=1518-2-D2b-Noise4-BaSpl-PdirectStck-rep1,1518-2-D2b-Noise4-BaSpl-PdirectStck-rep2,1518-2-D2b-Noise4-BaSpl-PdirectStck-rep3,1518-2-D2b-Noise4-BaSpl-PdirectStck-rep4,1518-2-D2b-Noise4-BaSpl-PdirectStck-rep5
methods_test_1518c=1518-3-P3D-Noise4-BaSpl-PdirectEncStck-rep1,1518-3-P3D-Noise4-BaSpl-PdirectEncStck-rep2,1518-3-P3D-Noise4-BaSpl-PdirectEncStck-rep3
methods_test_1518d=1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1,1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep2,1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep3,1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep4,1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep5
methods_test_1518=$methods_test_1518a,$methods_test_1518b,$methods_test_1518c,$methods_test_1518d

methods_abl=$methods_test_1513b,$methods_test_1518a,$methods_test_1518b,$methods_test_1518d

python -m src.a15_corr.inspect_tables comparison 1518_abl_obstacle --dataset_names ObstacleTrack-test --method_names $methods_abl
python -m src.a15_corr.inspect_tables comparison 1518_abl_LAF --dataset_names LostAndFound-testNoKnown --method_names $methods_abl


methods_test_dev=1513-4-P3D-Noise4-BaSpl-PdirectStck,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep1,1513-4-P3D-Noise4-BaSpl-PdirectStck-rep2
python -m src.a15_corr.inspect_tables comparison 1518_abl_obstacle --dataset_names ObstacleTrack-test --method_names $methods_test_dev



python -m src.a15_corr.inspect_tables comparison 1518pnf_abl_obstacle --dataset_names ObstacleTrack-test --method_names $methods_abl &
python -m src.a15_corr.inspect_tables comparison 1518pnf_abl_LAF --dataset_names LostAndFound-testNoKnown --method_names $methods_abl &



"""

import numpy as np
from matplotlib import pyplot
from pandas import DataFrame, Series
import pandas
from easydict import EasyDict
import click

from ..paths import DIR_DATA

from .file_io import hdf5_read_hierarchy_from_file, hdf5_write_hierarchy_to_file, hdf5_write_hierarchy_to_group
from .baseline_names import external_method_name_mapping, plot_prc_pretty, multi_dset_table

from .metrics import EvaluatorSystem_PerPixBinaryClassification
from .discrepancy_experiments import ObstaclePipelineSystem_Discrepancy



BLUR_TABLE_METHODS = """
	ImgVsInp-archResy-NoiseImg-last
	ImgBlurInfer03VsInp-archResy-NoiseImg-last
	ImgBlurInfer05VsInp-archResy-NoiseImg-last
	ImgBlurInfer07VsInp-archResy-NoiseImg-last

	ImgBlurTrain03VsInp-archResy-NoiseImg-last
	ImgBlur03VsInp-archResy-NoiseImg-last
	ImgBlur03Inf05VsInp-archResy-NoiseImg-last
	ImgBlur03Inf07VsInp-archResy-NoiseImg-last

	ImgBlurTrain05VsInp-archResy-NoiseImg-last
	ImgBlur05VsInp-archResy-NoiseImg-last
	ImgBlur05Inf03VsInp-archResy-NoiseImg-last
	ImgBlur05Inf07VsInp-archResy-NoiseImg-last
""".split()

@click.command()
def main():
	methods = BLUR_TABLE_METHODS
	dsets = ['FishyLAF-LafRoi', 'RoadObstacles2048p-full']

	blur_keys = {
		cn: ObstaclePipelineSystem_Discrepancy.default_cfgs_by_name[cn].get('blur_key', (1, 1))
		for cn in methods
	}

	curve_module = EvaluatorSystem_PerPixBinaryClassification.get_implementation('perpixAP')
	curve_infos = [c for ds in dsets for c in curve_module.load_curve_infos(ds, methods)]

	tables = []
	for ds in dsets:
		curve_infos = curve_module.load_curve_infos(ds, methods)

		entries = [
			(*blur_keys[ci.method_name], ci.area_average_precision)
			for ci in curve_infos
		]

		table = DataFrame(data = entries, columns = ['blur_train', 'blur_infer', f'ap_{ds}'])
		tables.append(table)

	def procrow(row):
		a, b = row['ap_FishyLAF-LafRoi'], row['ap_RoadObstacles2048p-full']
		return a, b

	table_all = tables[0].merge(tables[1])
	table_all['combined'] = table_all.apply(procrow, axis=1)

	table_split = table_all.pivot(index='blur_infer', columns='blur_train', values=['ap_FishyLAF-LafRoi', 'ap_RoadObstacles2048p-full'])
	print(table_split)

	#table_combined = table_all.pivot(index='blur_train', columns='blur_infer', values=['combined'])
	table_combined = table_all.pivot(index='blur_infer', columns='blur_train', values=['combined'])

	print(table_combined)

	def float_format(f):
		return '-' if np.isnan(f) else f'{100*f:.01f}'

	def pair_format(ab):
		return ' | '.join(f'{100*f:.01f}' for f in ab)

	formats = {
		'ap_FishyLAF-LafRoi': float_format,
		'ap_RoadObstacles2048p-full': float_format,
		'combined': pair_format,
	}

	table_html = table_split.to_html(
		float_format = float_format,
	) + table_combined.to_html(
		formatters = [pair_format]*3,
	)

	table_tex = table_split.to_latex(
		formatters = formats,
	) + table_combined.to_latex(
		formatters = [pair_format]*3,
	)

	dir_out = DIR_DATA / f'1210metrics-{curve_module.cfg.name}' / 'tables'
	dir_out.mkdir(parents=True, exist_ok=True)
	(dir_out / f'TableBlur.html').write_text(table_html)
	(dir_out / f'TableBlur.tex').write_text(table_tex)


if __name__ == '__main__':
	main()
	
#python -m src.a12_inpainting.blur_table



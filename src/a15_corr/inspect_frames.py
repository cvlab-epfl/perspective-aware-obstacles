
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np
from easydict import EasyDict

from road_anomaly_benchmark.datasets.dataset_registry import DatasetRegistry

from ..paths import DIR_DATA

from road_anomaly_benchmark.evaluation import Evaluation, DIR_OUTPUTS, log
from road_anomaly_benchmark.metrics import MetricRegistry
from road_anomaly_benchmark.metrics.base import EvaluationMetric
from road_anomaly_benchmark.metrics.pixel_classification_curves import curves_from_cmats
from road_anomaly_benchmark.datasets import DatasetObstacleTrack

from ..common.jupyter_show_image import imread, imwrite, adapt_img_data

import cv2 as cv
from tqdm import tqdm

from multiprocessing.dummy import Pool as Pool_thread

import shutil

@MetricRegistry.register_class()
class MetricInspection(EvaluationMetric):

	configs = [
		EasyDict(
			name = "inspect-PixAp",
			base_metric = "PixBinaryClass",
		),
		EasyDict(
			name = "inspect-Instance",
			base_metric = "SegEval-ObstacleTrack",
		),
	]

	def init(self, method_name, dataset_name):
	
		if not hasattr(self, 'bg_metric'):
			self.bg_metric = MetricRegistry.get(self.cfg.base_metric)
			self.bg_metric.init(method_name, dataset_name)

			if self.cfg.base_metric == "PixBinaryClass":
				self.threshold = self.bg_metric.load(method_name=method_name, dataset_name=dataset_name).best_f1_threshold
			else:
				self.threshold = self.bg_metric.cfg.thresh_p

	def process_frame(self, *args, fid : str, **kwargs):
		r = self.bg_metric.process_frame(*args, **kwargs)


		if self.cfg.base_metric == "PixBinaryClass":
			best_thr_idx = r.bin_edges.__len__() - np.searchsorted(r.bin_edges[::-1], self.threshold)
			best_thr_idx = min(best_thr_idx, r.cmat_sum.__len__()-1)
			#print(best_thr_idx, '/', r.cmat_sum.shape[0], r.bin_edges[0], r.bin_edges[best_thr_idx], r.bin_edges[-1])
			best_cmat = r.cmat_sum[best_thr_idx]

			return EasyDict(
				fid = fid,
				best_cmat = best_cmat,
				num_pos = r.num_pos,
				num_neg = r.num_neg,

				bg_metric_result = r,
			)


		elif self.cfg.base_metric == "SegEval-ObstacleTrack":
			
			return EasyDict(
				fid = fid,
				num_fp = r['fp_50'],
				num_fn = r['fn_50'],

				bg_metric_result = r,
			)




	def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
		"""
		@param frame_results: sequence of outputs of `process_frame` for the whole dataset
		@return: 
		"""

		"""
		We want to catch:
		- large area FP
		- high confidence FP
		"""

		self.init(method_name=method_name, dataset_name=dataset_name)

		if self.cfg.base_metric == "PixBinaryClass":

			num_positives = sum(r.num_pos for r in frame_results)
			num_negatives = sum(r.num_neg for r in frame_results)
			cmats = np.array([r.best_cmat for r in frame_results])

			tp = cmats[:, 0, 0]
			fp = cmats[:, 0, 1]
			fn = cmats[:, 1, 0]
			tn = cmats[:, 1, 1]

			# tp_rates = tp / (tp+fn)
			# fp_rates = fp / (fp+tn)

			# precisions = tp / (tp+fp)
			# recalls = tp / (tp+fn)
			# f1_scores = (2 * tp) / (2 * tp + fp + fn)

			frame_wrongness = fn / num_positives + fp / (np.sum(fp) + np.sum(tp))

		elif self.cfg.base_metric == "SegEval-ObstacleTrack":

			fp = np.array([r.num_fp for r in frame_results])
			fn = np.array([r.num_fn for r in frame_results])

			frame_wrongness = fn + fp

		frames_wrongest = np.argsort(frame_wrongness)[::-1]
		frames_wrongest = frames_wrongest[:50]

		print(frame_wrongness[frames_wrongest], 'vs', frame_wrongness[:20])

		fids = [frame_results[i].fid for i in frames_wrongest]
		fns = fn[frames_wrongest]
		fps = fp[frames_wrongest]

		bg_ag = self.bg_metric.aggregate([r.bg_metric_result for r in frame_results], method_name, dataset_name)

		return EasyDict(
			method_name = method_name,
			dataset_name = dataset_name,

			threshold = self.threshold,
			fids = fids,
			num_false_negatives = list(map(int, fns)),
			num_false_positives = list(map(int, fps)),

			bg_metric_ag = bg_ag,
		)

	
	def persistence_path_data(self, method_name, dataset_name):
		return DIR_DATA / '15xx_Inspect' / f'{self.cfg.name}_worst_{method_name}_{dataset_name}.json'
	
	def persistence_path_plot(self, method_name, dataset_name):
		return DIR_DATA / '15xx_Inspect' / f'{self.cfg.name}_worst_{method_name}_{dataset_name}'

	def save(self, aggregated_result, method_name : str, dataset_name : str, path_override : Path = None):
		self.bg_metric.save(aggregated_result.bg_metric_ag, method_name, dataset_name, path_override=path_override)
		del aggregated_result['bg_metric_ag']

		out_path = path_override or self.persistence_path_data(method_name, dataset_name)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(json.dumps(aggregated_result, indent='	'))


	def load(self, method_name : str, dataset_name : str, path_override : Path = None):
		out_path = path_override or self.persistence_path_data(method_name, dataset_name)
		return EasyDict(json.loads(out_path.read_text()))

	def plot_single(self, ag, close : bool = True):
		method_name = ag.method_name
		dataset_name = ag.dataset_name
		out_dir = self.persistence_path_plot(method_name, dataset_name)
		out_dir.mkdir(parents=True, exist_ok=True)

		ev = Evaluation(
			method_name = method_name, 
			dataset_name = dataset_name,
		)

		self.init(method_name=method_name, dataset_name=dataset_name)

		dset = DatasetRegistry.get(dataset_name)

		def draw_frame(i_and_fid):
			i, fid = i_and_fid
			fr = dset.get_frame(fid, 'image', 'label_pixel_gt')
			#fr = dset[fid]
			#print(fr.keys())
			fr.anomaly_p = ev.channels['anomaly_p'].read(
				method_name=method_name,
				dset_name=fr.dset_name,
				fid=fr.fid,
			)

			mask_roi = fr.label_pixel_gt < 255
			canvas = fr.image.copy()

			# if self.cfg.base_metric == "PixBinaryClass":
			heatmap_color = adapt_img_data(fr.anomaly_p)

			#heatmap_color = np.zeros_like(canvas)

			mask = fr.anomaly_p > self.threshold
			mask_inset = cv.erode(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(bool)
			mask_contour = mask != mask_inset

			heatmap_color[mask_contour] = (10, 255, 240)

			canvas[mask_roi] = canvas[mask_roi]//2 + heatmap_color[mask_roi]//2
	
			#img = imread(dir_vis / f'{fid}_demo_anomalyP.webp')

			caption = f'{ag.num_false_negatives[i]} FN + {ag.num_false_positives[i]} FP'

			# shadow
			cv.putText(canvas, method_name, (18, 50), cv.FONT_HERSHEY_DUPLEX, 2, color=(0, 0, 0), thickness=6)
			# foreground
			cv.putText(canvas, method_name, (18, 50), cv.FONT_HERSHEY_DUPLEX, 2, color=(250, 10, 200), thickness=2)

			cv.putText(canvas, caption, (18, 120), cv.FONT_HERSHEY_DUPLEX, 2, color=(0, 0, 0), thickness=6)
			# foreground
			cv.putText(canvas, caption, (18, 120), cv.FONT_HERSHEY_DUPLEX, 2, color=(250, 10, 200), thickness=2)


			out = out_dir / f'{i:03d}__{fid}.webp'
			# print(out)
			# shutil.copy((), out)
			imwrite(out, canvas)


		with Pool_thread(12) as pool:
			jobs = list(enumerate(ag.fids))

			for _ in tqdm(pool.imap(draw_frame, jobs), total=ag.fids.__len__()):
				...

			# for i, fid in tqdm(enumerate(ag.fids), total=ag.fids.__len__()):

	
import multiprocessing
from functools import partial

class EvaluationMultiMetric(Evaluation):

	

	def run_metric_single(self, metric_name, sample=None, frame_vis=False, default_instancer=True):
		raise NotImplementedError()

	@classmethod
	def metric_worker(cls, method_name, metric_names : list, frame_vis, default_instancer, dataset_name_and_frame_idx):
		try:
			dataset_name, frame_idx = dataset_name_and_frame_idx

			dset = DatasetRegistry.get(dataset_name)
			metrics = cls.get_metrics(metric_names)

			frame_vis_only = frame_vis == 'only'

			if default_instancer:
				for metric in metrics.values():
					metric.init(method_name, dataset_name)

			if not frame_vis_only:
				fr = dset[frame_idx]
			else:
				fr = dset.get_frame(frame_idx, 'image')

			frame = {"method_name": method_name, "dset_name": fr.dset_name, "fid": fr.fid}
			fr["mask_path"] = cls.channels['anomaly_mask_path'].format(**frame)

			if default_instancer:
				heatmap = cls.channels['anomaly_p'].read(
					method_name=method_name,
					dset_name=fr.dset_name,
					fid=fr.fid,
				)
				if heatmap.shape[1] < fr.image.shape[1]:
					heatmap = cv.resize(heatmap.astype(np.float32), fr.image.shape[:2][::-1], interpolation=cv.INTER_LINEAR)
			else:
				heatmap = None

			results = {
				metric_name: metric.process_frame(
					anomaly_p = heatmap,
					method_name = method_name, 
					visualize = frame_vis,
					**fr,
				)
				for metric_name, metric in metrics.items()
			}
			return results

		except Exception as e:
			log.exception(f'Metric worker {e}')
			return e

	@staticmethod
	def get_metrics(metric_name_or_list):
		if isinstance(metric_name_or_list, str):
			metric_names = [metric_name_or_list]
		else:
			metric_names = metric_name_or_list

		return {
			metric_name: MetricRegistry.get(metric_name)
			for metric_name in metric_names
		}

	def run_metric_parallel(self, metric_name, sample=None, frame_vis=False, default_instancer=True):
		metrics = self.get_metrics(metric_name)

		if sample is not None:
			dset_name, frame_indices = sample
		else:
			dset_name = self.dataset_name
			frame_indices = range(self.get_dataset().__len__())

		tasks = [
			(dset_name, idx)
			for idx in frame_indices
		]

		results_by_metric = {
			metric_name: []
			for metric_name in metrics.keys()
		}

		with multiprocessing.Pool() as pool:
			it = pool.imap_unordered(
				partial(self.metric_worker, self.method_name, metric_name, frame_vis, default_instancer),
				tasks,
				chunksize = 4,
			)
			
			for result in tqdm(it, total=tasks.__len__()):
				if isinstance(result, Exception):
					raise result
				else:
					for m, r in result.items():
						results_by_metric[m].append(r)

		ags = {
			m_name: m.aggregate(	
				results_by_metric[m_name],
				method_name = self.method_name,
				dataset_name = dset_name,
			)
			for m_name, m in metrics.items()
		}

		return ags

	def calculate_metric_from_saved_outputs(self, metric_name, sample=None, parallel=True, show_plot=False, frame_vis=False, default_instancer=True):
		metrics = self.get_metrics(metric_name)

		ags = self.run_metric_parallel(metric_name, sample, frame_vis, default_instancer)

		dset_name = sample[0] if sample is not None else self.dataset_name

		for m_name, m in metrics.items():
			m.save(
				ags[m_name], 
				method_name = self.method_name,
				dataset_name = dset_name,
			)

			#metric.plot_single(ag, close = not show_plot)

		return ags

import click

def name_list(name_list):
	return [name for name in name_list.split(',') if name]


def run_usual_metrics(method_name, dset_name, metrics = None, load=False):
	ev = EvaluationMultiMetric(
		method_name = method_name, 
		dataset_name = dset_name,
	)

	if metrics is None:
		('PixBinaryClass', 'inspect-Instance')
		
		ags = ev.calculate_metric_from_saved_outputs(
			['PixBinaryClass'],
			parallel = True,
			show_plot = False,
			frame_vis = False,
		)
		ags = ev.calculate_metric_from_saved_outputs(
			['inspect-Instance', 'SegEval-FixedThr99', 'SegEval-FixedThr80' , 'SegEval-FixedThr50'],
			parallel = True,
			show_plot = False,
			frame_vis = False,
		)

	else:
		for metric in metrics:
			#log.info(f'Metric: {metric} | Method : {method} | Dataset : {dset}')

			if not load:
				ag = ev.calculate_metric_from_saved_outputs(
					metric,
					sample = None,
					parallel = True,
					show_plot = False,
					frame_vis = False,
				)

				print(ag)

			else:
				m = MetricRegistry.get(metric)
				ag = m.load(method_name = method_name, dataset_name = dset_name)
				m.plot_single(ag)





from road_anomaly_benchmark.metrics import MetricSegment

@MetricRegistry.register_class()
class MetricSegmentFixedthr(MetricSegment):

	seg_defaults = dict(
		thresh_p=None,
		thresh_sIoU=np.linspace(0.25, 0.75, 11, endpoint=True),
		thresh_segsize=50,
		thresh_instsize=10,
	)

	configs = [
		EasyDict(
			name='SegEval-FixedThr99',
			thresh_for_dsets = {
				'LostAndFound': 0.01,
				'ObstacleTrack': 0.99,
			},
			**seg_defaults,
		),
		EasyDict(
			name='SegEval-FixedThr80',
			thresh_for_dsets = {
				'LostAndFound': 0.20,
				'ObstacleTrack': 0.80,
			},
			**seg_defaults,
		),
		EasyDict(
			name='SegEval-FixedThr50',
			thresh_for_dsets = {
				'LostAndFound': 0.5,
				'ObstacleTrack': 0.5,
			},
			**seg_defaults,
		),
	]

	def get_thresh_p_from_curve(self, method_name, dataset_name):
		for dsn, thr in self.cfg.thresh_for_dsets.items():
			if dataset_name.startswith(dsn):
				self.cfg.thresh_p = thr
				return self.cfg.thresh_p

		print(f'No fixed threhsold for dset {dataset_name}')
		super().get_thresh_p_from_curve(method_name, dataset_name)
		return self.cfg.thresh_p


@DatasetRegistry.register_class()
class DatasetObstacleTrackExtraSplits(DatasetObstacleTrack):
	configs = [
		dict(
			# default: exclude special weather and night
			name = 'ObstacleTrack-testNoShiny',
			scenes = DatasetObstacleTrack.SCENES_ALL.difference({'snowstorm1', 'snowstorm2', 'driveway', 'validation', 'one-way-street'}),
			#expected_length = 327,
			**DatasetObstacleTrack.DEFAULTS,
		),
	]



main = click.Group()

@main.command()
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
@click.option('--load/--no-load', default=False)
def worst(method_names, metric_names, dataset_names, load=False):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	for dset in dataset_names:
		for method in method_names:
			for metric in metric_names:

				#log.info(f'Metric: {metric} | Method : {method} | Dataset : {dset}')

				ev = Evaluation(
					method_name = method, 
					dataset_name = dset,
				)


				if not load:
					ag = ev.calculate_metric_from_saved_outputs(
						metric,
						sample = None,
						parallel = True,
						show_plot = False,
						frame_vis = False,
						default_instancer = True,
					)

					print(ag)
				else:
					m = MetricRegistry.get(metric)
					ag = m.load(method_name = method, dataset_name = dset)
					m.plot_single(ag)


from collections import Counter

@main.command()
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
def worst_framelist(method_names, metric_names, dataset_names, load=True):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	for dset in dataset_names:
		for metric in metric_names:
			m = MetricRegistry.get(metric)

			ag_by_name = {}
			for method in method_names:
				try:
					ag_by_name[method] = m.load(method_name = method, dataset_name = dset)
				except FileNotFoundError:
					...
						
			fid_counter = Counter()

			for ag in ag_by_name.values():
				fid_counter.update(ag.fids)

			Path(f'worst_frames_{dset}.json').write_text(json.dumps(dict(
				name_to_count = dict(fid_counter),
				sorted = fid_counter.most_common(50),
			)))

			print(fid_counter)

			print(fid_counter.most_common(50))

"""
python -m src.a15_corr.inspect_frames worst-framelist inspect-Instance $methods_reeval ObstacleTrack-test
"""






@main.command()
@click.argument('dataset_names', type=str)
@click.argument('method_names', type=str)
@click.option('--name', type=str, default='All')
def thresholds(dataset_names, method_names, name='ALL'):
	"""
	Inspect the thresholds used for binary segmentation in each method.
	The thresholds are derived from the pixel metrics.
	"""
	from pandas import DataFrame, Series
	from road_anomaly_benchmark.__main__ import wrap_html_table, DIR_OUTPUTS

	dataset_names = name_list(dataset_names)
	method_names = name_list(method_names)
	metric = MetricRegistry.get('SegEval-ObstacleTrack')

	columns = {}
	def get_col(name):
		c = columns.get(name)
		if c is not None:
			return c
		else:
			return columns.setdefault(name, Series(dtype=np.float64))

	for method in method_names:
		for dset in dataset_names:
			try:
				ag = metric.load(method_name = method, dataset_name = dset)
				thr = metric.get_thresh_p_from_curve(method_name = method, dataset_name = dset)

				get_col(f'{dset}.thr')[method] = thr
				get_col(f'{dset}.segF1')[method] = ag.f1_mean
			
			except FileNotFoundError:
				...


	table = DataFrame(data = columns)
	print(table)
	title = f'Thresholds_{name}'
	table_html = wrap_html_table(
		table = table.to_html(
			classes = ('display', 'compact'), 
			float_format = lambda f: f'{100*f:.01f}',
			na_rep = '-',
		),
		title = title,
	)
	out_f = DIR_OUTPUTS / 'tables' / title
	out_f.parent.mkdir(parents=True, exist_ok=True)
	out_f.with_suffix('.html').write_text(table_html)


@main.command()
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
def metrics(metric_names, method_names, dataset_names):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	for dset in dataset_names:
		for method in method_names:

			print(f'Metrics: {metric_names} | Method : {method} | Dataset : {dset}')

			ev = EvaluationMultiMetric(
				method_name = method, 
				dataset_name = dset,
			)

			try:
				ags = ev.calculate_metric_from_saved_outputs(
					metric_names,
					parallel = True,
					show_plot = False,
					frame_vis = False,
				)
			except FileNotFoundError as e:
				print('Skip, no file', e)


from ..a12_inpainting.discrepancy_experiments import ObstaclePipelineSystem
from ..a14_perspective.cityscapes_pitch_angles import invent_cam_and_persp_from_horizon, gen_perspective_scale_map
from ..a14_perspective.pos_enc_xy import get_yx_maps
from ..a12_inpainting.vis_imgproc import image_montage_same_shape


from matplotlib import cm
from time import sleep
from queue import Queue
from threading import Thread

from ..pipeline.frame import Frame

class VideoDset:
	dset = None # for code using dset.dset

	def __init__(self, video_path, step=2, name=None, horizon = None):
		video_path = Path(video_path)
		self.name = name or video_path.stem
		
		self.step = 2

		self.images = dict()
		self.vid_reader = cv.VideoCapture(str(video_path))
		self.video_path = video_path
		frame_idx = 0

		self.num_frames = int(self.vid_reader.get(cv.CAP_PROP_FRAME_COUNT))
		print('discovered frames ', self.num_frames)
		self.frame_idx = 0

		self.horizon = horizon
		if horizon is not None:
			self.pmap = None
			
		self.pos_encoding_X = self.pos_encoding_Y = None

		self.task_queue = Queue(maxsize=32)
		self.vid_reader_thread = Thread(target = self.video_extraction_thread)
		self.vid_reader_thread.start()


	def video_extraction_thread(self):
		while self.frame_idx < self.num_frames:
			status, image = self.vid_reader.read()

			if not status:
				print(f"Video {self.video_path} stopped after {self.frame_idx} frames")
				return
			
			self.frame_idx += 1
			if self.frame_idx % self.step != 0:
				continue

			image = image[:, :, ::-1].copy()

			fr = Frame(
				fid = f'{self.name}_{self.frame_idx//self.step:05d}',
				image = image,
				dset = None,
			)

			if self.horizon is not None:
				if self.pmap is None or self.pmap.shape[:2] != image.shape[:2]:
					self.cam_info, self.perspective_info = invent_cam_and_persp_from_horizon(image.shape[:2], self.horizon)

					self.pmap = gen_perspective_scale_map(
						image.shape[:2], 
						self.perspective_info.horizon_level, 
						self.perspective_info.pix_per_meter_slope,
					)
				fr.perspective_scale_map = self.pmap

			pos_YX = get_yx_maps(image.shape[:2])
			fr.pos_encoding_X = pos_YX[1]
			fr.pos_encoding_Y = pos_YX[0]

			self.task_queue.put(fr)
			

	def __len__(self):
		return self.num_frames // self.step

	def __getitem__(self, idx):
		fr = self.task_queue.get()
		self.task_queue.task_done()
		return fr



@main.command()
@click.argument('method_name', type=str)
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--name', type=str)
@click.option('--step', type=int, default=2)
@click.option('--horizon')
@click.option('--downsample', type=int, default=1)
def video(method_name, video_path, step, name=None, horizon=None, downsample=1):
	video_path = Path(video_path)
	name = name or video_path.stem
	dir_out = DIR_DATA / '1522_Video' / name / method_name

	if horizon is not None:
		horizon = int(horizon)

	# vid_reader = cv.VideoCapture(str(video_path))
	# frame_idx = 0

	# num_frames = int(vid_reader.get(cv.CAP_PROP_FRAME_COUNT))
	# print(f'Video {video_path} has {num_frames} frames')


	method = ObstaclePipelineSystem.get_implementation(method_name)
	method.load()

	vid_dset = VideoDset(video_path, step=step, name=name, horizon=horizon)


	cmap = cm.get_cmap('coolwarm') #cm.get_cmap('magma')

	with ThreadPoolExecutor(8) as tp:

		def write_job(image, anomaly_p, fid):
			if downsample != 1:
				anomaly_p = anomaly_p[::downsample, ::downsample]
			pred_image = cmap(anomaly_p, bytes=True)[:, :, :3]
			imwrite(dir_out / f'{fid}.webp', pred_image)

		def process_func(image, anomaly_p, fid, **_):
			tp.submit(write_job, image, anomaly_p, fid)

		method.predict_frames(vid_dset, process_func=process_func)


@main.command()
@click.argument('method_names', type=str)
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--name', type=str)
@click.option('--step', type=int, default=2)
@click.option('--downsample', type=int, default=1)
@click.option('--mask', type=click.Path(exists=True, dir_okay=False))
def video_fuse(method_names, video_path, step, name=None, horizon=None, downsample=1, mask=None):
	method_names = name_list(method_names)
	video_path = Path(video_path)
	name = name or video_path.stem
	dir_base = DIR_DATA / '1522_Video' / name
	dir_out = dir_base / 'fused'

	if mask is not None:
		mask = imread(mask)
		if mask.shape.__len__() > 2:
			mask = mask[:, :, 0]

		mask = mask > 128

		print(mask.shape)
		if downsample != 1:
			mask = mask[::downsample, ::downsample]

	vid_dset = VideoDset(video_path, step=step, name=name, horizon=horizon)

	m_rename = {
		'1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1': 'Ours',
		'1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1': 'No perspective aware synth',
		'Segmi_Entropy_max': 'Entropy maximization',
	}

	captions = [''] + [m_rename[n] for n in method_names]

	with ThreadPoolExecutor(8) as tp:

		def write_job(image, fid):
			frame_filename = f'{fid}.webp'

			if downsample == 2:
				image = cv.pyrDown(image)

			grid = [image]

			for met in method_names:
				pimg = imread(dir_base / met / frame_filename)
				# todo crop

				if pimg.shape[0] > image.shape[0]:
					pimg = pimg[::2, ::2]

				if mask is not None:
					ic = image.copy()
					ic[mask] = pimg[mask]
					pimg = ic

				grid.append(pimg)

			demo_img = image_montage_same_shape(grid, num_cols=2, border=4, captions=captions, caption_size=1)

			imwrite(dir_out / frame_filename, demo_img)

		for fr in tqdm(vid_dset):
			tp.submit(write_job, fr.image, fr.fid)
			#write_job(fr.image, fr.fid)


	# cmap = cm.get_cmap('magma')

	# for frame_idx in tqdm(range(num_frames)):
	# 	status, image = vid_reader.read()
	# 	image = image[:, :, ::-1].copy()

	# 	fr = EasyDict(
	# 		fid = f'{name}_{frame_idx:05d}',
	# 		image = image,
	# 	)

	# 	pred = method.predict_frames([fr])[0].anomaly_p
	# 	pred_image = cmap(pred, bytes=True)[:, :, :3]

	# 	img_out = np.concatenate([
	# 		image, pred_image,
	# 	], axis=1)

	# 	imwrite(dir_out / f'{name}_{frame_idx:05d}.webp', img_out)
			




if __name__ == '__main__':
	main()

# python -m src.a15_corr.inspect_frames worst inspect-PixAp Entropy_max Erasing-20  

# python -m src.a15_corr.inspect_frames worst inspect-PixAp 1500-1-CrBaseR101 ObstacleTrack-all



# python -m src.a15_corr.inspect_frames metrics SegEval-FixedThr99,SegEval-FixedThr80,SegEval-FixedThr50 Dummy ObstacleTrack-validation
# python -m road_anomaly_benchmark comparison FixedThrSegsDummy SegEval-FixedThr99,SegEval-FixedThr80,SegEval-FixedThr50 Dummy ObstacleTrack-validation --imports src.a15_corr.inspect_frames --names benchmark-names/names-fixedthr.json

"""




python -m src.a15_corr.inspect_frames video 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_10_1.mp4 --downsample 2 --horizon 250   
python -m src.a15_corr.inspect_frames video 1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_10_1.mp4 --downsample 2 --horizon 250   
python -m src.a15_corr.inspect_frames video Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_10_1.mp4 --horizon 250   
python -m src.a15_corr.inspect_frames video-fuse 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1,1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1,Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_10_1.mp4 --downsample 2 --mask /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_10_1_mask.png
ffmpeg -r 30 -f image2 -s 1924x1084 -i supcut_10_1/fused/supcut_10_1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p gen_supcut_10_1.mp4



#python -m src.a15_corr.inspect_frames video 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_02_loc1.mp4 --downsample 2 --horizon 250   
python -m src.a15_corr.inspect_frames video 1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_02_loc1.mp4 --downsample 2 --horizon 195   
python -m src.a15_corr.inspect_frames video Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_02_loc1.mp4 --horizon 195
python -m src.a15_corr.inspect_frames video-fuse 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1,1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1,Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_02_loc1.mp4 --downsample 2 --mask //cvlabdata2/home/lis/data/1522_Video/supcut_02_loc1/supcut_02_loc1.png
ffmpeg -r 30 -f image2 -s 1924x1084 -i supcut_02_loc1/fused/supcut_02_loc1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p gen_supcut_02_loc1.mp4


python -m src.a15_corr.inspect_frames video 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_05b.mp4 --horizon 320
python -m src.a15_corr.inspect_frames video 1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1 /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_05b.mp4 --horizon 320
python -m src.a15_corr.inspect_frames video Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_05b.mp4 --horizon 320   
python -m src.a15_corr.inspect_frames video-fuse 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1,1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1,Segmi_Entropy_max /cvlabsrc1/cvlab/dataset_RoadAnomalyVideos/supcut_05b.mp4 --downsample 2
ffmpeg -r 30 -f image2 -s 1924x1084 -i supcut_05b/fused/supcut_05b_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p gen_supcut_05b.mp4

cd /cvlabdata2/home/lis/data/1522_Video/supcut_10_1/


ffmpeg -r 30 -f image2 -s 960x540 -i 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1/supcut_10_1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p supcut_10_1_Ours.mp4
ffmpeg -r 30 -f image2 -s 960x540 -i 1518-5-D2b-Noise4-BaSpl-NoSumPdirect-rep1/supcut_10_1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p supcut_10_1_NoPSynth.mp4


ffmpeg 
	-r 30 -f image2 -s 960x540 -i 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1/supcut_10_1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p supcut_10_1_Ours.mp4
	-r 30 -f image2 -s 960x540 -i 1518-4-P3D-Noise4-BaSpl-NoSumPdirect-rep1/supcut_10_1_%05d.webp -vcodec libx264 -crf 20  -pix_fmt yuv420p supcut_10_1_Ours.mp4
	-i input1 -filter_complex "[0]pad=iw+5:color=black[left];[left][1]hstack=inputs=2" output

"""



from pathlib import Path
import json
import asyncio
import logging
from typing import List


import click
from tqdm import tqdm

log = logging.getLogger('compress_images')
log.setLevel(logging.DEBUG)

def async_map(func, tasks, num_concurrent=4):

	num_tasks = tasks.__len__()

	queue = asyncio.Queue()
	for idx, task in enumerate(tasks): queue.put_nowait((idx, task))

	results = [None] * num_tasks

	pbar = tqdm(total = num_tasks)
	
	async def async_worker():
		while not queue.empty():
			idx, task = queue.get_nowait()

			result = await func(task)
			results[idx] = result
			pbar.update(1)

			queue.task_done()

	joint_future = asyncio.gather(
		*( async_worker() for i in range(num_concurrent))
	)

	asyncio.get_event_loop().run_until_complete(joint_future)
	pbar.close()

	return results

async def run_external_program(cmd):

	try:
		proc = await asyncio.create_subprocess_exec(
			*map(str, cmd), # str to convert Path objects
			stdout=asyncio.subprocess.PIPE, 
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()
	except Exception as e:
		cmd_as_str = ' '.join(map(str, cmd))
		log.exception(f'run_external_program({cmd_as_str})')
		raise e

	if proc.returncode != 0:
		cmd_as_str = ' '.join(map(str, cmd))
		log.error(f"""Command {cmd_as_str} error, retcode = {proc.returncode}
--Stderr--
{stderr}
--Stdout--
{stdout}""")


# def cmd_jpg(src, dest):
# 	return ['convert', src, '-quality', '80', dest.with_suffix('.jpg')]

# def cmd_webp(src, dest):
# 	return ['cwebp',  
# 		src, '-o', dest.with_suffix('.webp'), 
# 		'-q', '90',
# 		'-sharp_yuv', 
# 		'-m', '6',
# 	]

# "{src} -o {dest} -q 90 -sharp_yuv -m 6"

CHANNEL_TEMPLATES = {
	'image': '{fid}.webp',
	'labels': '{fid}.labels/labels_semantic.png',
	'labels_color': '{fid}.labels/labels_semantic_color.png',
	'instances': '{fid}.labels/labels_instance.png',
}

CHANNEL_TEMPLATES_FLAT = {
	'image': 'images/{fid}.webp',
	'labels': 'labels_masks/{fid}_labels_semantic.png',
	'labels_color': 'labels_masks/{fid}_labels_semantic_color.png',
	'instances': 'labels_masks/{fid}_labels_instance.png',
}

# def cmd_resize_labels(src, dest):
# 	return f"""
# 	convert {src} -filter point -resize 2000x1500 {dest}
# 	""".strip().split()

CMD_RESIZE_LABEL = {
	'cwebp': "convert {src} -filter point -resize 2000x1500 {dest}",
	'pyrDown': "convert {src} -filter point -resize 2000x1500 {dest}",
	'2048': "python compress_2048.py {src} {dest} --label",
	'2048b3': "python compress_2048.py {src} {dest} --label",
	'2048p': "python compress_2048_padding.py {src} {dest} --label",
	'1920': "convert {src} -gravity South -extent 4000x2250 -filter Point -resize 1920x1080 {dest}",
}

CMD_RESIZE_IMAGE = {
	'cwebp': "cwebp {src} -o {dest} -q 95 -m 6 -resize 2000 1500",
	'pyrDown': "python compress_pyrDown.py {src} {dest}",
	'2048': "python compress_2048.py {src} {dest} --no-label --blur 5",
	'2048b3': "python compress_2048.py {src} {dest} --no-label --blur 3",
	'2048p': "python compress_2048_padding.py {src} {dest} --no-label",
	'1920': "convert {src} -gravity South -extent 4000x2250 -filter Gaussian -resize 1920x1080 -quality 85 {dest}",
}



def locate_frames(src_dir):
	src_dir = Path(src_dir)

	src_files = list(src_dir.glob('**/*.webp'))
	src_fids = [str((p.parent/p.stem).relative_to(src_dir)) for p in src_files]

	print('Found fids:', '\n	'.join(src_fids))

	# check if all files exist

	for fid in src_fids:
		for ch_name, ch_tmpl in CHANNEL_TEMPLATES.items():
			ch_path = src_dir / ch_tmpl.format(fid=fid)

			if not ch_path.is_file():
				raise FileNotFoundError(ch_path)
	
	return src_fids
	
def construct_conversion_jobs(fid_pairs, src_dir, dest_dir, method='pyrDown', chans_in=CHANNEL_TEMPLATES, chans_out=CHANNEL_TEMPLATES):
	"""
	@param fid_pairs: [(src_fid1, dest_fid1), (src_fid2, dest_fid2), ...]
	"""
	src_dir = Path(src_dir)
	dest_dir = Path(dest_dir)

	jobs_dirs = []
	jobs_images = []
	jobs_labels = []

	for (fid_src, fid_dest) in fid_pairs:

		for ch_name, ch_tmpl_in in chans_in.items():
			src_path = src_dir / ch_tmpl_in.format(fid=fid_src)
			ch_tmpl_out = chans_out[ch_name]
			dest_path = dest_dir / ch_tmpl_out.format(fid=fid_dest)

			if ch_name == 'image':
				jobs_images.append(
					CMD_RESIZE_IMAGE[method].format(
						src = src_path,
						dest = dest_path,
					).strip().split()
				)
			else:
				jobs_labels.append(
					CMD_RESIZE_LABEL[method].format(
						src = src_path,
						dest = dest_path,
					).strip().split()
				)
				
			jobs_dirs.append(dest_path.parent)

	return dict(
		dirs = jobs_dirs,
		images = jobs_images,
		labels = jobs_labels,
	)

# def construct_conversion_jobs(fids, src_dir, dest_dir, method='pyrDown'):
# 	src_dir = Path(src_dir)
# 	dest_dir = Path(dest_dir)

# 	paths = {
# 		'dirs':  set(),
# 		'images': set(),
# 		'labels': set(),
# 	}

# 	for fid in fids:
# 		for ch_name, ch_tmpl in CHANNEL_TEMPLATES.items():
# 			rel_name = ch_tmpl.format(fid=fid)
# 			ch_path = src_dir / rel_name
	
# 			if ch_name == 'image':
# 				paths['images'].add(rel_name)
# 			else:
# 				paths['labels'].add(rel_name)

# 			paths['dirs'].add(Path(rel_name).parent)

# 	jobs_dirs = [dest_dir / d for d in paths['dirs']]

# 	jobs_images = [
# 		CMD_RESIZE_IMAGE[method].format(
# 			src = src_dir / p,
# 			dest = dest_dir / p,
# 		).strip().split()
# 		for p in paths['images']
# 	]

# 	jobs_labels = [
# 		cmd_resize_labels(src = src_dir / p, dest = dest_dir / p)
# 		for p in paths['labels']
# 	]

# 	return dict(
# 		dirs = jobs_dirs,
# 		images = jobs_images,
# 		labels = jobs_labels,
# 	)


scene_replacements = {
	'gasp': 'greyasphalt',
	'dasp': 'darkasphalt',
	'gravel': 'gravel',
	'paving': 'paving',
	'dasp2': 'darkasphalt2',
	'motorway': 'motorway',
}

rename_replacement_rules = [
	# scene_gasp/gasp_obj_n -> gasp_obj_n
	(r'scene_(?P<scene>\w+)/\1_(?P<obj>\w+)', lambda scenes_repl, groups: f'{scenes_repl[groups.scene]}_{groups.obj}'),
	# scene_sc/obj_n -> Sc_obj_n
	(r'scene_(?P<scene>\w+)/(?P<obj>\w+)', lambda scenes_repl, groups: f'{scenes_repl[groups.scene]}_{groups.obj}'),
]

import re
from types import SimpleNamespace

def replace_fid(fid : str):
	for query, replacement_tmpl in rename_replacement_rules:
		m = re.match(query, fid)
		if m:
			groups = SimpleNamespace(**m.groupdict())
			repl = replacement_tmpl(
				scenes_repl = scene_replacements,
				groups = groups,
			)
			return repl

	raise RuntimeError(f'No replacement for {fid}')


def write_index(fids_dest : List[str], dest_dir : Path):

	splits_by_scene = {}

	for fid in fids_dest:
		scene, _ = fid.split('_', maxsplit=1)
		splits_by_scene.setdefault(scene, []).append(fid)

	splits = {
		f'scene_{sc_name}': list(sorted(sc_frames))
		for sc_name, sc_frames in splits_by_scene.items()
	}

	splits['full'] = list(sorted(fids_dest))

	splits['UnusualSurface'] = list(sorted(
		splits_by_scene['paving'] + splits_by_scene['gravel'] + splits_by_scene['darkasphalt2']
	))

	(dest_dir / 'splits.json').write_text(json.dumps(splits, indent='	'))


@click.command()
@click.argument('src_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('dest_dir', type=click.Path())
@click.option('--concurrent', default=12)
@click.option('--method', type=click.Choice(CMD_RESIZE_IMAGE.keys()), default='pyrDown')
@click.option('--rename/--no-rename', default=True)
@click.option('--flat/--no-flat', default=False)
def main(src_dir, dest_dir, method, concurrent, rename, flat):
	src_dir = Path(src_dir)
	dest_dir = Path(dest_dir)

	fids = locate_frames(src_dir)
	print(f'Found {fids.__len__()} frames')

	if rename:
		fid_pairs = [(fid, replace_fid(fid)) for fid in fids]
	else:
		fid_pairs = [(fid, fid) for fid in fids]

	jobs = construct_conversion_jobs(
		fid_pairs,
		src_dir, 
		dest_dir if flat else dest_dir / 'frames', 
		method=method, 
		chans_in = CHANNEL_TEMPLATES,
		chans_out = CHANNEL_TEMPLATES_FLAT if flat else CHANNEL_TEMPLATES,
	)

	for d in jobs['dirs']:
		d.mkdir(parents=True, exist_ok=True)


	fids_dest = [fid_dest for (fid_src, fid_dest) in fid_pairs]
	fids_dest.sort()
	write_index(fids_dest, dest_dir)

	# (dest_dir / 'frame_list.json').write_text(json.dumps(fids_dest, indent='	'))

	# print('\n'.join(map(str, jobs['images'])))
	# print('\n'.join(map(str, jobs['labels'])))
	# return

	cmds = jobs['images'] + jobs['labels']

	# print('\n'.join(cmds))

	async_map(run_external_program, cmds, num_concurrent=concurrent)



if __name__ == '__main__':
	main()

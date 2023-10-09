
import multiprocessing
import os
import numpy as np
from functools import partial
from tqdm import tqdm


def queue_parallelism_worker(worker_init_func, worker_task_func, task_queue, solution_queue):
	np.random.seed(os.getpid())

	worker_init_func()

	while not task_queue.empty():
		task = task_queue.get()
		result = worker_task_func(task)
		solution_queue.put(result)

def run_parallel(worker_init_func, worker_task_func, tasks, host_collect_func = None, num_workers=8, progress_bar=False):

	task_queue = multiprocessing.Queue()
	solution_queue = multiprocessing.Queue()

	worker_kwargs = dict(
		worker_init_func = worker_init_func,
		worker_task_func = worker_task_func,
		task_queue = task_queue,
		solution_queue = solution_queue,
	)

	for t in tasks:
		task_queue.put(t)

	workers = [
		multiprocessing.Process(target=queue_parallelism_worker, kwargs = worker_kwargs, daemon=True)
		for i in range(num_workers)
	]
	
	try:
		for w in workers:
			w.start()

		num_tasks = tasks.__len__()
		task_iter = tqdm(range(num_tasks)) if progress_bar else range(num_tasks)

		for i in task_iter:
			result = solution_queue.get()
		
			if host_collect_func is not None:
				host_collect_func(result)

	finally:
		for w in workers:
			if w.is_alive():
				w.terminate()

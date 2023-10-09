from pathlib import Path
import numpy as np
import h5py

class ExtraLoaderAttentropyForSampler:
				
	DIR_ATTN = Path('/cvlabdata2/home/lis/data/2010_SynthAtten/')
	
	FIELDS = ['attentropy']
	
	def __init__(self, attn_name):
		self.attn_name = attn_name
	
	def read_value(self, fid, dset, image, **_):
		subdir = dset.cfg.storage_dir.split('/')[-1]
		# print('fid', fid, 'subdir', subdir)
		data_path = self.DIR_ATTN / subdir / self.attn_name / f'{fid}_ent.hdf5'

		with h5py.File(data_path, 'r') as data_file:
			f = data_file['attention_layers'][:]

		f = f.astype(np.float32)
		
		ratio = image.shape[0] // f.shape[1]
		
		# f = cv2.resize(f, frame.image.shape[:2][::-1])
		f = np.repeat(f, ratio, axis=1)
		f = np.repeat(f, ratio, axis=2)
		f = np.transpose(f, [1,2,0]) # HWC for cropping
		
		return f
	

DIR_ATTN = Path('/cvlabdata2/home/lis/data/2010_SynthAtten/')


def load_attentropy(fid, dset_name, image, attn_name, **_):
	data_path = DIR_ATTN / dset_name / attn_name / f'{fid}_ent.hdf5'

	with h5py.File(data_path, 'r') as data_file:
		f = data_file['attention_layers'][:]

	f = f.astype(np.float32)
	
	ratio = image.shape[0] // f.shape[1]
	
	# f = cv2.resize(f, frame.image.shape[:2][::-1])
	f = np.repeat(f, ratio, axis=1)
	f = np.repeat(f, ratio, axis=2)
	f = np.transpose(f, [1,2,0]) # HWC for cropping
	
	return dict(
		attentropy = f,
	)

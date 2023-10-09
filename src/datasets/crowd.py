from .dataset import *

class DatasetVeniceSeq(DatasetBase):

	def __init__(self, dir_root='/cvlabdata1/home/lis/crowd/venice', seq='4895', b_cache=True):
		super().__init__()
		self.dir_root = dir_root
		self.seq = seq
		
		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/images/{dset.seq}/{fid}{channel.img_ext}', 
				img_ext = '.jpg',
			),
			head_labels = ChannelLoaderHDF5(
				file_path_tmpl = '{dset.dir_root}/labels/{dset.seq}.hdf5',
				var_name_tmpl = '{fid}',
			),
		)
		
		self.tr_post_load_pre_cache = TrsChain(
			self.tr_labels_to_count,
		)
		
		if seq == '4895':		
			
			sample_h5_file = pp(DIR_BASELINE_4895, 'sample.h5')
			self.add_channels(
				pred_scnn = ChannelLoaderNpy(file_path_tmpl = partial(self.path_for_prediction, 'scnn')),
				pred_mcnn = ChannelLoaderNpy(file_path_tmpl = partial(self.path_for_prediction, 'mcnn')),
# The sample.h5 file has 68 elements but the annotations have 24 elements
# Probably we have predictions for some unlabeled frames
#
# 				roi = ChannelLoaderHDF5(
# 					file_path_tmpl = sample_h5_file,
# 					var_name_tmpl = 'roi',
# 					index_func = self.fid_to_index,
# 				),
# 				count_from_sample = ChannelLoaderHDF5(
# 					file_path_tmpl = sample_h5_file,
# 					var_name_tmpl = 'count',
# 					index_func = self.fid_to_index,
# 				),
 				image_from_file = ChannelLoaderHDF5(
 					file_path_tmpl = sample_h5_file,
 					var_name_tmpl = 'image',
 					index_func = self.fid_to_index,
 				),
			)
		else:
			print('No predictions for sequence', seq)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(pp(self.dir_root, 'images', self.seq), self.channels['image'].img_ext)
		super().discover()
		return self
		
	@staticmethod
	def path_for_prediction(net_name, channel, dset, fid):
		return pp(DIR_BASELINE_4895, net_name, 'seq_{s}.npy'.format(s=fid.split('_')[1]))

	@staticmethod
	def fid_to_index(dset, fid, **_):
		return dset.get_idx_by_fid(fid)
			
	@staticmethod
	def tr_labels_to_count(head_labels, **_):
		return dict(head_count = head_labels.__len__())

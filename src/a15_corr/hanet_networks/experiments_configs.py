
from easydict import EasyDict


class ConfigFilter:

	def __init__(self, **kw):
		...
		self.defaults = {}

	def add_argument(self, name, default=None, type=None, action=None, help=None, nargs=None, required=None):

		if name.startswith('--'):
			name = name[2:]

		# if action == 'store_true' and bool is None:
		#     type = bool
		
		# if nargs is not None:
		#     inner_type = type
		#     type = lambda x: list(map(inner_type, x))

		self.defaults[name] = default

	def filter(self, full_cfg):
		out = EasyDict()

		for k in self.defaults.keys():
			out[k] = full_cfg.get(k, self.defaults[k])

		return out


parser = ConfigFilter()

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
	help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
	and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
	help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--cv', type=int, default=0,
	help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0,
	help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
	help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
	help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
	help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
	help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
	help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
	help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
	help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
	help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
	help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
	help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=True,
	help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
	help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
	help='0 means no aug, 1 means hard negative mining iter 1,' +
	'2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
	help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
	help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
	default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
	default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
	help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
	help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
	help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
	help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
	help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
	help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
	help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
	help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
	help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
	help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--snapshot_pe', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
	help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
	help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
	help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
	help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
	help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
	help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
	help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
	help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
	help='Minimum testing to verify nothing failed, ' +
	'Runs code for 1 epoch of train and val')
parser.add_argument('--wt_bound', type=float, default=1.0,
	help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
	help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
	help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1', type=str,
	help='url used to set up distributed training')
parser.add_argument('--hanet', nargs='*', type=int, default=[0,0,0,0,0],
	help='Row driven attention networks module')
parser.add_argument('--hanet_set', nargs='*', type=int, default=[0,0,0],
	help='Row driven attention networks module')
parser.add_argument('--hanet_pos', nargs='*', type=int, default=[0,0,0],
	help='Row driven attention networks module')
parser.add_argument('--pos_rfactor', type=int, default=0,
	help='number of position information, if 0, do not use')
parser.add_argument('--aux_loss', action='store_true', default=False,
	help='auxilliary loss on intermediate feature map')
parser.add_argument('--attention_loss', type=float, default=0.0)
parser.add_argument('--hanet_poly_exp', type=float, default=0.0)
parser.add_argument('--backbone_lr', type=float, default=0.0,
	help='different learning rate on backbone network')
parser.add_argument('--hanet_lr', type=float, default=0.0,
	help='different learning rate on attention module')
parser.add_argument('--hanet_wd', type=float, default=0.0001,
	help='different weight decay on attention module')	
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_noise', type=float, default=0.0)
parser.add_argument('--no_pos_dataset', action='store_true', default=False,
	help='get dataset with position information')
parser.add_argument('--use_hanet', action='store_true', default=False,
	help='use hanet')
parser.add_argument('--pooling', type=str, default='mean',
	help='pooling methods, average is better than max')


def extend_config(base, diff):
	result = base.copy()

	for k, val in diff.items():
		if isinstance(val, dict):
			baseval = base.get(k, None)
			if isinstance(baseval, dict):
				# merge a dict
				result[k] = extend_config(baseval, val)
			else:
				# overwrite any other type of value
				result[k] = val
		else:
			result[k] = val
	return result


def variants_gen():

	defaults_base = EasyDict(
		arch = 'network.deepv3.DeepR101V3PlusD_HANet_OS8',
		city_mode = 'train',
		fp16 = True,
		max_cu_epoch = 10000,
		# max_iter = 40000,
		max_iter = 20000,
		lr = 0.01,
		lr_schedule = 'poly',
		poly_exp = 0.9,
		syncbn = True,
		sgd = True,
		crop_size = 768,
		#scale_min = 0.5, # TODO reduce scale aug 
		#scale_max = 2.0,
		scale_min = 0.8,
		scale_max = 1.2,
		rrotate = 0,
		color_aug = 0.25,
		gblur = True,
		bs_mult = 4,
		pos_rfactor = 8,
		dropout = 0.1,
		pos_noise = 0.5,
		aux_loss = True,
		ckpt = './logs/',
		tb_path = './logs/',
	)

	defaults_hanet = EasyDict(
		hanet = [1, 1, 1, 1, 0],
		hanet_set = [3, 64, 3],
		hanet_pos = [2, 1],
		hanet_lr = 0.01,
		hanet_poly_exp = 0.9,
	)

	CFGS = {}

	def add(*blocks, **entries):
		cfg = EasyDict()
		for block in list(blocks) + [entries]:
			cfg = extend_config(cfg, block)

		cfg = extend_config(cfg, entries)

		cfg = EasyDict(cfg)

		CFGS[cfg.exp] = cfg

		return cfg

	dataset = 'synthobs__YX__Fblur5-v2b_cityscapes',
	exp = 'synthobsD2b_r101_os8_base',

	# Repeat legacy configs to run evals
	for dsetv in ['v2b', 'v3persp3D']:

		ds = f'Fblur5-{dsetv}_cityscapes'
		dsetvname = {
			'v2b': 'D2b',
			'v3persp3D': 'D3D'
		}[dsetv]

		add(
			defaults_base, 
			dataset = f'synthobs__YX__{ds}',
			exp = f'synthobs{dsetvname}_r101_os8_base',
			date = '1551',
		)

		add(
			defaults_base, 
			dataset = f'synthobs__YX__{ds}',
			exp = f'synthobs{dsetvname}_r101_os8_hanet',
			date = '1551',
		)

		add(
			defaults_base, 
			dataset = f'synthobs__YX__{ds}',
			exp = f'synthobs{dsetvname}_r101_os8_hanet_PX',
			date = '1551',
		)


	era = '1552'

	unet_shared = dict(
		arch_backbone = 'resnext101_32x8d',
		arch_core = 'v4split',
		arch_classifier = 'last',
	)
	unet_no_perspective = dict(
		perspective = False,
		arch_feature_procs = ['pass'],
		arch_feature_procs_sticky = [],
		**unet_shared,
	)
	unet_pmap = dict(
		perspective = True,
		arch_feature_procs = ['pass', 'persDirect'],
		arch_feature_procs_sticky = ['persDirect'],
		**unet_shared,
	)

	crop_args = {
		'DL101': {},
		'DL101sc': {
			'crop_size': 384,
		}
	}


	for dsetv in ['v2b', 'v3persp3D', 'v3p5bs', 'v3p4sc']:
		ds = f'Fblur5-{dsetv}_cityscapes'

		for archcore in ['DL101', 'DL101sc']:	

			# base, no hanet
			add(
				defaults_base, 
				dataset = f'synthobs__YX__{ds}',
				exp = f'{dsetv}_{archcore}_base',
				date = era,
				**crop_args[archcore],
			)

			# hanet YX
			add(
				defaults_base, defaults_hanet,
				dataset = f'synthobs__YX__{ds}',
				exp = f'{dsetv}_{archcore}_hanetYX',
				date = era,
				**crop_args[archcore],
			)

			# hanet PX
			add(
				defaults_base, defaults_hanet,
				dataset = f'synthobs__PX__{ds}',
				exp = f'{dsetv}_{archcore}_hanetPX',
				date = era,
				**crop_args[archcore],
			)

			# our Unet

			add(
				defaults_base,
				dataset = f'synthobs__P__{ds}',
				exp = f'{dsetv}_UNetA_Pers',
				arch = 'network.unknown_unet.WrapUnet',
				aux_loss = False,
				unet_config = unet_pmap,
				**crop_args[archcore],
			)
		
			add(
				defaults_base,
				dataset = f'synthobs__YX__{ds}',
				exp = f'{dsetv}_UNetA_base',
				arch = 'network.unknown_unet.WrapUnet',
				aux_loss = False,
				unet_config = unet_no_perspective,
				**crop_args[archcore],
			)


			# 
			for penc in ['persDirect', 'zoneFixed8p', 'persEncoder']:
				add(
					defaults_base,
					arch = f'network.deepv3_perspective.DeepR101V3PlusD_{penc}',
					dataset = f'synthobs__P__{ds}',
					exp = f'{dsetv}_{archcore}_{penc}',
					unet_config = unet_no_perspective,
					**crop_args[archcore],
				)

	

	return CFGS

CONFIGS = variants_gen()

def get_config_filtered_for_train(name):
	return parser.filter(CONFIGS[name])


def get_net_for_train(cfgname):

	args = get_config_filtered_for_train(cfgname)

	# get network class by name
	from . import deepv3
	clsname = args.arch.split('.')[-1]
	cls = getattr(deepv3, clsname)

	# create net
	net = cls(args=args, num_classes=2, criterion=None, criterion_aux=None)

	return net




from ..networks import ModsObstacleNet

def construct_DeepLabHanet(cfg):
	cfg = EasyDict(cfg)
	net = get_net_for_train(cfg.hanet_cfg)
	net.args.pos_encoding_names = cfg.pos_encodings

	return net


construct_DeepLabHanet.configs = [
	dict(
		name = 'hanetOrigBase',
		hanet_cfg = 'v3persp3D_DL101_base',
		pos_encodings = ['pos_encoding_Y', 'pos_encoding_X'],
	),
	dict(
		name = 'hanetOrigYX',
		hanet_cfg = 'v3persp3D_DL101_hanetYX',
		pos_encodings = ['pos_encoding_Y', 'pos_encoding_X'],
	),
	dict(
		name = 'hanetOrigPX',
		hanet_cfg = 'v3persp3D_DL101_hanetPX',
		pos_encodings = ['perspective_scale_map', 'pos_encoding_X'],
	),
]

ModsObstacleNet.register_class()(construct_DeepLabHanet)

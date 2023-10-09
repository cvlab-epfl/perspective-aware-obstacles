
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from ..datasets.cityscapes import cityscapes_num_trainids

from ..pytorch_semantic_segmentation import utils as ptseg_utils
initialize_weights = ptseg_utils.initialize_weights

class UNetAutoEncoder(nn.Module):
	class DownBlock(nn.Module):
		def __init__(self, in_channels, out_channels, dropout=0):
			super().__init__()
			layers = [
				nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
				nn.LeakyReLU(inplace=True),
				nn.BatchNorm2d(out_channels),
				nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
				nn.LeakyReLU(inplace=True),
				nn.BatchNorm2d(out_channels),
			]
			
			# maxpool replaced with stride
			#layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			
			if dropout > 0:
				layers.append(nn.Dropout(inplace=True, p=dropout))
			
			self.encode = nn.Sequential(*layers)

		def forward(self, x):
			return self.encode(x)


	class UpBlock(nn.Module):
		def __init__(self, in_channels, out_channels):
			super().__init__()
			self.decode = nn.Sequential(
				nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
				nn.LeakyReLU(inplace=True),
				nn.BatchNorm2d(in_channels),
				nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
				nn.LeakyReLU(inplace=True),
				nn.BatchNorm2d(in_channels),
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
			)

		def forward(self, x):
			return self.decode(x)
	
	def __init__(self, dropout=0.1, cn = [3, 16, 24, 48, 96, 192, 384]):
		super().__init__()
		
		#cn = [3, 32, 64, 128, 256]
		self.enc = nn.Sequential(
			self.DownBlock(cn[0], cn[1], dropout=dropout),
			self.DownBlock(cn[1], cn[2], dropout=dropout),
			self.DownBlock(cn[2], cn[3], dropout=dropout),
			self.DownBlock(cn[3], cn[4], dropout=dropout),
			self.DownBlock(cn[4], cn[5], dropout=dropout),
			self.DownBlock(cn[5], cn[6], dropout=dropout),
		)

		self.dec = nn.Sequential(
			self.UpBlock(cn[6], cn[5]),
			self.UpBlock(cn[5], cn[4]),
			self.UpBlock(cn[4], cn[3]),
			self.UpBlock(cn[3], cn[2]),
			self.UpBlock(cn[2], cn[1]),
			self.UpBlock(cn[1], cn[1] // 2),
		)

		self.final = nn.Conv2d(cn[1] // 2, cn[0], kernel_size=1)

		initialize_weights(self)

	def forward(self, x):
		x = self.enc(x)
		x = self.dec(x)
		return self.final(x)

class PretrainedNetCache:
	vgg_19_features = None
	
	@classmethod
	def get_vgg19_features(cls):
		if cls.vgg_19_features is None:
			mod = models.vgg19(pretrained=True).features
			for param in mod.parameters():
				param.requires_grad = False
			
			mod = mod.cuda() if torch.cuda.is_available() else mod
			cls.vgg_19_features = mod
			
		return cls.vgg_19_features

class VggFeatureExtractor(nn.Module):
	def __init__(self, layers_to_extract = [1, 6, 11, 20, 29]):
		super().__init__()
		
		layers_to_extract = layers_to_extract.copy()
		layers_to_extract.sort()
		last_layer = layers_to_extract[-1]
		self.layers_to_extract = set(layers_to_extract)
		
		# extract the necessary part of the feature network, store submodule
		self.features = PretrainedNetCache.get_vgg19_features()[:last_layer+1]	

		for param in self.parameters():
			param.requires_grad = False

	def forward(self, image):
		outputs = []
		
		value = image
		for idx, mod in enumerate(self.features):
			value = mod(value)
			if idx in self.layers_to_extract:
				outputs.append(value)

		return outputs

class PerceptualVggLoss(nn.Module):
	def __init__(self, layers, weights, distance='L1'):
		super().__init__()
		self.vgg_features = VggFeatureExtractor(layers)
		self.weights = weights
		#self.weights = torch.FloatTensor(weights)
		#self.weights.requires_grad = False
		
		if distance == 'L1':
			self.distance = nn.L1Loss()
		elif distance == 'L2':
			self.distance = nn.MSELoss()
		else:
			raise NotImplementedError('PerceptualVggLoss(distance={0})'.format(distance))

	def forward(self, pred_image_reconstr, image, **_):
		features_a = self.vgg_features(pred_image_reconstr)
		features_b = self.vgg_features(image.detach()) # no gradient wrt image

		loss = 0
		for (w, fa, fb) in zip(self.weights, features_a, features_b):
			loss += w * self.distance(fa, fb)
		
		return dict(
			loss = loss
		)

class PerceptualVggLossPixHd(PerceptualVggLoss):
	"""
	The layers and weights used in 
	https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py#L396
	"""
	def __init__(self):
		super().__init__(
			# 
			layers = [1, 6, 11, 20, 29],
			weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
			distance = 'L1',
		)
		
class TopologyLoss(PerceptualVggLoss):
	""" 
		Beyond pixel-wise loss for delineation...
	"""
	def __init__(self):
		super().__init__(
			layers = [3, 8, 17],
			weights = [0.5, 0.25, 0.08],
			distance = 'L2'
		)

class AutoEncoderLoss(nn.Module):
	def __init__(self, weight_perceptual = 0.8):
		super().__init__()
		
		self.perceptual_loss = PerceptualVggLossPixHd()
		self.l2_loss = nn.MSELoss()
		
		self.weight_perceptual = weight_perceptual
		
	def forward(self, pred_image_reconstr, image, **_):
		loss_l2 = self.l2_loss(pred_image_reconstr, image)
		loss_perc = self.perceptual_loss(pred_image_reconstr, image)['loss']

		return dict(
			loss =	loss_l2 * (1-self.weight_perceptual) + loss_perc * self.weight_perceptual,
			loss_l2  = loss_l2,
			loss_perc = loss_perc,
		)

class AutoEncoderL2OnlyLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.l2_loss = nn.MSELoss()

	def forward(self, prediction, input, batch):
		return dict(loss=(
			self.l2_loss(prediction, input)
		))

def conv_instancenorm_relu(in_ch, out_ch, ksize=3, stride=1, dropout=0):
	""" Convolution-InstanceNorm-ReLU
		with reflection borders
	"""
	return [
		nn.ReflectionPad2d(ksize//2), # use reflection padding to reduce boundary artifacts
		nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride),
		nn.InstanceNorm2d(out_ch),
		nn.ReLU(inplace=True),
	] + (
		[nn.Dropout(inplace=True, p=dropout)] if dropout > 0
		else []
	)

class CompAutoEncoder(nn.Module):
	""" Compression auto encoder inspired by
		"Generative Adversarial Networks for Extreme Learned Image Compression"
	"""
	class DownBlock(nn.Module):
		""" Convolution-InstanceNorm-ReLU
			ksize x ksize kernel
			k filters
			stride s
		"""
		def __init__(self, in_channels, out_channels, ksize=3, stride=1, dropout=0):
			super().__init__()

			self.encode = nn.Sequential(
				*conv_instancenorm_relu(in_channels, out_channels, ksize=ksize, stride=stride, dropout=dropout)
			)

		def forward(self, x):
			return self.encode(x)

	class ResnetBlock(nn.Module):
		def __init__(self, num_channels, ksize=3, dropout=0):
			super().__init__()

			self.conv_block = nn.Sequential(*[
				nn.ReflectionPad2d(ksize//2), # use reflection padding to reduce boundary artifacts
				nn.Conv2d(num_channels, num_channels, kernel_size=ksize),
				nn.InstanceNorm2d(num_channels),
				nn.ReLU(inplace=True),
				nn.ReflectionPad2d(ksize//2), # use reflection padding to reduce boundary artifacts
				nn.Conv2d(num_channels, num_channels, kernel_size=ksize),
				nn.InstanceNorm2d(num_channels),
			])

		def forward(self, x):
			return x + self.conv_block(x)

	class UpBlock(nn.Module):
		def __init__(self, in_channels, out_channels):
			super().__init__()
			self.decode = nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
				nn.InstanceNorm2d(out_channels),
				nn.ReLU(inplace=True)
			)

		def forward(self, x):
			return self.decode(x)

	def module_cs1(self, in_ch, out_ch, ksize):
		""" Let c7s1-k denote a 7x7 Convolution-InstanceNorm[53]-ReLU layer with k filters and stride 1. """
		return self.DownBlock(in_ch, out_ch, ksize=ksize)

	def module_d(self, in_ch, out_ch):
		""" dk denotes a 3x3 Convolution-InstanceNorm-ReLU layer with k filters, and stride 2.
			we use reflection padding to reduce boundary artifacts.
		"""
		return self.DownBlock(in_ch, out_ch, ksize=3, stride=2)

	def module_R(self, num_ch):
		""" Rk denotes a residual block that contains two 3x3 convolutional layers with the same number of filters on both layers. """
		return self.ResnetBlock(num_ch)

	def module_u(self, in_ch, out_ch):
		""" uk denotes a 3x3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters, and stride 1/2"""
		return self.UpBlock(in_ch, out_ch)

	def build_encoder(self):
		# Encoder GC:
		# c7s1-60,d120,d240,d480,d960,c3s1-C,1
		self.encoder = nn.Sequential(*[
			self.module_cs1(3, 60, ksize=7),
			self.module_d(60, 120),
			self.module_d(120, 240),
			self.module_d(240, 480),
			self.module_d(480, 960),
			self.module_cs1(960, self.num_intermediate_ch, ksize=3),
		])

	def build_decoder(self):
		# Generator/decoder:
		# c3s1-960,R960,R960,R960,R960,R960,R960,R960,R960,R960,u480,u240,u120,u60,c7s1-3
		self.decoder = nn.Sequential(*([
			self.module_cs1(self.num_intermediate_ch, 960, ksize=3),
		] + [
			self.module_R(960)
			for i in range(9)
		] + [
			self.module_u(960, 480),
			self.module_u(480, 240),
			self.module_u(240, 120),
			self.module_u(120, 60),
			self.module_cs1(60, 3, ksize=7),
		]))

	def __init__(self, num_intermediate_ch=4):
		super().__init__()

		self.num_intermediate_ch = num_intermediate_ch

		self.build_encoder()
		self.build_decoder()

	def forward(self, image, **_):
		intermediate = self.encoder(image)
		out = self.decoder(intermediate)
		return dict(
			pred_image_reconstr = out,
		)

class CompAutoEncoder2(CompAutoEncoder):

	def build_decoder(self, num_intermediate_ch):
		# Generator/decoder:
		# c3s1-960,R960,R960,R960,R960,R960,R960,R960,R960,R960,u480,u240,u120,u60,c7s1-3
		self.decoder = nn.Sequential(*([
			self.module_cs1(num_intermediate_ch, 960, ksize=3),
		] + [
			self.module_R(960)
			for i in range(9)
		] + [
			self.module_u(960, 480),
			self.module_u(480, 240),
			self.module_u(240, 120),
			self.module_u(120, 60),
			self.module_cs1(60, 3, ksize=7),
			# add a 1-1 convolution at the end to allow output to color scale,
			# which was probably clipped by instancenorm-relu in the version above
			nn.Conv2d(3, 3, kernel_size=1),
		]))

class CompAutoEncoderMask01(CompAutoEncoder):


	def __init__(self, num_intermediate_ch=4, num_semantic_classes=cityscapes_num_trainids):

		self.num_intermediate_ch = num_intermediate_ch
		self.num_semantic_classes = num_semantic_classes

		super().__init__()

	def build_encoder(self):
		# Encoders SC:	
		# Image encoder: c7s1-60,d120,d240,d480,c3s1-C,q,c3s1-480,d960
		self.encoder_img_and_sem_to_comp = nn.Sequential(
			self.module_cs1(3 + self.num_semantic_classes, 60, ksize=7),
			self.module_d(60, 120),
			self.module_d(120, 240),
			self.module_d(240, 480),
			self.module_cs1(480, self.num_intermediate_ch, ksize=3),
		)
		
		self.decoder_latent_to_merge = nn.Sequential(
			self.module_cs1(self.num_intermediate_ch, 480, ksize=3),
			self.module_d(480, 960),
		)
		
		# Semantic label map encoder: c7s1-60,d120,d240,d480,d960
		self.encoder_sem_to_merge = nn.Sequential(
			self.module_cs1(self.num_semantic_classes, 60, ksize=7),
			self.module_d(60, 120),
			self.module_d(120, 240),
			self.module_d(240, 480),
			self.module_d(480, 960),
		)

	def build_decoder(self):
		# Generator/decoder:
		# c3s1-960,R960,R960,R960,R960,R960,R960,R960,R960,R960,u480,u240,u120,u60,c7s1-3
		self.decoder = nn.Sequential(*([
			self.module_cs1(2*960, 960, ksize=3),
		] + [
			self.module_R(960)
			for i in range(9)
		] + [
			self.module_u(960, 480),
			self.module_u(480, 240),
			self.module_u(240, 120),
			self.module_u(120, 60),
			self.module_cs1(60, 3, ksize=7),
			# add a 1-1 convolution at the end to allow output to color scale,
			# which was probably clipped by instancenorm-relu in the version above
			nn.Conv2d(3, 3, kernel_size=1),
		]))
	
	def forward(self, image, labels_onehot, passthrough_mask_d08, **_):
		img_and_seg_comp = self.encoder_img_and_sem_to_comp(torch.cat((image, labels_onehot), 1))
		# mask the compressed representation
		comp_masked = img_and_seg_comp * passthrough_mask_d08.unsqueeze(1)

		compressed_merge = self.decoder_latent_to_merge(comp_masked)
		sem_merge = self.encoder_sem_to_merge(labels_onehot)

		out = self.decoder(torch.cat((compressed_merge, sem_merge), 1))
		return dict(
			pred_image_reconstr = out,
		)

	def blackout_latent(self, image, labels_onehot, passthrough_mask_d08, **_):
		enc_img_sem_latent = self.encoder_img[:5](torch.cat((image, labels_onehot), 1))
		latent_masked = enc_img_sem_latent * passthrough_mask_d08.unsqueeze(1)
		enc_img_sem = self.encoder_img[5:](latent_masked)

		enc_sem = self.encoder_sem(labels_onehot)

		out = self.decoder(torch.cat((enc_img_sem, enc_sem), 1))
		return dict(
			pred_image_reconstr = out,
		)

	def try_noise(self, labels_onehot, **_):

		enc_sem = self.encoder_sem(labels_onehot)

		noise_shape= (enc_sem.shape[0], self.num_intermediate_ch, enc_sem.shape[2]*2, enc_sem.shape[3]*2)
		noise = enc_sem.new_empty(noise_shape).normal_(0., 1.)

		enc_masked = self.encoder_img[5:](noise)

		print(enc_sem.shape, noise.shape)

		out = self.decoder(torch.cat((enc_masked, enc_sem), 1))
		return dict(
			pred_image_reconstr = out,
		)

class GaussianBlur(nn.Module):
	def __init__(self, ksize=5, std=1):
		super().__init__()

		k = self.gaussian_kernel_1d(ksize, std)
		k = k.repeat(3, 1, 1)

		self.kernel_1d = torch.nn.Parameter(k)
		self.kernel_1d.requires_grad = False

		self.padder = nn.ReplicationPad2d(ksize // 2)

	@staticmethod
	def gaussian_kernel_1d(ksize, std):
		x = torch.arange(ksize) + (0.5 - ksize * 0.5)
		kernel_1d = torch.exp(-0.5*x*x/(std*std))
		kernel_1d /= kernel_1d.sum()
		return kernel_1d

	def forward(self, x):
		num_ch = x.shape[1]
		#k = self.kernel_1d.expand(num_ch, 1, -1)
		k = self.kernel_1d

		x = self.padder(x)
		x = F.conv2d(x, k[:, :, :, None], groups=num_ch)
		x = F.conv2d(x, k[:, :, None, :], groups=num_ch)
		return x

class PhotographicGenerator(nn.Module):
	def __init__(self, num_ch_input = 3 + cityscapes_num_trainids, num_downsamples=7, b_test_dims=False):
		"""
		:param b_test_dims: print the sizes of images in the pyramid to check if it matches the original implementation
		TODO original had 7 levels
		"""
		super().__init__()
		self.num_downsamples = num_downsamples
		self.num_channels_input = num_ch_input
		self.num_ch = [1024]*5 + [512]*2 + [128]

		self.upsample_bilinear2x = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)
		self.downsample_avg = nn.AvgPool2d(2, stride=2)

		(out_ch, modules) = self.construct_recursive(self.num_downsamples)
		self.conv_blocks = nn.ModuleList(modules)
		self.final = nn.Conv2d(out_ch, 3, kernel_size=1)

		if b_test_dims:
			self.forward_test_dims(self.num_downsamples, 512)

		initialize_weights(self)

	def construct_recursive(self, step):
		""" :returns: (num output channels, list of modules) """
		if step == 0:
			return (self.num_channels_input, [None])

		else:
			(prev_ch, modules) = self.construct_recursive(step-1)
			in_ch = self.num_channels_input + prev_ch #becayse we cat the input to intermediate
			out_ch = self.num_ch[step]

			conv_block = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
				nn.LeakyReLU(inplace=True),
				#nn.LayerNorm(out_ch),
				nn.InstanceNorm2d(out_ch),
				nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
				nn.LeakyReLU(inplace=True),
				#nn.InstanceNorm2d(out_ch),
			)

			return (out_ch, modules + [conv_block])

	def forward_test_dims(self, step, input):
		if step == 0:
			intermediate = input
		else:
			intermediate = self.forward_test_dims(step-1, input*0.5)
			intermediate = 2*intermediate

		print(step, 'in:', input, '	net:', self.conv_blocks[step][0].weight.shape if self.conv_blocks[step] else '-')

		return intermediate

	def forward_recursive(self, step, input):
		input = input.detach()
		if step == 0:
			intermediate = input
		else:
			intermediate = self.forward_recursive(step-1, self.downsample_avg(input).detach())
			intermediate = self.upsample_bilinear2x(intermediate)
			intermediate = self.conv_blocks[step](torch.cat((input, intermediate), 1))

		#print(input.shape, '->', intermediate.shape)
		return intermediate

	def forward(self, image_blurred, labels_onehot, **_):
		# TODO blur
		input = torch.cat((image_blurred, labels_onehot), 1).detach()
		input.requires_grad = False
		intermediate = self.forward_recursive(self.num_downsamples, input)
		output = self.final(intermediate)
		return dict(
			pred_image_reconstr = output,
		)


	#def recursive_generator(label,sp):
		#dim=512 if sp>=128 else 1024
		#if sp==512:
			#dim=128
		#if sp==4:
			#input=label
		#else:
			#downsampled=tf.image.resize_area(label,(sp//2,sp),align_corners=False)
			#input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True),label],3)
		#net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
		#net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
		#if sp==512:
			#net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
			#net=(net+1.0)/2.0*255.0
		#return net

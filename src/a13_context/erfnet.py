
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from kornia.filters.gaussian import GaussianBlur2d

# from .padding import Padder
from ..common.util_networks import Padder
from ..pipeline.log import log

ACTIVATIONS = {
	'relu': nn.ReLU,
	'selu': nn.SELU,
}


class ErfFactoredConv(nn.Sequential):
	def __init__(self, num_chan : int, dilation_spacing : int = 1, intermediate_activation=nn.ReLU, bn_eps=1e-3, final_activation=True):
		nc = num_chan
		d = dilation_spacing

		mods = [
			nn.Conv2d(nc, nc, (3, 1), bias=True, padding = (d, 0), dilation = (d, 1)),
		]

		if intermediate_activation is not None:
			mods.append(intermediate_activation())
		
		mods.append(
			nn.Conv2d(nc, nc, (1, 3), bias=True, padding = (0, d), dilation = (1, d)),
		)

		if bn_eps is not None:
			mods.append(
				nn.BatchNorm2d(nc, eps=bn_eps),
			)

		if final_activation is not None:
			mods.append(final_activation())

		super().__init__(*mods)
		

	# def __repr__(self):
	# 	conv1 = self[0]
	# 	nc = conv1.in_channels
	# 	ks = conv1.kernel_size[0]
	# 	dil = conv1.dilation[0]
	# 	dilmsg = f', dil={dil}' if dil > 1 else ''
	# 	return f'ErfFactoredConv(chans={nc}, kernel={ks}{dilmsg})'


class ErfDownBlock(nn.Module):
	def __init__(self, num_chan_in : int, num_chan_out : int, activation='relu'):
		super().__init__()

		num_chan_extra = num_chan_out - num_chan_in

		self.conv = nn.Conv2d(num_chan_in, num_chan_extra, (3, 3), stride=2, padding=1, bias=True)
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.bn = nn.BatchNorm2d(num_chan_out, eps = 1e-3)
		self.activation = ACTIVATIONS[activation]()

	def forward(self, value):
		conv_and_pool = torch.cat([
			self.conv(value),
			self.pool(value),
		], dim=1)
		return self.activation(self.bn(conv_and_pool))


class ErfUpBlock(nn.Sequential):
	def __init__(self, num_chan_in : int, num_chan_out : int, activation='relu'):
		super().__init__(OrderedDict(
			upconv = nn.ConvTranspose2d(num_chan_in, num_chan_out, 3, stride=2, padding=1, output_padding=1, bias=True),
			bn = nn.BatchNorm2d(num_chan_out, eps=1e-3),
			activation = ACTIVATIONS[activation](),
		))


class ErfIntermediate(nn.Sequential):

	def __init__(self, num_chan : int, dilation_spacing : int = 1, dropout_p : float = 0., activation='relu'):
		nc = num_chan
		bn_eps = 1e-3

		act = ACTIVATIONS[activation]

		super().__init__(*[
			ErfFactoredConv(nc, bn_eps=bn_eps, intermediate_activation=act, final_activation=act),
			ErfFactoredConv(nc, bn_eps=bn_eps, intermediate_activation=act, final_activation=None, dilation_spacing=dilation_spacing),
		]
		+ [nn.Dropout2d(dropout_p)] if dropout_p > 1e-6 else []
		)

		self.activation = act()

	def forward(self, value):
		# residual
		value_with_residual = super().forward(value) + value
		return self.activation(value_with_residual)


class ErfNetBasic(nn.Module):

	feat_num_altered_inputs = (0, 0)

	# @staticmethod
	# def build_erf_net(num_classes : int, activation='relu') -> OrderedDict:
	# 	DownBlock = partial(ErfDownBlock, activation=activation)
	# 	Intermediate = partial(ErfIntermediate, activation=activation)
	# 	UpBlock = partial(ErfUpBlock, activation=activation)

	# 	return OrderedDict(
	# 		feat1 = DownBlock(3, 16),

	# 		feat2 = nn.Sequential(
	# 			*[
	# 				DownBlock(16, 64)
	# 			] + [
	# 				Intermediate(64, dropout_p = 0.03)
	# 				for i in range(5)
	# 			]
	# 		),

	# 		feat3 = nn.Sequential(
	# 			*[
	# 				DownBlock(64, 128)
	# 			] + [
	# 				# 8 intermediate blocks with dilation
	# 				Intermediate(64, dropout_p = 0.3, dilation_spacing=d)
	# 				for d in [2, 4, 8, 16, 2, 4, 8, 16]
	# 			]
	# 		),
			
	# 		up1 = nn.Sequential(OrderedDict(
	# 			up = UpBlock(128, 64),
	# 			layer1 = Intermediate(64),
	# 			layer2 = Intermediate(64),
	# 		)),

	# 		up2 = nn.Sequential(OrderedDict(
	# 			up = UpBlock(64, 16),
	# 			layer1 = Intermediate(16),
	# 			layer2 = Intermediate(16),
	# 		)),

	# 		classifier =  nn.ConvTranspose2d(
	# 			in_channels = 16, 
	# 			out_channels = num_classes, 
	# 			kernel_size = 2, 
	# 			stride = 2, 
	# 			bias=True,
	# 		),
	# 	)

	@property
	def num_class(self):
		return self.classifier.out_channels

	def __init__(self, num_class, activation='relu'):
		super().__init__()
		# super().__init__(self.build_erf_net(num_class, activation=activation))
		self.build(
			num_class = num_class, 
			activation = activation,
		)

	
	def build(self, num_class, activation='relu'):
		DownBlock = partial(ErfDownBlock, activation=activation)
		Intermediate = partial(ErfIntermediate, activation=activation)
		UpBlock = partial(ErfUpBlock, activation=activation)

		self.feat1 = DownBlock(3, 16)

		self.feat2 = nn.Sequential(
			*[
				DownBlock(16, 64 - self.feat_num_altered_inputs[0])
			] + [
				Intermediate(64, dropout_p = 0.03)
				for i in range(5)
			]
		)

		self.feat3 = nn.Sequential(
			*[
				DownBlock(64, 128 - self.feat_num_altered_inputs[1])
			] + [
				# 8 intermediate blocks with dilation
				Intermediate(128, dropout_p = 0.3, dilation_spacing=d)
				for d in [2, 4, 8, 16, 2, 4, 8, 16]
			]
		)
			
		self.up1 = nn.Sequential(OrderedDict(
			up = UpBlock(128, 64),
			layer1 = Intermediate(64),
			layer2 = Intermediate(64),
		))

		self.up2 = nn.Sequential(OrderedDict(
			up = UpBlock(64, 16),
			layer1 = Intermediate(16),
			layer2 = Intermediate(16),
		))

		self.classifier =  nn.ConvTranspose2d(
			in_channels = 16, 
			out_channels = num_class, 
			kernel_size = 2, 
			stride = 2, 
			bias=True,
		)


	def forward(self, image, **_):
		padder = Padder(image.shape, 16)
		
		value = padder.pad(image)
		for mod in [self.feat1, self.feat2, self.feat3, self.up1, self.up2, self.classifier]:
			value = mod(value)

		logits = padder.unpad(value)
		return logits



class ErfNetImageMultiscale(ErfNetBasic):

	feat_num_altered_inputs = (4, 8)

	def __init__(self, num_class, activation='relu', extra_blur_channels=False):
		super().__init__(num_class=num_class, activation=activation)

		self.b_extra_blur_channels = extra_blur_channels

		if self.b_extra_blur_channels:
			log.info('Extra blur channels activated')

		self.mod_blur = GaussianBlur2d((5, 5), (1.5, 1.5))

		img_in_ch = 6 if self.b_extra_blur_channels else 3

		self.img_ker1, self.img_ker2 =[
			nn.Sequential(
				nn.Conv2d(img_in_ch, nc_out, (3, 3), padding=1, bias=True),
				nn.BatchNorm2d(nc_out, eps = 1e-3),
				ACTIVATIONS[activation](),
			) 
			for nc_out in self.feat_num_altered_inputs
		]

	def forward(self, image, **_):
		padder = Padder(image.shape, 16)
		image = padder.pad(image)

		if self.b_extra_blur_channels:
			image_1_4th = nn.functional.avg_pool2d(image, 4)
			image_1_8th = nn.functional.avg_pool2d(image_1_4th, 2)

			image_1_4th = torch.cat([
				image_1_4th,
				self.mod_blur(image_1_4th),
			], dim=1)
			image_1_8th = torch.cat([
				image_1_8th,
				self.mod_blur(image_1_8th),
			], dim=1)

		else:
			image_1_4th = nn.functional.avg_pool2d(image, 4)
			image_1_8th = nn.functional.avg_pool2d(image_1_4th, 2)
		
			
		f1 = self.feat1(image)

		f2 = self.feat2[1:](torch.cat([
			self.feat2[0](f1),
			self.img_ker1(image_1_4th),
		], dim=1))

		f3 = self.feat3[1:](torch.cat([
			self.feat3[0](f2),
			self.img_ker2(image_1_8th),
		], dim=1))
		
		value = f3

		del f1, f2, f3

		for mod in [self.up1, self.up2, self.classifier]:
			value = mod(value)

		logits = padder.unpad(value)
		return logits
	
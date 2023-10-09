
import numpy as np
from piq import multi_scale_gmsd, gmsd
from kornia.utils import image_to_tensor
import torch
import cv2 as cv

class PixelDistanceFunctions:

	@staticmethod
	def L1(img, img_rec):
		return np.linalg.norm(img - img_rec, ord=1, axis=2)

	@staticmethod
	def L2(img, img_rec):
		return np.linalg.norm(img - img_rec, ord=2, axis=2)

	@staticmethod
	def MSGMS(img, img_rec):
		
		with torch.no_grad():
			img = image_to_tensor(img)[None]
			img_rec = image_to_tensor(img_rec)[None]
			# diff_torch = multi_scale_gmsd(img_rec, img, reduction='none')
			diff_torch = gmsd(img_rec, img, reduction='none')
			

			#print(img.shape, img_rec.shape, diff_torch.shape)
		diff_np = diff_torch[0].numpy()

		

		return cv.pyrUp(diff_np)
	

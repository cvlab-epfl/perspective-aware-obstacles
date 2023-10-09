
import torch
import kornia
from math import pi

def vis_centerpoint_offset(centerpoint_offset, MAX_LENGTH = 0.2, **_):
	
	_, h, w = centerpoint_offset.shape
	
	offset_length = centerpoint_offset.norm(dim=0)
	offset_angle = torch.atan2(centerpoint_offset[1], centerpoint_offset[0])

	vis_hsv = torch.stack([
		# hue from angle
		-offset_angle, #* /np.pi) + 0.5,
		# saturation from length
		torch.clamp(offset_length * (1/MAX_LENGTH), 0., 1.),
		# lightness = 0.5
		torch.full_like(offset_angle, fill_value=1),
	])

	# vis_rgb = kornia.color.hls_to_rgb(vis_hsl)
	vis_rgb = kornia.color.hsv_to_rgb(vis_hsv)

	vig_rgb_byte = (vis_rgb * 255).byte()
	vis_rgb_np = kornia.utils.tensor_to_image(vig_rgb_byte)
	
	return dict(
		vis_centerpoint_offset = vis_rgb_np,
	)

# def vis_sigma(centerpoint_vote_radius):


	

import torch, kornia

def offsets_to_egdes__blur(centerpoint_offset_batch, kernel_size=19):
		
	# 4 channels: [offset x y, offset abs x y]
	cp_offset_and_abs = torch.cat([
		centerpoint_offset_batch,
		centerpoint_offset_batch.abs(),
	], dim=1)
	
	cp_offset_and_abs_blurred = kornia.filters.gaussian_blur2d(
		cp_offset_and_abs, 
		kernel_size= (kernel_size, kernel_size),
		sigma = (kernel_size / 3, kernel_size / 3),
	)
		
	offset_len = cp_offset_and_abs_blurred[:, :2].pow(2).sum(dim=1, keepdim=True)
	offset_abs_len = cp_offset_and_abs_blurred[:, 2:4].pow(2).sum(dim=1, keepdim=True)
	diff = offset_abs_len - offset_len
	
# 	show([x[0, 0].cpu().numpy() for x in [offset_len, offset_abs_len]])
	
	return dict(
		edges = diff,
	#	edges_normed_by_len = diff / offset_len,
	)


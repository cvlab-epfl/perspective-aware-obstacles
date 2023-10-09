
import torch
from ..gluoncvth.models.deeplab import get_deeplab


def load_gluon_weight_file(net_name='deeplab_resnet50_citys'):
	from mxnet import ndarray
	from gluoncv.model_zoo.model_store import get_model_file
	
	weight_file_path = get_model_file(net_name, tag=True)	
	weight_dict = ndarray.load(weight_file_path)

	return {k: v.asnumpy() for k, v in weight_dict.items()}
	

def init_pytorch_deeplab():
	net = get_deeplab(dataset='citys', aux=False)
	net.pretrained.fc = None
	net.head.block[4] = torch.nn.Conv2d(256, 19, kernel_size=(1, 1), stride=(1, 1))
	net.eval()
	return net

def load_weights_into_pytorch_deeplab(weights, net=None):

	if isinstance(weights, str):
		weights = load_gluon_weight_file(weights)
	
	names_to_ignore = ('auxlayer',)
	renames = {
		'gamma': 'weight',
		'beta': 'bias',
		'concurent.': 'b'
	}
	add_prefix = ('conv1', 'bn1', 'layer')
	
	def proc_weight_name(name):
		if any(x in name for x in names_to_ignore):
			return None
		
		for rn_from, rn_to in renames.items():
			name = name.replace(rn_from, rn_to)
	
		if any(name.startswith(p) for p in add_prefix):
			name = f'pretrained.{name}'
		
		return name
	
	
	state_dict = {}
	for name, weight in weights.items():
		name_proc = proc_weight_name(name)
		if name_proc:
			state_dict[name_proc] = torch.from_numpy(weight)
		
	if net is None:
		net = init_pytorch_deeplab()

	#weight_file_names = list(state_dict.keys())
	#pytorch_net_names = dict(net.named_parameters()).keys()	
	#print('\n'.join(f'{n1:<40}{n2}' for n1, n2 in zip(weight_file_names, pytorch_net_names)))
	
	net.load_state_dict(state_dict)

	return net
	
	
#w1 = load_weight_file_deeplab()
#net_seg = load_weights_into_pytorch_deeplab(w1)

from ..gluoncvth.models.deeplab import DeepLabV3, DeepLabV3Head

class ObstacleHeadedDeeplab(torch.nn.Module):
	def __init__(self, cfg_net):
		super().__init__()

		self.deeplab = load_weights_into_pytorch_deeplab(
			'deeplab_resnet50_citys',
			net = init_pytorch_deeplab(),
		)
		self.deeplab.pretrained.fc = None
		
		self.obstacle_head = DeepLabV3Head(
			2048, 2, 
			norm_layer = torch.nn.BatchNorm2d, 
			up_kwargs = self.deeplab._up_kwargs,
		)

		self.default_freezing()


	def default_freezing(self):
		for p in self.deeplab.parameters():
			p.requires_grad = False

	def forward_backbone_feats(self, image):
		
		b = self.deeplab.pretrained
		
		x = image
		for layer in (b.conv1, b.bn1, b.relu, b.maxpool):
			x = layer(x)

		c1 = b.layer1(x)
		c2 = b.layer2(c1)
		c3 = b.layer3(c2)
		c4 = b.layer4(c3) 

		return dict(
			c1 = c1, c2 = c2, c3 = c3, c4 = c4,
		)

	def forward(self, image, **_):
		_, _, h, w = image.shape

		with torch.no_grad():
			backbone_feats = self.forward_backbone_feats(image)

		# obstacle head
		obst_logits = self.obstacle_head(backbone_feats['c4'])
		obst_logits = torch.nn.functional.interpolate(obst_logits, (h,w), **self.deeplab._up_kwargs)

		return obst_logits


	def forward_obst_and_semantic(self, image, **_):
		_, _, h, w = image.shape
		backbone_feats = self.forward_backbone_feats(image)

		# semantic head
		sem_logits = self.deeplab.head(backbone_feats['c4'])
		sem_logits = torch.nn.functional.interpolate(sem_logits, (h,w), **self.deeplab._up_kwargs)

		# obstacle head
		obst_logits = self.obstacle_head(backbone_feats['c4'])
		obst_logits = torch.nn.functional.interpolate(obst_logits, (h,w), **self.deeplab._up_kwargs)

		return dict(
			sem_logits = sem_logits,
			obst_logits = obst_logits,
		)



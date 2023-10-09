"""
Network Initializations
"""

import logging
import importlib
import torch



def get_net(args, criterion, criterion_aux=None):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion, criterion_aux=criterion_aux)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net



class FakeDataParallel(torch.nn.Module):

    def __init__(self, module, device_ids):
        super().__init__()

        print('FAKE DATA PARALLEL', device_ids)
        self.device = torch.device(f'cuda:{device_ids[0]}')
        self.module = module

    def forward(self, *args, **kw):

        d = self.device

        def map_obj(x):
            if isinstance(x, torch.Tensor):
                return x.to(d)
            elif isinstance(x, (tuple, list)):
                return list(map(map_obj, x))
            elif isinstance(x, dict):
                return {k: map_obj(v) for k, v in x.items()}
            else:
                return x

        return self.module(*map_obj(args), **map_obj(kw))


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    # torch.cuda.set_device(gpuid)
    # net.cuda(gpuid)    
    #net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])
    net = FakeDataParallel(net, device_ids=[gpuid])
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])#, find_unused_parameters=True)
    return net


def get_model(args, num_classes, criterion, criterion_aux=None):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(args=args, num_classes=num_classes, criterion=criterion, criterion_aux=criterion_aux)
    return net

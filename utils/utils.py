import torch
import numpy as np
from thop import profile
from thop import clever_format
import argparse
import os
from easydict import EasyDict as ed
import yaml
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    cuda_visible_devices = None
    local_rank = -1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == -1:
        device_num = 1
    elif cuda_visible_devices is None:
        device_num = torch.cuda.device_count()
    else:
        device_num = len(cuda_visible_devices)

    args.device_num = device_num
    args.local_rank = local_rank

    return args


def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))
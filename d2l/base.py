"""The base module contains some basic functions/classes for d2l"""
import time
import torch

__all__ = ['try_gpu', 'try_all_gpus', 'Benchmark']

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def try_all_gpus():
    """Return all available GPUs, or [torch device cpu] if there is no GPU."""
    if torch.cuda.is_available():
        devices = []
        for i in range(16):
            device = torch.device('cuda:'+str(i))
            devices.append(device)
    else:
        devices = [torch.device('cpu')]
    return devices

class Benchmark():
    """Benchmark programs."""
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))

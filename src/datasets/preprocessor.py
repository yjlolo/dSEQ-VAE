import sys
import torch


class LogCompress():
    def __call__(self, x):
        return torch.log(sys.float_info.epsilon + x)


class TakeExp():
    def __call__(self, x):
        return torch.exp(x)

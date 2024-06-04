import torch
import scipy
import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_cumsum_tensor(x, discount):

    if not len(x.shape) == 1:
        raise ValueError("tensor shape is: ", x.shape)

    xx = torch.flip(x, dims=[0])
    discount_sums = torch.zeros(xx.shape, dtype=torch.float32).to(x.device)
    for i in range(xx.shape[0]):
        if i == 0:
            discount_sums[0] = xx[0]
        else:
            discount_sums[i] = xx[i] + discount * discount_sums[i - 1]
    discount_sums = torch.flip(discount_sums, dims=[0])
    return discount_sums
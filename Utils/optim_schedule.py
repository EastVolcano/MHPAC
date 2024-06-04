'''A wrapper class for optimizer '''
import numpy as np
from mpi4py import MPI
import torch.distributed as dist


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps, torch_dist=False):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        if torch_dist:
            self.init_lr = 0.5 * dist.get_world_size() * np.power(d_model, -0.5)
        else:
            self.init_lr = 0.5 * MPI.COMM_WORLD.Get_size() * np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.35),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

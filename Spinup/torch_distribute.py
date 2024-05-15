"""torch.distributed部分功能的二次封装，可以直接对torch的tensor进行操作"""

import torch
import torch.distributed as dist


def statistics_scalar_torch(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An tensor containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x, dtype=torch.float32)
        except:
            print(f"debug: {x, type(x)}")
            x = torch.as_tensor(x, dtype=torch.float32)
    size = dist.get_world_size()

    global_sum = torch.sum(x)
    xt_len = torch.as_tensor(len(x), dtype=torch.int32).to(x.device)
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(xt_len, op=dist.ReduceOp.SUM)

    # print(global_sum, " ", xt_len)
    mean = global_sum / xt_len

    global_sum_sq = torch.sum((x - mean) ** 2)
    dist.all_reduce(global_sum_sq, op=dist.ReduceOp.SUM)
    std = torch.sqrt(global_sum_sq / xt_len)  # compute global std

    if with_min_and_max:
        global_min = torch.min(x) if len(x) > 0 else torch.as_tensor(torch.inf).to(x.device)
        global_max = torch.max(x) if len(x) > 0 else torch.as_tensor(-torch.inf).to(x.device)
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        return mean, std, global_min, global_max
    return mean, std


def average_gradients_torch(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def sum_x_torch(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x_sum = dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x_sum


def average_x_torch(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)

    size = float(dist.get_world_size())
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x_avg = x / size
    return x_avg


def sync_params_torch(module):
    """ Sync all parameters of module across all MPI processes. """
    if dist.get_world_size() == 1:
        return
    for p in module.parameters():
        p_data = p.data
        dist.broadcast(p_data, src=0, async_op=False)

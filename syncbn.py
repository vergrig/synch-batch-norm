#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float, eval=False):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`

        if eval:
            return (input - running_mean) / torch.sqrt(running_std + eps)

        # calculating statistics
        input_sum = torch.sum(input, dim=0)
        input_squared = torch.sum(input ** 2, dim=0)
        input_sz = input.shape[0]

        # flattening everything so to not use any additional memory duting all_reduce
        sum_shape = input_sum.shape
        input_sum, input_squared, input_sz = input_sum.flatten(), input_squared.flatten(), input.new([input_sz])
        input_len = input_sum.shape[0]

        # the all_reduce itself
        communicate = torch.cat((input_sum, input_squared, input_sz), 0)
        dist.all_reduce(communicate, op=dist.ReduceOp.SUM)

        # restoring dimensions
        all_sum, all_squared, all_sz = torch.split(communicate, input_len)
        all_sum, all_squared, all_sz = all_sum.unflatten(0, sum_shape), all_squared.unflatten(0, sum_shape), all_sz.item()

        # calculating mean and std
        all_mean = all_sum / all_sz
        all_std = all_squared / all_sz - all_mean ** 2

        running_mean[:] = momentum * all_mean + (1 - momentum) * running_mean
        running_std[:] = momentum * all_std + (1 - momentum) * running_std

        input_norm = (input - all_mean) / torch.sqrt(all_std + eps)
        ctx.save_for_backward(input_norm, all_mean, all_std, input.new([all_sz]), input.new([eps]))

        return input_norm
        


    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!

        input_norm, all_mean, all_std, all_sz, eps = ctx.saved_tensors
        all_sz, eps = all_sz.item(), eps.item()

        output_sum = torch.sum(grad_output, dim=0)
        output_squared = torch.sum(grad_output * input_norm, dim=0)

        communicate = torch.stack((output_sum, output_squared))
        dist.all_reduce(communicate, op=dist.ReduceOp.SUM)
        output_sum, output_squared = communicate[0], communicate[1]


        grad = ( (all_sz * grad_output - output_sum - input_norm * output_squared) / 
              (all_sz * torch.sqrt(all_std + eps)) )
        
        
        tuple_grad = tuple([grad] + [None] * 5)

        return tuple_grad


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.zeros((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum, not self.training)
        return output

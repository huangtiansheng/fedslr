from __future__ import print_function

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


from torch import Tensor


class MaskedModule(nn.Module):

    def __init__(self, weights, name, args, bias=None, device=None, zero_p_init=False):
        nn.Module.__init__(self)
        self.args = args
        self.device = device
        self.w = nn.Parameter(copy.deepcopy(weights), requires_grad=True)
        if zero_p_init:
            self.p = nn.Parameter(torch.zeros_like(weights), requires_grad=True)
        else:
            self.p = nn.Parameter(copy.deepcopy(weights), requires_grad=True)
        if bias != None:
            self.w_bias = nn.Parameter(copy.deepcopy(bias), requires_grad=True)
            if zero_p_init:
                self.p_bias = nn.Parameter(torch.zeros_like(bias), requires_grad=True)
            else:
                self.p_bias = nn.Parameter(copy.deepcopy(bias), requires_grad=True)
            self.has_bias = True
        else:
            self.w_bias = None
            self.p_bias = None
            self.has_bias = False
        self.layer_name = name
        # self.factorization = factorization
        self.status= "only_w"

    def set_train_status(self, status):
        self.status = status
        if  self.status == "only_w":
            self.w.requires_grad = True
            self.p.requires_grad = False
            if self.has_bias:
                self.w_bias.requires_grad = True
                self.p_bias.requires_grad = False
        else:
            self.w.requires_grad = False
            self.p.requires_grad = True
            if self.has_bias:
                self.w_bias.requires_grad = False
                self.p_bias.requires_grad = True

    def sparseFunction(self):
        if self.status == "only_w":
            effective_weights = self.w
            if self.has_bias == True:
                effective_bias = self.w_bias
            else:
                effective_bias = None

        elif self.status == "only_p":
            effective_weights = self.p
            if self.has_bias == True:
                effective_bias =  self.p_bias
            else:
                effective_bias = None
        else:
            effective_weights = self.w + self.p
            if self.has_bias == True:
                effective_bias = self.w_bias + self.p_bias
            else:
                effective_bias = None

        return effective_weights, effective_bias

    def get_sparse_num(self):
        effective_weights, effective_bias = self.sparseFunction()
        return (torch.numel(effective_weights) - torch.count_nonzero(effective_weights)).to("cpu")

class MaskedLinear(MaskedModule):
    """
    Which is a custom fully connected linear layer that its weights $W_f$
    remain constant once initialized randomly.
    A second weight matrix $W_m$ with the same shape as $W_f$ is used for
    generating a binary theta. This weight matrix can be trained through
    backpropagation. Each unit of $W_f$ may be passed through sigmoid
    function to generate the $p$ value of the $Bern(p)$ function.
    """

    def __init__(self, weights, name, args, bias=None, device=None,zero_p_init=False):
        super(MaskedLinear, self).__init__(weights, name, args, device=device, bias=bias,zero_p_init=zero_p_init)

        # del self.p
        # del self.p_bias

    def set_train_status(self, status):
        self.status = status
        if  self.status == "only_w":
            self.w.requires_grad = True
            self.p.requires_grad = False
            if self.has_bias:
                self.w_bias.requires_grad = True
                self.p_bias.requires_grad = False
        elif self.status == "train_p":
            self.w.requires_grad = False
            self.p.requires_grad = True
            if self.has_bias:
                self.w_bias.requires_grad = False
                self.p_bias.requires_grad = True
        else:
            self.w.requires_grad = False
            self.p.requires_grad = True
            if self.has_bias:
                self.w_bias.requires_grad = False
                self.p_bias.requires_grad = True

    def forward(self, x):
        effective_weights, effective_bias = self.sparseFunction()
        lin = F.linear(x, effective_weights, bias=effective_bias)
        return lin


class MaskedConv(MaskedModule):
    """
    Which is a custom fully connected linear layer that its weights $W_f$
    remain constant once initialized randomly.
    A second weight matrix $W_m$ with the same shape as $W_f$ is used for
    generating a binary theta. This weight matrix can be trained through
    backpropagation. Each unit of $W_f$ may be passed through sigmoid
    function to generate the $p$ value of the $Bern(p)$ function.
    """

    def __init__(self, weights, name, stride, padding, dilation, group, args, device=None,zero_p_init=False):
        self.out_channel, self.in_channel, self.kernel_size, _ = weights.shape
        theta_dimension = self.out_channel
        super(MaskedConv, self).__init__(weights, name, args, device=device,zero_p_init=zero_p_init)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.group = group

    def forward(self, x):
        effective_weights, effective_bias = self.sparseFunction()
        conv = F.conv2d(x, effective_weights, stride=self.stride, padding=self.padding, dilation=self.dilation,
                        groups=self.group, bias=effective_bias)
        return conv


# def spectral_init(weight, rank):
#     U, S, V = torch.svd(weight)
#     sqrtS = torch.sqrt(torch.diag(S[:rank]))
#     return torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T


# class MaskedLinear_UV(MaskedModule):
#     """
#     Which is a custom fully connected linear layer that its weights $W_f$
#     remain constant once initialized randomly.
#     A second weight matrix $W_m$ with the same shape as $W_f$ is used for
#     generating a binary theta. This weight matrix can be trained through
#     backpropagation. Each unit of $W_f$ may be passed through sigmoid
#     function to generate the $p$ value of the $Bern(p)$ function.
#     """
#
#     def __init__(self, weights, name, args, rank_scale=1, bias=None, device=None):
#         super(MaskedLinear_UV, self).__init__(weights, name, args, device=device, factorization=True, bias=bias)
#         self.out_channel, self.in_channel = weights.shape
#         self.rank = math.ceil(rank_scale * min(self.out_channel, self.in_channel))
#         self.w_U = nn.Parameter(torch.zeros(self.out_channel, self.rank))
#         self.w_VT = nn.Parameter(torch.zeros(self.rank, self.in_channel))
#         U, VT = spectral_init(weights, self.rank)
#         self.w_U.data = U
#         self.w_VT.data = VT
#
#         delattr(self, "w")
#
#     def set_train_status(self, status):
#         self.status = status
#         if self.status == "train_w":
#             self.w_U.requires_grad = True
#             self.w_VT.requires_grad = True
#             self.p.requires_grad = False
#             if self.has_bias:
#                 self.w_bias.requires_grad = True
#                 self.p_bias.requires_grad = False
#         else:
#             self.w_U.requires_grad = False
#             self.w_VT.requires_grad = False
#             self.p.requires_grad = True
#             if self.has_bias:
#                 self.w_bias.requires_grad = False
#                 self.p_bias.requires_grad = True
#
#     def forward(self, x):
#         if self.status == "train_w" or self.status == "only_w":
#             lin_global = F.linear(F.linear(x, self.w_VT), self.w_U, bias=self.w_bias)
#             lin = lin_global
#
#         elif self.status == "train_p":
#             lin_global = F.linear(F.linear(x, self.w_VT), self.w_U, bias=self.w_bias)
#             lin_sparse = F.linear(x, self.p, bias=self.p_bias)
#             lin = lin_global + lin_sparse
#         return lin


# class MaskedConv_UV(MaskedModule):
#     """
#     Which is a custom fully connected linear layer that its weights $W_f$
#     remain constant once initialized randomly.
#     A second weight matrix $W_m$ with the same shape as $W_f$ is used for
#     generating a binary theta. This weight matrix can be trained through
#     backpropagation. Each unit of $W_f$ may be passed through sigmoid
#     function to generate the $p$ value of the $Bern(p)$ function.
#     """
#
#     def __init__(self, weights, name, stride, padding, dilation, group, args, rank_scale=1, device=None):
#         self.out_channel, self.in_channel, self.kernel_size, _ = weights.shape
#         super(MaskedConv_UV, self).__init__(weights, name, args, factorization=True, device=device)
#         self.padding = padding
#         self.stride = stride
#         self.dilation = dilation
#         self.group = group
#         self.shape = weights.shape
#
#         dim1, dim2 = self.out_channel * self.kernel_size, self.in_channel * self.kernel_size
#         self.rank = math.ceil(rank_scale * min(dim1, dim2))  # int(round(rank_scale * min(dim1, dim2)))
#
#         self.w_U = nn.Parameter(torch.zeros(dim1, self.rank))
#         self.w_VT = nn.Parameter(torch.zeros(self.rank, dim2))
#
#         U, VT = spectral_init(weights.swapaxes(1, 2).reshape([dim1, dim2]), self.rank)
#         self.w_U.data = U
#         self.w_VT.data = VT
#         # print(torch.sum(weights.swapaxes(1, 2).reshape([dim1, dim2])-torch.matmul(self.w_U.data,self.w_VT)))
#         delattr(self, "w")
#
#     def set_train_status(self, status):
#         self.status = status
#         if self.status == "train_w":
#             self.w_U.requires_grad = True
#             self.w_VT.requires_grad = True
#             self.p.requires_grad = False
#             if self.has_bias:
#                 self.w_bias.requires_grad = True
#                 self.p_bias.requires_grad = False
#         else:
#             self.w_U.requires_grad = False
#             self.w_VT.requires_grad = False
#             self.p.requires_grad = True
#             if self.has_bias:
#                 self.w_bias.requires_grad = False
#                 self.p_bias.requires_grad = True
#
#
#     def forward(self, x):
#         # Apply the effective weight on the input data
#         # effective_theta = self.theta_to_theta()
#         # effective_weights  = self.w * effective_theta.unsqueeze(-1).unsqueeze(-1).expand( self.w.shape)
#         # effective_weights, effective_bias=  self.sparseFunction()
#         effective_w_VT = self.w_VT.reshape(self.rank, 1, self.in_channel, self.kernel_size).transpose(1, 2)
#         effective_U = self.w_U.reshape(self.out_channel, self.kernel_size, self.rank, 1).transpose(1, 2)
#         if self.status == "train_w" or self.status == "only_w":
#             conv_first = F.conv2d(x, effective_w_VT, stride=(1, self.stride[1]),
#                                   padding=(0, self.padding[1]),
#                                   dilation=(1, self.dilation[1]),
#                                   groups=self.group)
#             conv_second = F.conv2d(conv_first, effective_U, stride=(self.stride[0], 1),
#                                    padding=(self.padding[0], 0),
#                                    dilation=(self.dilation[0], 1),
#                                    groups=self.group, bias=self.w_bias)
#             return conv_second
#         else:
#             conv_first = F.conv2d(x, effective_w_VT, stride=(1, self.stride[1]),
#                                   padding=(0, self.padding[1]),
#                                   dilation=(1, self.dilation[1]),
#                                   groups=self.group)
#             conv_second = F.conv2d(conv_first, effective_U, stride=(self.stride[0], 1),
#                                    padding=(self.padding[0], 0),
#                                    dilation=(self.dilation[0], 1),
#                                    groups=self.group, bias=self.w_bias)
#             conv_p = F.conv2d(x, self.p, stride=self.stride, padding=self.padding,
#                               dilation=self.dilation,
#                               groups=self.group, bias=self.p_bias)
#             # print(conv_p.shape)
#             # print(conv_second.shape)
#             return conv_second + conv_p


class MaskedGroupNorm(MaskedModule):

    def __init__(self, num_groups: int, num_channels: int, weights, bias, name, args, eps: float = 1e-5,
                 affine: bool = True, device=None) -> None:
        MaskedModule.__init__(self, weights, name, args, bias=bias, device=device)
        self.p = nn.Parameter(torch.zeros_like(weights), requires_grad=True)
        if bias != None:
            self.p_bias = nn.Parameter(torch.zeros_like(bias), requires_grad=True)
            self.has_bias = True
        else:
            self.p_bias = None
            self.has_bias = False
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

    def set_train_status(self, status):
        self.status = status
        if  self.status == "only_w":
            self.w.requires_grad = True
            self.p.requires_grad = False
            if self.has_bias:
                self.w_bias.requires_grad = True
                self.p_bias.requires_grad = False
        elif self.status == "train_p":
            self.w.requires_grad = False
            self.p.requires_grad = True
            if self.has_bias:
                self.w_bias.requires_grad = False
                self.p_bias.requires_grad = True
        else:
            self.w.requires_grad = False
            self.p.requires_grad = True
            if self.has_bias:
                self.w_bias.requires_grad = False
                self.p_bias.requires_grad = True


    def forward(self, input: Tensor) -> Tensor:
        effective_weights, effective_bias = self.sparseFunction()
        return F.group_norm(
            input, self.num_groups,effective_weights, effective_bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


# class MaskedNorm(MaskedModule):
#
#     def __init__(self, num_features, weights, bias, name, args, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=False, device=None):
#
#         MaskedModule.__init__(self, weights, name, args, bias=bias, device=device)
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features))
#             self.register_buffer('running_var', torch.ones(num_features))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()
#
#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))
#
#     def reset_running_stats(self) -> None:
#         if self.track_running_stats:
#             # running_mean/running_var/num_batches... are registered at runtime depending
#             # if self.track_running_stats is on
#             self.running_mean.zero_()  # type: ignore[operator]
#             self.running_var.fill_(1)  # type: ignore[operator]
#             self.num_batches_tracked.zero_()  # type: ignore[operator]
#
#     def reset_parameters(self) -> None:
#         self.reset_running_stats()
#
#     def extra_repr(self):
#         return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'track_running_stats={track_running_stats}'.format(**self.__dict__)
#
#     # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#     #                           missing_keys, unexpected_keys, error_msgs):
#     #     version = local_metadata.get('version', None)
#     #
#     #     if (version is None or version < 2) and self.track_running_stats:
#     #         # at version 2: added num_batches_tracked buffer
#     #         #               this should have a default value of 0
#     #         num_batches_tracked_key = prefix + 'num_batches_tracked'
#     #         if num_batches_tracked_key not in state_dict:
#     #             state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
#     #
#     #     super._load_from_state_dict(
#     #         state_dict, prefix, local_metadata, strict,
#     #         missing_keys, unexpected_keys, error_msgs)
#
#     def forward(self, input: Tensor) -> Tensor:
#         self._check_input_dim(input)
#
#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that it gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#
#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore
#                 self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)
#
#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """
#         assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
#         assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
#         effective_weights, effective_bias = self.sparseFunction()
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean if not self.training or self.track_running_stats else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             effective_weights, effective_bias, bn_training, exponential_average_factor, self.eps)

class LinearLR(nn.Module):
    """
    Which is a custom fully connected linear layer that its weights $W_f$
    remain constant once initialized randomly.
    A second weight matrix $W_m$ with the same shape as $W_f$ is used for
    generating a binary theta. This weight matrix can be trained through
    backpropagation. Each unit of $W_f$ may be passed through sigmoid
    function to generate the $p$ value of the $Bern(p)$ function.
    """

    def __init__(self, weights, name, args, bias=None):
        super(LinearLR, self).__init__()
        U, S, VT = torch.linalg.svd(weights, full_matrices=False)
        position = torch.nonzero(S, as_tuple=True)[0]
        U_comm = U[:, position] @ torch.sqrt(torch.diag(S[position]))
        VT_comm = torch.sqrt(torch.diag(S[position])) @ VT[position, :]
        self.U = nn.Parameter(U_comm, requires_grad=True)
        self.V = nn.Parameter(VT_comm, requires_grad=True)

        if bias!=None:
            self.bias = nn.Parameter(copy.deepcopy(bias), requires_grad=True)
        else:
            self.bias =None
        # del self.p
        # del self.p_bias


    def forward(self, x):
        lin = F.linear(x, self.V)
        lin = F.linear(lin,  self.U)
        return lin


class ConvLR(nn.Module):
    """
    Which is a custom fully connected linear layer that its weights $W_f$
    remain constant once initialized randomly.
    A second weight matrix $W_m$ with the same shape as $W_f$ is used for
    generating a binary theta. This weight matrix can be trained through
    backpropagation. Each unit of $W_f$ may be passed through sigmoid
    function to generate the $p$ value of the $Bern(p)$ function.
    """

    def __init__(self, weights, name, stride, padding, dilation, group, args, rank_scale=0.2, device=None):
        self.out_channel, self.in_channel, self.kernel_size, _ = weights.shape
        super(ConvLR, self).__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.shape = weights.shape

        dim1, dim2 = self.out_channel * self.kernel_size, self.in_channel * self.kernel_size
        self.rank = math.ceil(rank_scale * min(dim1, dim2))  # int(round(rank_scale * min(dim1, dim2)))

        self.w_U = nn.Parameter(torch.zeros(dim1, self.rank))
        self.w_VT = nn.Parameter(torch.zeros(self.rank, dim2))

        # U, VT = spectral_init(weights.swapaxes(1, 2).reshape([dim1, dim2]), self.rank)
        self.w_U.data = self.w_U
        self.w_VT.data = self.w_VT
        # print(torch.sum(weights.swapaxes(1, 2).reshape([dim1, dim2])-torch.matmul(self.w_U.data,self.w_VT)))



    def forward(self, x):
        # Apply the effective weight on the input data
        # effective_theta = self.theta_to_theta()
        # effective_weights  = self.w * effective_theta.unsqueeze(-1).unsqueeze(-1).expand( self.w.shape)
        # effective_weights, effective_bias=  self.sparseFunction()
        effective_w_VT = self.w_VT.reshape(self.rank, 1, self.in_channel, self.kernel_size).transpose(1, 2)
        effective_U = self.w_U.reshape(self.out_channel, self.kernel_size, self.rank, 1).transpose(1, 2)
        conv_first = F.conv2d(x, effective_w_VT, stride=(1, self.stride[1]),
                              padding=(0, self.padding[1]),
                              dilation=(1, self.dilation[1]),
                              groups=self.group)
        conv_second = F.conv2d(conv_first, effective_U, stride=(self.stride[0], 1),
                               padding=(self.padding[0], 0),
                               dilation=(self.dilation[0], 1),
                               groups=self.group, )
        return conv_second

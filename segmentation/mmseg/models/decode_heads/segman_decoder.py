# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *

import math
from timm.models.layers import DropPath, trunc_normal_


import math
import time
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM as DWConv
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmcv.cnn import ConvModule
from natten import NeighborhoodAttention2D, use_fused_na, use_gemm_na
from natten.functional import na2d, na2d_av, na2d_qk, natten2dav, natten2dqkrpb
# from timm.models.vision_transformer import _cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from natten.flops import qk_2d_rpb_flop, av_2d_flop, add_natten_handle

try:
    from csm_triton import CrossScanTriton, CrossMergeTriton
except:
    from .csm_triton import CrossScanTriton, CrossMergeTriton

import selective_scan_cuda_oflex



# use_gemm_na(True)
# use_fused_na(True)

import torch
import numpy as np

#################################################################################
#               Mamba scan functions that preserve image continuity             #
#################################################################################

def get_continuous_paths(N):
    # Note that N is always even since we use image resolution of 256, 512, 1024 with the SD VAE encoder
    paths_lr = []
    reverse_lr = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (N - 1, 0, -1, 1),
    ]:
        path = lr_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_lr.append(path)
        reverse_lr.append(reverse_permut(path))
    

    paths_tb = []
    reverse_tb = []
    for start_row, start_col, dir_row, dir_col in [
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        path =tb_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_tb.append(path)
        reverse_tb.append(reverse_permut(path))
    
    
    return paths_lr, paths_tb, reverse_lr, reverse_tb
    

def lr_tranverse(N,start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for i in range(N):
        for j in range(N):
            # If the row number is even, move right; otherwise, move left
            col = j if i % 2 == 0 else N - 1 - j
            path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
    return path

def tb_tranverse(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for j in range(N):
        for i in range(N):
            # If the column number is even, move down; otherwise, move up
            row = i if j % 2 == 0 else N - 1 - i
            path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
    return path

def reverse_permut(permutation):
    n = len(permutation)
    reverse = [0] * n
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse


def reverse_index_select(x, indices, dim, original_size):
    """
    Reverses the torch.index_select operation.
    
    Args:
    x (torch.Tensor): The tensor that was output by index_select.
    indices (torch.Tensor): The indices used in the original index_select operation.
    dim (int): The dimension along which the indexing was done.
    original_size (int): The original size of the dimension that was indexed.
    
    Returns:
    torch.Tensor: A tensor with the same shape as the original tensor before index_select.
    """
    # Create an output tensor filled with zeros
    output_shape = list(x.shape)
    output_shape[dim] = original_size
    output = torch.zeros(output_shape, dtype=x.dtype, device=x.device)

    # Use indexing to place the values back in their original positions
    if dim == 0:
        output[indices] = x
    elif dim == 1:
        output[:, indices] = x
    else:
        # For higher dimensions, we need to construct the proper indexing
        idx = [slice(None)] * len(output_shape)
        idx[dim] = indices
        output[idx] = x

    return output

class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2): # [B, C, H, W] -> [B, 4, H/w * W/w]
        B, C, N_window, W, H = x.size()
        N= W


        C = int(C/num_scans)
        split_indexes = [C,C,C,C]
        x1, x2, x3, x4 = torch.split(x, split_indexes, dim=1)

        xs = x.new_empty((B, num_scans, C, N_window, H * W))

        ctx.shape = (B, num_scans, C, N_window, H * W)
        ctx.H = H
        ctx.W = W

        paths_lr, paths_tb, reverse_lr, reverse_tb = get_continuous_paths(N)
        paths_lr = torch.tensor(paths_lr, device=x.device, dtype=torch.long)
        paths_tb = torch.tensor(paths_tb, device=x.device, dtype=torch.long)
        reverse_lr = torch.tensor(reverse_lr, device=x.device, dtype=torch.long)
        reverse_tb = torch.tensor(reverse_tb, device=x.device, dtype=torch.long)
        

        xs[:, 0] = torch.index_select(x1.contiguous().flatten(-2,-1), -1, paths_lr[0])
        xs[:, 1] = torch.index_select(x2.contiguous().flatten(-2,-1), -1, paths_lr[1])
        xs[:, 2] = torch.index_select(x3.contiguous().flatten(-2,-1), -1, paths_tb[0])
        xs[:, 3] = torch.index_select(x4.contiguous().flatten(-2,-1), -1, paths_tb[1])
    
        

        return xs, paths_lr, paths_tb, reverse_lr, reverse_tb

    
    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor): # [B, 4, H/w * W/w] -> [B, C, H, W]

        # grad_xs has size B, num_scans, C, N_window, H*W

        B, num_scans, C, N_window, L = ctx.shape
        H = ctx.H 
        W = ctx.W

        grad_x = grad_xs.new_empty((B, 4, C, N_window, H, W))
        
        grad_xs = grad_xs.view(B, 4, C, N_window, H, W)
        
        grad_x[:, 0, :, :, :] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, 1, :, :, :] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, 2, :, :, :] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, 3, :, :, :] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        return grad_x, None 

class EfficientMerge(torch.autograd.Function): # [B, 4, C, H/w * W/w] -> [B, C, H*W]
    @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):
        B, K, C, L = ys.shape
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
        ctx.shape = (H, W)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size


        new_h = H * step_size
        new_w = W * step_size

        y = ys.new_empty((B, C, new_h, new_w))


        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        
        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        y = y.view(B, C, -1)
        return y
    
    @staticmethod
    def backward(ctx, grad_x: torch.Tensor): # [B, C, H*W] -> [B, 4, C, H/w * W/w]

        H, W = ctx.shape
        B, C, L = grad_x.shape
        step_size = ctx.step_size

        grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))  
        W = grad_x.shape[3]

        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))
        H = grad_x.shape[2]
        B, C, H, W = grad_x.shape
        H = H // step_size
        W = W // step_size
        grad_xs = grad_x.new_empty((B, 4, C, H*W)) 

        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(B, C, -1) 
        grad_xs[:, 1] = grad_x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 3] = grad_x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].reshape(B, C, -1)
        
        return grad_xs, None, None, None         


# Todo accelerate this, expecially backprop
def cross_scan_continuous(x, num_scans =4, split=False):
    B, C, N_window, W, H = x.size()
    N= W

    if split and C>1:
        C = int(C/num_scans)
        split_indexes = [C,C,C,C]
        x1, x2, x3, x4 = torch.split(x, split_indexes, dim=1)

    xs = x.new_empty((B, num_scans, C, N_window, H * W))
    
    paths_lr, paths_tb, reverse_lr, reverse_tb = get_continuous_paths(N)
    paths_lr = torch.tensor(paths_lr, device=x.device, dtype=torch.long)
    paths_tb = torch.tensor(paths_tb, device=x.device, dtype=torch.long)
    reverse_lr = torch.tensor(reverse_lr, device=x.device, dtype=torch.long)
    reverse_tb = torch.tensor(reverse_tb, device=x.device, dtype=torch.long)
    
    if split and C>1:
        xs[:, 0] = torch.index_select(x1.flatten(-2,-1), -1, paths_lr[0])
        xs[:, 1] = torch.index_select(x2.flatten(-2,-1), -1, paths_lr[1])
        xs[:, 2] = torch.index_select(x3.flatten(-2,-1), -1, paths_tb[0])
        xs[:, 3] = torch.index_select(x4.flatten(-2,-1), -1, paths_tb[1])
    
    else:
        for i in range(paths_lr.size(0)):
            xs[:, i] = torch.index_select(x.flatten(-2,-1), -1, paths_lr[i])
            
        for i in range(paths_tb.size(0)):
            xs[:, i+num_scans//2] = torch.index_select(x.flatten(-2,-1), -1, paths_tb[i])
    
    return xs, paths_lr, paths_tb, reverse_lr, reverse_tb
    
# Todo accelerate this, expecially backprop
def cross_merge_continuous(ys, paths_lr, paths_tb, reverse_lr, reverse_tb, split=False):
    B, K, D, N_window, H, W = ys.shape
    L = W*H


    ys = ys.view(B, K, D, N_window, -1)
    ys = ys.permute(0,2,3,1,4) # B, D, N_window, K, L
    
    corresponding_scan_paths = torch.concat([reverse_lr,reverse_tb], dim=0).view(1,1,1,K,L)
    corresponding_scan_paths = corresponding_scan_paths.repeat(B,D,N_window,1,1)
    y = torch.gather(ys, -1, corresponding_scan_paths)

    if split:
        return y.permute(0,3,1,2,4).reshape(B,4*D,N_window,L)

    y = torch.sum(y,dim=2) # B, D, L
    return y    


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        return (sin, cos)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
          
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
    
    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1 
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

class VSSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=1,
        expansion_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_groups=4,
        **kwargs,    
    ):
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio !=1:
            self.xproj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)
        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)
        
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True) # (K * D)
        
        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)
    
    
    def forward(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        xs = self._cross_scan(x)
        if self.expansion_ratio!=1:
            xs = self.xproj(xs.permute(0,1,3,2).contiguous()).permute(0,1,3,2).contiguous()
        xs = xs.reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)
        
        # y = ys.reshape(B, K, -1, L)
        # yf = F.conv1d(y[:, 0, ...], weight=self.factor1, groups=D)
        # yb = F.conv1d(y[:, 1, ...].flip([-1]), weight=self.factor2, groups=D)
        # y = yf + yb
        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        
        if self.expansion_ratio!=1:
            y = self.yproj(y.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        if to_dtype:
            y = y.to(x.dtype)
        
        return y
     

class Attention(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 window_size, 
                 window_dilation, 
                 global_mode=False, 
                 image_size=None, 
                 use_rpb=False, 
                 sr_ratio=1):
        
        super().__init__()
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation
        self.global_mode = global_mode
        self.sr_ratio = sr_ratio

        image_size=to_2tuple(image_size)
        self.image_size = image_size
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)]
            if global_mode: 
                rpb_list.append(nn.Parameter(torch.empty(num_heads, image_size[0]*image_size[1], image_size[0]*image_size[1]), requires_grad=True))

            self.rpb = nn.ParameterList(rpb_list)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.qkv.weight, gain=2**-2.5)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_normal_(self.proj.weight, gain=2**-2.5)
        nn.init.zeros_(self.proj.bias)
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item) # which better? nn.init.trunc_normal_(item, std=0.02)
    
    
    def forward(self, x, pos_enc):
        
        B, C, H, W = x.shape
        
        # attn time
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)
        
        sin, cos = pos_enc
        q = theta_shift(q, sin, cos) * self.scale
        k = theta_shift(k, sin, cos)
        
        if hasattr(self, 'rpb'):
            rpb = self.rpb[0]
        else:
            rpb = None
              
        attn = na2d_qk(q, k, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb)
        attn = torch.softmax(attn, dim=-1) # b, h, h, w, k^2
        x = na2d_av(attn, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation)

        x = rearrange(x, 'b n_h h w c -> b (n_h c) h w', n_h=self.num_heads).contiguous()
        x = x + lepe
        x = self.proj(x)
        
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class VSSMBlock(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 expansion_ratio=1,
                 channel_split=False,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1, 
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=VSSM,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d,
                 dropout=0.1):
        # retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.token_mixer = token_mixer(d_model=embed_dim,
                                    k_groups=4, 
                                    expansion_ratio=expansion_ratio,
                                    channel_split=channel_split,)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
            
        return x


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x


@HEADS.register_module()
class SegMANDecoder(BaseDecodeHead):
    def __init__(self, image_size=512,channel_split=False,short_cut=False, interpolate_mode='bilinear',use_rpb=False,
                  **kwargs):
        super(SegMANDecoder, self).__init__(input_transform="multiple_select", **kwargs) #input_transform='multiple_select'
        image_size = to_2tuple(image_size)
        image_size = [(image_size[0]//2**(i+2), image_size[1]//2**(i+2)) for i in range(4)]
        

        self.embed_dim = kwargs['channels']

        # downsample using convolutions
        self.conv_downsample_2 = ConvModule(
                        self.embed_dim, self.embed_dim*2, kernel_size=3, stride=2, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.conv_downsample_4 = ConvModule(
                        self.embed_dim, self.embed_dim*4, kernel_size=5, stride=4, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.feat_proj_dim = kwargs['feat_proj_dim']

        # try using all features at once
        self.short_cut = short_cut
        self.linear_c4 = MLP(self.in_channels[-1], self.feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], self.feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], self.feat_proj_dim)

        self.linear_fuse = ConvModule(
                        in_channels=self.feat_proj_dim*3,
                        out_channels=self.embed_dim,
                        kernel_size=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        

        self.reduce_channels = nn.ModuleList([ConvModule(in_channels=self.embed_dim*4*(2**i),
                                out_channels=self.embed_dim,kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)) for i in range(3)])
        vssm_dim = self.embed_dim*3

        self.vssm =VSSMBlock(embed_dim=vssm_dim,
                                    expansion_ratio=1,
                                    channel_split=False,)

        self.short_path = ConvModule(
                            in_channels=self.embed_dim,
                            out_channels=self.embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(self.embed_dim, self.embed_dim, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        self.proj_out = ConvModule(in_channels=vssm_dim,
                                out_channels=self.feat_proj_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))

        feat_concat_dim = self.embed_dim*(2+ 3) + self.feat_proj_dim*3
        self.cat = ConvModule(in_channels=feat_concat_dim,
                                out_channels=self.embed_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) 

        self.interpolate_mode = interpolate_mode


    def forward_mlp_decoder(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
   
        _c4 = resize(_c4, size=inputs[1].size()[2:],mode='bilinear',align_corners=False).contiguous()
        _c3 = resize(_c3, size=inputs[1].size()[2:],mode='bilinear',align_corners=False).contiguous()
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))
        
        return _c, _c2, _c3, _c4


    def forward_winssm(self, x: torch.Tensor, c2, c3, c4, c1=None):
        out = [self.short_path(x), 
                  resize(self.image_pool(x),
                        size=x.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners).contiguous()]

        B, C, H, W = x.size()

        # obtain multi scale features
        x_2 = self.conv_downsample_2(x) # 1/2 resolution
        x_4 = self.conv_downsample_4(x) # 1/4 resolution

        # unshuffle all features to size 1/4 resolution (16x16 for 512 input res)
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        # reduce channels
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        multi_x = torch.cat([x_unshuffle,x_2_unshuffle,x_4], dim=1)

        _out = self.vssm(multi_x)

        _out = resize(_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        _out_ = self.proj_out(_out)
        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_
 
        out += [_out, c2,c3,c4]

        out = self.cat(torch.cat(out, dim=1))

        return out

 
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x, c2, c3, c4 = self.forward_mlp_decoder(x)
        x = self.forward_winssm(x, c2, c3, c4)
        output = self.cls_seg(x)
        return output
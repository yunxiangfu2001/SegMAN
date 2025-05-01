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

        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)

        self.expansion_ratio = expansion_ratio

        if self.expansion_ratio != 1.0 :
            self.proj = nn.Linear(d_model,d_inner)
            self.yproj = nn.Linear(d_inner,d_model)
        
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
        # xs = self._cross_scan(x).reshape(B, -1, L)
        if self.expansion_ratio != 1.0:
            x = self.proj(x.permute(0,2,3,1)).permute(0,3,1,2)
            xs = self._cross_scan(x) # size b, 4, embed_dim, L
            xs = xs.reshape(B, -1, L)
        else:
            xs = self._cross_scan(x).reshape(B, -1, L)
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
        
        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, L)

        if self.expansion_ratio != 1.0:
            y = self.yproj(y.permute(0,2,1)).permute(0,2,1)
        
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
                 sr_ratio=1,
                 fused_na=False,
                 ssm_ratio=1):
        
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
        self.image_size = image_size
        self.fused_na = fused_na
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        if not global_mode:
            self.dwconv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.BatchNorm2d(embed_dim),
            )
            self.ssm = VSSM(d_model=embed_dim, expansion_ratio=ssm_ratio)
            self.norm = LayerNorm2d(embed_dim)
            
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
    
        if self.fused_na:
            q = rearrange(q, 'b n h w c -> b h w n c')
            k = rearrange(k, 'b n h w c -> b h w n c')
            v = rearrange(v, 'b n h w c -> b h w n c')

            x = na2d(q, k, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation, scale=float(q.size(-1)**0.5))
            q = rearrange(q, 'b h w n c -> b n h w c')
            k = rearrange(k, 'b h w n c -> b n h w c')
            x = rearrange(x, 'b h w n c -> b n h w c')

        else:
            attn = na2d_qk(q, k, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb)
            attn = torch.softmax(attn, dim=-1) # b, h, h, w, k^2
            x = na2d_av(attn, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation)
        
        if not self.global_mode:
            
            q = rearrange(q, 'b n h w c -> b n c h w')
            k = rearrange(k, 'b n h w c -> b n c h w')
            v = rearrange(x, 'b n h w c -> b n c h w')
            
            v_r = v.flatten(1, 2)
            v = self.dwconv(v_r)
            v = F.silu(v)
            v = self.ssm(v)
            
            v = self.norm(v.reshape(B, -1, H, W))
        
            x = v + v_r

        else:
            
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b n h w c -> b n (h w) c')
            
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')
            
            if hasattr(self, 'rpb'):
                attn = attn + self.rpb[-1]
                
            attn = torch.softmax(attn, dim=-1)  
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(B, -1, H, W)
        
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



class Block(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=256, 
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=Attention,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d):
        # retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim

        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = token_mixer(embed_dim, num_heads, window_size, window_dilation, global_mode, image_size, use_rpb, sr_ratio)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = channel_mixer(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc):
        
        x = x + self.cpe1(x)
        x = x + self.drop_path(self.layer_scale1(self.token_mixer(self.norm1(x), pos_enc)))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        
        # x = self.layer_scale1(x + self.cpe1(x))
        # x = x + self.drop_path(self.layer_scale1(self.token_mixer(self.norm1(x), pos_enc)))
        # x = self.layer_scale2(x + self.cpe2(x))
        # x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))  
            
        return x
    

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 image_size=None,
                 embed_dim=64, 
                 depth=4, 
                 num_heads=4,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=96, 
                 drop_path=0,
                 layerscale=False, 
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 use_checkpoint=0,
            ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # self.RoPE = RoPE(embed_dim, num_heads)

        self.rope = RoPE(embed_dim, num_heads)
        
        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_dim=embed_dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          window_dilation=window_dilation,
                          global_mode=global_mode,
                          ffn_dim=ffn_dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layerscale=layerscale,
                          layer_init_values=layer_init_values,
                          norm_layer=norm_layer,
                          image_size=image_size,
                          use_rpb=use_rpb,
                          sr_ratio=sr_ratio,
            )
            self.blocks.append(block)

    def forward(self, x):
        pos_enc = self.rope((x.shape[2:]))
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, pos_enc, use_reentrant=False)
            else:
                x = blk(x, pos_enc)
        return x


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )


class SegMANEncoder(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=[7, 7, 7, 7],
                 window_dilation=[1, 1, 1, 1],
                 use_rpb=False,
                 sr_ratio=[8, 4, 2, 1],
                 mlp_ratios=[4, 4, 4, 4], 
                 drop_path_rate=0,
                 projection=1024,
                 layerscales=[False, False, False, False],
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 drop_rate=0, 
                 use_checkpoint=[0, 0, 0, 0],
                 pretrain_path=None,):
        super().__init__()

        self.pretrained=pretrain_path
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = stem(in_chans=in_chans, embed_dim=embed_dims[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # input resolution
        image_size = to_2tuple(image_size)
        image_size = [(image_size[0]//2**(i+2), image_size[1]//2**(i+2)) for i in range(4)]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                window_dilation=window_dilation[i_layer],
                global_mode=(i_layer==3),
                use_rpb=use_rpb,
                sr_ratio=sr_ratio[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
                norm_layer=norm_layer,
                image_size=image_size[i_layer],
                use_checkpoint=use_checkpoint[i_layer],
            )
                       
            downsample = nn.Sequential(
                nn.Conv2d(embed_dims[i_layer], embed_dims[i_layer+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims[i_layer+1])
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            
            self.layers.append(layer)
            self.layers.append(downsample)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, projection, kernel_size=1),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity(),
        )

        self.apply(self._init_weights)
        self.init_pretrain_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
    def init_pretrain_weights(self, pretrained=None):
        if isinstance(self.pretrained, str):
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict_name = 'state_dict_ema'
            state_dict = checkpoint[state_dict_name]
            self.load_state_dict(state_dict, strict=False)
            print(f"loaded state dict using {state_dict_name} from {self.pretrained}")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x).flatten(1)
        return x
    
    def flops(self, shape=(3, 224, 224),batch_size=128):
        
        supported_ops={
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
            "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": av_2d_flop,
        }

        model = copy.deepcopy(self)
        
        if torch.cuda.is_available:
            model.cuda()
        model.eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        input = torch.randn((batch_size, *shape), device=next(model.parameters()).device)

        # latency
        start_time=time.time()
        for _ in range(128):
            _ = model(input)
        FPS = batch_size*128/(time.time()-start_time)

        del model, input
        return (sum(Gflops.values()), params, FPS)



def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',  # 'bilinear' or 'bicubic'
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }



@register_model
def SegMANEncoder_t(pretrained=None, pretrained_cfg=None, **args):
    model = SegMANEncoder(
        embed_dims=[32, 64, 144, 192],
        depths=[2, 2, 4, 2],
        num_heads=[1, 2, 4, 8],
        window_size=[11, 9, 9, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[False, False, False, False],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        pretrain_path=pretrained,
        **args,
    )
    model.default_cfg = _cfg()
    return model

@register_model
def SegMANEncoder_s(pretrained=None, pretrained_cfg=None, **args):
    model = SegMANEncoder(
        embed_dims=[64, 144, 288, 512],
        depths=[2, 2, 10, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3.4, 3.4],
        layerscales=[False, False, False, False],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        pretrain_path=pretrained,
        **args,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def SegMANEncoder_b(pretrained=None, pretrained_cfg=None, **args):
    model = SegMANEncoder(
        embed_dims=[96, 160, 364, 560],
        depths=[4, 4, 18, 4],
        num_heads=[4, 8, 13, 20],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[True, True, True, True],
        layer_init_values=1e-6,
        use_rpb=True,
        norm_layer=LayerNorm2d,
        pretrain_path=pretrained,
        **args,
    )
    model.default_cfg = _cfg(crop_pct=0.95)
    return model


@register_model
def SegMANEncoder_l(pretrained=None, pretrained_cfg=None, **args):
    model = SegMANEncoder(
        embed_dims=[96, 192, 432, 640],
        depths=[4, 4, 28, 4],
        num_heads=[4, 8, 12, 20],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[True, True, True, True],
        layer_init_values=1e-6,
        use_rpb=True,
        norm_layer=LayerNorm2d,
        pretrain_path=pretrained,
        **args,
    )
    model.default_cfg = _cfg(crop_pct=0.95)
    return model

if __name__ == "__main__":
    

    from mmcv.cnn import get_model_complexity_info
    from mmengine.runner import load_checkpoint
    from natten.flops import get_flops
    from timm.utils import random_seed
    
    random_seed(6)
    
    device = 'cuda'
        
    @torch.no_grad()
    def inference(model, input):
        model.eval()
        return model(input)
    
    model = vmamba_former_s2().to(device)
    print(model)
    
    # load_checkpoint(model, 'output/vmamba_former_v1_s1_128x16/20240626-191745-vmamba_former_v1_s1-224/checkpoints/checkpoint-12.pth.tar')

    x = torch.randn(1, 3, 224, 224).to(device)
    
    y = inference(model, x)
    print(y.flatten(0).sum())
    
    # flops = get_flops(model, input=torch.rand(1, 3, 224, 224, device=device))
    # print(f'FLOPs: {flops/1e9:.2f}')
    # print(f"Params: {sum(p.numel() for p in list(model.parameters()))/1e6:.2f}")
    
    flops, params = model.flops()
    print(f'GFLOPs = {flops / 1e9}')
    print('Params(M) =', params / 1e6)

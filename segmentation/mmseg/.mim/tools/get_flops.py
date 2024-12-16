# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import torch
import time

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor

import selective_scan_cuda_oflex
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from natten.flops import qk_2d_rpb_flop, av_2d_flop, add_natten_handle

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=1024,
        help='input image size')
    args = parser.parse_args()
    return args


def flops_fvcore(model2, shape=(3,512,512)):
    supported_ops={
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": qk_2d_rpb_flop,
        "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": av_2d_flop,
    }

    model = copy.deepcopy(model2)
    
    if torch.cuda.is_available:
        model.cuda()
    model.eval()

    batch_size=1
    input = torch.randn((batch_size, *shape), device=next(model.parameters()).device)
    params = parameter_count(model)[""]
    # Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    # latency
    start_time=time.time()
    t4_ssm=[]
    t4_attn=[]
    t3_ssm=[]
    t3_attn=[]
    t2_ssm=[]
    t2_attn=[]
    t1_ssm=[]
    t1_attn=[]
    for _ in range(1024*4):
        _, times = model(input)
        t4, t3, t2, t1 = times
        t4_ssm.append(t4[1])
        t4_attn.append(t4[0])
        t3_ssm.append(t3[1])
        t3_attn.append(t3[0])
        t2_ssm.append(t2[1])
        t2_attn.append(t2[0])
        t1_ssm.append(t1[1])
        t1_attn.append(t1[0])

    fps = (batch_size*256)/(time.time()-start_time)

    
    s4_time_total_attn = ( sum(t4_attn))/(batch_size*256)
    s3_time_total_attn = (sum(t3_attn))/(batch_size*256)
    s2_time_total_attn = (sum(t2_attn))/(batch_size*256)
    s1_time_total_attn = (sum(t1_attn))/(batch_size*256)

    s4_time_total_ssm = (sum(t4_ssm) )/(batch_size*256)
    s3_time_total_ssm = (sum(t3_ssm) )/(batch_size*256)
    s2_time_total_ssm = (sum(t2_ssm) )/(batch_size*256)
    s1_time_total_ssm = (sum(t1_ssm) )/(batch_size*256)

    total_time_attn = sum(t4_attn) + sum(t3_attn) + sum(t2_attn) + sum(t1_attn)
    total_time_ssm = sum(t4_ssm) + sum(t3_ssm) + sum(t2_ssm) + sum(t1_ssm)

    print('s4_time_total_attn', s4_time_total_attn)
    print('s3_time_total_attn', s3_time_total_attn)
    print('s2_time_total_attn', s2_time_total_attn)
    print('s1_time_total_attn', s1_time_total_attn)

    print('s4_time_total_ssm', s4_time_total_ssm)
    print('s3_time_total_ssm', s3_time_total_ssm)
    print('s2_time_total_ssm', s2_time_total_ssm)
    print('s1_time_total_ssm', s1_time_total_ssm)

    print('total_time_attn',total_time_attn)
    print('total_time_ssm',total_time_ssm)

    del model, input

    print(f'fvcore GFLOPs: {sum(Gflops.values())}')
    print(f'fvcore Params: {params/10**6}M')
    print(f'FPS: {fps}')

    return (sum(Gflops.values()), params, fps)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    print(split_line)
    print('using fvcore to compute model complexity:')
    flops_fvcore(model, input_shape)
    print(split_line)


if __name__ == '__main__':
    main()

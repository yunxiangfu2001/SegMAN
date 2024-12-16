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

from collections import OrderedDict

def count_parameters(model):
    # Create an OrderedDict to store parameter names and their counts
    param_count = OrderedDict()
    
    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count[name] = param.numel()
    
    # Sort the OrderedDict by parameter count (descending order)
    sorted_param_count = OrderedDict(sorted(param_count.items(), key=lambda x: x[1], reverse=True))
    
    # Calculate total parameters
    total_params = sum(sorted_param_count.values())
    
    # Print results
    print(f"{'Parameter Name':<50} {'Number of Parameters':<20} {'Percentage':<10}")
    print("-" * 80)
    
    for name, count in sorted_param_count.items():
        percentage = (count / total_params) * 100
        print(f"{name:<50} {count:<20,d} {percentage:.2f}%")
    
    print("-" * 80)
    print(f"{'Total':<50} {total_params:<20,d} 100.00%")

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
    parser.add_argument('--bs', help='batch size',default=64)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=1024,
        help='input image size')
    args = parser.parse_args()
    return args


def flops_fvcore(model2, shape=(3,512,512),args=None):
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

    batch_size=int(args.bs)
    input = torch.randn((1, *shape), device=next(model.parameters()).device)
    print(input.size())
    params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    # latency
    start_time=time.time()
    input = torch.randn((batch_size, *shape), device=next(model.parameters()).device)
    for _ in range(128):
        _ = model(input)
    
    fps = (batch_size*128)/(time.time()-start_time)

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

    count_parameters(model.decode_head)


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
    flops_fvcore(model, input_shape, args=args)
    print(split_line)


if __name__ == '__main__':
    main()

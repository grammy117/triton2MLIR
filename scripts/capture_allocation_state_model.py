#!/usr/bin/env python3
"""Capture real model calls whose tensor values supply aliased allocation roots."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture_updates(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/readonly_alias_kernels.py"
    spec = importlib.util.spec_from_file_location("allocation_model_kernels",source)
    kernels = importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
    qualname = "t2m_demo::allocation_state"

    @triton_op(qualname,mutates_args={})
    def allocation_state(state:torch.Tensor,n:int,steps:int,read_origin:int,first_origin:int,
                         second_origin:int,capacity:int)->torch.Tensor:
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        wrap_triton(kernels.aliased_updates)[((n+7)//8,)](
            flat[read_origin:],flat[first_origin:],flat[second_origin:],n,steps,B=8)
        return updated

    @allocation_state.register_kernel("cpu")
    def reference(state,n,steps,read_origin,first_origin,second_origin,capacity):
        if state.numel() != capacity:
            raise ValueError("source allocation capacity differs from its logical model value")
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        for iteration in range(steps):
            value = flat[read_origin:read_origin+n].clone()
            flat[first_origin:first_origin+n] = value+3
            forwarded = flat[read_origin:read_origin+n].clone()
            flat[second_origin:second_origin+n] = forwarded*2
        return updated

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(48,dtype=torch.int64).reshape(3,16)+2**53+1,requires_grad=False)
            self.bias = torch.nn.Parameter(torch.arange(48,dtype=torch.int64).reshape(3,16)%7-3,requires_grad=False)

        def forward(self,x):
            first = allocation_state(x+self.weight,17,2,0,0,0,48)
            second = allocation_state(first,7,2,4,0,8,48)
            empty = allocation_state(x[:0],0,2,0,0,0,0)
            return second+self.bias,first,empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model,values = Model().eval(),(torch.arange(48,dtype=torch.int64).reshape(3,16)%5,)
        out.mkdir(parents=True,exist_ok=False)
        result = m2m.convert(model,values,backend="fx_importer",weights_path=str(out / "weights.safetensors"),capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model,values,out,capture_regions=False,capture_trace=True,
                     source_path=Path(__file__),conversion_result=result)
    finally:
        if previous is None: KERNEL_REGISTRY.pop(qualname,None)
        else: KERNEL_REGISTRY[qualname] = previous
    return out


def capture_dot(out,dtype="f32"):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    storage = {"f32":torch.float32,"f16":torch.float16,"bf16":torch.bfloat16}[dtype]
    source = Path(__file__).resolve().parents[1] / "t2m/examples/strided_memory_kernels.py"
    spec = importlib.util.spec_from_file_location("allocation_dot_kernels",source)
    kernels = importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
    qualname = "t2m_demo::allocation_dot"

    @triton_op(qualname,mutates_args={})
    def allocation_dot(x:torch.Tensor,weight:torch.Tensor,state:torch.Tensor,m:int,n:int,k:int,
        a_row:int,a_k:int,b_k:int,b_col:int,a_base:int,b_base:int,a_count:int,b_count:int,
        out_row:int,out_col:int,out_base:int,out_count:int,c_origin:int)->torch.Tensor:
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        wrap_triton(kernels.strided_output_dot)[((m+15)//16,(n+15)//16)](
            x.contiguous(),weight.contiguous(),flat[c_origin:],flat,m,n,k,a_row,a_k,b_k,b_col,
            a_base,b_base,a_count,b_count,out_row,out_col,out_base,out_count,BM=16,BN=16,BK=32)
        return updated

    @allocation_dot.register_kernel("cpu")
    def reference(x,weight,state,m,n,k,a_row,a_k,b_k,b_col,a_base,b_base,a_count,b_count,
                  out_row,out_col,out_base,out_count,c_origin):
        if (tuple(x.shape) != (m,k) or tuple(weight.shape) != (k,n) or tuple(state.shape) != (m,n)
            or (a_row,a_k,b_k,b_col,a_base,b_base,a_count,b_count,out_row,out_col,out_base,out_count,c_origin)
                != (k,1,n,1,0,0,m*k,k*n,n,1,0,m*n,0)):
            raise ValueError("reference requires the declared dense allocation views")
        result = state.clone()
        left,right = x.float(),weight.float()
        for inner in range(((k+31)//32)*32):
            product = left[:,inner:inner+1]*right[inner:inner+1,:] if inner < k else torch.zeros_like(result)
            result = result+product
        return result

    m,n,k = 17,19,37
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(((torch.arange(k*n).reshape(k,n)*7%31-15).float()/8).to(storage),requires_grad=False)
            initial = (torch.arange(m*n).reshape(m,n)%7-3).float()/8
            initial[0,0] = 2**24
            self.initial = torch.nn.Parameter(initial,requires_grad=False)
            self.bias = torch.nn.Parameter((torch.arange(n)%5-2).float()/8,requires_grad=False)

        def forward(self,x):
            full = allocation_dot(x,self.weight,self.initial,m,n,k,k,1,n,1,0,0,m*k,k*n,n,1,0,m*n,0)
            empty_k = allocation_dot(x[:,:0],self.weight[:0],self.initial,m,n,0,0,1,n,1,0,0,0,0,n,1,0,m*n,0)
            return full+self.bias,empty_k,self.initial+self.bias

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (((torch.arange(m*k).reshape(m,k)%17-8).float()/8).to(storage),)
        out.mkdir(parents=True,exist_ok=False)
        result = m2m.convert(model,values,backend="fx_importer",weights_path=str(out / "weights.safetensors"),capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model,values,out,capture_regions=False,capture_trace=True,
                     source_path=Path(__file__),conversion_result=result)
    finally:
        if previous is None: KERNEL_REGISTRY.pop(qualname,None)
        else: KERNEL_REGISTRY[qualname] = previous
    return out


def capture(out,kind="updates"):
    return capture_updates(out) if kind == "updates" else capture_dot(out,{"dot":"f32","half-dot":"f16","bfloat-dot":"bf16"}[kind])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--kind",choices=("updates","dot","half-dot","bfloat-dot"),default="updates")
    args = parser.parse_args()
    print(capture(args.out,args.kind))

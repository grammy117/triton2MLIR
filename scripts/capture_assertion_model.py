"""Capture a functional checked kernel with captured state and tuple results."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    path = Path(__file__).resolve().parents[1] / "t2m/examples/assertion_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_assertion_sources",path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::checked_copy"

    @triton_op(qualname,mutates_args={})
    def implementation(x:torch.Tensor,initial:torch.Tensor,n:int,steps:int,limit:int)->torch.Tensor:
        out = initial.clone()
        wrap_triton(kernels.checked_copy)[(3,)](x,out,n,steps,limit,B=8,MODE=2,RANK=2,debug=True)
        return out

    @implementation.register_kernel("cpu")
    def reference(x,initial,n,steps,limit):
        out = initial.clone()
        for step in range(steps):
            if step % 2 == 0:
                if n < 0:
                    raise RuntimeError("source count must be nonnegative")
                if torch.any(x[:n] > limit):
                    raise RuntimeError("source tile exceeds limit")
                wide = x[:n].to(torch.int64) + step
                if torch.any((wide < -(2**31)) | (wide >= 2**31)):
                    raise RuntimeError("int32 overflow detected for operation add")
                out[:n] = wide.to(torch.int32)
        return out

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.initial = torch.nn.Parameter(torch.arange(17,dtype=torch.int32) - 77,requires_grad=False)
        def forward(self,x):
            return implementation(x,self.initial,13,3,5), x.clone(), self.initial.clone()

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model,values = Model().eval(), (torch.arange(17,dtype=torch.int32) % 9 - 4,)
        out.mkdir(parents=True)
        result = m2m.convert(model,values,backend="fx_importer",weights_path=str(out / "weights.safetensors"),capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model,values,out,conversion_result=result,capture_trace=True,capture_regions=False,source_path=Path(__file__))
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname,None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    print(capture(parser.parse_args().out))

"""Capture public precise math with learned bounds and complete tuple results."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op,wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY,KernelSpec


def capture(out):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    path = Path(__file__).resolve().parents[1] / "t2m/examples/scalar_math_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_scalar_math_sources",path)
    kernels = importlib.util.module_from_spec(spec);spec.loader.exec_module(kernels)
    qualname = "t2m_demo::precise_values"

    @triton_op(qualname,mutates_args={})
    def implementation(x:torch.Tensor,y:torch.Tensor,low:torch.Tensor,high:torch.Tensor,n:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        outputs = [torch.empty_like(x) for _ in range(4)]
        wrap_triton(kernels.precise_values)[((n+7)//8,)](x,y,low,high,*outputs,n,B=8)
        return tuple(outputs)

    @implementation.register_kernel("cpu")
    def reference(x,y,low,high,n):
        # Public div_rn/sqrt_rn require correct f32 rounding. Torch's vector
        # f32 sqrt may differ by one ulp; upstream Triton tests use f64 then
        # narrow for the precise builtins' independent reference.
        return (torch.fmin(torch.fmax(x,low),high),torch.minimum(torch.maximum(x,low),high),
            (x.double()/y.double()).float(),torch.sqrt(x.double()).float())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.low = torch.nn.Parameter(torch.full((97,),-1.0),requires_grad=False)
            self.high = torch.nn.Parameter(torch.full((97,),1.0),requires_grad=False)
        def forward(self,x,y):
            return (*implementation(x,y,self.low,self.high,97),x.clone(),y.clone(),self.low.clone(),self.high.clone())

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        x,y = torch.linspace(-4,4,97),torch.arange(97,dtype=torch.float32)%7-3
        x[0],x[1],x[2],x[3],x[4] = 0.0,-0.0,float("inf"),float("-inf"),float("nan")
        model,values = Model().eval(),(x,y)
        out.mkdir(parents=True)
        result = m2m.convert(model,values,backend="fx_importer",weights_path=str(out / "weights.safetensors"),capture_trace=True)
        if not result.ok: raise ValueError(result.diagnostics)
        write_bundle(model,values,out,conversion_result=result,capture_trace=True,capture_regions=False,source_path=Path(__file__))
    finally:
        if previous is None: KERNEL_REGISTRY.pop(qualname,None)
        else: KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(capture(args.out))

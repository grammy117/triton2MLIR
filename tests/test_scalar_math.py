"""Strict math uses upstream conversions and independent exact-rational oracles."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.effect_bundle import write_effect_bundle
from t2m.examples.scalar_math_reference import arrays,reference,rounded,fields,inputs,recurrence
from t2m.frontend import capture_ttir
from t2m.kernel_consumer import qualify_kernel,verify_kernel_qualification
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(key) for key in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing native/Merlin environments required")


def typed_source(dtype,*,word_copy=False,control=False):
    """Legal typed TTIR, including types absent from Python's f32-only builtins."""
    names = ["x","y","low","high","no_nan","all_nan","divided","root"]
    if word_copy: names.append("raw")
    args = ",".join(f"%{name}: !tt.ptr<{dtype}>" for name in names)
    extra = ",%steps:i32,%enabled:i1" if control else ""
    lines = [f"module {{ tt.func @typed_math({args},%n:i32{extra}) {{",
        "%pid = tt.get_program_id x : i32", "%B = arith.constant 8 : i32",
        "%base = arith.muli %pid,%B : i32", "%r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>",
        "%base_vec = tt.splat %base : i32 -> tensor<8xi32>",
        "%lane = arith.addi %base_vec,%r : tensor<8xi32>", "%nvec = tt.splat %n : i32 -> tensor<8xi32>",
        "%mask = arith.cmpi slt,%lane,%nvec : tensor<8xi32>",
        f"%zero = arith.constant dense<0.0> : tensor<8x{dtype}>",
        f"%one = arith.constant dense<1.0> : tensor<8x{dtype}>",
        f"%minus = arith.constant dense<-1.0> : tensor<8x{dtype}>"]
    for name in names:
        lines.extend([f"%{name}_base = tt.splat %{name} : !tt.ptr<{dtype}> -> tensor<8x!tt.ptr<{dtype}>>",
            f"%{name}_ptr = tt.addptr %{name}_base,%lane : tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>"])
    for name,fill in zip(names[:4],["zero","one","minus","one"],strict=True):
        lines.append(f"%{name}_v = tt.load %{name}_ptr,%mask,%{fill} : tensor<8x!tt.ptr<{dtype}>>")
    value = "x_v"
    if control:
        lines.extend(["%c0 = arith.constant 0 : i32","%c1 = arith.constant 1 : i32",
            f"%state = scf.for %iv = %c0 to %steps step %c1 iter_args(%carry = %x_v) -> (tensor<8x{dtype}>) : i32 {{",
            f"%next = scf.if %enabled -> (tensor<8x{dtype}>) {{",
            f"%q = tt.precise_divf %carry,%y_v : tensor<8x{dtype}>",
            f"%rroot = tt.precise_sqrt %q : tensor<8x{dtype}>",
            f"scf.yield %rroot : tensor<8x{dtype}>","} else {",
            f"scf.yield %carry : tensor<8x{dtype}>","}",f"scf.yield %next : tensor<8x{dtype}>","}"])
        value = "state"
    lines.extend([f"%no_nan_v = tt.clampf %{value},%low_v,%high_v,propagateNan=none : tensor<8x{dtype}>",
        f"%all_nan_v = tt.clampf %{value},%low_v,%high_v,propagateNan=all : tensor<8x{dtype}>",
        f"%divided_v = tt.precise_divf %{value},%y_v : tensor<8x{dtype}>",
        f"%root_v = tt.precise_sqrt %{value} : tensor<8x{dtype}>"])
    for name in names[4:]:
        value = "x_v" if name == "raw" else name+"_v"
        lines.append(f"tt.store %{name}_ptr,%{value},%mask : tensor<8x!tt.ptr<{dtype}>>")
    return "\n".join([*lines,"tt.return", "} }"])


def convert(root,profile,dtype,*,public=False,control=False):
    from t2m.dynamic_abi import GridExtent
    path = root / "source.ttir"
    names = ["x","y","low","high","no_nan","all_nan","divided","root"]
    entry = "typed_math"
    if public:
        from t2m.examples.scalar_math_kernels import precise_values
        assert dtype == "f32"
        entry = precise_values.__name__
        path.write_text(capture_ttir(precise_values,{**{name:"*fp32" for name in names},"n":"i32","B":"constexpr"},{"B":8}))
    else:
        path.write_text(typed_source(dtype,control=control))
    abi = TensorABI(entry,tuple(TensorBuffer(name,("n",),"read" if index < 4 else "write",dtype) for index,name in enumerate(names)),
        grid=(GridExtent("n",8),),runtime_scalars=(RuntimeScalar("n","i32",8,bounds=(0,127)),
            *((RuntimeScalar("steps","i32",9,bounds=(0,3)),RuntimeScalar("enabled","i1",10)) if control else ())))
    return convert_shared(path,abi,root / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


def qualify(result,root,dtype,*,control=False):
    values = inputs(dtype)
    goldens = reference(recurrence(values,dtype,2,True) if control else values,dtype)
    ref = {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
        "results":{"archive":"reference.npz","comparisons":[{"kind":"values","rtol":0,"atol":0},*[{"kind":"ieee"}]*3]}}
    capture,qualified = root / "capture",root / "qualified"
    write_effect_bundle(result,capture,[*values,np.asarray(len(values[0]),np.int32),
        *([np.asarray(2,np.int32),np.asarray(True)] if control else [])],ref,references=goldens,
        work=root / "native",python=NATIVE)
    verify_bundle(capture)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    opcodes = {record["opcode"] for record in result.trace["source_operations"]}
    assert {"tt.clampf","tt.precise_divf","tt.precise_sqrt"} <= opcodes
    assert "arith.divf" in result.mlir_text and "math.sqrt" in result.mlir_text
    assert "arith.maximumf" in result.mlir_text and "arith.maxnumf" in result.mlir_text
    assert json.loads((qualified / "evaluation/evaluation.json").read_text())["invocations"] == 1
    for op in result.module.walk():
        if op.name in ("arith.divf","math.sqrt","arith.maximumf","arith.maxnumf"):
            flags = op.properties.get("fastmath")
            assert flags is None or not flags.data
    if dtype == "bf16":
        assert "emulate_bfloat_elementwise" in json.loads((qualified / "host/compilation.json").read_text())["features"]
    return capture,qualified


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_bfloat_recurrence_rounds_each_iteration_and_retains_runtime_control(tmp_path,profile):
    result = convert(tmp_path,profile,"bf16",control=True)
    assert "scf.for" in result.mlir_text and "scf.if" in result.mlir_text
    _,qualified = qualify(result,tmp_path,"bf16",control=True)
    from t2m.invocation import invoke_kernel,verify_invocation
    for index,(n,steps,enabled) in enumerate([(0,3,True),(1,0,True),(13,3,False),(127,1,True)]):
        values = inputs("bf16",n)
        expected = reference(recurrence(values,"bf16",steps,enabled),"bf16")
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate([
            *values,np.asarray(n,np.int32),np.asarray(steps,np.int32),np.asarray(enabled)])})
        np.savez(caller / "reference.npz",**{f"out{i}":value for i,value in enumerate(expected)})
        (caller / "reference.json").write_text(json.dumps({"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
            "results":{"archive":"reference.npz","comparisons":[{"kind":"values","rtol":0,"atol":0},*[{"kind":"ieee"}]*3]}}))
        out = tmp_path / f"invoked-{index}"
        record = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert record["native_compilation"] == "reused-qualified-library"


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_public_triton_precise_math_and_both_nan_modes_reach_merlin(tmp_path,profile):
    qualify(convert(tmp_path,profile,"f32",public=True),tmp_path,"f32")


def test_visible_public_math_demo_has_complete_ieee_references(tmp_path,capsys):
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo","--kernel","precise-math","--out",str(out),"--optimizer",OPT,"--python",NATIVE]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["native_numerical_check"] == "passed" and summary["result_count"] == 4
    assert summary["result_policies"][1:] == [{"kind":"ieee"}]*3
    verify_bundle(out)


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("dtype",["f16","bf16","f32","f64"])
def test_all_standard_ttir_float_types_keep_exact_rounding_and_exception_classes(tmp_path,profile,dtype):
    _,qualified = qualify(convert(tmp_path,profile,dtype),tmp_path,dtype)
    from t2m.invocation import invoke_kernel,verify_invocation
    from t2m.bundle import _record
    library = _record(qualified / "host/model.so")
    for n in (0,1,13,127):
        caller = tmp_path / f"caller-{n}";caller.mkdir()
        values = inputs(dtype,n)
        goldens = reference(values,dtype)
        ref = {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
            "results":{"archive":"reference.npz","comparisons":[{"kind":"values","rtol":0,"atol":0},*[{"kind":"ieee"}]*3]}}
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate([*values,np.asarray(n,np.int32)])})
        np.savez(caller / "reference.npz",**{f"out{i}":value for i,value in enumerate(goldens)})
        (caller / "reference.json").write_text(json.dumps(ref))
        out = tmp_path / f"invoked-{n}"
        record = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert record["result_count"] == 4 and record["native_compilation"] == "reused-qualified-library"
        assert _record(out / "source/host/model.so") == library


def test_bfloat_arithmetic_keeps_every_word_of_adjacent_uncomputed_transport(tmp_path):
    from t2m.dynamic_abi import GridExtent
    n = 65536
    names = ["x","y","low","high","no_nan","all_nan","divided","root","raw"]
    source = tmp_path / "source.ttir";source.write_text(typed_source("bf16",word_copy=True))
    abi = TensorABI("typed_math",tuple(TensorBuffer(name,("n",),"read" if index < 4 else "write","bf16") for index,name in enumerate(names)),
        grid=(GridExtent("n",8),),runtime_scalars=(RuntimeScalar("n","i32",9,bounds=(0,n)),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    values = [np.arange(n,dtype=np.uint16),np.full(n,0x4040,np.uint16),
        np.full(n,0xff80,np.uint16),np.full(n,0x7f80,np.uint16)]
    goldens = [*reference(values,"bf16"),values[0].copy()]
    ref = {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
        "results":{"archive":"reference.npz","comparisons":[{"kind":"values","rtol":0,"atol":0},
            *[{"kind":"ieee"}]*3,{"kind":"words"}]}}
    capture,qualified = tmp_path / "capture",tmp_path / "qualified"
    write_effect_bundle(result,capture,[*values,np.asarray(n,np.int32)],ref,references=goldens,work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(capture);verify_kernel_qualification(qualified)
    for base in (capture / "native",qualified / "evaluation"):
        with np.load(base / "outputs.npz",allow_pickle=False) as output:
            np.testing.assert_array_equal(output["out4"],values[0])

"""Strict external identity plus complete-result native/Merlin execution."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects import builtin
from xdsl.ir import Block

from t2m.diagnostics import ConversionError
from t2m.external_math import PROFILE, SYMBOLS, contract, expand_missing
from t2m.ir import text
from t2m.shared_tensor import opcode, parse_shared

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))


def declaration(symbol="__nv_tanh", *, tensor=True):
    _,dtype,arity = SYMBOLS[symbol]
    type_ = builtin.TensorType(dtype,(3,5)) if tensor else dtype
    block = Block(arg_types=[type_]*arity)
    op = builtin.UnregisteredOp.with_name("tt.extern_elementwise").create(
        operands=list(block.args),result_types=[type_],attributes={
            "pure":builtin.IntegerAttr(1,builtin.i1),"libname":builtin.StringAttr(""),
            "libpath":builtin.StringAttr(""),"symbol":builtin.StringAttr(symbol)})
    block.add_op(op)
    return op


@pytest.mark.parametrize("symbol",SYMBOLS)
@pytest.mark.parametrize("tensor",[False,True])
def test_all_default_prototypes_are_checked_without_granting_custom_semantics(symbol,tensor):
    op = declaration(symbol,tensor=tensor)
    record = contract(op,"source-external")
    assert record["profile"] == PROFILE and record["target"] == SYMBOLS[symbol][0]


@pytest.mark.parametrize("name,value",[
    ("pure",builtin.IntegerAttr(0,builtin.i1)),
    ("pure",builtin.IntegerAttr(1,builtin.i32)),
    ("libpath",builtin.StringAttr("custom.bc")),
    ("libname",builtin.StringAttr("custom")),
    ("symbol",builtin.StringAttr("unknown")),
    ("symbol",builtin.StringAttr("__nv_tanhf")),
    ("fastmath",builtin.IntegerAttr(1,builtin.i32)),
])
def test_external_contract_refuses_changed_library_purity_or_prototype(name,value):
    op = declaration()
    op.attributes[name] = value
    with pytest.raises(ConversionError):
        contract(op,"located-source")


def test_table_completion_validates_every_candidate_before_editing():
    module = parse_shared('''module {func.func @f(%x:tensor<8xf64>)->tensor<8xf64> {
      %a="tt.extern_elementwise"(%x) {libname="",libpath="",pure=true,symbol="__nv_tanh"}:(tensor<8xf64>)->tensor<8xf64>
      %b="tt.extern_elementwise"(%a) {libname="other",libpath="",pure=true,symbol="__nv_atanh"}:(tensor<8xf64>)->tensor<8xf64>
      return %b:tensor<8xf64>
    }}''')
    before = text(module)
    with pytest.raises(ConversionError,match="external implementation"):
        expand_missing(module,lambda loc:[])
    assert text(module) == before


@pytest.mark.parametrize("extra", [[], ["--math-expand-ops=ops=rsqrt"], ["--math-expand-ops"],
                                    ["--math-expand-ops=ops=sin"], ["--math-expand-ops=ops=rsqrt"]*2])
def test_source_word_math_pipeline_is_exact_and_legacy_custody_is_retained(extra):
    from t2m.word_evidence import COMPUTE_FLAGS, POINTER_FLAGS, check_word_stage, record
    flags = [*POINTER_FLAGS, COMPUTE_FLAGS[0], *extra, *COMPUTE_FLAGS[1:]]
    stage = {"schema":"t2m.triton-shared-stage.v1","stage":"source-word-tensor","stage_completed":True,
             "command":["optimizer",*flags,"source.ttir"],
             "artifacts":{"source.ttir":record("source"),"lowered.mlir":record("output")}}
    if extra in ([], ["--math-expand-ops=ops=rsqrt"]):
        check_word_stage(stage,"source","output")
    else:
        with pytest.raises(ValueError,match="declared pipeline"):
            check_word_stage(stage,"source","output")


@pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")
@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("dtype",["f32","f64"])
def test_all_math_results_and_runtime_regions_reach_reused_merlin_library(tmp_path,profile,dtype):
    from test_mixed_effects import qualify
    from t2m.external_math_demo import convert
    from t2m.examples.external_math_reference import NAMES,SIGNED_ZERO,inputs,policy,reference
    from t2m.invocation import invoke_kernel,verify_invocation

    result = convert(tmp_path / "source",dtype,optimizer=OPT,memory_profile=profile,
                     address_policy="source-words" if profile == "guarded-buffer" else "structured")
    declarations = [record["external_math"] for record in result.trace["source_operations"] if "external_math" in record]
    assert len(declarations) == 27 and len({record["symbol"] for record in declarations}) == 26
    assert all(opcode(op) != "tt.extern_elementwise" for op in result.module.walk())
    assert "scf.for" in result.mlir_text and "scf.if" in result.mlir_text
    assert any("prov.external_math_contracts" in op.attributes for op in result.module.walk())
    x,y = inputs(dtype)
    _,qualified = qualify(result,tmp_path,[x,y,np.asarray(len(x),np.int32),np.asarray(2,np.int32),np.asarray(True)],
                         policy(dtype),reference(x,y,dtype,2))
    for index,(n,steps,enabled) in enumerate([(0,3,True),(1,0,True),(127,0,False),(37,3,False),(37,1,True)]):
        x,y = inputs(dtype,n)
        if n == 1:
            x.fill(-0.)
            y.fill(-3.)
        expected = reference(x,y,dtype,steps,enabled)
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",in0=x,in1=y,in2=np.asarray(n,np.int32),in3=np.asarray(steps,np.int32),in4=np.asarray(enabled))
        np.savez(caller / "reference.npz",**{f"out{i}":value for i,value in enumerate(expected)})
        (caller / "reference.json").write_text(json.dumps(policy(dtype)))
        out = tmp_path / f"invoked-{index}"
        report = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                               merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        with np.load(out / "evaluation/outputs.npz",allow_pickle=False) as actual:
            assert len(actual.files) == 26
            for i,name in enumerate(NAMES):
                if name in SIGNED_ZERO:
                    zeros = expected[i] == 0
                    np.testing.assert_array_equal(np.signbit(actual[f"out{i}"][zeros]),np.signbit(expected[i][zeros]))

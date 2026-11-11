"""Fresh reduction tensors retain their actual allocation and lifetime semantics."""

import os

import numpy as np
import pytest

from t2m.guarded_memory import check_tensor_allocation
from t2m.shared_tensor import opcode, parse_shared


def allocation(*, type_="tensor<f32>", operands="", arguments="", operand_type="index", segments="0,0,0", extra="", attrs=""):
    module = parse_shared(f'''module {{func.func @test({arguments}) {{
      %t="bufferization.alloc_tensor"({operands}) <{{operandSegmentSizes=array<i32:{segments}>{extra}}}>
        {attrs}: ({operand_type if operands else ''})->{type_}
      func.return
    }} }}''')
    return next(op for op in module.walk() if opcode(op) == "bufferization.alloc_tensor")


@pytest.mark.parametrize("type_", ("tensor<f32>", "tensor<4xi64>", "tensor<0xbf16>"))
def test_static_dense_default_allocations_are_retained_without_rewrite(type_):
    op = allocation(type_=type_)
    original = str(op)
    check_tensor_allocation(op)
    assert str(op) == original


@pytest.mark.parametrize("options", (
    {"type_": "tensor<?xf32>", "operands": "%n", "arguments": "%n:index", "segments": "1,0,0"},
    {"operands": "%copy", "arguments": "%copy:tensor<f32>", "operand_type": "tensor<f32>",
     "segments": "0,1,0", "attrs": '{"prov.fresh"=true}'},
    {"extra": ",memory_space=1:i64"}, {"attrs": "{memory_space=1:i64}"},
    {"operands": "%n", "arguments": "%n:index", "segments": "0,0,1"}, {"segments": "0,1,0"},
))
def test_copy_hints_dynamic_sizes_and_physical_spaces_gain_no_permission(options):
    with pytest.raises(ValueError, match="static dense copy-free"):
        check_tensor_allocation(allocation(**options))


TOOLS = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
                                       "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.mark.skipif(not TOOLS, reason="existing explicit native/Merlin toolchains required")
@pytest.mark.parametrize("n", (13, 37))
@pytest.mark.parametrize("address", ("structured", "source-words"))
def test_guarded_softmax_reductions_reach_actual_merlin_libraries(tmp_path, n, address):
    from t2m.examples.tensor_kernels import row_softmax
    from t2m.frontend import capture_ttir
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(row_softmax, {"x": "*fp32", "out": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
                                   {"N": n, "BLOCK": 1 << (n - 1).bit_length()}))
    abi = TensorABI("row_softmax", (TensorBuffer("x", (n,)), TensorBuffer("out", (n,), "write")))
    result = convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy=address)
    assert "bufferization.alloc_tensor" in result.mlir_text
    assert any(opcode(op) == "linalg.reduce" for op in result.module.walk())
    x = (np.arange(n, dtype=np.float32) % 11 - 5) / 8
    values = np.exp(x - np.max(x))
    reference = values / np.sum(values)
    write_shared_bundle(result, tmp_path / "package", [x], [reference], work=tmp_path / "native",
                         python=os.environ["T2M_NATIVE_PYTHON"], rtol=1e-5, atol=2e-6)
    qualify_kernel(tmp_path / "package", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"], rtol=1e-5, atol=2e-6)
    verify_kernel_qualification(tmp_path / "merlin")

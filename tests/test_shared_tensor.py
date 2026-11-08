"""Source capture -> actual shared passes -> functional tensor IR -> native JIT."""

import os
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.shared_tensor import (TensorABI, TensorBuffer, _functionalize, _preflight,
                               convert_shared, opcode, parse_shared)

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


def source_case(name, tmp_path):
    from t2m.examples import tensor_kernels as kernels
    if name == "matrix_add":
        constants, shapes = {"M": 4, "N": 8}, [(4, 8)] * 3
    elif name == "row_softmax":
        constants, shapes = {"N": 7, "BLOCK": 8}, [(7,)] * 2
    else:
        constants, shapes = {"M": 16, "N": 16, "K": 16}, [(16, 16)] * 3
    names = ["x", "y", "out"] if len(shapes) == 3 else ["x", "out"]
    buffers = tuple(TensorBuffer(n, s, "readwrite" if n == "out" else "read")
                    for n, s in zip(names, shapes, strict=True))
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(getattr(kernels, name),
                      {**{n: "*fp32" for n in names}, **{n: "constexpr" for n in constants}}, constants))
    return source, TensorABI(name, buffers, intent="caller-provided example description")


def test_dense_hex_constants_are_ieee_bits_and_preserve_signed_zero():
    for literal, expected in [("0xFF800000", -np.inf), ("0x80000000", -0.0), ("0x3F800000", 1.0)]:
        module = parse_shared(f'module {{ %0 = arith.constant dense<{literal}> : tensor<4xf32> }}')
        value = next(iter(module.ops)).value.get_values()[0]
        assert value == expected and np.signbit(value) == np.signbit(expected)
    with pytest.raises(Exception, match="NaN constant payload"):
        parse_shared('module { %0 = arith.constant dense<0x7FC00001> : tensor<4xf32> }')


def test_tensor_abi_does_not_silently_serialize_parallel_writes():
    with pytest.raises(ValueError, match="single logical program"):
        TensorABI("f", (TensorBuffer("out", (8,), "readwrite"),), grid=(2,))


@pytest.mark.skipif(not OPT, reason="existing shared optimizer not configured")
@pytest.mark.parametrize("name", ["matrix_add", "row_softmax", "block_matmul"])
def test_real_source_to_functional_tensor_and_native_results(tmp_path, name):
    source, abi = source_case(name, tmp_path)
    result = convert_shared(source, abi, tmp_path / "conversion", optimizer=OPT)
    names = {opcode(op) for op in result.module.walk()}
    assert "tensor.extract_slice" in names and "tensor.insert_slice" in names
    assert not any(n.split(".")[0] in ("tt", "tts", "memref", "gpu", "llvm", "bufferization") for n in names)
    assert "linalg.matmul" in names if name == "block_matmul" else "linalg.generic" in names
    original_names = {r["opcode"] for r in result.trace["source_operations"]}
    if name == "row_softmax":
        assert {"tt.call", "tt.reduce"} <= original_names  # Private combiner helpers are captured too.
        assert "linalg.reduce" in names and "tensor.pad" in names
    assert any(e["source_ids"] for e in result.trace["emitted_operations"])
    assert result.trace["numeric_equivalence"] == "requires-qualification-against-source"
    if not NATIVE:
        return  # IR conversion is separately measured from optional host execution.
    rng = np.random.default_rng(17)
    arrays = [rng.normal(size=b.shape).astype("float32") for b in abi.buffers]
    before = [a.copy() for a in arrays]
    (actual,), report = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE)
    if name == "matrix_add":
        expected = arrays[0] + arrays[1]
    elif name == "block_matmul":
        expected = arrays[0] @ arrays[1]
    else:
        e = np.exp(arrays[0] - arrays[0].max())
        expected = e / e.sum()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
    for after, saved in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(after, saved)
    assert report["llvm_lowering_verified"]


@pytest.mark.skipif(not OPT, reason="existing shared optimizer not configured")
def test_bounds_effects_and_dot_numeric_policy_refuse(tmp_path):
    source, abi = source_case("matrix_add", tmp_path)
    convert_shared(source, abi, tmp_path / "good", optimizer=OPT)
    lowered = (tmp_path / "good/tensor-ir/lowered.mlir").read_text()
    smaller = TensorABI(abi.entry, (TensorBuffer("x", (2, 8)), *abi.buffers[1:]))
    with pytest.raises(ConversionError, match="exceeds.*bounds"):
        _functionalize(parse_shared(lowered), smaller)
    readonly = TensorABI(abi.entry, (TensorBuffer("x", (4, 8), "readwrite"),
                                    TensorBuffer("y", (4, 8)), TensorBuffer("out", (4, 8))))
    with pytest.raises(ConversionError, match="read-only"):
        _functionalize(parse_shared(lowered), readonly)
    dot_dir = tmp_path / "dot"
    dot_dir.mkdir()
    source, abi = source_case("block_matmul", dot_dir)
    convert_shared(source, abi, dot_dir / "good", optimizer=OPT)
    source_ir = (dot_dir / "good/source/lowered.mlir").read_text()
    module = parse_shared(source_ir)
    dot = next(op for op in module.walk() if opcode(op) == "tt.dot")
    dot.operands[2].owner.properties["value"] = DenseIntOrFPElementsAttr.from_list(TensorType(f32, [16, 16]), [1.0] * 256)
    with pytest.raises(ConversionError, match="nonzero dot accumulation"):
        _preflight(module, abi)
    with pytest.raises(ConversionError, match="precision must explicitly be IEEE"):
        _preflight(parse_shared(source_ir.replace("inputPrecision = 2", "inputPrecision = 0")), abi)


def test_unregistered_standard_spelling_is_not_tensor_coverage():
    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name = "f", function_type = (!tt.ptr<f32>) -> ()}> ({
      ^bb0(%p: !tt.ptr<f32>):
        "linalg.unqualified"() : () -> ()
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()''')
    with pytest.raises(ConversionError, match="requiring a tensor/effect adapter"):
        _functionalize(module, TensorABI("f", (TensorBuffer("out", (8,), "readwrite"),)))

"""Runtime scalar values and interleaved source argument positions reach Merlin."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
NATIVE_ONLY = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environment required")
MERLIN = pytest.mark.skipif(not all(os.environ.get(n) for n in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")), reason="existing Merlin host toolchain required")


def abi(dtype="f32"):
    return TensorABI("runtime_affine", (TensorBuffer("x", (13,), dtype=dtype), TensorBuffer("out", (13,), "write", dtype)),
        grid=(2,), runtime_scalars=(RuntimeScalar("alpha", dtype, 1), RuntimeScalar("bias", dtype, 3)))


def test_runtime_scalar_schema_is_distinct_and_preserves_source_positions():
    spec = abi()
    assert spec.as_dict()["schema"] == "t2m.shared-tensor-abi.v3"
    assert [kind for kind, _ in spec.source_bindings] == ["buffer", "runtime", "buffer", "runtime"]
    assert [v.name for v in spec.inputs] == ["x", "alpha", "bias"]
    assert TensorABI.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    old = TensorABI("copy", (TensorBuffer("x", (2,)), TensorBuffer("out", (2,), "write")))
    assert old.as_dict()["schema"] == "t2m.shared-tensor-abi.v2"
    assert "runtime_scalars" not in old.as_dict()
    mixed = TensorABI("mixed", old.buffers, scalar_constants=(13,), runtime_scalars=(RuntimeScalar("scale", "f32", 1),))
    assert [kind for kind, _ in mixed.source_bindings] == ["buffer", "runtime", "buffer", "constant"]


@pytest.mark.parametrize("change", ["duplicate_position", "outside_signature", "duplicate_name", "dtype", "old_schema"])
def test_invalid_runtime_scalar_declarations_refuse(change):
    value = abi().as_dict()
    value = json.loads(json.dumps(value))
    if change == "duplicate_position":
        value["runtime_scalars"][1]["source_arg"] = 1
    elif change == "outside_signature":
        value["runtime_scalars"][0]["source_arg"] = 5
    elif change == "duplicate_name":
        value["runtime_scalars"][0]["name"] = "x"
    elif change == "dtype":
        value["runtime_scalars"][0]["dtype"] = "f16"
    else:
        value["schema"] = "t2m.shared-tensor-abi.v2"
    with pytest.raises(ValueError):
        TensorABI.from_dict(value)


def convert(root, dtype="f32"):
    from t2m.examples.tensor_kernels import runtime_affine
    path = root / "source.ttir"
    native = {"f32": "fp32", "f64": "fp64", "i32": "i32"}[dtype]
    path.write_text(capture_ttir(runtime_affine,
        {"x": "*" + native, "alpha": native, "out": "*" + native, "bias": native, "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    return convert_shared(path, abi(dtype), root / "translation", optimizer=OPT)


@NATIVE_ONLY
@pytest.mark.parametrize("dtype,numpy", [("f32", np.float32), ("f64", np.float64), ("i32", np.int32)])
def test_interleaved_runtime_values_are_typed_inputs_and_not_specializations(tmp_path, dtype, numpy):
    result = convert(tmp_path, dtype)
    entry = next(iter(result.module.ops))
    assert [str(t) for t in entry.function_type.inputs] == [f"tensor<13x{dtype}>", dtype, dtype]
    x = np.arange(-6, 7, dtype=numpy)
    if dtype == "i32":
        x[[0, 1]] = [np.iinfo(numpy).min, np.iinfo(numpy).max]
    alpha, bias = np.asarray(-2, dtype=numpy), np.asarray(3, dtype=numpy)
    expected = x * alpha + bias
    write_shared_bundle(result, tmp_path / "bundle", [x, alpha, bias], [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(tmp_path / "bundle")
    assert result.spec.scalar_constants == ()
    with np.load(tmp_path / "bundle/inputs.npz", allow_pickle=False) as data:
        assert data["in1"].shape == data["in2"].shape == ()
    # Same source/serialized IR accepts independent values without conversion.
    (actual,), _ = execute_native(result.mlir_text, [x, np.asarray(0, dtype=numpy), np.asarray(-9, dtype=numpy)],
                                  tmp_path / "changed-values", python=NATIVE)
    np.testing.assert_array_equal(actual, np.full(13, -9, dtype=numpy))


@NATIVE_ONLY
@pytest.mark.parametrize("flag", [False, True])
def test_boolean_by_value_argument_controls_source_regions(tmp_path, flag):
    from t2m.examples.tensor_kernels import runtime_branch
    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(runtime_branch,
        {"x": "*fp32", "flag": "i1", "out": "*fp32", "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    spec = TensorABI("runtime_branch", (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")),
                     grid=(2,), runtime_scalars=(RuntimeScalar("flag", "i1", 1),))
    result = convert_shared(path, spec, tmp_path / "translation", optimizer=OPT)
    x = np.arange(13, dtype=np.float32)
    bundle = tmp_path / "bundle"
    write_shared_bundle(result, bundle, [x, np.asarray(flag)], [x + (2 if flag else -3)], work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    assert any(op.name == "scf.if" for op in result.module.walk())
    if all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        from t2m.kernel_consumer import qualify_kernel
        qualify_kernel(bundle, tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                       compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])


@NATIVE_ONLY
def test_source_type_and_runtime_value_types_are_checked(tmp_path):
    result = convert(tmp_path)
    source = tmp_path / "source.ttir"
    wrong = TensorABI("runtime_affine", result.spec.buffers, grid=(2,),
        runtime_scalars=(RuntimeScalar("alpha", "i32", 1), RuntimeScalar("bias", "f32", 3)))
    with pytest.raises(ConversionError, match="runtime scalar type differs"):
        convert_shared(source, wrong, tmp_path / "wrong-declaration", optimizer=OPT)
    x = np.arange(13, dtype=np.float32)
    for name, scalar in (("dtype", np.asarray(2, dtype=np.float64)), ("rank", np.asarray([2], dtype=np.float32))):
        with pytest.raises(ValueError, match="input shape/dtype"):
            execute_native(result.mlir_text, [x, scalar, np.asarray(0, dtype=np.float32)], tmp_path / name, python=NATIVE)


@NATIVE_ONLY
@MERLIN
@pytest.mark.parametrize("dtype,numpy", [("f32", np.float32), ("f64", np.float64), ("i32", np.int32)])
def test_runtime_scalars_stage_and_execute_through_real_merlin_by_value(tmp_path, dtype, numpy):
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    result = convert(tmp_path, dtype)
    x, alpha, bias = np.arange(13, dtype=numpy), np.asarray(-2, dtype=numpy), np.asarray(3, dtype=numpy)
    bundle, out = tmp_path / "bundle", tmp_path / "merlin"
    write_shared_bundle(result, bundle, [x, alpha, bias], [x * alpha + bias], work=tmp_path / "native", python=NATIVE)
    original = _record(bundle / "capture_receipt.json")
    qualify_kernel(bundle, out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                   compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(out)
    assert _record(bundle / "capture_receipt.json") == original
    signature = json.loads((out / "compiler/signature.json").read_text())
    assert signature["inputs"][1:] == [{"shape": [], "dtype": dtype, "kind": "scalar"}] * 2
    assert signature["scalar_abi"] == "native-by-value; invocation-data"

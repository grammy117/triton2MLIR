"""Guarded runtime memory domains retain tiles, fills and preserved outputs."""

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
NATIVE_ONLY = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")
MERLIN = pytest.mark.skipif(not all(os.environ.get(n) for n in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")), reason="existing Merlin native toolchain required")


def abi(dtype="i32", *, bounds=(0, 13), effect="readwrite"):
    return TensorABI("runtime_prefix", (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), effect)),
        grid=(4,), runtime_scalars=(RuntimeScalar("n", dtype, 1, bounds=bounds),))


def test_bounded_runtime_schema_is_explicit_and_preserves_older_specs():
    spec = abi()
    assert spec.as_dict()["schema"] == "t2m.shared-tensor-abi.v4"
    assert TensorABI.from_dict(json.loads(json.dumps(spec.as_dict()))) == spec
    old = abi(bounds=None)
    assert old.as_dict()["schema"] == "t2m.shared-tensor-abi.v3"
    assert "bounds" not in old.as_dict()["runtime_scalars"][0]
    wrong = spec.as_dict()
    wrong["schema"] = "t2m.shared-tensor-abi.v3"
    with pytest.raises(ValueError, match="explicit ABI v4"):
        TensorABI.from_dict(wrong)


@pytest.mark.parametrize("dtype,bounds", [("f32", (0, 1)), ("i32", (2, 1)), ("i32", (0, 2**31)), ("i64", (False, 1))])
def test_runtime_bounds_must_be_typed_representable_integer_intervals(dtype, bounds):
    with pytest.raises(ValueError, match="inclusive range"):
        RuntimeScalar("n", dtype, 1, bounds=bounds)


@pytest.fixture(params=("i32", "i64"))
def translated(request, tmp_path):
    from t2m.examples.tensor_kernels import runtime_prefix
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_prefix,
        {"x": "*fp32", "n": request.param, "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    return convert_shared(source, abi(request.param), tmp_path / "translation", optimizer=OPT)


def arrays(n, dtype):
    x, old = np.arange(13, dtype=np.float32), np.arange(13, dtype=np.float32) * 7 + 99
    expected = old.copy()
    expected[:max(0, n)] = x[:max(0, n)] + 2
    return [x, old, np.asarray(n, dtype=dtype)], expected


@NATIVE_ONLY
def test_runtime_prefix_domains_include_empty_tails_and_inactive_programs(tmp_path, translated):
    dtype = np.int32 if translated.spec.runtime_scalars[0].dtype == "i32" else np.int64
    entry = next(iter(translated.module.ops))
    operations = list(entry.body.block.ops)
    assert next(i for i, op in enumerate(operations) if op.name == "cf.assert") < next(i for i, op in enumerate(operations) if op.name == "scf.forall")
    assert "parameters" in entry.attributes["prov.output_ownership"].data
    for n in (0, 1, 7, 8, 13):
        inputs, expected = arrays(n, dtype)
        (actual,), _ = execute_native(translated.mlir_text, inputs, tmp_path / f"native-{n}", python=NATIVE)
        np.testing.assert_array_equal(actual, expected)
    assert {"tensor.pad", "scf.forall", "linalg.generic", "tensor.parallel_insert_slice"} <= {op.name for op in translated.module.walk()}


@NATIVE_ONLY
def test_invalid_runtime_domain_trips_executable_entry_guard(tmp_path, translated):
    dtype = np.int32 if translated.spec.runtime_scalars[0].dtype == "i32" else np.int64
    for n in (-1, 14):
        inputs, _ = arrays(0, dtype)
        inputs[-1] = np.asarray(n, dtype=dtype)
        with pytest.raises(ValueError, match="native host execution failed"):
            execute_native(translated.mlir_text, inputs, tmp_path / f"invalid-{n}", python=NATIVE)


@NATIVE_ONLY
@pytest.mark.parametrize("change", ["unbounded", "outside_allocation", "fresh_partial_output"])
def test_unproved_runtime_memory_contracts_refuse(tmp_path, change):
    from t2m.examples.tensor_kernels import runtime_prefix
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_prefix, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    spec = abi(bounds=None if change == "unbounded" else (0, 14) if change == "outside_allocation" else (0, 13),
               effect="write" if change == "fresh_partial_output" else "readwrite")
    with pytest.raises(ConversionError):
        convert_shared(source, spec, tmp_path / "translation", optimizer=OPT)


@NATIVE_ONLY
@pytest.mark.parametrize("bounds,values", [((-13, 13), (-13, -1, 0, 13)), ((-13, -1), (-13, -1))])
def test_negative_limits_and_proved_empty_launches_preserve_all_memory(tmp_path, bounds, values):
    from t2m.examples.tensor_kernels import runtime_prefix
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_prefix, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    result = convert_shared(source, abi(bounds=bounds), tmp_path / "translation", optimizer=OPT)
    for n in values:
        inputs, expected = arrays(n, np.int32)
        (actual,), _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{n}", python=NATIVE)
        np.testing.assert_array_equal(actual, expected)
    if bounds[1] < 0:
        ownership = result.trace["memory"]["output_ownership"]["out"]
        assert ownership["write_independence"] == "empty-access-domain"
    if all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        from t2m.kernel_consumer import qualify_kernel
        inputs, expected = arrays(values[0], np.int32)
        capture = tmp_path / "capture"
        write_shared_bundle(result, capture, inputs, [expected], work=tmp_path / "package-native", python=NATIVE)
        qualify_kernel(capture, tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                       compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])


@NATIVE_ONLY
@MERLIN
def test_one_merlin_library_handles_variable_domains_and_preserves_inputs(tmp_path, translated):
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    dtype = np.int32 if translated.spec.runtime_scalars[0].dtype == "i32" else np.int64
    inputs, expected = arrays(7, dtype)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(translated, capture, inputs, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python = os.environ["T2M_MERLIN_ROOT"], os.environ["T2M_MERLIN_PYTHON"]
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])
    original = verify_kernel_qualification(compiled)
    library = _record(compiled / "host/model.so")
    signature = json.loads((compiled / "compiler/signature.json").read_text())
    assert signature["inputs"][-1]["bounds"] == [0, 13]
    for n in (0, 1, 8, 13):
        values, expected = arrays(n, dtype)
        arguments, reference = tmp_path / f"args-{n}.npz", tmp_path / f"ref-{n}.npz"
        np.savez(arguments, **{f"in{i}": a for i, a in enumerate(values)})
        np.savez(reference, out0=expected)
        out = tmp_path / f"replay-{n}"
        report = invoke_kernel(compiled, arguments, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["shared_library"] == library
        assert report["evaluation"]["functional_inputs_preserved"] is True
    for n in (-1, 14):
        values, expected = arrays(0, dtype)
        values[-1] = np.asarray(n, dtype=dtype)
        arguments, reference = tmp_path / f"args-{n}.npz", tmp_path / f"ref-{n}.npz"
        np.savez(arguments, **{f"in{i}": a for i, a in enumerate(values)})
        np.savez(reference, out0=expected)
        out = tmp_path / f"failed-{n}"
        with pytest.raises(ValueError, match="guarded ABI bounds"):
            invoke_kernel(compiled, arguments, reference, out, merlin_root=root, python=python)
        assert not (out / "invocation.json").exists()
    assert verify_kernel_qualification(compiled) == original

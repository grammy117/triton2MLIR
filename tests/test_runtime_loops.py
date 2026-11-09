"""Runtime value recurrences retain loops through native and Merlin execution."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN, PYTHON, CLANG = (os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")


def convert(root, profile="tensor", *, memory=False, bounds=(0, 5), capacity=65, dtype="i32"):
    from t2m.examples.tensor_kernels import runtime_memory_loop, runtime_recurrence
    kernel = runtime_memory_loop if memory else runtime_recurrence
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*fp32", "iterations": dtype, "out": "*fp32",
        "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    spec = TensorABI(kernel.__name__, (TensorBuffer("x", (capacity if memory else 13,)), TensorBuffer("out", (13,), "write")),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", dtype, 1, bounds=bounds),))
    return convert_shared(source, spec, root / "translation", optimizer=OPT, memory_profile=profile)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_runtime_register_loop_keeps_ssa_bounds_and_reuses_merlin_library(tmp_path, profile):
    result = convert(tmp_path, profile)
    loops = [op for op in result.module.walk() if op.name == "scf.for"]
    assert len(loops) == 1
    assert loops[0].ub is next(iter(result.module.ops)).body.block.args[-1]
    assert result.trace["memory"]["source_loops"][0]["memory_effects"] == "no-buffer-accesses"
    x = np.arange(-6, 7, dtype=np.float32)

    def reference(count):
        value = x.copy()
        for k in range(count):
            value = value * np.float32(0.5) + np.float32(k)
        return value

    capture = tmp_path / "capture"
    write_shared_bundle(result, capture, [x, np.asarray(1, dtype=np.int32)], [reference(1)], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    if all((MERLIN, PYTHON, CLANG)):
        from t2m.invocation import invoke_kernel, verify_invocation
        from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
        compiled = tmp_path / "merlin"
        qualify_kernel(capture, compiled, merlin_root=MERLIN, python=PYTHON, compiler_python=NATIVE, clang=CLANG)
        receipt = verify_kernel_qualification(compiled)
        compilation = json.loads((compiled / "host/compilation.json").read_text())
        assert compilation["features"] == ["reuse_loop_tensor_destinations"]
        library = _record(compiled / "host/model.so")
        for count in (0, 5):
            inputs, expected = (tmp_path / (name + str(count) + ".npz") for name in ("inputs", "reference"))
            np.savez(inputs, in0=x, in1=np.asarray(count, dtype=np.int32))
            np.savez(expected, out0=reference(count))
            out = tmp_path / ("invoke" + str(count))
            report = invoke_kernel(compiled, inputs, expected, out, merlin_root=MERLIN, python=PYTHON)
            verify_invocation(out)
            assert report["native_compilation"] == "reused-qualified-library"
            assert report["evaluation"]["functional_inputs_preserved"]
        assert verify_kernel_qualification(compiled) == receipt
        assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_runtime_buffer_accesses_still_require_iteration_domain_proof(tmp_path, profile):
    with pytest.raises(ConversionError, match="range proof"):
        convert(tmp_path, profile, memory=True, bounds=None)
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("dtype,numpy", [("i32", np.int32), ("i64", np.int64)])
def test_runtime_read_loops_keep_exact_masks_and_zero_trips_through_merlin(tmp_path, profile, dtype, numpy):
    result = convert(tmp_path, profile, memory=True, bounds=(-2, 5), dtype=dtype)
    loop = result.trace["memory"]["source_loops"][0]
    assert loop["bounds"] == "original-runtime-SSA"
    assert loop["minimum_iterations"] == 0 and loop["maximum_iterations"] == 5
    assert "iterations" not in loop and "upper" not in loop
    x = np.arange(-33, 32, dtype=np.float32)

    def reference(count):
        expected = np.zeros(13, dtype=np.float32)
        for k in range(count):
            expected += x[k * 13:(k + 1) * 13]
        return expected

    capture = tmp_path / "capture"
    write_shared_bundle(result, capture, [x, np.asarray(3, dtype=numpy)], [reference(3)],
                        work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    if all((MERLIN, PYTHON, CLANG)):
        from t2m.invocation import invoke_kernel, verify_invocation
        from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
        compiled = tmp_path / "merlin"
        qualify_kernel(capture, compiled, merlin_root=MERLIN, python=PYTHON, compiler_python=NATIVE, clang=CLANG)
        receipt = verify_kernel_qualification(compiled)
        library = _record(compiled / "host/model.so")
        for count in (-2, 0, 1, 5):
            inputs, expected = (tmp_path / (name + str(count) + ".npz") for name in ("inputs", "reference"))
            np.savez(inputs, in0=x, in1=np.asarray(count, dtype=numpy))
            np.savez(expected, out0=reference(count))
            out = tmp_path / ("invoke" + str(count))
            report = invoke_kernel(compiled, inputs, expected, out, merlin_root=MERLIN, python=PYTHON)
            verify_invocation(out)
            assert report["native_compilation"] == "reused-qualified-library"
            assert report["evaluation"]["functional_inputs_preserved"]
        assert verify_kernel_qualification(compiled) == receipt
        assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_runtime_read_loop_range_is_checked_for_every_admitted_count(tmp_path, profile):
    with pytest.raises(ConversionError, match="bounds"):
        convert(tmp_path, profile, memory=True, capacity=64)
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("effect", ["write", "readwrite"])
def test_dynamic_loop_stores_require_a_conditional_state_and_initialization_proof(tmp_path, profile, effect):
    from t2m.examples.tensor_kernels import runtime_memory_store
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_memory_store, {"x": "*fp32", "iterations": "i32", "out": "*fp32",
        "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    abi = TensorABI(runtime_memory_store.__name__, (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), effect)),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", "i32", 1, bounds=(0, 5)),))
    with pytest.raises(ConversionError, match="conditional initialization/effect proof"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.parametrize("kernel", ["runtime-recurrence", "runtime-memory-loop"])
def test_runtime_loops_have_reproducible_cli_demos(tmp_path, kernel):
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", kernel, "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE, "--memory-profile", "buffer"]) == 0
    verify_bundle(out)

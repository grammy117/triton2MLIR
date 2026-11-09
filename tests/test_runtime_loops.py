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


def convert(root, profile="tensor", *, memory=False):
    from t2m.examples.tensor_kernels import runtime_memory_loop, runtime_recurrence
    kernel = runtime_memory_loop if memory else runtime_recurrence
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*fp32", "iterations": "i32", "out": "*fp32",
        "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    spec = TensorABI(kernel.__name__, (TensorBuffer("x", (65 if memory else 13,)), TensorBuffer("out", (13,), "write")),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", "i32", 1, bounds=(0, 5)),))
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
    with pytest.raises(ConversionError, match="source memory loop bounds need static specialization"):
        convert(tmp_path, profile, memory=True)
    assert not (tmp_path / "translation/memory-ir").exists()


def test_runtime_recurrence_has_a_reproducible_cli_demo(tmp_path):
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", "runtime-recurrence", "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE, "--memory-profile", "buffer"]) == 0
    verify_bundle(out)

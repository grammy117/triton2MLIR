"""Runtime tensor dimensions are allocation sizes, never static capacities."""

import json
import os

import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")


def spec(*, maximum=13, divisor=8):
    return TensorABI("dynamic_count", (TensorBuffer("x", ("n",)), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", divisor),), runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=(0, maximum)),))


def test_dynamic_shape_schema_carries_symbols_and_grid_without_invocation_values():
    abi = spec()
    assert abi.as_dict()["schema"] == "t2m.shared-tensor-abi.v5"
    assert abi.as_dict()["grid"] == ({"dimension": "n", "divisor": 8},)
    assert abi.grid_bounds == (2,)
    assert str(abi.buffers[0].type) == "tensor<?xf32>"
    assert TensorABI.from_dict(json.loads(json.dumps(abi.as_dict()))) == abi
    old = abi.as_dict()
    old["schema"] = "t2m.shared-tensor-abi.v4"
    with pytest.raises(ValueError, match="explicit ABI v5"):
        TensorABI.from_dict(old)


@pytest.mark.parametrize("change", ["unbounded", "negative", "unknown"])
def test_unproved_dynamic_dimension_contracts_refuse(change):
    declaration = spec().as_dict()
    if change == "unbounded":
        declaration["runtime_scalars"][0].pop("bounds")
    elif change == "negative":
        declaration["runtime_scalars"][0]["bounds"] = [-1, 13]
    elif change == "unknown":
        declaration["buffers"][0]["shape"] = ["other"]
    with pytest.raises(ValueError):
        TensorABI.from_dict(declaration)


@pytest.mark.skipif(not OPT, reason="existing shared optimizer required")
@pytest.mark.parametrize("maximum,divisor", [(13, 8), (7, 8), (13, 4)])
def test_tensor_conversion_keeps_dynamic_types_allocations_and_logical_launch(tmp_path, maximum, divisor):
    from t2m.examples.tensor_kernels import dynamic_count
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_count, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    result = convert_shared(source, spec(maximum=maximum, divisor=divisor), tmp_path / "translation", optimizer=OPT)
    entry = next(iter(result.module.ops))
    assert str(entry.function_type.outputs) == "[tensor<?xf32>]"
    names = {op.name for op in result.module.walk()}
    assert {"tensor.dim", "arith.ceildivui", "cf.assert", "scf.forall"} <= names
    loop = next(op for op in result.module.walk() if op.name == "scf.forall")
    assert len(loop.operands) == 2  # Runtime upper bound plus tensor output state.


@pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_native_dynamic_shapes_and_counts_are_not_specialized(tmp_path, profile):
    import numpy as np
    from t2m.examples.tensor_kernels import dynamic_count
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_count, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    abi = spec()
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    for n in (0, 1, 7, 8, 9, 13):
        x = np.arange(n, dtype=np.float32)
        (actual,), _ = execute_native(result.mlir_text, [x, np.asarray(n, dtype=np.int32)],
            tmp_path / f"native-{n}", python=NATIVE, shape_abi=abi.runtime_shapes())
        np.testing.assert_array_equal(actual, x + np.float32((n + 7) // 8))


@pytest.mark.skipif(not OPT, reason="existing shared optimizer required")
def test_a_runtime_launch_that_can_leave_fresh_holes_refuses(tmp_path):
    from t2m.diagnostics import ConversionError
    from t2m.examples.tensor_kernels import dynamic_count
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_count, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        convert_shared(source, spec(divisor=16), tmp_path / "translation", optimizer=OPT, memory_profile="buffer")
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.skipif(not all(os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")), reason="existing native/Merlin toolchains required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_one_merlin_library_handles_distinct_input_result_and_launch_shapes(tmp_path, profile):
    import numpy as np
    from t2m.bundle import _record, verify_bundle
    from t2m.examples.tensor_kernels import dynamic_count
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_count, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    result = convert_shared(source, spec(), tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    x = np.arange(7, dtype=np.float32)
    write_shared_bundle(result, capture, [x, np.asarray(7, dtype=np.int32)], [x + 1], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    signature = json.loads((compiled / "compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v2"
    assert signature["inputs"][0]["shape"] == signature["outputs"][0]["shape"] == ["n"]
    for n in (0, 1, 8, 9, 13):
        x = np.arange(n, dtype=np.float32)
        inputs, reference = (tmp_path / (name + str(n) + ".npz") for name in ("inputs", "reference"))
        np.savez(inputs, in0=x, in1=np.asarray(n, dtype=np.int32))
        np.savez(reference, out0=x + np.float32((n + 7) // 8))
        out = tmp_path / ("invoke" + str(n))
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(out / "evaluation/outputs.npz", allow_pickle=False) as values:
            assert values["out0"].shape == (n,)
    inputs, reference = tmp_path / "invalid-inputs.npz", tmp_path / "invalid-reference.npz"
    np.savez(inputs, in0=np.arange(6, dtype=np.float32), in1=np.asarray(7, dtype=np.int32))
    np.savez(reference, out0=np.arange(7, dtype=np.float32) + 1)
    with pytest.raises(ValueError, match="shape differs from its scalar dimension binding"):
        invoke_kernel(compiled, inputs, reference, tmp_path / "mismatched-shape", merlin_root=root, python=python)
    assert not (tmp_path / "mismatched-shape/invocation.json").exists()
    # Bypass the evaluator's pre-call validation to test the actual compiled
    # guard. The child disables core dumps and C stdout buffering before abort.
    import signal
    import subprocess
    script = tmp_path / "raw-guard.py"
    script.write_text('''import ctypes, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.path.insert(0, sys.argv[1] + '/src')
import numpy as np
from merlin.llvmlower.abi import HostModel, ScalarArg
libc = ctypes.CDLL(None)
libc.setbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libc.setbuf.restype = None
libc.setbuf(ctypes.c_void_p.in_dll(libc, 'stdout'), None)
x, out = np.arange(6, dtype=np.float32), np.zeros(7, dtype=np.float32)
HostModel.load(sys.argv[2])([(x.ctypes.data, x.shape), ScalarArg(7, 'i32'), (out.ctypes.data, out.shape)])
''')
    process = subprocess.run([python, str(script), root, str((compiled / "host/model.so").resolve())],
        capture_output=True, text=True, timeout=10,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == -signal.SIGABRT
    assert "Triton runtime dimension differs from input shape: x" in process.stdout
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
def test_dynamic_count_cli_demo_keeps_a_symbolic_package(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", "dynamic-count", "--memory-profile", "buffer", "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE]) == 0
    verify_bundle(out)
    assert json.loads((out / "spec.json").read_text())["buffers"][0]["shape"] == ["n"]
    assert "tensor<?xf32>" in (out / "model.mlir").read_text()


@pytest.mark.skipif(not all(os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")), reason="existing native/Merlin toolchains required")
def test_scalar_only_generator_allocates_declared_shape_and_refuses_over_budget(tmp_path):
    import numpy as np
    from t2m.bundle import verify_bundle
    from t2m.examples.tensor_kernels import dynamic_generate
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_generate, {"n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("dynamic_generate", (TensorBuffer("out", ("n",), "write"),), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 0, bounds=(0, 2**28)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="buffer")
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, [np.asarray(0, dtype=np.int32)], [np.zeros(0, dtype=np.float32)],
                        work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    original = verify_kernel_qualification(compiled)
    inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
    np.savez(inputs, in0=np.asarray(3, dtype=np.int32))
    np.savez(reference, out0=np.full(3, 7, dtype=np.float32))
    invoke_kernel(compiled, inputs, reference, tmp_path / "three", merlin_root=root, python=python)
    np.savez(inputs, in0=np.asarray(40_000_000, dtype=np.int32))
    # Only a scalar and a tiny reference archive are created. Allocation must
    # refuse before constructing the requested 160MB result or calling MLIR.
    with pytest.raises(ValueError, match="array budget"):
        invoke_kernel(compiled, inputs, reference, tmp_path / "over-budget", merlin_root=root, python=python)
    assert not (tmp_path / "over-budget/invocation.json").exists()
    assert verify_kernel_qualification(compiled) == original

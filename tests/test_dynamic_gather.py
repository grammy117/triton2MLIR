"""Indexed dataflow uses actual allocation extents, including empty storage."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def convert(tmp_path, profile, dtype="i32", maximum=(7, 11), shape=("m", "n")):
    from t2m.examples.tensor_kernels import dynamic_indexed_gather
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_indexed_gather,
        {"x": "*fp32", "indices": "*" + dtype, "m": "i32", "n": "i32", "count": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("dynamic_indexed_gather", (TensorBuffer("x", shape),
        TensorBuffer("indices", ("count",), dtype=dtype), TensorBuffer("out", ("count",), "write")),
        grid=(GridExtent("count", 8),), runtime_scalars=tuple(RuntimeScalar(name, "i32", arg, bounds=(0, hi))
            for name, arg, hi in (("m", 2, maximum[0]), ("n", 3, maximum[1]), ("count", 4, 19))))
    return convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile), abi


def arguments(m, n, count, dtype):
    dtype = np.int32 if dtype == "i32" else np.int64
    x = (np.arange(m * n, dtype=np.float32) / 2).reshape(m, n)
    indices = np.full(count, np.iinfo(dtype).min, dtype=dtype)
    if m * n and count:
        indices[:] = (np.arange(count) * 3) % (m * n)
        indices[::3] = -1
        indices[-1] = m * n - 1
    golden = np.full(count, -7, dtype=np.float32)
    selected = indices >= 0
    golden[selected] = x.ravel()[indices[selected]]
    return [x, indices, *[np.asarray(v, dtype=np.int32) for v in (m, n, count)]], golden


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_one_merlin_library_gathers_from_changing_and_empty_allocations(tmp_path, profile, dtype):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result, abi = convert(tmp_path, profile, dtype)
    if profile == "tensor":
        assert result.trace["memory"]["source_gathers"]
        assert any(op.name == "tensor.collapse_shape" for op in result.module.walk())
        for extract in (op for op in result.module.walk() if op.name == "tensor.extract"):
            assert len(extract.indices) == 1
            assert not {"arith.divui", "arith.remui"} & {op.name for op in extract.parent_op().walk()}
    arrays, golden = arguments(3, 5, 13, dtype)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, arrays, [golden], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for m, n, count in ((0, 0, 0), (0, 7, 9), (5, 0, 3), (1, 1, 1), (3, 5, 13), (7, 11, 19)):
        arrays, golden = arguments(m, n, count, dtype)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=golden)
        out = tmp_path / f"invoke-{m}-{n}-{count}"
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("m,n,index", [(1, 1, 1), (0, 7, 0), (5, 0, 0)])
def test_gather_guard_uses_actual_extent_before_extract(tmp_path, m, n, index):
    from t2m.native import execute_native
    result, abi = convert(tmp_path, "tensor")
    arrays, _ = arguments(m, n, 1, "i32")
    arrays[1][0] = index
    # Each index is below the declared maximum allocation of 77 elements,
    # but outside this invocation's actual storage, including empty storage.
    with pytest.raises(ValueError, match="native host execution failed"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())


@pytest.mark.parametrize("maximum,shape", [((2**20, 2**20), ("m", "n")), ((0, 2**20), ("m", "n", "n"))])
def test_gather_extent_proof_refuses_overflow_without_allocating_data(tmp_path, maximum, shape):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="32-bit index domain"):
        convert(tmp_path, "buffer", maximum=maximum, shape=shape)
    assert not (tmp_path / "translation/memory-ir").exists()


def test_dynamic_gather_cli_keeps_allocation_and_result_symbols(tmp_path):
    import json
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", "dynamic-gather", "--memory-profile", "buffer", "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE]) == 0
    verify_bundle(out)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["buffers"][0]["shape"] == ["m", "n"]
    assert spec["buffers"][2]["shape"] == ["count"]


def test_native_dense_gather_view_does_not_copy_source_allocation(tmp_path):
    import subprocess
    from t2m.shared_tensor import parse_shared
    result, _ = convert(tmp_path, "tensor")
    source, output, script = (tmp_path / name for name in ("tensor.mlir", "bufferized.mlir", "bufferize.py"))
    source.write_text(result.mlir_text)
    script.write_text('''import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    pipeline = "builtin.module(scf-forall-to-for,canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops})"
    passmanager.PassManager.parse(pipeline).run(module.operation)
    Path(sys.argv[2]).write_text(module.operation.get_asm(print_generic_op_form=True))
''')
    subprocess.run([NATIVE, str(script), str(source), str(output)], check=True, capture_output=True, text=True, timeout=30,
                   env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    module = parse_shared(output.read_text())
    entry = next(op for op in module.ops if op.name == "func.func")
    original = entry.body.block.args[0]
    def root(value):
        while getattr(value.owner, "name", "") in ("memref.cast", "memref.subview", "memref.collapse_shape", "memref.reinterpret_cast"):
            value = value.owner.operands[0]
        return value
    views = [op for op in module.walk() if op.name == "memref.collapse_shape" and root(op.operands[0]) is original]
    assert views and all(len(op.results[0].type.get_shape()) == 1 for op in views)
    assert not any(op.name == "memref.copy" and root(op.operands[0]) is original for op in module.walk())

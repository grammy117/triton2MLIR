"""Dense dynamic axes retain tiles and execute changing shapes through Merlin."""

import json
import os

import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def spec():
    return TensorABI("dynamic_matrix_add", (
        TensorBuffer("x", ("m", "n")), TensorBuffer("y", ("m", "n")),
        TensorBuffer("out", ("m", "n"), "write")),
        grid=(GridExtent("m", 4), GridExtent("n", 8)), runtime_scalars=(
            RuntimeScalar("m", "i32", 2, bounds=(0, 9)), RuntimeScalar("n", "i32", 3, bounds=(0, 13))))


def convert(tmp_path, profile):
    from t2m.examples.tensor_kernels import dynamic_matrix_add
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_matrix_add,
        {"x": "*fp32", "y": "*fp32", "m": "i32", "n": "i32", "out": "*fp32", "BM": "constexpr", "BN": "constexpr"},
        {"BM": 4, "BN": 8}))
    return convert_shared(source, spec(), tmp_path / "translation", optimizer=OPT, memory_profile=profile)


@pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_dynamic_dense_matrix_axes_and_empty_tails(tmp_path, profile):
    import numpy as np
    from t2m.native import execute_native
    result = convert(tmp_path, profile)
    assert "tensor<?x?xf32>" in result.mlir_text
    if profile == "tensor":
        names = {op.name for op in result.module.walk()}
        assert {"linalg.generic", "tensor.extract_slice", "tensor.parallel_insert_slice"} <= names
    for m, n in ((0, 0), (0, 7), (5, 0), (1, 1), (5, 9), (9, 13)):
        x = np.arange(m * n, dtype=np.float32).reshape(m, n)
        y = -x / 2 + np.float32(3)
        (actual,), _ = execute_native(result.mlir_text, [x, y, np.asarray(m, dtype=np.int32), np.asarray(n, dtype=np.int32)],
            tmp_path / f"native-{m}-{n}", python=NATIVE, shape_abi=spec().runtime_shapes())
        np.testing.assert_array_equal(actual, x + y)


@pytest.mark.skipif(not OPT, reason="existing shared optimizer required")
def test_runtime_matrix_stride_disagreement_refuses_before_memory_continuation(tmp_path):
    from dataclasses import replace
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import convert_shared
    convert(tmp_path, "tensor")
    abi = spec()
    wrong = replace(abi, buffers=tuple(replace(b, shape=("n", "m")) for b in abi.buffers))
    with pytest.raises(ConversionError, match="do not match the declared dense"):
        convert_shared(tmp_path / "source.ttir", wrong, tmp_path / "wrong", optimizer=OPT, memory_profile="buffer")
    assert not (tmp_path / "wrong/memory-ir").exists()


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin environments required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_one_merlin_library_handles_runtime_matrix_strides(tmp_path, profile):
    import numpy as np
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = convert(tmp_path, profile)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    write_shared_bundle(result, capture, [x, x + 1, np.asarray(3, dtype=np.int32), np.asarray(5, dtype=np.int32)],
                        [2 * x + 1], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    signature = json.loads((compiled / "compiler/signature.json").read_text())
    assert signature["inputs"][0]["shape"] == signature["outputs"][0]["shape"] == ["m", "n"]
    for m, n in ((0, 0), (0, 7), (5, 0), (1, 1), (5, 9), (9, 13)):
        x = np.arange(m * n, dtype=np.float32).reshape(m, n)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, in0=x, in1=-x + 3, in2=np.asarray(m, dtype=np.int32), in3=np.asarray(n, dtype=np.int32))
        np.savez(reference, out0=np.full((m, n), 3, dtype=np.float32))
        out = tmp_path / f"invoke-{m}-{n}"
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin environments required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("transposed", [False, True])
def test_one_merlin_gemm_library_retains_runtime_k_loop_and_tiling(tmp_path, profile, transposed):
    import numpy as np
    from t2m.bundle import _record, verify_bundle
    from t2m.examples.tensor_kernels import dynamic_matmul
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(dynamic_matmul,
        {"x": "*fp32", "y": "*fp32", "m": "i32", "n": "i32", "k": "i32", "out": "*fp32",
         "BM": "constexpr", "BN": "constexpr", "BK": "constexpr", "TRANSPOSED": "constexpr"},
        {"BM": 16, "BN": 16, "BK": 16, "TRANSPOSED": transposed}))
    abi = TensorABI("dynamic_matmul", (TensorBuffer("x", ("m", "k")),
        TensorBuffer("y", ("n", "k") if transposed else ("k", "n")),
        TensorBuffer("out", ("m", "n"), "write")), grid=(GridExtent("m", 16), GridExtent("n", 16)),
        runtime_scalars=tuple(RuntimeScalar(name, "i32", arg, bounds=(0, high))
            for name, arg, high in (("m", 2, 35), ("n", 3, 21), ("k", 4, 37))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    matmul = next(op for op in result.module.walk() if op.name == "linalg.matmul")
    assert matmul.parent_op().name == "scf.for"
    assert str(matmul.operands[0].type) == "tensor<16x16xf32>"
    if profile == "tensor":
        loop = result.trace["memory"]["source_loops"][0]
        assert loop["upper_interval"] == [0, 37] and loop["step"] == 16
        assert loop["bounds"] == "original-runtime-SSA"
    rng = np.random.default_rng(103)
    def arguments(m, n, k):
        x = rng.normal(size=(m, k)).astype(np.float32)
        y = rng.normal(size=(n, k) if transposed else (k, n)).astype(np.float32)
        return [x, y, *[np.asarray(v, dtype=np.int32) for v in (m, n, k)]], x @ (y.T if transposed else y)
    inputs, golden = arguments(7, 9, 17)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, inputs, [golden], work=tmp_path / "native", python=NATIVE, rtol=1e-5, atol=6e-6)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang, rtol=1e-5, atol=6e-6)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for m, n, k in ((0, 0, 0), (0, 7, 3), (5, 0, 3), (5, 9, 0), (1, 1, 1), (16, 16, 16), (35, 21, 37)):
        arrays, golden = arguments(m, n, k)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=golden)
        out = tmp_path / f"invoke-{m}-{n}-{k}"
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python, rtol=1e-5, atol=6e-6)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(out / "evaluation/outputs.npz", allow_pickle=False) as values:
            assert values["out0"].shape == (m, n)
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
def test_runtime_matmul_cli_keeps_symbolic_dimensions_and_source_k_loop(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", "dynamic-matmul", "--memory-profile", "buffer", "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE]) == 0
    verify_bundle(out)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["buffers"][0]["shape"] == ["m", "k"]
    assert spec["buffers"][2]["shape"] == ["m", "n"]
    assert "tensor<?x?xf32>" in (out / "model.mlir").read_text()

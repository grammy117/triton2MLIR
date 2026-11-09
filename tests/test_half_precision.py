"""Half tensor storage retains typed arithmetic through the real host ABI."""

import os

import numpy as np
import pytest

from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin toolchains required")


def qualify(result, arrays, expected, tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    return compiled, verify_kernel_qualification(compiled), root, python


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_half_tensor_rounding_through_native_and_reused_merlin_library(tmp_path, profile):
    from t2m.bundle import _record
    from t2m.examples.tensor_kernels import runtime_affine
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_affine,
        {"x": "*fp16", "alpha": "fp32", "out": "*fp16", "bias": "fp32", "N": "constexpr", "B": "constexpr"},
        {"N": 13, "B": 8}))
    abi = TensorABI("runtime_affine", (TensorBuffer("x", (13,), dtype="f16"), TensorBuffer("out", (13,), "write", "f16")),
        grid=(2,), runtime_scalars=(RuntimeScalar("alpha", "f32", 1), RuntimeScalar("bias", "f32", 3)))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    assert "f16" in result.mlir_text and "arith.truncf" in result.mlir_text
    x = np.array([1, 1 + 2**-10, -0.0, 2**-24, 2**-14, -2**-24, 65504, -65504,
                  np.inf, -np.inf, np.nan, 0.33325, -0.33325], dtype=np.float16)
    def reference(a, b):
        with np.errstate(all="ignore"):
            return ((x.astype(np.float32) * a) + b).astype(np.float16)
    alpha, bias = np.asarray(1, dtype=np.float32), np.asarray(2**-11, dtype=np.float32)
    expected = reference(alpha, bias)
    assert expected[0] == np.float16(1) and expected[1] == np.float16(1 + 2**-9)
    compiled, receipt, root, python = qualify(result, [x, alpha, bias], expected, tmp_path)
    library = _record(compiled / "host/model.so")
    for ordinal, (a, b) in enumerate(((1, 2**-11), (0.5, -0.25), (2, 0), (-1, 0), (0, 0))):
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        golden = reference(a, b)
        inputs, references = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, in0=x, in1=a, in2=b)
        np.savez(references, out0=golden)
        out = tmp_path / ("invoke" + str(ordinal))
        report = invoke_kernel(compiled, inputs, references, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(out / "evaluation/outputs.npz", allow_pickle=False) as values:
            actual = values["out0"]
            assert actual.dtype == np.float16
            selected = ~np.isnan(golden)
            np.testing.assert_array_equal(actual.view(np.uint16)[selected], golden.view(np.uint16)[selected])
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_half_operands_keep_the_source_nonzero_float32_dot_accumulator(tmp_path, profile):
    from t2m.examples.tensor_kernels import accumulated_matmul
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(accumulated_matmul,
        {"x": "*fp16", "y": "*fp16", "initial": "*fp32", "out": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr"},
        {"M": 16, "N": 16, "K": 16}))
    abi = TensorABI("accumulated_matmul", (TensorBuffer("x", (16, 16), dtype="f16"),
        TensorBuffer("y", (16, 16), dtype="f16"), TensorBuffer("initial", (16, 16)), TensorBuffer("out", (16, 16), "write")))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    matmul = next(op for op in result.module.walk() if op.name == "linalg.matmul")
    assert str(matmul.operands[0].type) == "tensor<16x16xf16>"
    assert str(matmul.operands[2].type) == "tensor<16x16xf32>"
    rng = np.random.default_rng(81)
    x, y = (rng.integers(-2, 3, (16, 16)).astype(np.float16) for _ in range(2))
    initial = rng.integers(-5, 6, (16, 16)).astype(np.float32)
    # Integer-valued half operands make all intermediate products/sums exact;
    # this oracle needs no assumption about a backend's reduction tree.
    expected = initial + x.astype(np.float32) @ y.astype(np.float32)
    qualify(result, [x, y, initial], expected, tmp_path)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_half_masked_matrix_state_keeps_runtime_extents_and_inactive_bits(tmp_path, profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.tensor_kernels import masked_matrix_store
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel
    from t2m.shared_tensor import convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(masked_matrix_store,
        {"x": "*fp16", "out": "*fp16", "m": "i32", "n": "i32", "BM": "constexpr", "BN": "constexpr",
         "TRANSPOSED": "constexpr", "TRAVERSE_TRANSPOSED": "constexpr", "BAD_MASK": "constexpr"},
        {"BM": 4, "BN": 8, "TRANSPOSED": True, "TRAVERSE_TRANSPOSED": True, "BAD_MASK": False}))
    abi = TensorABI("masked_matrix_store", (TensorBuffer("x", ("m", "n"), dtype="f16"),
        TensorBuffer("out", ("n", "m"), "readwrite", "f16")), grid=(GridExtent("m", 4), GridExtent("n", 8)),
        runtime_scalars=(RuntimeScalar("m", "i32", 2, bounds=(0, 9)), RuntimeScalar("n", "i32", 3, bounds=(0, 13))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    def arguments(m, n):
        x = np.linspace(-2, 2, m * n, dtype=np.float16).reshape(m, n)
        old = np.arange(m * n, dtype=np.float16).reshape(n, m)
        if old.size:
            old.view(np.uint16).flat[0] = 0x8000
        if old.size > 1:
            old.view(np.uint16).flat[1] = 0x7E35
        r, c = np.indices((m, n))
        selected = ((r + c) % 3 != 1) & (x > 0)
        expected = old.T.copy()
        expected[selected] = (x[selected].astype(np.float32) + np.float32(2)).astype(np.float16)
        return [x, old, np.asarray(m, dtype=np.int32), np.asarray(n, dtype=np.int32)], expected.T.copy()
    arrays, expected = arguments(5, 9)
    compiled, _, root, python = qualify(result, arrays, expected, tmp_path)
    for m, n in ((0, 0), (0, 7), (5, 0), (1, 1), (9, 13)):
        arrays, expected = arguments(m, n)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / f"invoke-{m}-{n}"
        invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        with np.load(out / "evaluation/outputs.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(values["out0"].view(np.uint16), expected.view(np.uint16))


def test_model_manifest_materializes_half_weights_and_inputs_without_widening(tmp_path):
    """Protocol-only materialization fixture, not a claimed FX capture."""
    import json
    import subprocess
    from pathlib import Path
    import t2m
    capture, work = tmp_path / "capture", tmp_path / "materialized"
    capture.mkdir()
    work.mkdir()
    x = np.array([0, -0.0, 2**-24, 1], dtype=np.float16)
    np.savez(capture / "inputs.npz", in0=x)
    np.savez(capture / "extra.npz")
    (capture / "input_order.json").write_text(json.dumps({"x": 0}))
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "name": "x"}, "1": {"kind": "param", "weight": "w"}}))
    (work / "abi.json").write_text(json.dumps([{"shape": [4], "dtype": "f16"}] * 2))
    script = tmp_path / "materialize.py"
    script.write_text('''import sys
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
sys.path.insert(0, sys.argv[1])
from t2m.model_inputs import main
capture, work = Path(sys.argv[2]), Path(sys.argv[3])
weight = np.array([1, 2, 3, 4], dtype=np.float16)
save_file({"w": weight}, capture / "weights.safetensors")
main(capture, work)
''')
    subprocess.run([NATIVE, str(script), str(Path(t2m.__file__).resolve().parent.parent), str(capture), str(work)],
        capture_output=True, text=True, check=True, timeout=30,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    with np.load(work / "arguments.npz", allow_pickle=False) as values:
        assert values["in0"].dtype == values["in1"].dtype == np.float16
        np.testing.assert_array_equal(values["in0"].view(np.uint16), x.view(np.uint16))
        np.testing.assert_array_equal(values["in1"], np.asarray([1, 2, 3, 4], dtype=np.float16))

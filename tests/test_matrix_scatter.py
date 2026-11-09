"""Multiaxis tile proofs remain valid through broadcasts and transpositions."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")


def convert(root, profile, transposed, traversal, *, bad_mask=False):
    from t2m.examples.tensor_kernels import masked_matrix_store
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import convert_shared
    path = root / "source.ttir"
    path.write_text(capture_ttir(masked_matrix_store,
        {"x": "*fp32", "out": "*fp32", "m": "i32", "n": "i32", "BM": "constexpr", "BN": "constexpr",
         "TRANSPOSED": "constexpr", "TRAVERSE_TRANSPOSED": "constexpr", "BAD_MASK": "constexpr"},
        {"BM": 4, "BN": 8, "TRANSPOSED": transposed, "TRAVERSE_TRANSPOSED": traversal, "BAD_MASK": bad_mask}))
    abi = TensorABI("masked_matrix_store", (TensorBuffer("x", ("m", "n")),
        TensorBuffer("out", ("n", "m") if transposed else ("m", "n"), "readwrite")),
        grid=(GridExtent("m", 4), GridExtent("n", 8)), runtime_scalars=(
            RuntimeScalar("m", "i32", 2, bounds=(0, 9)), RuntimeScalar("n", "i32", 3, bounds=(0, 13))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile=profile)


def arguments(m, n, transposed):
    x = np.linspace(-2, 2, m * n, dtype=np.float32).reshape(m, n)
    old = np.arange(m * n, dtype=np.float32).reshape((n, m) if transposed else (m, n)) + 100
    r, c = np.indices((m, n))
    selected = ((r + c) % 3 != 1) & (x > 0)
    expected = old.T.copy() if transposed else old.copy()
    expected[selected] = x[selected] + np.float32(2)
    return [x, old, np.asarray(m, dtype=np.int32), np.asarray(n, dtype=np.int32)], expected.T.copy() if transposed else expected


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("traversal", [False, True])
def test_runtime_matrix_masks_keep_axis_correspondence_through_merlin(tmp_path, profile, transposed, traversal):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = convert(tmp_path, profile, transposed, traversal)
    assert result.trace["memory"]["source_scatters"]
    arrays, expected = arguments(5, 9, transposed)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for m, n in ((0, 0), (0, 7), (5, 0), (1, 1), (5, 9), (9, 13)):
        arrays, expected = arguments(m, n, transposed)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / f"invoke-{m}-{n}"
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("transposed", [False, True])
def test_a_missing_logical_axis_bound_is_not_inferred_from_flat_storage(tmp_path, transposed):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="mask does not prove active lanes") as error:
        convert(tmp_path, "buffer", transposed, False, bad_mask=True)
    assert error.value.source_id.startswith("ttir.op.")
    assert error.value.opcode == "tts.scatter"
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.parametrize("kernel", ["masked-store", "masked-matrix"])
def test_masked_write_cli_packages_keep_original_predicated_memory(tmp_path, kernel):
    import json
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    out = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", kernel, "--memory-profile", "buffer", "--out", str(out),
                 "--optimizer", OPT, "--python", NATIVE]) == 0
    verify_bundle(out)
    trace = json.loads((out / "provenance.json").read_text())
    assert trace["memory"]["source_scatters"]
    assert "memref.store" in (out / "model.mlir").read_text()
    assert "preserve-inactive-initialized-values" in (out / "memory-witness.mlir").read_text()

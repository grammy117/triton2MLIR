"""End-to-end checked packages from the real reusable frontend and LLVM JIT."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing optimizer and native environment required")


@pytest.fixture(scope="module")
def sequenced(tmp_path_factory):
    from t2m.examples.tensor_kernels import sequenced_outputs
    root = tmp_path_factory.mktemp("sequenced")
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(sequenced_outputs,
                      {"x": "*fp32", "out": "*fp32", "second": "*fp32", "N": "constexpr"}, {"N": 8}))
    abi = TensorABI("sequenced_outputs", (TensorBuffer("x", (8,)),
                    TensorBuffer("out", (8,), "readwrite"), TensorBuffer("second", (8,), "readwrite")))
    return convert_shared(source, abi, root / "conversion", optimizer=OPT)


def inputs():
    x = np.arange(8, dtype=np.float32)
    return [x, np.full_like(x, 99), np.full_like(x, 42)], [(x + 1) * 2, x - 2]


def test_sequenced_memory_and_multiple_results_package(tmp_path, sequenced):
    arrays, golden = inputs()
    bundle = tmp_path / "bundle"
    write_shared_bundle(sequenced, bundle, arrays, golden, work=tmp_path / "native", python=NATIVE)
    receipt = verify_bundle(bundle)
    assert receipt["schema"] == "t2m.capture-receipt.v2"
    assert receipt["materialized_abi"] == {"complete": True, "inputs": 3, "outputs": 2}
    np.testing.assert_array_equal(np.load(bundle / "golden.npy"), golden[0])
    with np.load(bundle / "extra.npz") as saved:
        np.testing.assert_array_equal(saved["out1"], golden[1])
    report = json.loads((bundle / "execution.json").read_text())
    assert report["execution"] == "native-host-LLVM-JIT" and report["reference_comparison"] == "passed"
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["consumer_execution"] == "not-performed"
    assert "universal-equivalence-not-established" in provenance["numeric_equivalence"]
    (bundle / "model.mlir").write_text((bundle / "model.mlir").read_text() + "\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_bundle(bundle)


def test_wrong_golden_never_publishes_a_bundle(tmp_path, sequenced):
    arrays, golden = inputs()
    golden[0][:] = 0
    with pytest.raises(AssertionError):
        write_shared_bundle(sequenced, tmp_path / "bad", arrays, golden, work=tmp_path / "native", python=NATIVE)
    assert not (tmp_path / "bad").exists()


@pytest.mark.skipif(not all(os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")), reason="existing Merlin host toolchain required")
def test_readwrite_memory_versions_and_multiple_results_use_real_merlin(tmp_path, sequenced):
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    arrays, golden = inputs()
    bundle, out = tmp_path / "bundle", tmp_path / "merlin"
    write_shared_bundle(sequenced, bundle, arrays, golden, work=tmp_path / "native", python=NATIVE)
    qualify_kernel(bundle, out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                   compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(out)
    with np.load(out / "host/outputs.npz", allow_pickle=False) as actual:
        for i, expected in enumerate(golden):
            np.testing.assert_array_equal(actual[f"out{i}"], expected)


def test_mutated_ir_and_invalid_tolerance_are_rejected_before_execution(tmp_path, sequenced):
    arrays, golden = inputs()
    with pytest.raises(ValueError, match="finite nonnegative"):
        write_shared_bundle(sequenced, tmp_path / "bad", arrays, golden, work=tmp_path / "native", python=NATIVE, rtol=np.inf)
    clone = type(sequenced)(sequenced.module.clone(), sequenced.spec, sequenced.trace, sequenced.source_ttir)
    from xdsl.dialects.builtin import StringAttr
    clone.module.attributes["test.changed"] = StringAttr("changed")
    with pytest.raises(ValueError, match="IR changed"):
        write_shared_bundle(clone, tmp_path / "bad", arrays, golden, work=tmp_path / "native", python=NATIVE)
    assert not (tmp_path / "native").exists()

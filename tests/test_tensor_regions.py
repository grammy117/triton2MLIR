"""Real source regions retain local tensor state and nested contractions."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

REQUIRED = all(os.environ.get(key) for key in (
    "TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG",
))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing shared/native/Merlin tools required")


def capture(root, kind, dtype, profile):
    from t2m.examples.tensor_region_kernels import tensor_region_bitsets

    source = root / "source.ttir"
    source.write_text(capture_ttir(tensor_region_bitsets, {
        "x": "*" + dtype, "out": "*" + dtype, "n": "i32",
        "B": "constexpr", "KIND": "constexpr", "WHILE": "constexpr",
        "REVERSE": "constexpr",
    }, {"B": 8, "KIND": "map" if kind == "map" else kind.split("-")[0],
        "WHILE": "while" in kind, "REVERSE": "reverse" in kind}))
    abi = TensorABI("tensor_region_bitsets", (
        TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("out", ("n",), "write", dtype),
    ), grid=(GridExtent("n", 8),), runtime_scalars=(
        RuntimeScalar("n", "i32", 2, bounds=(0, 31)),
    ))
    return convert_shared(source, abi, root / "translation",
                          optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, kind, dtype):
    bits = int(dtype[1:])
    words = [0, -1, -(1 << (bits - 1)), (1 << (bits - 1)) - 1,
             -2, -16, (1 << (bits - 2)) + 37, -128, 0x12345]
    # Shared nonzero bits keep each tile's result meaningful while the sign,
    # high words and remaining lanes still vary independently.
    words = [word | ((1 << (bits - 2)) + 5) for word in words]
    x = np.resize(np.asarray(words, dtype=dtype.replace("i", "int")), n)
    out = x.copy()
    for start in range(0, n, 8):
        stop = min(start + 8, n)
        if kind.startswith("reduce"):
            value = -1
            for word in x[start:stop]:
                value &= int(word)
            out[start:stop] = value
        elif kind.startswith("scan"):
            order = list(range(start, stop))
            if "reverse" in kind:
                order.reverse()
            value = -1
            for index in order:
                value &= int(x[index])
                out[index] = value
    return [x, np.asarray(n, np.int32)], [out]


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("kind", ["reduce-for", "scan-while-reverse", "map"])
def test_local_tensor_state_and_nested_regions_reach_merlin_and_spike(tmp_path, kind, dtype, profile):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, kind, dtype, profile)
    assert any(item["kind"] == "source-region-tensor-dataflow" for item in result.trace["transformations"])
    assert "linalg.reduce" in result.mlir_text
    assert not any(operation.name.startswith("gpu.") for operation in result.module.walk())
    args, outputs = arguments(13, kind, dtype)
    policy = expected("", [{"kind": "words"}])
    captured, compiled = qualify(result, tmp_path, args, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, n in enumerate((0, 1, 8, 31)):
        args, outputs = arguments(n, kind, dtype)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
        np.savez(caller / "reference.npz", out0=outputs[0])
        (caller / "reference.json").write_text(json.dumps(policy))
        out = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", out,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert _record(out / "source/host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    target.qualify(captured, tmp_path / "target", **options)
    target.verify(tmp_path / "target", **options)

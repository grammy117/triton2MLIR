"""Retain coarse source math/dataflow through real reusable consumer programs."""

import json
import os

import numpy as np
import pytest
from test_tensor_regions import REQUIRED

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing shared/native/Merlin tools required")
CASES = [("histogram", "i32"), ("histogram", "i64"), ("dot", "f32"),
         ("dot-zero", "f32"), ("scaled-dot", "f32"), ("gather", "i64"),
         ("logical", "i32"), ("logical-control", "i32"), ("math", "f32"), ("math", "f64")]


def capture(root, family, dtype, profile):
    from t2m.examples.numeric_region_kernels import numeric_region_values

    signature = "*" + {"f32": "fp32", "f64": "fp64"}.get(dtype, dtype)
    source = root / "source.ttir"
    source.write_text(capture_ttir(numeric_region_values, {
        **{name: signature for name in ("x", "y", "initial", "out")},
        "n": "i32", "B": "constexpr", "FAMILY": "constexpr",
    }, {"B": 8, "FAMILY": family}))
    abi = TensorABI("numeric_region_values", tuple(
        TensorBuffer(name, ("n",), "write" if name == "out" else "read", dtype)
        for name in ("x", "y", "initial", "out")
    ), grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                          memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, family, dtype):
    storage = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}[dtype]
    if dtype.startswith("i"):
        bits = int(dtype[1:])
        x = np.resize(np.asarray([-1, 0, 1, -(1 << (bits - 1)), (1 << (bits - 1)) - 1,
                                  9, 0x12345, -9], storage), n)
        y, c = x[::-1].copy(), np.zeros(n, storage)
        def signed(value):
            value &= (1 << bits) - 1
            return value - (1 << bits) if value >> (bits - 1) else value
        if family == "histogram":
            result = [(int(value) & ((1 << bits) - 1)).bit_count() for value in x]
        elif family == "logical":
            result = [signed(int(value) + (index // 8) * 17 + (n + 7) // 8)
                      for index, value in enumerate(x)]
        elif family == "logical-control":
            result = [signed(int(value) + ((index // 8) * 17 if int(value) & 1 else 0)
                             + (int(value) & 3) * ((n + 7) // 8 + 1))
                      for index, value in enumerate(x)]
        else:
            result = [signed(sum((int(value) + ((index + (int(value) & 7)) & 7)) * (index + 1)
                                 for index in range(8))) for value in x]
        out = np.asarray(result, storage)
    else:
        x = np.resize(np.asarray([-.75, -.5, -.25, .125, .25, .5, .75, 1], storage), n)
        y = np.resize(np.asarray([-.5, .125, .25, .5, .75, 1, 1.25, 1.5], storage), n)
        c = np.resize(np.asarray([.125, -.25, .5, 1, -2, 3, .75, -.5], storage), n)
        out = np.tanh(x) * storage(8) if family == "math" else (
            (storage(32 if family == "scaled-dot" else 16) * x * y
             + (storage(0) if family == "dot-zero" else c)) * storage(256)
        )
    return [x, y, c, np.asarray(n, np.int32)], [out]


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("family,dtype", CASES)
def test_coarse_numeric_regions_reach_native_merlin_and_spike(tmp_path, family, dtype, profile):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, family, dtype, profile)
    if family.startswith("dot") or family == "scaled-dot":
        assert "linalg.matmul" in result.mlir_text
    if family == "dot":
        assert any(item["kind"] == "restore-source-contraction-accumulator"
                   for item in result.trace["transformations"])
    if family == "histogram":
        assert "indexed-integer-accumulation" in result.mlir_text
    if family.startswith("logical"):
        assert any(item["kind"] == "uniform-region-launch-query"
                   for item in result.trace["transformations"])
    comparison = {"kind": "words"} if family != "math" else {
        "kind": "values", "rtol": 2e-6 if dtype == "f32" else 1e-12,
        "atol": 1e-7 if dtype == "f32" else 1e-13,
    }
    policy = expected("", [comparison])
    args, outputs = arguments(13, family, dtype)
    captured, compiled = qualify(result, tmp_path, args, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, n in enumerate((0, 1, 8, 31)):
        args, outputs = arguments(n, family, dtype)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
        np.savez(caller / "reference.npz", out0=outputs[0])
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(captured, tmp_path / "target", **options)
    assert target.verify(tmp_path / "target", **options) == report

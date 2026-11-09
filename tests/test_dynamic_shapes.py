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


@pytest.mark.parametrize("change", ["unbounded", "negative", "unknown", "rank"])
def test_unproved_dynamic_dimension_contracts_refuse(change):
    declaration = spec().as_dict()
    if change == "unbounded":
        declaration["runtime_scalars"][0].pop("bounds")
    elif change == "negative":
        declaration["runtime_scalars"][0]["bounds"] = [-1, 13]
    elif change == "unknown":
        declaration["buffers"][0]["shape"] = ["other"]
    else:
        declaration["buffers"][0]["shape"] = ["n", 8]
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

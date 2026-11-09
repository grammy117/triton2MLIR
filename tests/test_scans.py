import os

import numpy as np
import pytest

from t2m.frontend import capture_ttir
from t2m.diagnostics import ConversionError
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, _source_ids, convert_shared, parse_shared
from t2m.scans import expand_scans

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


def translate(kernel, buffers, constants, grid, root):
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*" + ("fp32" if b.dtype == "f32" else b.dtype) for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    return convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), root / "translation", optimizer=OPT)


@pytest.mark.parametrize("kind", [0, 1, 2], ids=["sum", "product", "maximum"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", ["f32", "i32"])
def test_scan_keeps_parallel_stages_and_exact_tail_padding(tmp_path, kind, reverse, dtype):
    from t2m.examples.tensor_kernels import scan_tiles
    buffers = (TensorBuffer("x", (53,), dtype=dtype), TensorBuffer("out", (53,), "write", dtype=dtype))
    result = translate(scan_tiles, buffers, {"N": 53, "B": 16, "KIND": kind, "REVERSE": reverse}, (4,), tmp_path)
    rng = np.random.default_rng(30)
    x = rng.uniform(0.7, 1.3, 53).astype("float32") if dtype == "f32" else rng.integers(-1, 3, 53, dtype="int32")
    expected = np.empty_like(x)
    for start in range(0, len(x), 16):
        tile = x[start:start + 16]
        if reverse:
            tile = tile[::-1]
        scanned = (np.cumsum(tile) if kind == 0 else np.cumprod(tile) if kind == 1 else np.maximum.accumulate(tile))
        expected[start:start + 16] = scanned[::-1] if reverse else scanned
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
    scan = next(t for t in result.trace["transformations"] if t["kind"] == "associative-scan-network")
    assert scan["stages"] == 4 and scan["reverse"] == reverse
    stages = [op for op in result.module.walk() if "prov.scan_distance" in op.attributes]
    assert len(stages) == 4 and all(op.name == "linalg.generic" for op in stages)
    assert len({op.attributes["prov.composite_id"].data for op in stages}) == 1
    assert not any(op.name.startswith(("tt.", "ttx.", "gpu.")) for op in result.module.walk())


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_tuple_scan_preserves_noncommutative_combiner_order_on_each_axis(tmp_path, axis, reverse):
    from t2m.examples.tensor_kernels import tuple_scan
    buffers = tuple(TensorBuffer(n, (2, 8), "write" if n.startswith("out") else "read")
                    for n in ("a", "b", "out_a", "out_b"))
    result = translate(tuple_scan, buffers, {"R": 2, "S": 8, "AXIS": axis, "REVERSE": reverse}, (1,), tmp_path)
    a = np.tile(np.array([1, 2, 1, 0, 2, 1, 1, 2], dtype="float32"), (2, 1))
    b = np.arange(-8, 8, dtype="float32").reshape(2, 8)
    expected_a, expected_b = np.empty_like(a), np.empty_like(b)
    previous = None
    order = list(range(a.shape[axis]))
    for index in reversed(order) if reverse else order:
        current_a, current_b = np.take(a, index, axis=axis), np.take(b, index, axis=axis)
        if previous is not None:
            current_a, current_b = current_a * previous[0], current_a * previous[1] + current_b
        target = [slice(None)] * a.ndim
        target[axis] = index
        expected_a[tuple(target)], expected_b[tuple(target)] = current_a, current_b
        previous = current_a, current_b
    actual, _ = execute_native(result.mlir_text, [a, b], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual[0], expected_a)
    np.testing.assert_array_equal(actual[1], expected_b)


def test_single_element_scan_does_not_invent_a_zero_identity(tmp_path):
    from t2m.examples.tensor_kernels import scan_tiles
    buffers = (TensorBuffer("x", (1,)), TensorBuffer("out", (1,), "write"))
    result = translate(scan_tiles, buffers, {"N": 1, "B": 1, "KIND": 0, "REVERSE": False}, (1,), tmp_path)
    x = np.array([-0.0], dtype="float32")
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual.view("uint32"), x.view("uint32"))


@pytest.mark.parametrize("changed", ["axis", "effect"])
def test_unqualified_scan_combiner_or_axis_has_a_located_refusal(changed):
    axis = 3 if changed == "axis" else 0
    extra = '"tt.print"() <{prefix = "effect", hex = false}> : () -> ()' if changed == "effect" else ""
    module = parse_shared(f'''module {{
      func.func @f(%x: tensor<8xf32>) -> tensor<8xf32> {{
        %out = "tt.scan"(%x) <{{axis = {axis} : i32, reverse = false}}> ({{
        ^bb0(%a: f32, %b: f32):
          {extra}
          %sum = arith.addf %a, %b : f32
          "tt.scan.return"(%sum) : (f32) -> ()
        }}) : (tensor<8xf32>) -> tensor<8xf32> loc("ttir.op.9999")
        return %out : tensor<8xf32>
      }}
    }}''')
    with pytest.raises(ConversionError, match="ttir.op.9999"):
        expand_scans(module, _source_ids)

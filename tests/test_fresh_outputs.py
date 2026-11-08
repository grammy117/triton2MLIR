import os

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not OPT, reason="existing shared optimizer required")


def capture(kernel, root, buffers, constants, grid=(1,)):
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI(kernel.__name__, tuple(buffers), grid=grid)
    return convert_shared(source, abi, root / "conversion", optimizer=OPT)


def test_fresh_outputs_use_only_actual_inputs_and_preserve_local_updates(tmp_path):
    from t2m.examples.tensor_kernels import sequenced_outputs
    result = capture(sequenced_outputs, tmp_path,
                     [TensorBuffer("x", (8,)), TensorBuffer("out", (8,), "write"),
                      TensorBuffer("second", (8,), "write")], {"N": 8})
    function = next(iter(result.module.ops))
    assert len(function.body.block.args) == 1 and len(function.function_type.outputs) == 2
    assert set(result.trace["memory"]["output_initialization"]) == {"out", "second"}
    assert result.spec.as_dict()["schema"] == "t2m.shared-tensor-abi.v2"
    assert TensorABI.from_dict(result.spec.as_dict()) == result.spec
    if NATIVE:
        x = np.arange(8, dtype=np.float32)
        write_shared_bundle(result, tmp_path / "bundle", [x], [(x + 1) * 2, x - 2],
                            work=tmp_path / "native", python=NATIVE)
        assert verify_bundle(tmp_path / "bundle")["materialized_abi"] == {"complete": True, "inputs": 1, "outputs": 2}


def test_fresh_output_initial_read_is_refused(tmp_path):
    from t2m.examples.tensor_kernels import nested_updates
    with pytest.raises(ConversionError, match="read before.*initialized"):
        capture(nested_updates, tmp_path, [TensorBuffer("out", (16,), "write")], {"B": 8}, (2,))


def test_disjoint_writes_with_unwritten_holes_are_not_complete_outputs(tmp_path):
    from t2m.examples.tensor_kernels import tiled_add
    with pytest.raises(ConversionError, match="complete overwrite"):
        capture(tiled_add, tmp_path, [TensorBuffer("x", (12, 8)), TensorBuffer("y", (12, 8)),
                TensorBuffer("out", (12, 8), "write")], {"N": 8, "BM": 4, "BN": 8}, (2, 1))


@pytest.mark.skipif(not NATIVE, reason="existing native environment required")
def test_fresh_k_loop_matmul_omits_destination_argument(tmp_path):
    from t2m.examples.tensor_kernels import loop_matmul
    result = capture(loop_matmul, tmp_path, [TensorBuffer("x", (32, 32)), TensorBuffer("y", (32, 32)),
                     TensorBuffer("out", (32, 32), "write")],
                     {"N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 16}, (2, 2))
    rng = np.random.default_rng(23)
    x, y = [rng.normal(size=(32, 32)).astype("float32") for _ in range(2)]
    write_shared_bundle(result, tmp_path / "bundle", [x, y], [x @ y], work=tmp_path / "native",
                        python=NATIVE, rtol=1e-5, atol=5e-6)
    assert verify_bundle(tmp_path / "bundle")["materialized_abi"]["inputs"] == 2

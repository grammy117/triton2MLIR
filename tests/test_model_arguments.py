"""Call attributes supply compiler values; tensors prove specialized shapes."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding, bind_model
from t2m.memory_contract import GUARDED_PROFILES
from t2m.native import execute_native
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not OPT, reason="existing shared optimizer required")


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def dynamic_implementation(tmp_path_factory, request):
    from t2m.examples.tensor_kernels import dynamic_matrix_add

    root = tmp_path_factory.mktemp("model-argument-implementation")
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            dynamic_matrix_add,
            {
                "x": "*fp32",
                "y": "*fp32",
                "m": "i32",
                "n": "i32",
                "out": "*fp32",
                "BM": "constexpr",
                "BN": "constexpr",
            },
            {"BM": 4, "BN": 8},
        )
    )
    abi = TensorABI(
        "dynamic_matrix_add",
        (
            TensorBuffer("x", ("m", "n")),
            TensorBuffer("y", ("m", "n")),
            TensorBuffer("out", ("m", "n"), "write"),
        ),
        grid=(GridExtent("m", 4), GridExtent("n", 8)),
        runtime_scalars=(
            RuntimeScalar("m", "i32", 2, bounds=(0, 8)),
            RuntimeScalar("n", "i32", 3, bounds=(0, 16)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "convert",
        optimizer=OPT,
        memory_profile=request.param,
        address_policy="source-words"
        if request.param == "guarded-buffer"
        else "structured",
    )


def model(m=3, n=5, *, result_n=None, dtype="f32", rows=None):
    result_n = n if result_n is None else result_n
    input_type, result_type = (
        f"tensor<{m}x{n}x{dtype}>",
        f"tensor<{m}x{result_n}x{dtype}>",
    )
    rows = f"{m}:i64" if rows is None else rows
    return f"""module {{
      func.func @forward(%x:{input_type},%y:{input_type})->{result_type} {{
        %r=func.call @kernel(%x,%y) {{prov.kernel="test::dynamic",prov.kernel_operands=["x","y"],prov.kernel_args={{rows={rows},columns={n}:i64}}}}:({input_type},{input_type})->{result_type}
        return %r:{result_type}
      }}
      func.func private @kernel({input_type},{input_type})->{result_type} attributes {{prov.kernel="test::dynamic"}}
    }}"""


def binding(implementation, m=3, n=5, *, mapping=None, operands=None):
    return KernelBinding(
        "test::dynamic",
        implementation,
        scalar_arguments={"rows": m, "columns": n},
        runtime_arguments={"m": "rows", "n": "columns"} if mapping is None else mapping,
        operands={} if operands is None else operands,
    )


@pytest.mark.parametrize("m,n", [(3, 5), (0, 5), (8, 16)])
def test_runtime_dimensions_are_specialized_from_call_contract(
    tmp_path, dynamic_implementation, m, n
):
    bound = bind_model(model(m, n), [binding(dynamic_implementation, m, n)])
    assert '"func.call"' not in bound.mlir_text
    assert bound.trace["model_abi"] == "unchanged"
    call = bound.trace["calls"][0]
    assert set(call["runtime_scalar_bindings"]) == {"m", "n"}
    assert {item["kind"] for item in call["argument_adaptations"]} == {
        "tensor-shape-cast",
        "captured-runtime-scalar",
        "result-shape-cast",
    }
    if dynamic_implementation.trace.get("profile") in GUARDED_PROFILES:
        assert bound.trace.get("guard_storage_bytes", 0) == m * n * 16
    if NATIVE:
        x = np.arange(m * n, dtype=np.float32).reshape(m, n)
        y = np.full((m, n), 3, np.float32)
        output, _ = execute_native(
            bound.mlir_text, [x, y], tmp_path / "native", python=NATIVE
        )
        np.testing.assert_array_equal(output[0], x + y)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "schema-name",
        "bounds",
        "boolean",
        "tensor-shape",
        "result-shape",
        "dtype",
    ],
)
def test_runtime_adaptation_cannot_invent_missing_or_conflicting_facts(
    dynamic_implementation, change
):
    implementation = binding(dynamic_implementation)
    source = model()
    if change == "missing":
        implementation.runtime_arguments.pop("n")
    elif change == "extra":
        implementation.runtime_arguments["other"] = "rows"
    elif change == "schema-name":
        implementation.runtime_arguments["m"] = "missing"
    elif change == "bounds":
        implementation.scalar_arguments["rows"] = 9
        source = model(9, 5)
    elif change == "boolean":
        implementation.scalar_arguments["rows"] = True
        source = model(rows="true")
    elif change == "tensor-shape":
        source = model().replace("tensor<3x5xf32>", "tensor<4x5xf32>")
    elif change == "result-shape":
        source = model(result_n=6)
    else:
        source = model(dtype="f64")
    with pytest.raises(ConversionError):
        bind_model(source, [implementation])


def test_same_runtime_body_accounts_distinct_captured_call_shapes(
    tmp_path, dynamic_implementation
):
    source = """module {
      func.func @forward(%x:tensor<3x5xf32>,%y:tensor<3x5xf32>,%u:tensor<2x4xf32>,%v:tensor<2x4xf32>)->(tensor<3x5xf32>,tensor<2x4xf32>) {
        %a=func.call @first(%x,%y) {prov.kernel="test::dynamic",prov.kernel_operands=["x","y"],prov.kernel_args={rows=3:i64,columns=5:i64}}:(tensor<3x5xf32>,tensor<3x5xf32>)->tensor<3x5xf32>
        %b=func.call @second(%u,%v) {prov.kernel="test::dynamic",prov.kernel_operands=["x","y"],prov.kernel_args={rows=2:i64,columns=4:i64}}:(tensor<2x4xf32>,tensor<2x4xf32>)->tensor<2x4xf32>
        return %a,%b:tensor<3x5xf32>,tensor<2x4xf32>
      }
      func.func private @first(tensor<3x5xf32>,tensor<3x5xf32>)->tensor<3x5xf32> attributes {prov.kernel="test::dynamic"}
      func.func private @second(tensor<2x4xf32>,tensor<2x4xf32>)->tensor<2x4xf32> attributes {prov.kernel="test::dynamic"}
    }"""
    result = bind_model(
        source, [binding(dynamic_implementation), binding(dynamic_implementation, 2, 4)]
    )
    assert [call["parent_call_ordinal"] for call in result.trace["calls"]] == [0, 1]
    if dynamic_implementation.trace.get("profile") in GUARDED_PROFILES:
        assert result.trace["guard_storage_bytes"] == 16 * (3 * 5 + 2 * 4)
    if NATIVE:
        inputs = [
            np.arange(m * n, dtype=np.float32).reshape(m, n) + offset
            for m, n, offset in [(3, 5, 0), (3, 5, 1), (2, 4, 2), (2, 4, 3)]
        ]
        output, _ = execute_native(
            result.mlir_text, inputs, tmp_path / "native", python=NATIVE
        )
        for actual, expected in zip(
            output, (inputs[0] + inputs[1], inputs[2] + inputs[3]), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def scalar_model(size, attributes):
    return f"""module {{func.func @forward(%x:tensor<{size}xf32>)->tensor<{size}xf32>{{
      %r=func.call @kernel(%x) {{prov.kernel="test::scalar",prov.kernel_operands=["x"],prov.kernel_args={{{attributes}}}}}:(tensor<{size}xf32>)->tensor<{size}xf32>
      return %r:tensor<{size}xf32>
    }} func.func private @kernel(tensor<{size}xf32>)->tensor<{size}xf32> attributes {{prov.kernel="test::scalar"}}}}"""


@pytest.mark.parametrize("kind", ["float", "boolean"])
def test_runtime_float_rounding_and_boolean_types_retain_source_behavior(
    tmp_path, kind
):
    from t2m.examples.tensor_kernels import runtime_affine, runtime_branch

    kernel = runtime_affine if kind == "float" else runtime_branch
    constants = {"N": 8, "B": 8}
    if kind == "float":
        signature = {
            "x": "*fp32",
            "alpha": "fp32",
            "out": "*fp32",
            "bias": "i32",
            "N": "constexpr",
            "B": "constexpr",
        }
        scalars = (
            RuntimeScalar("alpha", "f32", 1),
            RuntimeScalar("bias", "i32", 3, bounds=(-10, 10)),
        )
        arguments, mapping = (
            {"scale": 0.1, "offset": 2},
            {"alpha": "scale", "bias": "offset"},
        )
        attributes = "scale=0.1:f64,offset=2:i64"
    else:
        signature = {
            "x": "*fp32",
            "flag": "i1",
            "out": "*fp32",
            "N": "constexpr",
            "B": "constexpr",
        }
        scalars = (RuntimeScalar("flag", "i1", 1),)
        arguments, mapping = {"enabled": True}, {"flag": "enabled"}
        attributes = "enabled=true"
    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(kernel, signature, constants))
    abi = TensorABI(
        kernel.__name__,
        (TensorBuffer("x", (8,)), TensorBuffer("out", (8,), "write")),
        runtime_scalars=scalars,
    )
    implementation = convert_shared(path, abi, tmp_path / "convert", optimizer=OPT)
    bound = bind_model(
        scalar_model(8, attributes),
        [
            KernelBinding(
                "test::scalar",
                implementation,
                scalar_arguments=arguments,
                runtime_arguments=mapping,
            )
        ],
    )
    if kind == "float":
        assert '"arith.truncf"' in bound.mlir_text
        assert "rtne-f64-to-f32" in json.dumps(bound.trace)
    if NATIVE:
        x = np.arange(8, dtype=np.float32)
        output, _ = execute_native(
            bound.mlir_text, [x], tmp_path / "native", python=NATIVE
        )
        expected = x * np.float32(0.1) + np.float32(2) if kind == "float" else x + 2
        np.testing.assert_array_equal(output[0], expected)

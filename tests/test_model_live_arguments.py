"""Live scalar edges come from original SSA; samples and tags cannot supply facts."""

import hashlib
from copy import deepcopy

import pytest
import test_model_arguments as argument_fixtures

from t2m.diagnostics import ConversionError
from t2m.model_binding import KernelBinding, bind_model
from t2m.model_live_arguments import PROFILE, validate_model
from t2m.shared_tensor import parse_shared

NATIVE = argument_fixtures.NATIVE
dynamic_implementation = argument_fixtures.dynamic_implementation
pytestmark = pytest.mark.skipif(not argument_fixtures.OPT, reason="existing shared optimizer required")


def source():
    return '''module {
      func.func @forward(%x:tensor<3x5xf32>,%y:tensor<3x5xf32>)->tensor<3x5xf32> {
        %zero=arith.constant 0:index
        %one=arith.constant 1:index
        %rows=tensor.dim %x,%zero:tensor<3x5xf32>
        %columns=tensor.dim %y,%one:tensor<3x5xf32>
        %out=func.call @kernel(%x,%y,%rows,%columns) {prov.kernel="test::live",
          prov.kernel_operands=["x","y","rows","columns"]}:
          (tensor<3x5xf32>,tensor<3x5xf32>,index,index)->tensor<3x5xf32>
        return %out:tensor<3x5xf32>
      }
      func.func private @kernel(tensor<3x5xf32>,tensor<3x5xf32>,index,index)->tensor<3x5xf32>
        attributes {prov.kernel="test::live"}
    }'''


def binding(implementation):
    return KernelBinding("test::live", implementation, runtime_operands={"m": "rows", "n": "columns"})


def test_live_shape_operands_preserve_actual_ssa_and_original_interface(tmp_path, dynamic_implementation):
    result = bind_model(source(), [binding(dynamic_implementation)])
    assert result.trace["argument_binding_profile"] == PROFILE
    record = result.trace["calls"][0]
    assert record["runtime_scalar_bindings"] == {}
    assert {name: item["constant_integer"] for name, item in record["runtime_operand_bindings"].items()} == {"m": 3, "n": 5}
    assert "tensor.dim" in result.mlir_text and "arith.index_cast" in result.mlir_text
    assert not any(item["kind"] == "captured-runtime-scalar" for item in record["argument_adaptations"])
    validate_model(result.trace, parse_shared(source()))
    if NATIVE:
        import numpy as np

        from t2m.native import execute_native
        x = np.arange(15, dtype=np.float32).reshape(3, 5)
        values, _ = execute_native(result.mlir_text, [x, -2 * x], tmp_path / "native", python=NATIVE)
        np.testing.assert_array_equal(values[0], -x)


@pytest.mark.parametrize("damage", ["missing", "overlap", "duplicate", "absent", "tensor-overlap"])
def test_live_mapping_cannot_omit_duplicate_or_reclassify_schema_values(dynamic_implementation, damage):
    selected = binding(dynamic_implementation)
    if damage == "missing": selected.runtime_operands.pop("n")
    elif damage == "overlap": selected.runtime_arguments["m"] = "rows"
    elif damage == "duplicate": selected.runtime_operands["n"] = "rows"
    elif damage == "absent": selected.runtime_operands["n"] = "absent"
    else: selected.runtime_operands["m"] = "x"
    with pytest.raises(ConversionError):
        bind_model(source(), [selected])


def test_one_actual_scalar_can_supply_two_formals_without_dropping_schema_operands(dynamic_implementation):
    original = source().replace("3x5", "3x3").replace(
        '@kernel(%x,%y,%rows,%columns)', '@kernel(%x,%y,%rows)').replace(
        '["x","y","rows","columns"]', '["x","y","rows"]').replace(
        'tensor<3x3xf32>,tensor<3x3xf32>,index,index', 'tensor<3x3xf32>,tensor<3x3xf32>,index')
    selected = binding(dynamic_implementation)
    selected.runtime_operands["n"] = "rows"
    result = bind_model(original, [selected])
    validate_model(result.trace, parse_shared(original))
    assert result.trace["calls"][0]["runtime_operand_bindings"]["n"]["constant_integer"] == 3


@pytest.mark.parametrize("damage", ["profile", "fact", "adapter", "operand", "extra", "caller-type"])
def test_rehashed_live_records_cannot_replace_the_actual_parent_ssa(dynamic_implementation, damage):
    trace = deepcopy(bind_model(source(), [binding(dynamic_implementation)]).trace)
    record = trace["calls"][0]["runtime_operand_bindings"]["m"]
    if damage == "profile": trace["argument_binding_profile"] = "t2m.captured-runtime-arguments.v1"
    elif damage == "fact": record["constant_integer"] = 4
    elif damage == "adapter": record["adapter"] = "exact-scalar"
    elif damage == "operand": record["schema_operand"] = "absent"
    elif damage == "caller-type": record["caller_type"] = "i64"
    else: record["alignment_permission"] = True
    with pytest.raises(ValueError):
        validate_model(trace, parse_shared(source()))


def test_prior_kernel_inlining_cannot_create_a_fact_about_an_original_opaque_result(tmp_path):
    from test_live_scalar_model import implementation

    from t2m.ir import text
    from t2m.shared_tensor import SharedTensorResult, TensorABI, TensorBuffer

    # A typed binding unit interface deliberately exposes a literal only after
    # inlining. The original parent declaration supplies no scalar value fact.
    module = parse_shared('''module {func.func @literal()->tensor<1xi32> {
      %seven=arith.constant 7:i32
      %value="tensor.from_elements"(%seven):(i32)->tensor<1xi32>
      return %value:tensor<1xi32>
    }}''')
    spec = TensorABI("literal", (TensorBuffer("out", (1,), "write", "i32"),))
    constant = SharedTensorResult(module, spec, {"spec": spec.as_dict(),
        "mlir_sha256": hashlib.sha256(text(module).encode()).hexdigest(), "source_sha256": "0" * 64}, "typed-unit-interface")
    root = tmp_path / "scalar"
    root.mkdir()
    scalar = implementation(root, "i32")
    original = '''module {func.func @forward(%x:tensor<13xi32>)->tensor<13xi32> {
      %control=func.call @literal() {prov.kernel="test::literal",prov.kernel_operands=[]}:()->tensor<1xi32>
      %out=func.call @scalar(%x,%control) {prov.kernel="test::scalar",prov.kernel_operands=["x","control"]}:
          (tensor<13xi32>,tensor<1xi32>)->tensor<13xi32>
      return %out:tensor<13xi32>
    }
    func.func private @literal()->tensor<1xi32> attributes {prov.kernel="test::literal"}
    func.func private @scalar(tensor<13xi32>,tensor<1xi32>)->tensor<13xi32> attributes {prov.kernel="test::scalar"}
    }'''
    result = bind_model(original, [KernelBinding("test::literal", constant),
        KernelBinding("test::scalar", scalar, runtime_operands={"control": "control"})])
    record = result.trace["calls"][1]["runtime_operand_bindings"]["control"]
    assert "constant_integer" not in record
    validate_model(result.trace, parse_shared(original))

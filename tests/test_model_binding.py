import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding, bind_model
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not OPT, reason="existing shared optimizer required")


@pytest.fixture(scope="module")
def implementation(tmp_path_factory):
    from t2m.examples.tensor_kernels import matrix_add
    root = tmp_path_factory.mktemp("binding-implementation")
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(matrix_add,
        {"x": "*fp32", "y": "*fp32", "out": "*fp32", "M": "constexpr", "N": "constexpr"}, {"M": 4, "N": 8}))
    abi = TensorABI("matrix_add", (TensorBuffer("x", (4, 8)), TensorBuffer("y", (4, 8)),
                                  TensorBuffer("out", (4, 8), "write")))
    return convert_shared(source, abi, root / "translated", optimizer=OPT)


def model(*, alpha=1, names='["x", "y"]', declaration="test::kernel", two_calls=False):
    extra = ('%second = func.call @arbitrary_symbol_29(%y, %x) '
             '{prov.kernel = "test::kernel", prov.kernel_operands = ["x", "y"], prov.kernel_args = {alpha = 1 : i64}} '
             ': (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>') if two_calls else ""
    output_types = "(tensor<4x8xf32>, tensor<4x8xf32>)" if two_calls else "tensor<4x8xf32>"
    returns = "%out, %second : tensor<4x8xf32>, tensor<4x8xf32>" if two_calls else "%out : tensor<4x8xf32>"
    return f'''module {{
      func.func @forward(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> {output_types} {{
        %out = func.call @arbitrary_symbol_29(%x, %y) {{prov.kernel = "test::kernel", prov.kernel_operands = {names},
                 prov.kernel_args = {{alpha = {alpha} : i64}}}} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        {extra}
        return {returns}
      }}
      func.func private @arbitrary_symbol_29(tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        attributes {{prov.kernel = "{declaration}"}}
    }}'''


def test_real_translated_body_binds_by_identity_and_preserves_call_instances(tmp_path, implementation):
    binding = KernelBinding("test::kernel", implementation, scalar_arguments={"alpha": 1})
    result = bind_model(model(two_calls=True), [binding])
    assert len(list(result.module.ops)) == 1
    assert all(op.name != "func.call" for op in result.module.walk())
    assert len(result.trace["calls"]) == 2
    ids = [op.attributes["prov.emitted_id"].data for op in result.module.walk() if "prov.emitted_id" in op.attributes]
    assert len(ids) == len(set(ids))
    if NATIVE:
        x, y = np.ones((4, 8), np.float32), np.full((4, 8), 3, np.float32)
        outputs, _ = execute_native(result.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
        assert len(outputs) == 2
        for output in outputs:
            np.testing.assert_array_equal(output, x + y)


@pytest.mark.parametrize("change", ["scalars", "names", "identity", "missing", "ambiguous"])
def test_mismatched_model_call_contract_cannot_select_an_implementation(implementation, change):
    bindings = [KernelBinding("test::kernel", implementation, scalar_arguments={"alpha": 1})]
    source = model()
    if change == "scalars":
        source = model(alpha=2)
    elif change == "names":
        source = model(names='["x", "other"]')
    elif change == "identity":
        source = model(declaration="other::kernel")
    elif change == "missing":
        bindings = []
    else:
        bindings *= 2
    with pytest.raises(ConversionError):
        bind_model(source, bindings)


def test_bindings_cannot_drop_schema_operands_or_results(implementation):
    with pytest.raises(ConversionError, match="every source tensor input"):
        bind_model(model(), [KernelBinding("test::kernel", implementation, operands={"x": "x"})])
    with pytest.raises(ConversionError, match="every source result"):
        bind_model(model(), [KernelBinding("test::kernel", implementation, results=("unknown",))])


@pytest.mark.parametrize("selected_first", (True, False))
def test_selected_specialization_owns_its_operand_correspondence(implementation, selected_first):
    selected = KernelBinding("test::kernel", implementation, scalar_arguments={"alpha": 1})
    unrelated = KernelBinding("other::kernel", implementation, operands={"x": "y", "y": "x"}, scalar_arguments={"alpha": 1})
    bindings = [selected, unrelated] if selected_first else [unrelated, selected]
    bound = bind_model(model(two_calls=True), bindings)
    assert [call["source_operand_bindings"] for call in bound.trace["calls"]] == [{"x": "x", "y": "y"}] * 2


def test_guarded_scratch_counts_each_inlined_call_not_just_model_outputs(tmp_path, implementation):
    path = tmp_path / "source.ttir"
    path.write_text(implementation.source_ttir)
    guarded = convert_shared(path, implementation.spec, tmp_path / "guarded", optimizer=OPT, memory_profile="guarded-buffer")
    bound = bind_model(model(two_calls=True), [KernelBinding("test::kernel", guarded, scalar_arguments={"alpha": 1})])
    entry = next(iter(bound.module.ops))
    assert entry.attributes["prov.memory_guard_scratch_bytes"].value.data == 2 * 32 * 16
    assert bound.trace["guard_storage_bytes"] == 2 * 32 * 16
    if NATIVE:
        x, y = np.ones((4, 8), np.float32), np.full((4, 8), 3, np.float32)
        outputs, _ = execute_native(bound.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
        for output in outputs:
            np.testing.assert_array_equal(output, x + y)


@pytest.mark.skipif(not NATIVE, reason="existing native environment required")
def test_guarded_model_scratch_budget_refuses_before_allocation(tmp_path):
    source = '''module {
      func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32>
        attributes {prov.memory_guard_scratch_bytes = 67108864 : i64} {
          return %x : tensor<1xf32>
        }
    }'''
    with pytest.raises(ValueError, match="native results exceed the 64MiB budget"):
        execute_native(source, [np.ones(1, dtype=np.float32)], tmp_path / "native", python=NATIVE)

"""Moving a call hint adds no numeric or effect permissions."""

import pytest
from xdsl.dialects import builtin
from xdsl.ir import Block, Region

from t2m.call_intent import expose_defined_helpers
from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.package_compatibility import kernel_representation_differences


def helper(hint=None, *, body=True, name="helper"):
    block = Block(arg_types=[builtin.i32])
    block.add_op(builtin.UnregisteredOp.with_name("tt.return").create(operands=[block.args[0]]))
    return builtin.UnregisteredOp.with_name("tt.func").create(
        properties={"sym_name": builtin.StringAttr(name), "function_type": builtin.FunctionType.from_lists([builtin.i32], [builtin.i32])},
        attributes={} if hint is None else {"noinline": hint}, regions=[Region(block) if body else Region()])


@pytest.mark.parametrize("hint", (None, builtin.IntegerAttr(0, builtin.i1)))
def test_absent_and_false_hints_leave_the_exact_program_unchanged(hint):
    module = builtin.ModuleOp([helper(hint)])
    before = text(module)
    assert expose_defined_helpers(module, lambda _: ["original-function"]) == []
    assert text(module) == before


def test_true_hint_retains_complete_body_types_and_descriptive_identity():
    function = helper(builtin.IntegerAttr(1, builtin.i1))
    operations = tuple(function.walk())
    signature = function.properties["function_type"]
    reports = expose_defined_helpers(builtin.ModuleOp([function]), lambda _: ["original-function"])
    assert len(reports) == 1 and reports[0]["function"] == "helper"
    assert reports[0]["source_noinline"] and reports[0]["grants_semantics"] is False
    assert reports[0]["body_source_ids"] == ["original-function"]
    assert "noinline" not in function.attributes
    assert tuple(function.walk()) == operations and function.properties["function_type"] == signature
    notes = kernel_representation_differences({"transformations": reports})
    assert any("physical call boundaries, code size and register pressure" in note for note in notes)
    assert not any("source helpers marked noinline" in note for note in
                   kernel_representation_differences({"transformations": []}))


@pytest.mark.parametrize("bad", (builtin.StringAttr("true"), builtin.IntegerAttr(1, builtin.i32), builtin.UnitAttr()))
def test_bad_semantic_field_type_refuses_before_any_hint_is_changed(bad):
    module = builtin.ModuleOp([helper(builtin.IntegerAttr(1, builtin.i1)), helper(bad, name="other")])
    before = text(module)
    with pytest.raises(ConversionError, match="native Boolean"):
        expose_defined_helpers(module, lambda _: ["original-function"])
    assert text(module) == before


def test_hint_cannot_supply_a_missing_external_implementation():
    module = builtin.ModuleOp([helper(builtin.IntegerAttr(1, builtin.i1), body=False)])
    before = text(module)
    with pytest.raises(ConversionError, match="actual portable function implementation"):
        expose_defined_helpers(module, lambda _: ["original-function"])
    assert text(module) == before

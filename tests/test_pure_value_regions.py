"""Every actual numeric descendant must be pure, including unused instructions."""

import pytest
from test_value_region_contract import source

from t2m.ir import text
from t2m.shared_tensor import _source_ids, opcode, parse_shared
from t2m.value_regions import expand_value_regions, pure_value_region


def root(module):
    return next(op for op in module.walk() if opcode(op) == "tt.reduce")


def test_nested_source_regions_and_existing_standard_expansions_are_functional():
    module = parse_shared(source())
    before = text(module)
    assert pure_value_region(root(module))
    assert text(module) == before
    expand_value_regions(module, _source_ids)
    containers = [op for op in module.walk() if op.name in ("linalg.reduce", "linalg.generic")]
    assert containers and all(pure_value_region(op) for op in containers)


@pytest.mark.parametrize("effect", [
    '"tt.future_operation"() {prov.family="pure"}:()->()',
    '%ok=arith.constant true\ncf.assert %ok,"source assertion"',
    '"func.call"() <{callee=@unknown}> {prov.family="pure"}:()->()',
    '"arith.future_operation"() {prov.family="pure"}:()->()',
    '"tt.print"(%left) <{prefix="observed",hex=false,isSigned=array<i32:1>}>:(i32)->()',
])
def test_parent_purity_and_annotations_never_hide_unused_nested_effects(effect):
    module = parse_shared(source(effect))
    before = text(module)
    assert not pure_value_region(root(module))
    assert text(module) == before


@pytest.mark.parametrize("instruction", [
    '%value="tt.load"(%ptr):(!tt.ptr<i32>)->i32',
    '"tt.store"(%ptr,%left):(!tt.ptr<i32>,i32)->()',
    '%unused="tt.splat"(%ptr):(!tt.ptr<i32>)->tensor<4x!tt.ptr<i32>>',
])
def test_captured_addresses_and_memory_state_cannot_become_numeric_purity(instruction):
    raw = source(instruction).replace('%input:tensor<4xi32>', '%input:tensor<4xi32>,%ptr:!tt.ptr<i32>')
    assert not pure_value_region(root(parse_shared(raw)))


def test_budget_exhaustion_and_unknown_container_properties_grant_no_admission():
    from xdsl.dialects import builtin

    module = parse_shared(source())
    operation = root(module)
    assert not pure_value_region(operation, maximum_operations=1)
    assert pure_value_region(operation)
    operation.attributes["unimplemented_effect"] = builtin.UnitAttr()
    assert not pure_value_region(operation)

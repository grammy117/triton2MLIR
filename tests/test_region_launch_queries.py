"""Logical launch values may move; source control and effects must not move."""

import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.region_launch_queries import normalize_region_launch_queries
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def source(name="first", query="tt.get_program_id", axis=0):
    return f'''func.func @{name}(%n:index) -> i32 {{
      %zero=arith.constant 0:index
      %one=arith.constant 1:index
      scf.for %i=%zero to %n step %one {{
        %id="{query}"() <{{axis={axis}:i32}}>:()->i32 loc("ttir.op.{name}")
        func.call @observe(%id):(i32)->()
      }}
      %count="tt.get_num_programs"() <{{axis=0:i32}}>:()->i32
      return %count:i32
    }}'''


@pytest.mark.parametrize("query", ["tt.get_program_id", "tt.get_num_programs"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_uniform_values_keep_function_scope_locations_uses_and_source_effect_positions(query, axis):
    module = parse_shared("module {" + source(query=query, axis=axis)
                          + source("second") + "func.func private @observe(i32)}")
    original = [op for op in module.walk() if _source_ids(op.location)]
    effects = [op for op in module.walk() if op.name == "func.call"]
    parents = [op.parent for op in effects]
    locations = [op.location for op in original]
    functions = [op for op in module.walk() if op.name == "func.func" and not op.is_declaration]
    reports = normalize_region_launch_queries(module, _source_ids)
    module.verify()
    assert len(reports) == 2
    for operation, function, location, report in zip(original, functions, locations, reports, strict=True):
        assert operation.parent is function.body.block
        assert operation.location == location
        assert report["source_ids"] == _source_ids(location)
        assert report["scope"] == "one-logical-program-invocation"
        assert operation.results[0] in effects[functions.index(function)].operands
    assert [op.parent for op in effects] == parents
    assert all(op.parent_op().name == "scf.for" for op in effects)
    # Already-entry queries stay in their original positions. A second call is
    # inert, without creating new source operations or moving side effects.
    before = text(module)
    assert normalize_region_launch_queries(module, _source_ids) == []
    assert text(module) == before


@pytest.mark.parametrize("invalid", ["axis", "type", "field", "operand"])
def test_invalid_later_query_cannot_move_an_earlier_valid_query(invalid):
    module = parse_shared("module {" + source() + source("second")
                          + "func.func private @observe(i32)}")
    queries = [op for op in module.walk() if _source_ids(op.location)]
    bad = queries[1]
    if invalid == "axis":
        bad.properties["axis"] = builtin.IntegerAttr(3, builtin.i32)
    elif invalid == "type":
        # No native verification or provenance tag may confer a different
        # source query result prototype at this transformation boundary.
        bad.results[0]._type = builtin.i64
    elif invalid == "operand":
        bad.operands = [next(op for op in module.walk() if op.name == "arith.constant").results[0]]
    else:
        bad.attributes["unknown_placement"] = builtin.UnitAttr()
        bad.attributes["prov.family"] = builtin.StringAttr("pure")
    before = text(module)
    with pytest.raises(ConversionError, match="exact i32") as error:
        normalize_region_launch_queries(module, _source_ids)
    assert error.value.source_id == "ttir.op.second"
    assert queries[0].parent_op().name == "scf.for"
    assert text(module) == before


def test_query_relocation_is_discarded_with_a_later_invalid_value_region():
    from test_value_region_contract import source as region_source

    from t2m.value_regions import expand_value_regions

    module = parse_shared(region_source('''%id="tt.get_num_programs"() <{axis=0:i32}>:()->i32
        "tt.future_effect"() {prov.family="pure"}:()->()'''))
    query = next(op for op in module.walk() if opcode(op) == "tt.get_num_programs")
    parent = query.parent
    before = text(module)
    with pytest.raises(ConversionError, match="pure scalar"):
        expand_value_regions(module, _source_ids)
    assert query.parent is parent
    assert text(module) == before

"""Nested region admission is complete, transactional and annotation independent."""

import pytest

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.shared_tensor import _source_ids, opcode, parse_shared
from t2m.value_regions import expand_value_regions


def source(effect=""):
    return f'''module {{func.func @forward(%input:tensor<4xi32>)->i32 {{
      %out="tt.reduce"(%input) <{{axis=0:i32}}> ({{
      ^bb0(%a:i32,%b:i32):
        %tile="tt.splat"(%a):(i32)->tensor<4xi32>
        %scanned="tt.scan"(%tile) <{{axis=0:i32,reverse=false}}> ({{
        ^bb1(%left:i32,%right:i32):
          %v=arith.addi %left,%right:i32
          "tt.scan.return"(%v):(i32)->()
        }}):(tensor<4xi32>)->tensor<4xi32>
        %sum="tt.reduce"(%scanned) <{{axis=0:i32}}> ({{
        ^bb2(%left:i32,%right:i32):
          {effect}
          %v=arith.addi %left,%right:i32
          "tt.reduce.return"(%v):(i32)->()
        }}):(tensor<4xi32>)->i32
        %value=arith.addi %sum,%b:i32
        "tt.reduce.return"(%value):(i32)->()
      }}):(tensor<4xi32>)->i32
      return %out:i32
    }} }}'''


def test_mixed_nesting_uses_one_postorder_without_losing_original_scalar_bodies():
    module = parse_shared(source())
    reductions, scans, maps, tensors = expand_value_regions(module, _source_ids)
    module.verify()
    assert len(reductions) == 2 and len(scans) == 1 and not maps and len(tensors) == 1
    assert not any(opcode(op) in ("tt.reduce", "tt.scan") for op in module.walk())
    outer = next(op for op in module.walk() if op.name == "linalg.reduce")
    assert any(op.name == "linalg.reduce" and op is not outer for op in outer.walk())
    assert any(opcode(op) == "tt.splat" for op in outer.walk())  # Shared supplies its conversion.


@pytest.mark.parametrize("effect", [
    '"tt.future_operation"() {prov.family="pure"}:()->()',
    '%ok=arith.constant true\ncf.assert %ok,"source assertion"',
    '"func.call"() <{callee=@unknown}> {prov.family="pure"}:()->()',
])
def test_an_invalid_nested_body_publishes_none_of_the_prepared_expansion(effect):
    module = parse_shared(source(effect))
    before = text(module)
    with pytest.raises(ConversionError, match="pure scalar"):
        expand_value_regions(module, _source_ids)
    assert text(module) == before


def test_a_later_invalid_map_cannot_publish_an_earlier_valid_container():
    module = parse_shared(source())
    function = next(iter(module.ops))
    original = next(op for op in function.walk() if opcode(op) == "tt.reduce")
    invalid = parse_shared('''module {func.func @bad(%x:tensor<4xi32>)->tensor<4xi32> {
      %out="tt.map_elementwise"(%x) <{pack=1:i32}> ({
      ^bb0(%a:i32):
        "tt.future_operation"() : ()->()
        "tt.map_elementwise.return"(%a):(i32)->()
      }):(tensor<4xi32>)->tensor<4xi32>
      return %out:tensor<4xi32>
    }}''')
    # Put the invalid candidate before the valid reduction. Reversed traversal
    # prepares the valid one first, so refusal must discard actual prior edits.
    bad = next(iter(invalid.ops))
    bad.detach()
    module.body.block.insert_op_before(bad, function)
    assert original.parent is not None
    before = text(module)
    with pytest.raises(ConversionError):
        expand_value_regions(module, _source_ids)
    assert text(module) == before


@pytest.mark.parametrize("property_", ["unqualified_property", "efficient_layout"])
def test_shape_operation_tags_do_not_grant_source_fields(property_):
    from xdsl.dialects import builtin
    module = parse_shared(source())
    splat = next(op for op in module.walk() if opcode(op) == "tt.splat")
    splat.attributes[property_] = builtin.UnitAttr()
    before = text(module)
    with pytest.raises(ConversionError, match="pure scalar"):
        expand_value_regions(module, _source_ids)
    assert text(module) == before


@pytest.mark.parametrize("operation,first,last", [
    ("extsi", "i32", "i64"), ("extui", "i32", "i64"), ("trunci", "i64", "i32"),
])
def test_standard_integer_cast_pattern_retains_original_scalar_operation(operation, first, last):
    module = parse_shared(f'''module {{func.func @forward(%input:tensor<4x{first}>)->tensor<4x{last}> {{
      %out="tt.map_elementwise"(%input) <{{pack=1:i32}}> ({{
      ^bb0(%a:{first}):
        %tile="tt.splat"(%a):({first})->tensor<4x{first}>
        %cast="arith.{operation}"(%tile) {{prov.op="original-cast"}}:(tensor<4x{first}>)->tensor<4x{last}>
        %one=arith.constant 0:index
        %scalar=tensor.extract %cast[%one]:tensor<4x{last}>
        "tt.map_elementwise.return"(%scalar):({last})->()
      }}):(tensor<4x{first}>)->tensor<4x{last}>
      return %out:tensor<4x{last}>
    }} }}''')
    expand_value_regions(module, _source_ids)
    module.verify()
    casts = [op for op in module.walk() if op.name == "arith." + operation]
    assert len(casts) == 1
    assert str(casts[0].operands[0].type) == first and str(casts[0].results[0].type) == last
    assert casts[0].attributes["prov.op"].data == "original-cast"


@pytest.mark.parametrize("serialized", [False, True])
def test_guard_authority_follows_actual_nested_clone_edges_only(serialized):
    from t2m.scalar_tile_reads import expand_scalar_tile_reads

    effect = '''%table=arith.constant dense<7>:tensor<4xi32>
      %zero=arith.constant 0:i32
      %index="tt.splat"(%zero):(i32)->tensor<1xi32>
      %selected="tt.gather"(%table,%index) {axis=0:i32}:(tensor<4xi32>,tensor<1xi32>)->tensor<1xi32>
      %unused="tt.unsplat"(%selected):(tensor<1xi32>)->i32'''
    module = parse_shared(source(effect))
    guards = set()
    expand_scalar_tile_reads(module, _source_ids, generated_guards=guards)
    assert len(guards) == 1
    if serialized:
        copied = parse_shared(text(module))
        before = text(copied)
        with pytest.raises(ConversionError, match="pure scalar"):
            expand_value_regions(copied, _source_ids, generated_guards=guards)
        assert text(copied) == before
    else:
        expand_value_regions(module, _source_ids, generated_guards=guards)
        module.verify()
        assert any(op.name == "cf.assert" for op in module.walk())

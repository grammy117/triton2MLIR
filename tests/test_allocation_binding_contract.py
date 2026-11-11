"""Allocation bindings are explicit typed call operands, independent of data."""

from dataclasses import replace

import pytest
from xdsl.dialects import builtin
from xdsl.ir import Block, Region
from xdsl.parser import Parser

from t2m.allocation_bindings import (
    ARGUMENTS_SCHEMA,
    PointerArgument,
    ScalarArgument,
    arguments_from_dict,
    bind_module,
)
from t2m.diagnostics import ConversionError
from t2m.ir import context
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, opcode


def case():
    ctx = context();ctx.allow_unregistered = True
    pointer = Parser(ctx,"!tt.ptr<i64>").parse_type()
    body = Block(arg_types=[pointer,pointer,builtin.i32])
    body.add_op(builtin.UnregisteredOp.with_name("tt.return").create())
    source = builtin.UnregisteredOp.with_name("tt.func").create(
        properties={"sym_name":builtin.StringAttr("source"),"sym_visibility":builtin.StringAttr("public"),
                    "function_type":builtin.FunctionType.from_lists([arg.type for arg in body.args],[])},
        attributes={"prov.intent":builtin.StringAttr("source body retained")},regions=[Region(body)])
    module = builtin.ModuleOp([source])
    abi = TensorABI("bound",(TensorBuffer("state",(48,),"readwrite","i64"),),runtime_scalars=(
        RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("origin","i64",2,bounds=(-32,63))))
    bindings = (PointerArgument("state"),PointerArgument("state","origin"),ScalarArgument("n"))
    return module,abi,bindings


def test_binding_constructs_real_pointer_sharing_without_mutating_source():
    source,abi,bindings = case()
    result,report = bind_module(source,"source",abi,bindings)
    wrapper,body = list(result.ops)
    call = next(op for op in wrapper.walk() if opcode(op) == "tt.call")
    args = wrapper.regions[0].block.args
    assert call.operands[0] is args[0] and call.operands[2] is args[1]
    assert opcode(call.operands[1].owner) == "tt.addptr"
    assert tuple(call.operands[1].owner.operands) == (args[0],args[2])
    assert body.properties["sym_visibility"] == builtin.StringAttr("private")
    original = next(iter(source.ops))
    assert original.properties["sym_visibility"] == builtin.StringAttr("public")
    assert body.attributes == original.attributes
    assert report["grants_semantics"] is False


def test_equal_width_source_views_use_one_root_before_offset_and_bitcast():
    module, abi, bindings = case()
    abi = replace(abi, buffers=(replace(abi.buffers[0], dtype="f64"),))
    result, report = bind_module(module, "source", abi, bindings)
    wrapper = next(iter(result.ops))
    assert str(wrapper.regions[0].block.args[0].type) == "!tt.ptr<f64>"
    call = next(op for op in wrapper.walk() if opcode(op) == "tt.call")
    first, second = call.operands[:2]
    assert opcode(first.owner) == opcode(second.owner) == "tt.bitcast"
    assert first.owner.operands[0] is wrapper.regions[0].block.args[0]
    offset = second.owner.operands[0]
    assert opcode(offset.owner) == "tt.addptr" and str(offset.type) == "!tt.ptr<f64>"
    assert report["bindings"][0]["source_type"] == "!tt.ptr<i64>"
    assert report["bindings"][0]["representation"] == "equal-width-word-view-v1"


@pytest.mark.parametrize("source_type", ("tensor<2x!tt.ptr<i64>>", "!tt.ptr<i64, 3>", "!tt.ptr<i1>"))
def test_alias_binding_does_not_grant_tensor_pointer_address_space_or_padding_views(source_type):
    from t2m.shared_tensor import parse_shared
    module = parse_shared(f'''builtin.module {{ "tt.func"() <{{sym_name="source", function_type=({source_type})->()}}> ({{
      ^bb0(%p:{source_type}): "tt.return"():()->() }}) : ()->() }}''')
    abi = TensorABI("bound", (TensorBuffer("state", (48,), "readwrite", "i64"),))
    with pytest.raises(ConversionError, match="equal byte-aligned"):
        bind_module(module, "source", abi, (PointerArgument("state"),))


@pytest.mark.parametrize("binding,message",[
    (PointerArgument("missing"),"declared allocation"),
    (PointerArgument("state","missing"),"declared i32/i64"),
    (PointerArgument("state",1.5),"declared i32/i64"),
    (PointerArgument("state",True),"declared i32/i64"),
    (PointerArgument("state",2**63),"signed i64"),
    (ScalarArgument("missing"),"exact declared"),
    (ScalarArgument("origin"),"exact declared"),
    (2**31,"signed source type"),
    (True,"explicit typed"),
])
def test_invalid_pointer_scalar_and_constant_bindings_refuse(binding,message):
    module,abi,bindings = case()
    index = 2 if isinstance(binding,(ScalarArgument,int)) and not isinstance(binding,PointerArgument) else 0
    changed = list(bindings);changed[index] = binding
    with pytest.raises(ConversionError,match=message):
        bind_module(module,"source",abi,changed)


def test_incomplete_source_roster_and_type_changes_refuse():
    module,abi,bindings = case()
    with pytest.raises(ConversionError,match="complete void source signature"):
        bind_module(module,"source",abi,bindings[:-1])
    with pytest.raises(ConversionError,match="same element type"):
        bind_module(module,"source",replace(abi,buffers=(replace(abi.buffers[0],dtype="f32"),)),bindings)
    with pytest.raises(ConversionError,match="every declared allocation"):
        bind_module(module,"source",replace(abi,buffers=(*abi.buffers,TensorBuffer("other",(48,),"readwrite","i64"))),bindings)
    with pytest.raises(ConversionError,match="fresh"):
        bind_module(module,"source",replace(abi,entry="source"),bindings)


@pytest.mark.parametrize("origin",[-7,0,11])
def test_signed_constant_origins_and_specialized_source_scalars_are_executable(origin):
    module,abi,bindings = case()
    result,_ = bind_module(module,"source",abi,(PointerArgument("state",origin),bindings[1],7))
    wrapper = next(iter(result.ops))
    call = next(op for op in wrapper.walk() if opcode(op) == "tt.call")
    constant = call.operands[2].owner
    assert constant.value.value.data == 7 and constant.result.type == builtin.i32
    if origin:
        assert call.operands[0].owner.operands[1].owner.value.value.data == origin
    else:
        assert call.operands[0] is wrapper.regions[0].block.args[0]


def test_saved_argument_configuration_has_exact_source_order_and_no_ignored_fields():
    value = {"schema":ARGUMENTS_SCHEMA,"source_entry":"source","arguments":[
        {"kind":"pointer","allocation":"state","offset":"origin"},
        {"kind":"scalar","name":"n"},{"kind":"constant","value":7}]}
    entry,arguments = arguments_from_dict(value)
    assert entry == "source" and arguments == (PointerArgument("state","origin"),ScalarArgument("n"),7)
    for bad in [{**value,"ignored":True},{**value,"schema":"unknown"},
                {**value,"arguments":[None]},
                {**value,"arguments":[{"kind":"constant","value":True}]},
                {**value,"arguments":[{**value["arguments"][0],"ignored":True}]},
                {**value,"arguments":[{"kind":"unknown"}]}]:
        with pytest.raises(ValueError):
            arguments_from_dict(bad)

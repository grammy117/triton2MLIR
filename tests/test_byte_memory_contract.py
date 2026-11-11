"""Protocol fixtures for byte views; real execution is tested separately."""

import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.pointer_control import normalize_pointer_control
from t2m.shared_tensor import _source_ids, opcode, parse_shared
from t2m.word_events import print_ir


def source(root="i8", view="i32", atomic=False):
    event = (
        f'%r="tt.atomic_rmw"(%p,%one):(!tt.ptr<{view}>,{view})->{view}'
        if atomic
        else f'%r="tt.load"(%p)<{{operandSegmentSizes=array<i32:1,0,0>}}>:(!tt.ptr<{view}>)->{view}'
    )
    return parse_shared(f"""builtin.module {{"tt.func"()<{{sym_name="source",function_type=(!tt.ptr<{root}>)->()}}>({{
      ^bb0(%base:!tt.ptr<{root}>):
        %origin=arith.constant 3:i32
        %base3="tt.addptr"(%base,%origin):(!tt.ptr<{root}>,i32)->!tt.ptr<{root}>
        %cast="tt.bitcast"(%base3):(!tt.ptr<{root}>)->!tt.ptr<{view}> loc("ttir.op.cast")
        %offset=arith.constant 2:i32
        %p="tt.addptr"(%cast,%offset):(!tt.ptr<{view}>,i32)->!tt.ptr<{view}>
        %one=arith.constant 1:{view}
        {event} loc("ttir.op.event")
        "tt.return"():()->()
    }}):()->()}}""")


@pytest.mark.parametrize("order", ("little", "big"))
def test_one_original_word_load_has_one_guardable_byte_plane(order):
    module = source()
    reports = normalize_pointer_control(module, _source_ids, byte_order=order)
    report = next(r for r in reports if r["kind"] == "logical-byte-buffer-word-event")
    assert report["source_ids"] == ["ttir.op.event"]
    assert report["byte_tile"] == [4] and report["source_tile"] == []
    loads = [op for op in module.walk() if opcode(op) == "tt.load"]
    assert len(loads) == 1 and loads[0].results[0].type == builtin.TensorType(
        builtin.i8, (4,)
    )
    assert all(
        "!tt.ptr<i32>" not in str(v.type) for op in module.walk() for v in op.results
    )


@pytest.mark.parametrize("order", ("native", "host", False, 1))
def test_unknown_byte_order_refuses_without_edits(order):
    module = source()
    before = print_ir(module)
    with pytest.raises(ValueError, match="byte order"):
        normalize_pointer_control(module, _source_ids, byte_order=order)
    assert print_ir(module) == before


@pytest.mark.parametrize(
    "root,view,order,message",
    (
        ("i8", "i32", None, "equal byte-aligned"),
        ("f32", "i16", "little", "i8 allocation roots"),
        ("i1", "i16", "little", "supported scalar|i8 allocation roots"),
    ),
)
def test_other_storage_contracts_are_not_inferred(root, view, order, message):
    module = source(root, view)
    before = print_ir(module)
    with pytest.raises(ConversionError, match=message) as error:
        normalize_pointer_control(module, _source_ids, byte_order=order)
    assert error.value.source_id == "ttir.op.cast" and print_ir(module) == before


def test_byte_backed_atomic_cannot_become_separate_byte_effects():
    module = source(atomic=True)
    before = print_ir(module)
    with pytest.raises(ConversionError, match="indivisible word-effect") as error:
        normalize_pointer_control(
            module, _source_ids, byte_order="little", typed_atomics=True
        )
    assert error.value.source_id == "ttir.op.event" and print_ir(module) == before

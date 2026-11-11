"""Current standard byte-atomic bodies must match their complete recipe."""

import os

import pytest
from xdsl.dialects import arith, builtin, cf, memref
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.rewriter import Rewriter

from t2m.byte_atomic_events import view_info
from t2m.byte_atomic_match import match
from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.standard_memref import GenericAtomicRMWOp
from t2m.upstream import lower_with_shared

OPT = os.environ.get("TRITON_SHARED_OPT")


@pytest.mark.parametrize("width,operation", ((32, "add"), (64, "xor"), (32, "exch"), (16, "cas")))
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("native", (False, True))
def test_complete_recipe_matches_actual_standard_and_upstream_bodies(tmp_path, width, operation, order, native):
    from test_byte_atomic_words import module

    from t2m.buffer_memory import parse_memory

    program = module(width, operation, order)
    if native:
        if not OPT:
            pytest.skip("existing pinned shared optimizer required")
        path = tmp_path / "source.mlir"
        path.write_text(text(program))
        lower_with_shared(path, tmp_path / "native", optimizer=OPT, stage="memory-effects")
        program = parse_memory((tmp_path / "native/lowered.mlir").read_text())
    generics = [op for op in program.walk() if isinstance(op, GenericOp) and "prov.byte_word_atomic" in op.attributes]
    assert len(generics) == 2
    for generic in generics:
        match(generic, {"operation": operation, "byte_view": view_info(generic.results[0].type.element_type, order)}, masked=True)


@pytest.mark.parametrize("change", ("update", "old", "alignment", "address", "capacity", "literal", "probe-assert", "dead", "order"))
def test_metadata_cannot_authorize_changed_atomic_codec_or_probe(change):
    from test_byte_atomic_words import module

    program = module(32, "add", "little")
    generic = next(op for op in program.walk() if isinstance(op, GenericOp) and "prov.byte_word_atomic" in op.attributes)
    atomic = next(op for op in generic.walk() if isinstance(op, GenericAtomicRMWOp))
    info = {"operation": "add", "byte_view": view_info(builtin.i32, "little")}
    if change == "update":
        update = next(op for op in atomic.walk() if isinstance(op, arith.AddiOp))
        Rewriter.replace_op(update, arith.XOrIOp(*update.operands))
    elif change == "old":
        branch = generic.body.block.last_op.operands[0].owner
        branch.true_region.block.last_op.operands = [atomic.result]
    elif change == "alignment":
        assertion = next(op for op in generic.walk() if isinstance(op, cf.AssertOp))
        assertion.parent.erase_op(assertion)
    elif change == "address":
        viewed = atomic.memref.owner
        one = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        viewed.parent.insert_op_before(one, viewed)
        viewed.operands = [viewed.source, one.result, *viewed.sizes]
    elif change == "capacity":
        viewed = atomic.memref.owner
        one = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        viewed.parent.insert_op_before(one, viewed)
        viewed.operands = [viewed.source, viewed.byte_shift, one.result]
    elif change == "literal":
        probe = next(op for op in program.walk() if isinstance(op, memref.AllocOp)
                     and op.memref.type == builtin.MemRefType(builtin.i8, [4]))
        store = next(use.operation for use in probe.memref.uses if isinstance(use.operation, memref.StoreOp))
        two = arith.ConstantOp.from_int_and_width(2, builtin.i8)
        store.parent.insert_op_before(two, store)
        store.operands = [two.result, *store.operands[1:]]
    elif change == "probe-assert":
        assertion = next(op for op in program.walk() if isinstance(op, cf.AssertOp)
                         and op.msg.data == "Unsupported standard integer byte representation")
        assertion.parent.erase_op(assertion)
    elif change == "dead":
        extra = arith.AddiOp(atomic.body.block.args[0], atomic.body.block.args[0])
        atomic.body.block.insert_op_before(extra, atomic.body.block.last_op)
    else:
        info["byte_view"]["byte_order"] = "big"
    # Tags are deliberately unchanged; they cannot grant the modified recipe.
    with pytest.raises(ConversionError, match="complete typed word/byte recipe"):
        match(generic, info, masked=True)


def test_executable_recipe_does_not_require_provenance_tags():
    from test_byte_atomic_words import module

    program = module(32, "add", "little")
    generics = [op for op in program.walk() if isinstance(op, GenericOp) and "prov.byte_word_atomic" in op.attributes]
    for op in program.walk():
        for key in list(op.attributes):
            if key.startswith("prov."):
                del op.attributes[key]
    for generic in generics:
        match(generic, {"operation": "add", "byte_view": view_info(builtin.i32, "little")}, masked=True)

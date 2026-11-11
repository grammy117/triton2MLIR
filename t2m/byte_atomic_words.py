"""Standard word atomics over byte storage, with executable byte order.

LLVM's MemRefOps.td defines a pure scalar update region on one integer word;
MemRefToLLVM supplies its compare/exchange lowering. A standard memref.view
aliases the bytes. A private literal probe establishes that view's byte order,
so encoding inside the atomic region is independent of the selected target.
This builder does not by itself admit a TTIR event or its ownership effects.
"""

import json

from xdsl.dialects import arith, builtin, cf, memref, scf, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from .atomic_word_views import UPDATES
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp


def probe(word_type, append):
    """Establish local standard integer storage using only private literal bytes."""
    if word_type not in (builtin.i16, builtin.i32, builtin.i64):
        raise ValueError("byte word atomics require integer i16/i32/i64 words")
    size = word_type.bitwidth // 8

    def emit(op):
        op.attributes["prov.byte_order_probe"] = builtin.StringAttr(
            f"private-literal-i{word_type.bitwidth}"
        )
        append(op)
        return op.results[0] if op.results else None

    def constant(n, type_):
        return emit(arith.ConstantOp.from_int_and_width(n, type_))

    raw = emit(memref.AllocOp.get(builtin.i8, shape=[size], alignment=8))
    zero = constant(0, builtin.IndexType())
    for byte in range(size):
        index = zero if byte == 0 else constant(byte, builtin.IndexType())
        emit(memref.StoreOp.get(constant(int(byte == 0), builtin.i8), raw, [index]))
    word = emit(memref.ViewOp(raw, zero, [], builtin.MemRefType(word_type, [1])))
    loaded = emit(memref.LoadOp.get(word, [zero]))
    little = emit(arith.CmpiOp(loaded, constant(1, word_type), "eq"))
    big = emit(arith.CmpiOp(loaded, constant(1 << (8 * (size - 1)), word_type), "eq"))
    valid = emit(arith.OrIOp(little, big))
    emit(cf.AssertOp(valid, "Unsupported standard integer byte representation"))
    return little


def encode(value, little, byte_order, append):
    """Involutive local-word/logical-word transport, with explicit source order."""
    if (
        value.type not in (builtin.i16, builtin.i32, builtin.i64)
        or little.type != builtin.i1
        or byte_order not in ("little", "big")
    ):
        raise ValueError("byte word atomics require explicit little or big byte order")
    type_ = value.type
    size = type_.bitwidth // 8

    def emit(op):
        append(op)
        return op.results[0] if op.results else None

    def constant(n):
        return emit(arith.ConstantOp.from_int_and_width(n, type_))

    reversed_ = constant(0)
    mask = constant(255)
    for byte in range(size):
        piece = value if byte == 0 else emit(arith.ShRUIOp(value, constant(byte * 8)))
        piece = emit(arith.AndIOp(piece, mask))
        shift = (size - byte - 1) * 8
        if shift:
            piece = emit(arith.ShLIOp(piece, constant(shift)))
        reversed_ = emit(arith.OrIOp(reversed_, piece))
    first, second = (value, reversed_) if byte_order == "little" else (reversed_, value)
    return emit(arith.SelectOp(little, first, second))


def event(
    storage,
    offsets,
    values,
    little,
    append,
    *,
    operation,
    byte_order,
    mask=None,
    desired=None,
):
    """One indivisible aligned word update per active original lane.

    Offsets are signed i64 byte addresses. Bounds cover the complete word;
    inactive lanes neither inspect their addresses nor execute an atomic.
    Whole-event source ownership/initialization admission remains a caller gate.
    The caller supplies private byte storage aligned for the selected word type
    and runs the normal ownership/deallocation pipeline for the literal probe.
    """
    if (
        not isinstance(storage.type, builtin.MemRefType)
        or storage.type.element_type != builtin.i8
        or len(storage.type.get_shape()) != 1
        or storage.type.get_offset() != 0
    ):
        raise ValueError("byte atomics require a rank-one zero-offset byte allocation")
    type_ = values.type
    if (
        not isinstance(type_, builtin.TensorType)
        or type_.element_type not in (builtin.i16, builtin.i32, builtin.i64)
        or offsets.type != builtin.TensorType(builtin.i64, type_.get_shape())
        or mask is not None
        and mask.type != builtin.TensorType(builtin.i1, type_.get_shape())
        or operation not in (*UPDATES, "exch", "cas")
        or (desired is not None) != (operation == "cas")
        or desired is not None
        and desired.type != type_
        or little.type != builtin.i1
        or byte_order not in ("little", "big")
    ):
        raise ValueError(
            "byte atomic operation differs from its typed word/offset/mask schema"
        )
    word_type, shape = type_.element_type, type_.get_shape()
    size = word_type.bitwidth // 8

    def emit(op):
        append(op)
        return op.results[0] if op.results else None

    zero = emit(arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
    extent = emit(memref.DimOp.from_source_and_index(storage, zero))
    size_index = emit(arith.ConstantOp.from_int_and_width(size, builtin.IndexType()))
    capacity = emit(arith.DivUIOp(extent, size_index))
    viewed = emit(
        memref.ViewOp(
            storage,
            zero,
            [capacity],
            builtin.MemRefType(word_type, [builtin.DYNAMIC_INDEX]),
        )
    )
    extent64 = emit(arith.IndexCastOp(extent, builtin.i64))
    dynamic = []
    for axis, bound in enumerate(shape):
        if bound == builtin.DYNAMIC_INDEX:
            index = emit(arith.ConstantOp.from_int_and_width(axis, builtin.IndexType()))
            dynamic.append(emit(tensor.DimOp(values, index)))
    empty = emit(tensor.EmptyOp(dynamic, type_))
    inputs = [
        offsets,
        values,
        *([desired] if desired is not None else []),
        *([mask] if mask is not None else []),
    ]
    block = Block(arg_types=[v.type.element_type for v in inputs] + [word_type])
    active = Block() if mask is not None else block

    def scalar(op):
        active.add_op(op)
        return op.results[0] if op.results else None

    def constant(n, type_=builtin.i64):
        return scalar(arith.ConstantOp.from_int_and_width(n, type_))

    factor = constant(size)
    aligned = scalar(
        arith.CmpiOp(scalar(arith.RemSIOp(block.args[0], factor)), constant(0), "eq")
    )
    scalar(cf.AssertOp(aligned, "Triton byte atomic address is not word aligned"))
    nonnegative = scalar(arith.CmpiOp(block.args[0], constant(0), "sge"))
    limit = scalar(arith.SubiOp(extent64, factor))
    within = scalar(arith.CmpiOp(block.args[0], limit, "sle"))
    scalar(
        cf.AssertOp(
            scalar(arith.AndIOp(nonnegative, within)),
            "Triton active memory index outside logical allocation",
        )
    )
    index = scalar(
        arith.IndexCastOp(
            scalar(arith.DivSIOp(block.args[0], factor)), builtin.IndexType()
        )
    )
    update = Block(arg_types=[word_type])
    logical = (
        encode(update.args[0], little, byte_order, update.add_op)
        if operation != "exch"
        else None
    )
    if operation == "cas":
        equal = arith.CmpiOp(logical, block.args[1], "eq")
        result = arith.SelectOp(equal.result, block.args[2], logical)
        update.add_ops([equal, result])
        changed = result.result
    elif operation == "exch":
        changed = block.args[1]
    else:
        result = UPDATES[operation](logical, block.args[1])
        update.add_op(result)
        changed = result.result
    stored = encode(changed, little, byte_order, update.add_op)
    update.add_op(AtomicYieldOp.create(operands=[stored]))
    atomic = GenericAtomicRMWOp.create(
        operands=[viewed, index], result_types=[word_type], regions=[Region(update)]
    )
    raw_old = scalar(atomic)
    old = encode(raw_old, little, byte_order, active.add_op)
    if mask is not None:
        active.add_op(scf.YieldOp(old))
        inactive = Block()
        ignored = arith.ConstantOp.from_int_and_width(0, word_type)
        inactive.add_ops([ignored, scf.YieldOp(ignored.result)])
        branch = scf.IfOp(block.args[-2], [word_type], Region(active), Region(inactive))
        block.add_op(branch)
        old = branch.results[0]
    block.add_op(YieldOp(old))
    generic = GenericOp(
        inputs,
        [empty],
        Region(block),
        [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * (len(inputs) + 1),
        [IteratorTypeAttr.parallel()] * len(shape),
        [type_],
    )
    generic.attributes["prov.byte_word_atomic"] = builtin.StringAttr(
        json.dumps(
            {
                "schema": "t2m.byte-word-atomic-intent.v1",
                "operation": operation,
                "word_bits": word_type.bitwidth,
                "byte_order": byte_order,
                "address_units": "signed-i64-bytes",
                "alignment_bytes": size,
                "atomicity": "one-standard-word-region-per-active-lane",
                "grants_semantics": False,
            },
            sort_keys=True,
        )
    )
    atomic.attributes["prov.byte_word_atomic"] = generic.attributes[
        "prov.byte_word_atomic"
    ]
    return emit(generic)

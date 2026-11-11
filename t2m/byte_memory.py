"""Logical byte-buffer word events with explicit byte order and one byte plane.

Ascend 8f0ade66 CommonPtrRules.cpp and PtrAnalysisBitcastTest.cpp establish
the source-pointee unit rule. Keep logical byte addresses rather than adopting
its physical pointer conversions. Meta's existing native event conversion and
guarded memory then handle the complete byte plane at each original event.
"""

import json

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def expand_byte_events(state, source_ids, byte_order, *, byte_atomics=False):
    from .pointer_views import element
    from .shared_tensor import _array, opcode
    from .word_events import MEMORY_EVENTS

    if byte_order not in ("little", "big"):
        raise ValueError("byte-buffer views require explicit little or big byte order")
    plans = []
    for op in state.operations:
        if opcode(op) not in MEMORY_EVENTS or not state.byte_roots.intersection(
            state.bases[op.operands[0]]
        ):
            continue
        view = element(state.types[op.operands[0]])
        if view.bitwidth == 8:
            continue

        def refuse(message, op=op):
            raise ConversionError(
                message,
                opcode=opcode(op),
                source_id=next(iter(source_ids(op.location)), None),
            )

        if opcode(op) not in ("tt.load", "tt.store"):
            if byte_atomics:
                from .atomic_memory import contract
                from .byte_atomic_events import view_info
                contract(op, source_ids, typed=True)
                try:
                    view_info(view, byte_order)
                except ValueError:
                    refuse("byte atomics require an integer word numeric profile")
                continue
            refuse(
                "different-width byte atomics require an indivisible word-effect implementation"
            )
        if opcode(op) == "tt.load":
            segments = _array(op, "operandSegmentSizes")
            if (
                len(segments) != 3
                or segments[0] != 1
                or sum(segments) != len(op.operands)
            ):
                refuse(
                    "byte-buffer loads require their complete mask/fill operand schema"
                )
        shape = (
            state.types[op.operands[0]].get_shape()
            if isinstance(state.types[op.operands[0]], builtin.TensorType)
            else ()
        )
        if any(n <= 0 for n in shape):
            refuse("byte-buffer event tiles require static positive extents")
        roots = state.bases[op.operands[0]]
        if any(element(root.type) != builtin.i8 for root in roots):
            refuse("byte-buffer events require actual i8 owning storage")
        plans.append((op, view, tuple(shape), next(iter(roots)).type))
    return [
        _expand_event(
            op,
            view,
            shape,
            pointer,
            state.bases[op.operands[0]],
            source_ids,
            byte_order,
        )
        for op, view, shape, pointer in plans
    ]


def _expand_event(op, view, shape, byte_pointer, roots, source_ids, byte_order):
    from .shared_tensor import _array, opcode

    size = view.bitwidth // 8
    plane_shape = (*shape, size)
    word_type = builtin.IntegerType(view.bitwidth)
    plane_type = builtin.TensorType(builtin.i8, plane_shape)
    emitted = []

    def emit(item):
        item.location = op.location
        emitted.append(item)
        return item.results[0] if item.results else None

    def tt(name, operands, type_, properties=None):
        return emit(
            builtin.UnregisteredOp.with_name(name).create(
                operands=operands, result_types=[type_], properties=properties or {}
            )
        )

    def broadcast(value):
        if not shape:
            return tt("tt.splat", [value], builtin.TensorType(value.type, plane_shape))
        expanded = tt(
            "tt.expand_dims",
            [value],
            builtin.TensorType(value.type.element_type, (*shape, 1)),
            {"axis": builtin.IntegerAttr(len(shape), builtin.i32)},
        )
        return tt(
            "tt.broadcast",
            [expanded],
            builtin.TensorType(value.type.element_type, plane_shape),
        )

    pointer_type = builtin.TensorType(byte_pointer, shape) if shape else byte_pointer
    cast = tt("tt.bitcast", [op.operands[0]], pointer_type)
    pointer = broadcast(cast)
    lanes = tt(
        "tt.make_range",
        [],
        builtin.TensorType(builtin.i32, (size,)),
        {
            "start": builtin.IntegerAttr(0, builtin.i32),
            "end": builtin.IntegerAttr(size, builtin.i32),
        },
    )
    for _ in shape:
        lanes = tt(
            "tt.expand_dims",
            [lanes],
            builtin.TensorType(builtin.i32, (1, *lanes.type.get_shape())),
            {"axis": builtin.IntegerAttr(0, builtin.i32)},
        )
    if shape:
        lanes = tt(
            "tt.broadcast", [lanes], builtin.TensorType(builtin.i32, plane_shape)
        )
    pointer = tt(
        "tt.addptr", [pointer, lanes], builtin.TensorType(byte_pointer, plane_shape)
    )
    load = opcode(op) == "tt.load"
    segments = _array(op, "operandSegmentSizes") if load else None
    mask = (
        op.operands[1]
        if load and segments[1]
        else op.operands[2]
        if not load and len(op.operands) == 3
        else None
    )
    plane_mask = broadcast(mask) if mask is not None else None
    info = {
        "kind": "logical-byte-buffer-word-event",
        "source_ids": source_ids(op.location),
        "opcode": opcode(op),
        "byte_order": byte_order,
        "value_dtype": str(view),
        "word_bits": view.bitwidth,
        "source_tile": list(shape),
        "byte_tile": list(plane_shape),
        "roots": sorted(root.index for root in roots),
        "expansion": "single-byte-plane-transfer-and-standard-word-pack",
        "address_units": "bytes-after-source-pointee-scaled-addptr",
        "grants_semantics": False,
    }
    marker = builtin.StringAttr(json.dumps(info, sort_keys=True))
    if load:
        properties = dict(op.properties)
        properties["operandSegmentSizes"] = builtin.DenseArrayBase.from_list(
            builtin.i32, [1, int(mask is not None), int(mask is not None)]
        )
        operands = [pointer]
        if plane_mask is not None:
            zero = emit(
                arith.ConstantOp(
                    builtin.DenseIntOrFPElementsAttr.from_list(plane_type, [0])
                )
            )
            operands.extend((plane_mask, zero))
        loaded = emit(
            type(op).create(
                operands=operands,
                result_types=[plane_type],
                properties=properties,
                attributes=dict(op.attributes),
            )
        )
        body = Block(arg_types=[view]) if shape else None

        def scalar(item):
            item.location = op.location
            if body is None:
                return emit(item)
            body.add_op(item)
            return item.results[0] if item.results else None

        coordinates = [scalar(IndexOp(axis)) for axis in range(len(shape))]
        word = scalar(arith.ConstantOp.from_int_and_width(0, word_type))
        for byte in range(size):
            index = scalar(
                arith.ConstantOp.from_int_and_width(byte, builtin.IndexType())
            )
            piece = scalar(tensor.ExtractOp(loaded, [*coordinates, index], builtin.i8))
            piece = scalar(arith.ExtUIOp(piece, word_type))
            shift = (byte if byte_order == "little" else size - byte - 1) * 8
            if shift:
                amount = scalar(arith.ConstantOp.from_int_and_width(shift, word_type))
                piece = scalar(arith.ShLIOp(piece, amount))
            word = scalar(arith.OrIOp(word, piece))
        if view != word_type:
            word = scalar(arith.BitcastOp(word, view))
        if shape:
            scalar(YieldOp(word))
            type_ = builtin.TensorType(view, shape)
            empty = emit(tensor.EmptyOp([], type_))
            packed = GenericOp(
                [],
                [empty],
                Region(body),
                [builtin.AffineMapAttr(AffineMap.identity(len(shape)))],
                [IteratorTypeAttr.parallel()] * len(shape),
                [type_],
            )
            packed.attributes["prov.byte_word_event"] = marker
            word = emit(packed)
        if mask is not None and segments[2]:
            word = emit(arith.SelectOp(mask, word, op.operands[2]))
        Rewriter.replace_op(op, emitted, [word])
    else:
        body = Block(arg_types=[builtin.i8])

        def scalar(item):
            item.location = op.location
            body.add_op(item)
            return item.results[0] if item.results else None

        coordinates = [scalar(IndexOp(axis)) for axis in range(len(plane_shape))]
        word = (
            scalar(tensor.ExtractOp(op.operands[1], coordinates[:-1], view))
            if shape
            else op.operands[1]
        )
        if view != word_type:
            word = scalar(arith.BitcastOp(word, word_type))
        index = scalar(arith.IndexCastOp(coordinates[-1], word_type))
        if byte_order == "big":
            last = scalar(arith.ConstantOp.from_int_and_width(size - 1, word_type))
            index = scalar(arith.SubiOp(last, index))
        eight = scalar(arith.ConstantOp.from_int_and_width(8, word_type))
        shift = scalar(arith.MuliOp(index, eight))
        byte = scalar(arith.ShRUIOp(word, shift))
        byte = scalar(arith.TruncIOp(byte, builtin.i8))
        scalar(YieldOp(byte))
        empty = emit(tensor.EmptyOp([], plane_type))
        unpacked = GenericOp(
            [],
            [empty],
            Region(body),
            [builtin.AffineMapAttr(AffineMap.identity(len(plane_shape)))],
            [IteratorTypeAttr.parallel()] * len(plane_shape),
            [plane_type],
        )
        unpacked.attributes["prov.byte_word_event"] = marker
        values = emit(unpacked)
        operands = [
            pointer,
            values,
            *([plane_mask] if plane_mask is not None else []),
        ]
        emit(
            type(op).create(
                operands=operands,
                properties=dict(op.properties),
                attributes=dict(op.attributes),
            )
        )
        Rewriter.replace_op(op, emitted)
    return info

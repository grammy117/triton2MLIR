"""Logical word views of byte-aligned scalar storage, before memory lowering.

Meta 17f0c234 TritonToUnstructuredPass.cpp propagates offsets through pointer
bitcasts when storage strides agree (bitcast_ptr_unstructured.mlir). Reuse that
base/offset boundary, but keep our public storage element and bitcast each
loaded/stored value. This avoids physical pointer casts and whole-buffer copies.
Sub-byte types, changed strides and differently typed selected roots need a
separate address/representation contract.
"""

from functools import lru_cache

from xdsl.dialects import arith, builtin
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import ParseError

from .diagnostics import ConversionError


@lru_cache(maxsize=128)
def element(pointer):
    from .ir import context
    from .shared_tensor import DTYPES, SharedParser, _pointer_type
    scalar = pointer.element_type if isinstance(pointer, builtin.TensorType) else pointer
    if not _pointer_type(scalar):
        raise ValueError("expected a TT scalar pointer type")
    # The pinned binding represents TT pointer parameters as an unregistered
    # attribute body. Parse that type parameter structurally with the same
    # parser; canonical equality also excludes explicit address-space suffixes.
    result = SharedParser(context(), scalar.value.data).parse_type()
    if result not in DTYPES.values() or scalar.value.data != str(result) or result.bitwidth % 8:
        raise ValueError("pointer views require a supported byte-aligned scalar and implicit logical address space")
    return result


def compatible(left, right):
    try:
        return element(left).bitwidth == element(right).bitwidth
    except (ValueError, ParseError):
        # Parser diagnostics are not source-located; the caller supplies its
        # original pointer operation and ID. No fallback interpretation follows.
        return False


def normalize(state, source_ids):
    from .shared_tensor import _array, opcode
    from .word_events import MEMORY_EVENTS
    def refuse(op, message):
        raise ConversionError(message, opcode=opcode(op), source_id=next(iter(source_ids(op.location)), None))
    plan = []
    for op in state.operations:
        if opcode(op) not in MEMORY_EVENTS:
            continue
        pointer = op.operands[0]
        roots = state.bases[pointer]
        scalar = state.types[pointer].element_type if isinstance(state.types[pointer], builtin.TensorType) else state.types[pointer]
        root, *others = sorted(roots, key=lambda v: v.index)
        if any(other.type != root.type for other in others):
            refuse(op, "selected pointer views need one logical storage element type across all roots")
        if scalar == root.type:
            continue
        if opcode(op) not in ("tt.load", "tt.store"):
            refuse(op, "atomic word views require a separately qualified typed atomic region expansion")
        if not compatible(scalar, root.type):
            refuse(op, "pointer views require equal byte-aligned element widths without address-space changes")
        fill = None
        if opcode(op) == "tt.load":
            segments = _array(op, "operandSegmentSizes")
            if len(segments) != 3 or sum(segments) != len(op.operands):
                refuse(op, "logical word load requires explicit mask/fill segments")
            if segments[2]:
                fill = len(op.operands) - 1
        plan.append((op, root, element(scalar), element(root.type), fill))
    reports = []
    for op, root, view, storage, fill in plan:
        def cast(value, element_, before=True):
            type_ = builtin.TensorType(element_, value.type.get_shape()) if isinstance(value.type, builtin.TensorType) else element_
            result = arith.BitcastOp(value, type_)
            result.location = op.location
            (op.parent.insert_op_before if before else op.parent.insert_op_after)(result, op)
            return result.result
        if opcode(op) == "tt.store":
            op.operands = [op.operands[0], cast(op.operands[1], storage), *op.operands[2:]]
        else:
            original_fill = original_mask = None
            if fill is not None:
                original_fill, original_mask = op.operands[fill], op.operands[1]
                # An integer fill may encode a signaling NaN in the storage
                # type. Keep those bits in the value domain rather than letting
                # native constant folding quiet/canonicalize a floating literal.
                zero = (arith.ConstantOp(builtin.FloatAttr(0.0, storage)) if isinstance(storage, builtin.AnyFloat)
                        else arith.ConstantOp.from_int_and_width(0, storage))
                zero.location = op.location
                op.parent.insert_op_before(zero, op)
                value = zero.result
                if isinstance(original_fill.type, builtin.TensorType):
                    splat = builtin.UnregisteredOp.with_name("tt.splat").create(operands=[value],
                        result_types=[builtin.TensorType(storage, original_fill.type.get_shape())], location=op.location)
                    op.parent.insert_op_before(splat, op)
                    value = splat.results[0]
                operands = list(op.operands)
                operands[fill] = value
                op.operands = operands
            old = op.results[0]
            type_ = builtin.TensorType(storage, old.type.get_shape()) if isinstance(old.type, builtin.TensorType) else storage
            loaded = Rewriter.replace_value_with_new_type(old, type_)
            restored = cast(loaded, view, before=False)
            loaded.replace_uses_with_if(restored, lambda use: use.operation is not restored.owner)
            if original_fill is not None:
                selected = arith.SelectOp(original_mask, restored, original_fill)
                selected.location = op.location
                op.parent.insert_op_after(selected, restored.owner)
                restored.replace_uses_with_if(selected.result, lambda use: use.operation is not selected)
        reports.append({"kind": "logical-pointer-word-view", "source_ids": source_ids(op.location),
            "opcode": opcode(op), "storage_dtype": str(storage), "view_dtype": str(view),
            "element_bits": storage.bitwidth, "roots": sorted(v.index for v in state.bases[op.operands[0]]),
            "expansion": "original-storage-memory-event-with-register-bitcasts", "offsets": "unchanged-source-word-elements"})
    # Only saved pointer types change. Live pointer SSA becomes i64 offsets in
    # the existing analysis; reconstruction uses each root's actual storage.
    for value in state.values:
        root = min(state.bases[value], key=lambda v: v.index)
        if all(v.type == root.type for v in state.bases[value]):
            type_ = state.types[value]
            state.types[value] = builtin.TensorType(root.type, type_.get_shape()) if isinstance(type_, builtin.TensorType) else root.type
    return reports

"""Checked byte-address/word-template boundary for isolated TTIR atomics.

Meta 17f0c234 TritonToUnstructuredPass.cpp refuses changed-size pointer
bitcasts. Keep its word-typed event converter: expose a word formal only in
the isolated template, after checking the original byte address is divisible
by the word size. This module does not grant byte ownership or payload admission.
"""

from xdsl.dialects import arith, builtin, cf, scf, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from .diagnostics import ConversionError


def boundary(op, source_ids, byte_order):
    """Recognize a real byte addptr followed by a word pointer bitcast.

    The cast's position matters: an addptr *after* a word cast has word units.
    No annotation can turn that address into a byte offset.
    """
    from .pointer_views import byte_compatible, element
    from .shared_tensor import opcode

    pointer = op.operands[0]
    cast = pointer.owner
    if (opcode(op) not in ("tt.atomic_rmw", "tt.atomic_cas")
            or not hasattr(cast, "operands") or opcode(cast) != "tt.bitcast"
            or len(cast.operands) != 1
            or not byte_compatible(pointer.type, cast.operands[0].type)):
        return None

    def refuse(message):
        raise ConversionError(message, opcode=opcode(op),
                              source_id=next(iter(source_ids(op.location)), None))

    if byte_order not in ("little", "big"):
        refuse("byte atomic templates require explicit little or big byte order")
    view = element(pointer.type)
    cas = opcode(op) == "tt.atomic_cas"
    if view not in ((builtin.i16, builtin.i32, builtin.i64) if cas else (builtin.i32, builtin.i64)):
        refuse("byte atomic templates require qualified integer RMW/CAS word types")
    address = cast.operands[0].owner
    if (element(cast.operands[0].type) != builtin.i8
            or not hasattr(address, "operands") or opcode(address) != "tt.addptr"
            or len(address.operands) != 2):
        refuse("byte atomic templates require an actual byte-address addptr before the word cast")
    offset = address.operands[1]
    shape = pointer.type.get_shape() if isinstance(pointer.type, builtin.TensorType) else ()
    expected = builtin.TensorType(builtin.i64, shape) if shape else builtin.i64
    if (offset.type != expected or any(n <= 0 for n in shape)
            or isinstance(cast.operands[0].type, builtin.TensorType) != bool(shape)
            or shape and cast.operands[0].type.get_shape() != shape):
        refuse("byte atomic templates require matching static tiles and signed i64 byte addresses")
    return address, {
        "storage_dtype": "i8", "value_dtype": str(view), "word_bits": view.bitwidth,
        "byte_order": byte_order, "address_units": "signed-i64-bytes",
        "template_address_units": "word-elements-after-active-alignment",
        "alignment_bytes": view.bitwidth // 8,
    }


def prefix(body, size, mask, location):
    """Validate active byte alignment, then derive the isolated word indices."""
    offset = body.args[1]
    shape = offset.type.get_shape() if isinstance(offset.type, builtin.TensorType) else ()
    constants = {}

    def emit(block, op):
        op.location = location
        block.add_op(op)
        return op.results[0] if op.results else None

    def constant(block, n, type_=None):
        type_ = type_ or builtin.IndexType()
        key = type_, n
        if key not in constants:
            op = (arith.ConstantOp(builtin.DenseIntOrFPElementsAttr.from_list(type_, [n]))
                  if isinstance(type_, builtin.TensorType)
                  else arith.ConstantOp.from_int_and_width(n, type_))
            op.location = location
            if body.first_op is None:
                body.add_op(op)
            else:
                body.insert_op_before(op, body.first_op)
            constants[key] = op.result
        return constants[key]

    parent, indices = body, []
    for bound in shape:
        lo, hi, step = (constant(parent, n) for n in (0, bound, 1))
        nested = Block(arg_types=[builtin.IndexType()])
        emit(parent, scf.ForOp(lo, hi, step, [], Region(nested)))
        if parent is not body:
            emit(parent, scf.YieldOp())
        parent = nested
        indices.append(nested.args[0])
    scalar_offset = emit(parent, tensor.ExtractOp(offset, indices, builtin.i64)) if shape else offset
    active = parent
    if mask is not None:
        selected = emit(parent, tensor.ExtractOp(mask, indices, builtin.i1)) if shape else mask
        active = Block()
        emit(parent, scf.IfOp(selected, [], Region(active), Region()))
    remainder = emit(active, arith.RemSIOp(scalar_offset, constant(active, size, builtin.i64)))
    equal = emit(active, arith.CmpiOp(remainder, constant(active, 0, builtin.i64), "eq"))
    emit(active, cf.AssertOp(equal, "Triton byte atomic address is not word aligned"))
    if active is not parent:
        emit(active, scf.YieldOp())
    if parent is not body:
        emit(parent, scf.YieldOp())
    divisor = constant(body, size, builtin.i64)
    if not shape:
        return emit(body, arith.DivSIOp(offset, divisor))
    scalar = Block(arg_types=[builtin.i64, builtin.i64])
    quotient = emit(scalar, arith.DivSIOp(scalar.args[0], divisor))
    emit(scalar, YieldOp(quotient))
    empty = emit(body, tensor.EmptyOp([], offset.type))
    return emit(body, GenericOp(
        [offset], [empty], Region(scalar),
        [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * 2,
        [IteratorTypeAttr.parallel()] * len(shape), [offset.type],
    ))


def check_prefix(body, info, masked):
    """Reconstruct alignment/division from current typed formals, not tags."""
    expected = Block(arg_types=[arg.type for arg in body.args])
    word_index = prefix(expected, info["alignment_bytes"], expected.args[-1] if masked else None,
                        body.parent_op().location)
    context = {expected: body, **dict(zip(expected.args, body.args, strict=True))}
    wanted, actual = list(expected.ops), list(body.ops)
    # Native greedy rewriting pools/reorders constants independently of its
    # pointer patterns. Rebind literal definitions by full typed value; compare
    # every remaining operation and SSA edge in order.
    for constant in (op for op in wanted if isinstance(op, arith.ConstantOp)):
        matches = [op for op in actual if isinstance(op, arith.ConstantOp)
                   and tuple(v.type for v in op.results) == tuple(v.type for v in constant.results)
                   and op.attributes == constant.attributes and op.properties == constant.properties]
        if len(matches) != 1:
            break
        context[constant.result] = matches[0].result
    else:
        wanted = [op for op in wanted if not isinstance(op, arith.ConstantOp)]
        actual = [op for op in actual if not isinstance(op, arith.ConstantOp)]
        if len(actual) >= len(wanted) and all(
            [tuple(v.type for v in op.results) for op in a.walk()]
            == [tuple(v.type for v in op.results) for op in b.walk()]
            and a.is_structurally_equivalent(b, context)
            for a, b in zip(wanted, actual[:len(wanted)], strict=True)
        ):
            return context[word_index]
    raise ConversionError("native byte atomic template changed active alignment or byte-to-word division")


def check_event(body, info, masked, event):
    """The native memory event must consume the checked quotient and formals."""
    from .shared_tensor import _array, opcode

    quotient = check_prefix(body, info, masked)
    returned = tuple(body.last_op.operands)
    if opcode(event) == "tts.unstructured_atomic_rmw":
        if (tuple(event.operands) == (body.args[0], quotient, *body.args[2:])
                and returned == (event.results[0],)):
            return
    elif opcode(event) == "tt.atomic_cas":
        descriptor = event.operands[0].owner

        def flat(value, source):
            if value is source:
                return True
            owner = value.owner
            return (hasattr(owner, "operands") and tuple(owner.operands) == (source,)
                    and opcode(owner) in ("tt.reshape", "tensor.from_elements"))

        return_ok = returned == (event.results[0],)
        if len(returned) == 1 and returned[0].type == body.args[2].type:
            owner = returned[0].owner
            if isinstance(owner, tensor.ExtractOp):
                index = owner.indices[0].owner if len(owner.indices) == 1 else None
                return_ok = (owner.tensor is event.results[0] and isinstance(index, arith.ConstantOp)
                             and isinstance(index.value, builtin.IntegerAttr) and index.value.value.data == 0)
            elif hasattr(owner, "operands") and opcode(owner) == "tt.reshape":
                return_ok = tuple(owner.operands) == (event.results[0],)
        if (return_ok and hasattr(descriptor, "operands") and opcode(descriptor) == "tts.make_gather_scatter_tptr"
                and len(descriptor.operands) == 2 and descriptor.operands[0] is body.args[0]
                and flat(descriptor.operands[1], quotient)
                and _array(descriptor, "static_strides") == (1,)
                and _array(descriptor, "static_offsets") == (0,)
                and _array(descriptor, "operandSegmentSizes") == (1, 1, 0, 0, 0)
                and all(flat(v, arg) for v, arg in zip(event.operands[1:], body.args[2:], strict=True))):
            return
    raise ConversionError("native byte atomic event bypassed its checked word index, source value/mask or old result")

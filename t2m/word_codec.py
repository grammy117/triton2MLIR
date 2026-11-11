"""Bit-preserving standard tensor transport between words and byte planes."""

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap


def transport(value, target, byte_order, append):
    """Append executable transport; the final byte axis is storage, not a lane map.

    Floating values use integer bitcasts exclusively, preserving every untouched
    word, including signaling NaNs. Dynamic dimensions come from actual tensors.
    """
    if byte_order not in ("little", "big"):
        raise ValueError("word transport requires explicit little or big byte order")
    source = value.type
    unpack = (
        target.element_type == builtin.i8
        and len(target.get_shape()) == len(source.get_shape()) + 1
    )
    logical = source if unpack else target
    width = logical.element_type.bitwidth
    if width not in (16, 32, 64):
        raise ValueError("word transport requires a 16/32/64-bit logical element")
    size = width // 8
    byte_type = builtin.TensorType(builtin.i8, (*logical.get_shape(), size))
    if (target if unpack else source) != byte_type:
        raise ValueError("word transport byte plane differs from its logical tensor")

    def emit(op):
        append(op)
        return op.results[0] if op.results else None

    dynamic = []
    for axis, extent in enumerate(target.get_shape()):
        if extent == builtin.DYNAMIC_INDEX:
            index = emit(arith.ConstantOp.from_int_and_width(axis, builtin.IndexType()))
            dynamic.append(emit(tensor.DimOp(value, index)))
    empty = emit(tensor.EmptyOp(dynamic, target))
    block = Block(arg_types=[target.element_type])

    def scalar(op):
        block.add_op(op)
        return op.results[0] if op.results else None

    def constant(n, type_):
        return scalar(arith.ConstantOp.from_int_and_width(n, type_))

    coordinates = [scalar(IndexOp(axis)) for axis in range(len(logical.get_shape()))]
    integer = builtin.IntegerType(width)
    if unpack:
        word = scalar(tensor.ExtractOp(value, coordinates, logical.element_type))
        if logical.element_type != integer:
            word = scalar(arith.BitcastOp(word, integer))
        byte = scalar(IndexOp(len(coordinates)))
        byte = scalar(arith.IndexCastOp(byte, integer))
        if byte_order == "big":
            byte = scalar(arith.SubiOp(constant(size - 1, integer), byte))
        shift = scalar(arith.MuliOp(byte, constant(8, integer)))
        result = scalar(arith.TruncIOp(scalar(arith.ShRUIOp(word, shift)), builtin.i8))
    else:
        result = constant(0, integer)
        for byte in range(size):
            index = constant(byte, builtin.IndexType())
            piece = scalar(tensor.ExtractOp(value, [*coordinates, index], builtin.i8))
            piece = scalar(arith.ExtUIOp(piece, integer))
            shift = (byte if byte_order == "little" else size - byte - 1) * 8
            if shift:
                piece = scalar(arith.ShLIOp(piece, constant(shift, integer)))
            result = scalar(arith.OrIOp(result, piece))
        if logical.element_type != integer:
            result = scalar(arith.BitcastOp(result, logical.element_type))
    scalar(YieldOp(result))
    generic = GenericOp(
        [],
        [empty],
        Region(block),
        [builtin.AffineMapAttr(AffineMap.identity(len(target.get_shape())))],
        [IteratorTypeAttr.parallel()] * len(target.get_shape()),
        [target],
    )
    generic.attributes["prov.word_transport"] = builtin.StringAttr(
        f"{'unpack' if unpack else 'pack'}-{byte_order}-{width}"
    )
    return emit(generic)

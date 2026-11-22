"""Typed storage ports for original TTIR value arguments and results.

Use ordinary TTIR loads, calls, stores and ordered reshapes before upstream
inlining. No low-precision value crosses a native by-value entry ABI.
"""

import math
from dataclasses import dataclass

from xdsl.dialects import arith, builtin

from .diagnostics import ConversionError

ARGUMENTS_SCHEMA = "t2m.source-allocation-arguments.v3"


def _validate(port):
    if (type(port.allocation) is not str or not port.allocation.isidentifier()
            or type(port.per_program) is not bool
            or not (type(port.offset) is int and -(2**63) <= port.offset < 2**63
                    or type(port.offset) is str and port.offset.isidentifier())):
        raise ValueError("entry value ports require an allocation, i64/scalar offset and Boolean program selection")


@dataclass(frozen=True)
class ValueArgument:
    allocation: str
    offset: int | str = 0
    per_program: bool = False

    def __post_init__(self):
        _validate(self)


@dataclass(frozen=True)
class ResultBinding:
    allocation: str
    offset: int | str = 0
    per_program: bool = True

    def __post_init__(self):
        _validate(self)


def checked_type(type_, port, roots, scalars, *, result=False):
    root = roots.get(port.allocation)
    shaped = isinstance(type_, builtin.TensorType)
    element = type_.element_type if shaped else type_
    if (root is None or str(element) != root.dtype
            or not isinstance(element, (builtin.IntegerType, builtin.AnyFloat))
            or result and root.effect == "read" or not result and root.effect == "write"):
        raise ConversionError("entry value port requires exact numeric storage and its read/write effect")
    if type(port.offset) is str and (port.offset not in scalars or scalars[port.offset].dtype not in ("i32", "i64")):
        raise ConversionError("entry value origin requires a declared i32/i64 runtime scalar")
    shape = tuple(type_.get_shape()) if shaped else None
    count = math.prod(shape) if shaped else 1
    if shape is not None and any(n <= 0 for n in shape) or not 0 < count < 2**31:
        raise ConversionError("entry value tile requires static positive extents and an i32-size range")
    return root, shape, count


class Ports:
    def __init__(self, block, abi, values):
        self.block, self.abi, self.values = block, abi, values

    def emit(self, name, operands=(), types=(), properties=None):
        operation = builtin.UnregisteredOp.with_name(name).create(
            operands=operands, result_types=types, properties=properties or {})
        self.block.add_op(operation)
        return operation.results[0] if operation.results else None

    def number(self, value):
        op = arith.ConstantOp.from_int_and_width(value, builtin.i64)
        self.block.add_op(op)
        return op.result

    def query(self, name, axis):
        value = self.emit(name, types=[builtin.i32], properties={"axis": builtin.IntegerAttr(axis, builtin.i32)})
        return self.emit("arith.extsi", [value], [builtin.i64])

    def pointer(self, port, count, shaped):
        if port.per_program and count * math.prod(self.abi.grid_bounds) >= 2**63:
            raise ConversionError("entry value program/tile domain must fit signed i64 logical addressing")
        offset = self.number(port.offset) if type(port.offset) is int else self.values[port.offset]
        if offset.type == builtin.i32:
            offset = self.emit("arith.extsi", [offset], [builtin.i64])
        if port.per_program:
            linear = self.query("tt.get_program_id", 0)
            stride = self.number(1)
            for axis in range(1, len(self.abi.grid)):
                stride = self.emit("arith.muli", [stride, self.query("tt.get_num_programs", axis - 1)], [builtin.i64])
                term = self.emit("arith.muli", [self.query("tt.get_program_id", axis), stride], [builtin.i64])
                linear = self.emit("arith.addi", [linear, term], [builtin.i64])
            offset = self.emit("arith.addi", [offset,
                self.emit("arith.muli", [linear, self.number(count)], [builtin.i64])], [builtin.i64])
        root = self.values[port.allocation]
        pointer = self.emit("tt.addptr", [root, offset], [root.type])
        if shaped:
            pointer_type = builtin.TensorType(root.type, [count])
            pointer = self.emit("tt.splat", [pointer], [pointer_type])
            offsets = self.emit("tt.make_range", types=[builtin.TensorType(builtin.i32, [count])],
                properties={"start": builtin.IntegerAttr(0, builtin.i32), "end": builtin.IntegerAttr(count, builtin.i32)})
            pointer = self.emit("tt.addptr", [pointer, offsets], [pointer_type])
        return pointer

    def load(self, port, type_):
        shaped = isinstance(type_, builtin.TensorType)
        count = math.prod(type_.get_shape()) if shaped else 1
        flat = builtin.TensorType(type_.element_type, [count]) if shaped else type_
        pointer = self.pointer(port, count, shaped)
        value = self.emit("tt.load", [pointer], [flat],
            {"isVolatile": builtin.IntegerAttr(0, builtin.i1),
             "operandSegmentSizes": builtin.DenseArrayBase.from_list(builtin.i32, [1, 0, 0])})
        return value if flat == type_ else self.emit("tt.reshape", [value], [type_])

    def store(self, port, value):
        shaped = isinstance(value.type, builtin.TensorType)
        count = math.prod(value.type.get_shape()) if shaped else 1
        pointer = self.pointer(port, count, shaped)
        flat = builtin.TensorType(value.type.element_type, [count]) if shaped else value.type
        if value.type != flat:
            value = self.emit("tt.reshape", [value], [flat])
        self.emit("tt.store", [pointer, value])

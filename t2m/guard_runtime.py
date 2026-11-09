"""Private guard storage and completion checks at the functional ABI boundary."""

from xdsl.dialects import arith, builtin, cf, memref, scf
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Block, Region

from .guarded_memory import MAP_TYPE, argument_type, internal_arguments


def allocate(abi, dimensions, entry, body, append):
    values, owners = {}, {}
    def constant(n, type_=IndexType()):
        return append(entry, arith.ConstantOp.from_int_and_width(n, type_)).result
    one = constant(1)
    for buffer in abi.buffers:
        count = one
        # Check before every product/allocation; maxima are not allocation sizes.
        size = 16 if buffer.effect != "read" else max(1, buffer.type.element_type.compile_time_size)
        limit = constant(((1 << 63) - 1) // size)
        for dimension in buffer.shape:
            dimension = dimensions[dimension] if isinstance(dimension, str) else constant(dimension)
            divisor = append(entry, arith.MaxUIOp(dimension, one)).result
            maximum = append(entry, arith.DivUIOp(limit, divisor)).result
            safe = append(entry, arith.CmpiOp(count, maximum, "ule")).result
            append(entry, cf.AssertOp(safe, "Triton guarded allocation exceeds its signed index/byte domain"))
            count = append(entry, arith.MuliOp(count, dimension)).result
        values["extent", buffer.name] = append(entry, arith.IndexCastOp(count, builtin.i64)).result
        if buffer.effect == "read":
            continue
        for kind, initial in (("owners", -1), ("visited", 0)):
            value = append(entry, memref.AllocOp.build(operands=[[count], []], result_types=[MAP_TYPE]),
                           role="private-memory-guard-state").memref
            append(entry, FillOp([constant(initial, builtin.i64)], [value], res=[]), role="initialize-memory-guard-state")
            values[kind, buffer.name] = value
            if kind == "owners":
                owners[buffer.name] = value, count
    values["program", None] = append(body, arith.IndexCastOp(body.args[0], builtin.i64), role="logical-guard-owner").result
    arguments = [values[arg["kind"], arg.get("buffer")] for arg in internal_arguments(abi)]
    if [value.type for value in arguments] != [argument_type(arg) for arg in internal_arguments(abi)]:
        raise ValueError("internal guard bindings differ from their typed declaration")
    return arguments, owners


def finish(abi, entry, state, append):
    _, owners = state
    def constant(n, type_=IndexType()):
        return append(entry, arith.ConstantOp.from_int_and_width(n, type_)).result
    zero, one = constant(0), constant(1)
    for buffer in abi.outputs:
        if buffer.effect != "write":
            continue
        owner, count = owners[buffer.name]
        body = Block(arg_types=[IndexType()])
        value = memref.LoadOp.build(operands=[owner, [body.args[0]]], result_types=[builtin.i64])
        unowned = arith.ConstantOp.from_int_and_width(0, builtin.i64)
        complete = arith.CmpiOp(value.results[0], unowned.result, "sge")
        check = cf.AssertOp(complete.result, "Triton fresh output has an unwritten logical element: " + buffer.name)
        body.add_ops([value, unowned, complete, check, scf.YieldOp()])
        append(entry, scf.ForOp(zero, count, one, [], Region(body)), role="complete-output-initialization-guard")
    # Ownership-based buffer deallocation owns frees. Pre-inserting deallocs
    # prevents that native pass from establishing its lifetime contract.

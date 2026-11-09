"""Functional tensor ABI around the complete shared standard-memory body.

Source memory admission precedes this builder: either a tensor proof witness or
the explicit executable guarded-memory profile. This module preserves the
upstream computation, supplies typed buffers and binds its logical launch ABI.
"""

import math

from xdsl.dialects import arith, bufferization, builtin, cf, func, math as math_dialect, memref
from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IndexType, IntegerAttr, MemRefType, ModuleOp, StringAttr, UnitAttr, UnrankedMemRefType
from xdsl.ir import Block, Dialect, Region

from .diagnostics import ConversionError
from .ir import context
from .scalar_abi import guard_scalar
from .dynamic_abi import GridExtent
from .shape_lowering import bind_dimensions, launch_sizes, product, program_id
from .shared_tensor import SharedParser, opcode
from .standard_bufferization import MaterializeInDestinationOp, ToTensorOp
from .standard_memref import memref_dialect
from .standard_ops import ForallOp, InParallelOp


def parse_memory(source):
    ctx = context()
    for dialect in (memref_dialect(), cf.Cf, math_dialect.Math):
        ctx.load_dialect(dialect)
    ctx.load_dialect(Dialect("bufferization", [MaterializeInDestinationOp if op is bufferization.MaterializeInDestinationOp
        else ToTensorOp if op is bufferization.ToTensorOp else op for op in bufferization.Bufferization.operations], []))
    ctx.allow_unregistered = True
    module = SharedParser(ctx, source).parse_module()
    for op in module.walk():
        if op.name == "builtin.unregistered" or op.name.split(".")[0] not in {
                "builtin", "func", "arith", "math", "tensor", "linalg", "scf", "cf", "memref", "bufferization"}:
            raise ConversionError("memory continuation retains an unresolved operation", opcode=opcode(op))
        if op.name == "func.call":
            raise ConversionError("memory continuation requires complete helper inlining", opcode=op.name)
    module.verify()
    return module


def functionalize_memory(module, abi, *, guards=None):
    functions = list(module.ops)
    if len(functions) != 1 or not isinstance(functions[0], func.FuncOp):
        raise ConversionError("memory continuation requires one standard kernel function")
    original = functions[0]
    from .guarded_memory import internal_arguments
    internal = internal_arguments(abi) if guards is not None else []
    if guards is not None and guards.get("internal_arguments") != internal:
        raise ConversionError("memory guard ABI differs from its internal declaration")
    if (original.sym_name.data != abi.entry or len(original.body.blocks) != 1
            or original.function_type.outputs or len(original.body.block.args) != len(abi.source_bindings) + len(internal) + 6):
        raise ConversionError("memory continuation differs from the declared source/launcher ABI")
    entry = Block(arg_types=[value.type for value in abi.inputs])
    inputs = {value.name: arg for value, arg in zip(abi.inputs, entry.args, strict=True)}
    for scalar in abi.runtime_scalars:
        guard_scalar(scalar, inputs[scalar.name], entry, original.location)
    dimensions = bind_dimensions(abi, inputs, entry, original.location)

    def append(block, op, *, role="functional-memory-boundary"):
        op.attributes.update({"prov.family": StringAttr("abi"), "prov.op": StringAttr(role)})
        block.add_op(op)
        return op

    body = Block(arg_types=[IndexType()])
    launch = launch_sizes(abi, dimensions, entry, original.location) if abi.dynamic else None
    guard_state = None
    if guards is not None:
        from .guard_runtime import allocate
        guard_state = allocate(abi, dimensions, entry, body, append)
    values, states = [], {}
    for source, (kind, declared) in zip(original.body.block.args, abi.source_bindings):
        if kind == "constant":
            if not isinstance(source.type, builtin.IntegerType):
                raise ConversionError("specialized scalar type changed during memory continuation")
            values.append(append(entry, arith.ConstantOp(IntegerAttr(declared, source.type))).result)
            continue
        if kind == "runtime":
            if source.type != declared.type:
                raise ConversionError("runtime scalar type changed during memory continuation")
            value = inputs[declared.name]
            values.append(value)
            continue
        expected = UnrankedMemRefType.from_type(declared.type.element_type)
        if source.type != expected:
            raise ConversionError("raw memory argument type differs from the logical buffer ABI")
        type_ = MemRefType(declared.type.element_type, declared.type.get_shape())
        incoming = None
        if declared.effect != "write":
            incoming = append(entry, bufferization.ToBufferOp.build(operands=[inputs[declared.name]],
                result_types=[type_], properties={"read_only": UnitAttr()})).memref
        value = incoming
        if declared.effect != "read":
            dynamic = [dimensions[n] for n in declared.shape if isinstance(n, str)]
            value = append(entry, memref.AllocOp.build(operands=[dynamic, []], result_types=[type_]), role="owned-output-state").memref
            if incoming is not None:
                append(entry, memref.CopyOp(incoming, value), role="preserve-incoming-output-state")
            states[declared.name] = value
        if len(declared.shape) > 1:
            reassociation = ArrayAttr([ArrayAttr(IntegerAttr(i, builtin.i64) for i in range(len(declared.shape)))])
            value = append(entry, memref.CollapseShapeOp.build(operands=[value],
                result_types=[MemRefType(type_.element_type, [builtin.DYNAMIC_INDEX if any(isinstance(n, str) for n in declared.shape) else math.prod(declared.shape)])],
                properties={"reassociation": reassociation}), role="linear-raw-pointer-view").result
        values.append(append(entry, memref.CastOp.get(value, expected), role="bind-upstream-raw-buffer").dest)
    if guard_state is not None:
        values.extend(guard_state[0])
    counts = (*abi.grid_bounds, *((1,) * (3 - len(abi.grid))))
    if launch is not None:
        one = append(entry, arith.ConstantOp.from_int_and_width(1, IndexType())).result
        for value in (*launch, *((one,) * (3 - len(launch)))):
            values.append(append(body, arith.IndexCastOp(value, builtin.i32), role="logical-program-count").result)
        for axis in range(3):
            value = program_id(axis, body.args[0], launch, lambda op: append(body, op))
            values.append(append(body, arith.IndexCastOp(value, builtin.i32), role="logical-program-id").result)
    for count in counts:
        if launch is not None:
            break
        values.append(append(body, arith.ConstantOp.from_int_and_width(count, builtin.i32), role="logical-program-count").result)
    for axis, count in enumerate(counts):
        if launch is not None:
            break
        pid = body.args[0]
        divisor = math.prod(counts[axis + 1:])
        if divisor != 1:
            n = append(body, arith.ConstantOp.from_int_and_width(divisor, IndexType())).result
            pid = append(body, arith.DivUIOp(pid, n), role="logical-program-id").result
        n = append(body, arith.ConstantOp.from_int_and_width(count, IndexType())).result
        pid = append(body, arith.RemUIOp(pid, n), role="logical-program-id").result
        values.append(append(body, arith.IndexCastOp(pid, builtin.i32), role="logical-program-id").result)
    if [value.type for value in values] != list(original.function_type.inputs):
        raise ConversionError("memory continuation launcher types differ from the checked ABI")
    mapper = dict(zip(original.body.block.args, values, strict=True))
    for op in original.body.block.ops:
        if not isinstance(op, func.ReturnOp):
            body.add_op(op.clone(value_mapper=mapper))
    body.add_op(InParallelOp.build(regions=[Region(Block())]))
    dynamic_launch = any(isinstance(g, GridExtent) for g in abi.grid)
    upper = product(launch, lambda op: append(entry, op)) if dynamic_launch else None
    append(entry, ForallOp.build(operands=[[], [upper] if dynamic_launch else [], [], []], result_types=[[]], regions=[Region(body)], properties={
        "staticLowerBound": DenseArrayBase.from_list(builtin.i64, [0]),
        "staticUpperBound": DenseArrayBase.from_list(builtin.i64, [builtin.DYNAMIC_INDEX if dynamic_launch else math.prod(abi.grid_bounds)]),
        "staticStep": DenseArrayBase.from_list(builtin.i64, [1])}), role="unmapped-logical-launch")
    if guard_state is not None:
        from .guard_runtime import finish
        finish(abi, entry, guard_state, append)
    outputs = [append(entry, ToTensorOp.build(operands=[states[value.name]], result_types=[value.type],
                properties={"restrict": UnitAttr(), "writable": UnitAttr()}), role="functional-output-state").tensor
               for value in abi.outputs]
    entry.add_op(func.ReturnOp(*outputs))
    function = func.FuncOp(abi.entry, ([value.type for value in abi.inputs], [value.type for value in abi.outputs]), Region(entry))
    if guards is not None:
        function.attributes["prov.memory_guard_bytes_per_result_element"] = IntegerAttr(guards["guard_storage_bytes_per_output_element"], builtin.i64)
    function.attributes["prov.logical_grid_bounds" if abi.dynamic else "prov.logical_grid"] = DenseArrayBase.from_list(builtin.i64, abi.grid_bounds)
    result = ModuleOp([function])
    result.attributes.update({"prov.level": StringAttr("functional-tensor-with-standard-memory"),
                              "prov.frontend": StringAttr("triton2mlir.shared")})
    result.verify()
    return result

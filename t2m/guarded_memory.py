"""Executable memory obligations before the full upstream memory continuation.

Meta 17f0c234 PtrState::accumulateTargetOffset and ScatterConverter define
the address/mask boundary reused here. Guards consume the exact original SSA
indices and predicates. Payload operations remain unchanged and parallel;
only compiler-owned validation metadata is updated by the guard loops.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin, cf, memref, scf, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, IndexType, IntegerAttr, MemRefType, StringAttr, TensorType
from xdsl.ir import Block, Region

from .diagnostics import ConversionError
from .memory_contract import guarded_arguments

# MLIR arith::AtomicRMWKind::assign (ArithBase.td), not a device opcode.
ATOMIC_ASSIGN = 3
MAP_TYPE = MemRefType(builtin.i64, [builtin.DYNAMIC_INDEX])


def internal_arguments(abi):
    return guarded_arguments(abi.as_dict())


def argument_type(argument):
    return MAP_TYPE if argument["kind"] in ("owners", "visited") else builtin.i64


@dataclass
class Access:
    buffer: object
    sizes: tuple
    index: object = None
    mask: object = None
    offsets: tuple = ()
    strides: tuple = ()
    prefixes: tuple = ()


def normalize_scalars(module, source_ids):
    """Lift scalar memory events into the upstream rank-one tensor converter.

    Upstream GatherConverter/ScatterConverter explicitly skip scalar offsets.
    A one-element tensor preserves the event, predicate and word/value types;
    no arithmetic, memory observation or grid iteration is added or removed.
    """
    from .shared_tensor import _array, opcode
    reports = []
    for op in list(module.walk()):
        name = opcode(op)
        if name not in ("tts.gather", "tts.scatter") or isinstance(op.operands[1].type, TensorType):
            continue
        def refuse(message):
            raise ConversionError(message, source_id=next(iter(source_ids(op.location)), None), opcode=name)
        if op.operands[1].type not in (builtin.i32, builtin.i64):
            refuse("scalar memory offsets require signed i32/i64 source words")
        write = name == "tts.scatter"
        positions = [1, 2] if write else [1]
        if write:
            if len(op.operands) not in (3, 4) or isinstance(op.operands[2].type, TensorType):
                refuse("scalar scatter requires one scalar value and optional scalar mask")
            if len(op.operands) == 4:
                positions.append(3)
        else:
            segments = _array(op, "operandSegmentSizes")
            if (len(segments) != 4 or segments[:2] != (1, 1) or any(n not in (0, 1) for n in segments[2:])
                    or sum(segments) != len(op.operands) or len(op.results) != 1 or isinstance(op.results[0].type, TensorType)):
                refuse("scalar gather requires one scalar result and explicit optional operands")
            if segments[2]:
                positions.append(2)
        if len(positions) > (2 if write else 1) and op.operands[positions[-1]].type != builtin.i1:
            refuse("scalar memory predicate must be i1")
        operands, before = list(op.operands), []
        for position in positions:
            wrapped = tensor.FromElementsOp(operands[position])
            before.append(wrapped)
            operands[position] = wrapped.result
        converted = type(op).create(operands=operands, result_types=[] if write else [TensorType(op.results[0].type, [1])],
            properties=op.properties, attributes=op.attributes, location=op.location)
        before.append(converted)
        if not write:
            zero = arith.ConstantOp.from_int_and_width(0, IndexType())
            extracted = tensor.ExtractOp(converted.results[0], [zero.result], op.results[0].type)
            before.extend([zero, extracted])
            op.results[0].replace_all_uses_with(extracted.result)
        for generated in before:
            generated.location = op.location
        op.parent.insert_ops_before(before, op)
        reports.append({"opcode": name, "source_ids": source_ids(op.location), "normalization": "scalar-event-to-unit-tensor"})
        op.parent.erase_op(op)
    return reports


def instrument(module, abi, source_ids):
    from .shared_tensor import _array, _pointer_type, opcode
    function = next(iter(module.ops))
    entry = function.regions[0].block
    scalar_normalizations = normalize_scalars(module, source_ids)
    original = list(function.walk())
    buffers = {arg: value for arg, (kind, value) in zip(entry.args, abi.source_bindings, strict=True) if kind == "buffer"}
    descriptors = {}
    supported = {"tt.func", "tt.return", "tt.get_program_id", "tt.get_num_programs", "tts.make_tptr",
                 "tts.make_gather_scatter_tptr", "tts.load", "tts.store", "tts.gather", "tts.scatter"}
    controls = {"scf.for", "scf.if", "scf.while", "scf.condition", "scf.yield"}

    def refuse(op, message):
        raise ConversionError(message, source_id=next(iter(source_ids(op.location)), None), opcode=opcode(op))

    # Reject source memory/calls/physical pointers before adding our own memrefs.
    # An input attribute cannot authorize a synthetic runtime guard operation.
    for op in original:
        name = opcode(op)
        if name not in supported | controls and name.split(".")[0] not in ("arith", "math", "tensor", "linalg", "cf"):
            refuse(op, "guarded memory has an unqualified operation/effect")
        if name.startswith("cf.") and name != "cf.assert":
            refuse(op, "guarded memory requires structured source control flow")
        for value in op.results:
            if _pointer_type(value.type) and name not in ("tts.make_tptr", "tts.make_gather_scatter_tptr"):
                refuse(op, "guarded pointer state needs explicit base/offset normalization")
    for value in (*buffers, *(v for op in original for v in op.results if _pointer_type(v.type))):
        for use in value.uses:
            if use.index != 0 or opcode(use.operation) not in {
                    "tts.make_tptr", "tts.make_gather_scatter_tptr", "tts.load", "tts.store", "tts.gather", "tts.scatter"}:
                refuse(use.operation, "pointer escaped the guarded logical buffer boundary")

    arguments = internal_arguments(abi)
    extra = {}
    for argument in arguments:
        extra[argument["kind"], argument.get("buffer")] = entry.insert_arg(argument_type(argument), len(entry.args))
    function.properties["function_type"] = FunctionType.from_lists([v.type for v in entry.args], [])
    for container in (function.properties, function.attributes):
        if "arg_attrs" in container:
            container["arg_attrs"] = ArrayAttr([*container["arg_attrs"], *[DictionaryAttr({}) for _ in arguments]])

    def mixed(op, field, operands):
        dynamic = iter(operands)
        values = tuple(next(dynamic) if n == builtin.DYNAMIC_INDEX else n for n in _array(op, field))
        if next(dynamic, None) is not None:
            refuse(op, "guarded descriptor dynamic segment mismatch")
        return values

    def describe(op, write):
        name = opcode(op)
        if name in ("tts.gather", "tts.scatter"):
            if op.operands[0] not in buffers:
                refuse(op, "indirect memory base must be a declared logical buffer")
            buffer, index = buffers[op.operands[0]], op.operands[1]
            if not isinstance(index.type, TensorType):
                refuse(op, "scalar indirect memory requires unit-tensor normalization")
            if write:
                if len(op.operands) not in (3, 4):
                    refuse(op, "scatter requires explicit index/value/mask operands")
                mask = op.operands[3] if len(op.operands) == 4 else None
                type_ = op.operands[2].type
            else:
                segments = _array(op, "operandSegmentSizes")
                if len(segments) != 4 or segments[:2] != (1, 1) or any(n not in (0, 1) for n in segments[2:]) or sum(segments) != len(op.operands):
                    refuse(op, "gather operand schema is not qualified")
                mask = op.operands[2] if segments[2] else None
                type_ = op.results[0].type
            if (not isinstance(type_, TensorType) or type_.get_shape() != index.type.get_shape()
                    or type_.element_type != buffer.type.element_type
                    or mask is not None and mask.type != TensorType(builtin.i1, index.type.get_shape())):
                refuse(op, "indirect memory index/value/mask types do not match")
            if index.type.element_type not in (builtin.i32, builtin.i64):
                refuse(op, "indirect indices require signed i32/i64 source words")
            return Access(buffer, index.type.get_shape(), index=index, mask=mask)
        pointer = descriptors.get(op.operands[0])
        if pointer is None:
            refuse(op, "structured memory requires a normalized declared base pointer")
        buffer, sizes, offsets, strides, gather = pointer
        if write:
            # StoreOp has one variadic operand group, hence no segment array.
            # LoadOp also carries optional `other` and uses three segments.
            if len(op.operands) < 2:
                refuse(op, "structured store requires one tensor value")
            mask_operands = op.operands[2:]
        else:
            segments = _array(op, "operandSegmentSizes")
            if len(segments) != 3 or segments[0] != 1 or sum(segments) != len(op.operands):
                refuse(op, "structured memory operand schema is not qualified")
            mask_operands = op.operands[1:1 + segments[1]]
        prefixes = mixed(op, "static_mask_dims", mask_operands)
        if prefixes and len(prefixes) != len(sizes):
            refuse(op, "structured prefix mask rank differs from its tile")
        type_ = op.operands[1].type if write else op.results[0].type
        if not isinstance(type_, TensorType) or type_.get_shape() != sizes or type_.element_type != buffer.type.element_type:
            refuse(op, "structured memory value differs from its declared pointer tile")
        return Access(buffer, sizes, index=gather, offsets=offsets, strides=strides, prefixes=prefixes)

    def pointer(op):
        base = buffers.get(op.operands[0])
        if base is None:
            refuse(op, "guarded descriptors require a declared base argument")
        sizes = _array(op, "sizes")
        if opcode(op) == "tts.make_gather_scatter_tptr":
            dimension = op.properties.get("gather_scatter_dim")
            if (len(sizes) != 1 or not isinstance(dimension, IntegerAttr) or dimension.value.data != 0
                    or _array(op, "operandSegmentSizes") != (1, 1, 0, 0, 0)
                    or _array(op, "static_strides") != (1,) or _array(op, "static_offsets") != (0,)):
                refuse(op, "gather/scatter descriptor needs its rank-one zero-base unit-stride form")
            index_type = op.operands[1].type
            if (not isinstance(index_type, TensorType) or index_type.get_shape() != sizes
                    or index_type.element_type not in (builtin.i32, builtin.i64)):
                refuse(op, "gather/scatter descriptor requires a matching signed i32/i64 index tile")
            return base, sizes, (), (), op.operands[1]
        segments = _array(op, "operandSegmentSizes")
        if len(segments) != 4 or segments[0] != 1 or sum(segments) != len(op.operands):
            refuse(op, "structured pointer segments differ from the supported schema")
        parts, start = [], 1
        for length in segments[1:]:
            parts.append(op.operands[start:start + length])
            start += length
        strides, offsets, shape = (mixed(op, field, part) for field, part in zip(
            ("static_strides", "static_offsets", "static_shape"), parts, strict=True))
        if any(not isinstance(n, int) or n != 0 for n in shape) or _array(op, "order"):
            refuse(op, "wrapping structured pointers require a separate guarded address adapter")
        if len(strides) != len(sizes) or len(offsets) != len(sizes):
            refuse(op, "structured address axes differ from the pointer tile")
        return base, sizes, offsets, strides, None

    accesses = []
    for op in original:
        name = opcode(op)
        if name in ("tts.make_tptr", "tts.make_gather_scatter_tptr"):
            descriptors[op.results[0]] = pointer(op)
            continue
        if name not in ("tts.load", "tts.store", "tts.gather", "tts.scatter"):
            continue
        write = name in ("tts.store", "tts.scatter")
        access = describe(op, write)
        if not access.sizes or any(n <= 0 for n in access.sizes):
            refuse(op, "memory guards require a positive static tile shape")
        if write and access.buffer.effect == "read":
            refuse(op, "source stores into a declared read-only buffer")
        if not write and access.buffer.effect != "read":
            refuse(op, "guarded writable-buffer reads require an observation/initialization contract")
        before = guard_loops(op, access, extra, write=write)
        op.parent.insert_ops_before(before, op)
        if write:
            op.parent.insert_ops_after(guard_loops(op, access, extra, write=True, cleanup=True), op)
        accesses.append({"source_ids": source_ids(op.location), "opcode": name, "buffer": access.buffer.name,
                         "effect": "write" if write else "read", "tile": list(access.sizes),
                         "guard": "bounds-owner-lane-uniqueness" if write else "active-bounds",
                         "mask": "source-per-lane" if access.mask is not None else "source-prefix" if access.prefixes else "unmasked"})
    return {"schema": "t2m.memory-admission.v1", "strategy": "runtime-guarded-private-output-state",
            "spec": abi.as_dict(), "internal_arguments": arguments, "accesses": accesses,
            "scalar_normalizations": scalar_normalizations,
            "control_flow": [{"opcode": opcode(op), "source_ids": source_ids(op.location)} for op in original
                             if opcode(op) in ("scf.for", "scf.while", "scf.if")],
            "writable_reads": "refused", "fresh_initialization": "post-grid-complete-write-coverage",
            "parallelism": "original-logical-grid-and-payload-retained",
            "guard_storage_bytes_per_output_element": 16}


def guard_loops(source, access, extra, *, write=False, cleanup=False):
    outer = Block()
    def add(block, op, role="memory_guard"):
        op.location = source.location
        op.attributes.update({"prov.family": StringAttr("guard"), "prov.op": StringAttr(role)})
        block.add_op(op)
        return op.results[0] if op.results else None
    def constant(block, n, type_=IndexType()):
        return add(block, arith.ConstantOp.from_int_and_width(n, type_))
    def index(block, value):
        if isinstance(value, int):
            return constant(block, value)
        return value if value.type == IndexType() else add(block, arith.IndexCastOp(value, IndexType()))
    body, indices = outer, []
    prefixes = tuple(index(outer, value) for value in access.prefixes)
    if not cleanup:
        for prefix, size in zip(prefixes, access.sizes):
            low = add(outer, arith.CmpiOp(prefix, constant(outer, 0), "sge"))
            high = add(outer, arith.CmpiOp(prefix, constant(outer, size), "sle"))
            add(outer, cf.AssertOp(add(outer, arith.AndIOp(low, high)), "Triton structured prefix outside tile"))
    for size in access.sizes:
        lo, hi, step = constant(body, 0), constant(body, size), constant(body, 1)
        nested = Block(arg_types=[IndexType()])
        add(body, scf.ForOp(lo, hi, step, [], Region(nested)), "memory_guard_iteration")
        if body is not outer:
            body.add_op(scf.YieldOp())
        body = nested
        indices.append(body.args[0])
    selected = None
    if access.mask is not None:
        selected = add(body, tensor.ExtractOp(access.mask, indices, builtin.i1))
    for iv, prefix in zip(indices, prefixes):
        test = add(body, arith.CmpiOp(iv, prefix, "slt"))
        selected = test if selected is None else add(body, arith.AndIOp(selected, test))
    parent = body
    if selected is not None:
        body = Block()
        add(parent, scf.IfOp(selected, [], Region(body), Region()), "active_memory_guard")
    if access.index is not None:
        offset = add(body, tensor.ExtractOp(access.index, indices, access.index.type.element_type))
        offset = add(body, arith.ExtSIOp(offset, builtin.i64)) if offset.type == builtin.i32 else offset
    else:
        offset = constant(body, 0)
        for value in access.offsets:
            offset = add(body, arith.AddiOp(offset, index(body, value)))
        for iv, stride in zip(indices, access.strides, strict=True):
            offset = add(body, arith.AddiOp(offset, add(body, arith.MuliOp(iv, index(body, stride)))))
        offset = add(body, arith.IndexCastOp(offset, builtin.i64))
    if not cleanup:
        low = add(body, arith.CmpiOp(offset, constant(body, 0, builtin.i64), "sge"))
        high = add(body, arith.CmpiOp(offset, extra["extent", access.buffer.name], "slt"))
        add(body, cf.AssertOp(add(body, arith.AndIOp(low, high)), "Triton active memory index outside logical allocation"))
    if write:
        position = add(body, arith.IndexCastOp(offset, IndexType()))
        visited = extra["visited", access.buffer.name]
        if cleanup:
            add(body, memref.StoreOp.build(operands=[constant(body, 0, builtin.i64), visited, [position]]), "release_store_lane_marks")
        else:
            pid, owners = extra["program", None], extra["owners", access.buffer.name]
            def exchange(value, buffer):
                return add(body, memref.AtomicRMWOp.build(operands=[value, buffer, [position]], result_types=[builtin.i64],
                    properties={"kind": IntegerAttr(ATOMIC_ASSIGN, builtin.i64)}), "claim_memory_obligation")
            owner = exchange(pid, owners)
            empty = add(body, arith.CmpiOp(owner, constant(body, -1, builtin.i64), "eq"))
            local = add(body, arith.CmpiOp(owner, pid, "eq"))
            add(body, cf.AssertOp(add(body, arith.OrIOp(empty, local)), "Triton write conflicts across logical programs"))
            previous = exchange(constant(body, 1, builtin.i64), visited)
            unique = add(body, arith.CmpiOp(previous, constant(body, 0, builtin.i64), "eq"))
            add(body, cf.AssertOp(unique, "Triton scatter has conflicting active lanes"))
    body.add_op(scf.YieldOp())
    if body is not parent:
        parent.add_op(scf.YieldOp())
    operations = list(outer.ops)
    for op in operations:
        outer.detach_op(op)
    return operations

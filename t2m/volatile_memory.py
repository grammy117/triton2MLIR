"""Portable exactly-once volatile reads over guarded logical memory.

MemRefToLLVM's ordinary LoadOpLowering explicitly emits nonvolatile loads.
The leaf below obtains the real descriptor base/offset/strides and emits an
actual standard LLVM volatile load. It never exposes an address as a source
value or fixes a GPU layout. Tensor tiles and logical control stay outside it.
"""

import json

from xdsl.dialects import arith, builtin, func, llvm, memref, scf
from xdsl.ir import Block, Region

from .diagnostics import ConversionError
from .standard_llvm import LoadOp

EFFECT_PROFILE = "private-invocation-volatile-loads-v1"


def contract(op, source_id):
    flag = op.properties.get("isVolatile", op.attributes.get("isVolatile"))
    if (not isinstance(flag, builtin.IntegerAttr) or flag.type != builtin.i1 or flag.value.data != -1
            or len(op.results) != 1):
        raise ConversionError("volatile read needs the actual pinned source boolean and one typed result",
                              source_id=source_id, opcode="tt.load")
    return {"schema": "t2m.volatile-read.v1", "source_id": source_id,
            "result_type": str(op.results[0].type), "scope": EFFECT_PROFILE,
            "observation": "one-actual-volatile-read-per-active-logical-lane",
            "external_participants": "excluded-by-private-buffer-ABI", "physical_mapping": "undecided"}


def adapt_payload(module, abi, records, source_ids):
    """Recover source reads before any nonvolatile cleanup can run."""
    function = next(iter(module.ops))
    buffers = {arg: value for arg, (kind, value) in zip(function.body.block.args, abi.source_bindings)
               if kind == "buffer"}
    contracts = {row["id"]: row["volatile_read"] for row in records if "volatile_read" in row}
    views = {"memref.cast", "memref.subview", "memref.reinterpret_cast", "memref.collapse_shape",
             "memref.expand_shape", "memref.memory_space_cast", "memref.view"}

    def root(value):
        while value not in buffers:
            op = value.owner
            if not hasattr(op, "name") or op.name not in views:
                return None
            value = op.operands[0]
        return buffers[value]

    reports, covered = [], set()
    for op in list(function.walk()):
        if not isinstance(op, memref.LoadOp) or (buffer := root(op.memref)) is None:
            continue
        selected = [sid for sid in source_ids(op.location) if sid in contracts]
        if not selected:
            continue
        if len(selected) != 1 or selected[0] in covered:
            raise ConversionError("volatile source read has an ambiguous or duplicated payload event")
        covered.update(selected)
        row = {"source_ids": selected, "buffer": buffer.name, "dtype": str(op.results[0].type),
               "descriptor_rank": len(op.memref.type.get_shape()), "representation": "standard-LLVM-volatile-descriptor-read"}
        phase = lower_load(op)
        phase.attributes["prov.volatile_read"] = builtin.StringAttr(json.dumps(row, sort_keys=True))
        reports.append(row)
    if covered != set(contracts):
        raise ConversionError("volatile source read lost its actual standard-memory observation")
    return reports


def protect_events(module, contracts, source_ids):
    """Keep source observations live using complete compiler-only identities.

    Native func.call remains conservative until the actual volatile leaf is
    present. No external call or numeric assertion supplies extra semantics.
    These calls and identity bodies never reach exported IR.
    """
    from .shared_tensor import opcode

    function = next(iter(module.ops))
    covered = set()
    occupied = {op.properties["sym_name"].data for op in module.ops
                if isinstance(op.properties.get("sym_name"), builtin.StringAttr)}
    for op in list(function.walk()):
        selected = set(source_ids(op.location)) & set(contracts)
        if opcode(op) != "tts.gather" or not selected:
            continue
        if len(selected) != 1 or selected & covered:
            raise ConversionError("volatile event anchoring requires one original load identity")
        covered |= selected
        type_ = op.results[0].type
        ordinal = len(covered)
        name = "__t2m_volatile_identity_" + str(ordinal)
        while name in occupied:
            ordinal += 1
            name = "__t2m_volatile_identity_" + str(ordinal)
        occupied.add(name)
        body = Block(arg_types=[type_])
        body.add_op(func.ReturnOp(body.args[0]))
        helper = func.FuncOp(name, ([type_], [type_]), Region(body), visibility="private")
        anchor = builtin.StringAttr(next(iter(selected)))
        helper.attributes["prov.volatile_anchor"] = anchor
        module.body.block.add_op(helper)
        call = func.CallOp(name, [op.results[0]], [type_])
        call.attributes["prov.volatile_anchor"] = anchor
        call.location = op.location
        op.parent_block().insert_op_after(call, op)
    if covered != set(contracts):
        raise ConversionError("volatile source load lost its typed pointer event before anchoring")


def checked_anchors(module):
    """Validate real identity bodies, all call types and private closed uses."""
    helpers, calls = {}, []
    for op in list(module.ops)[1:]:
        if (not isinstance(op, func.FuncOp) or op.sym_visibility != builtin.StringAttr("private")
                or not isinstance(op.attributes.get("prov.volatile_anchor"), builtin.StringAttr) or len(op.body.blocks) != 1):
            raise ConversionError("volatile intermediate has an unknown extra function")
        body = op.body.block
        if (len(body.args) != 1 or len(list(body.ops)) != 1 or not isinstance(body.last_op, func.ReturnOp)
                or tuple(body.last_op.arguments) != tuple(body.args)
                or list(op.function_type.outputs) != [body.args[0].type]):
            raise ConversionError("volatile intermediate anchor is not a complete pure identity")
        helpers[op.sym_name.data] = op
    for op in next(iter(module.ops)).walk():
        if not isinstance(op, func.CallOp):
            continue
        helper = helpers.get(op.callee.root_reference.data)
        if (helper is None or op.attributes.get("prov.volatile_anchor") != helper.attributes["prov.volatile_anchor"]
                or [v.type for v in op.operands] != list(helper.function_type.inputs)
                or [v.type for v in op.results] != list(helper.function_type.outputs)):
            raise ConversionError("volatile intermediate has an unresolved or changed anchor call")
        calls.append(op)
    if len(calls) != len(helpers) or {op.callee.root_reference.data for op in calls} != set(helpers):
        raise ConversionError("volatile intermediate anchors have missing or duplicate uses")
    return helpers, calls


def remove_anchors(module):
    helpers, calls = checked_anchors(module)
    restored = {}
    for phase in next(iter(module.ops)).walk():
        marker = phase.attributes.get("prov.volatile_read")
        if not isinstance(marker, builtin.StringAttr):
            continue
        row = json.loads(marker.data)
        loads = [op for op in phase.walk() if op.name == "llvm.load" and op.properties.get("volatile_") == builtin.UnitAttr()]
        if len(loads) == 1 and len(row["source_ids"]) == 1:
            sid = row["source_ids"][0]
            restored[sid] = restored.get(sid, 0) + 1
    if (len(restored) != len(calls) or any(
            restored.get(call.attributes["prov.volatile_anchor"].data) != 1 for call in calls)):
        raise ConversionError("volatile anchors cannot be removed before actual observations are restored")
    for call in calls:
        call.results[0].replace_all_uses_with(call.operands[0])
        call.parent_block().erase_op(call)
    for helper in helpers.values():
        module.body.block.erase_op(helper)


def validate_evidence(memory, admission, records, source, stage):
    """Tie the explicit profile and every observation to actual typed source."""
    from .shared_tensor import _source_ids, opcode, parse_shared

    if (memory.get("effect_profile") != EFFECT_PROFILE or admission.get("effect_profile") != EFFECT_PROFILE
            or memory.get("volatile_adaptations") != admission.get("volatile_adaptations")
            or not isinstance(admission.get("volatile_adaptations"), list)
            or stage.get("schema") != "t2m.source-word-stages.v3"):
        raise ValueError("volatile memory lacks its selected source/effect/stage contract")
    declared = {r["id"]: r["volatile_read"] for r in records if "volatile_read" in r}
    actual = {}
    module = parse_shared(source.decode() if isinstance(source, bytes) else source)
    for op in module.walk():
        flag = op.properties.get("isVolatile")
        if opcode(op) != "tt.load" or not isinstance(flag, builtin.IntegerAttr) or not flag.value.data:
            continue
        selected = set(_source_ids(op.location)) & set(declared)
        if len(selected) != 1:
            raise ValueError("volatile typed source has an unaccounted observation")
        sid = next(iter(selected))
        if sid in actual or contract(op, sid) != declared[sid]:
            raise ValueError("volatile source contract differs from its actual typed read")
        type_ = op.results[0].type
        actual[sid] = str(type_.element_type if isinstance(type_, builtin.TensorType) else type_)
    payload = admission["volatile_adaptations"]
    seen = set()
    for row in payload:
        ids = row.get("source_ids")
        if (not isinstance(ids, list) or len(ids) != 1 or ids[0] not in actual or ids[0] in seen
                or set(row) != {"source_ids", "buffer", "dtype", "descriptor_rank", "representation"}
                or type(row.get("descriptor_rank")) is not int or row["descriptor_rank"] < 0
                or row.get("dtype") != actual[ids[0]]
                or row.get("representation") != "standard-LLVM-volatile-descriptor-read"
                or not any(a.get("effect") == "read" and a.get("buffer") == row.get("buffer")
                           and ids[0] in a.get("source_ids", []) for a in admission["accesses"])):
            raise ValueError("volatile observations differ from guarded source memory accesses")
        seen.update(ids)
    if seen != set(declared) or set(actual) != set(declared):
        raise ValueError("volatile source/effect inventory lost an observation")


def validate_payload(module, expected):
    """Check actual volatile loads and the complete descriptor address recipe."""
    phases = [op for op in module.walk() if "prov.volatile_read" in op.attributes]
    if len(phases) != len(expected):
        raise ValueError("volatile executable payload lost an observation")
    all_leaves = set()
    for phase, row in zip(phases, expected, strict=True):
        if (not isinstance(phase, scf.ExecuteRegionOp) or len(phase.regions) != 1 or len(phase.regions[0].blocks) != 1
                or phase.regions[0].block.args or phase.attributes.get("no_inline") != builtin.UnitAttr()
                or json.loads(phase.attributes["prov.volatile_read"].data) != row):
            raise ValueError("volatile read lost its exactly-once typed region")
        ops = list(phase.regions[0].block.ops)
        rank = row["descriptor_rank"]
        names = ["memref.extract_strided_metadata", *[n for _ in range(rank) for n in ("arith.muli", "arith.addi")],
                 "arith.constant", "arith.muli", "memref.extract_aligned_pointer_as_index", "arith.addi",
                 "arith.index_cast", "llvm.inttoptr", "llvm.load", "scf.yield"]
        if [op.name for op in ops] != names:
            raise ValueError("volatile read changed its standard descriptor recipe")
        for op in ops:
            if op.name in ("arith.addi", "arith.muli"):
                flags = op.properties.get("overflowFlags")
                if not isinstance(flags, arith.IntegerOverflowAttr) or flags.data:
                    raise ValueError("volatile descriptor arithmetic cannot invent overflow promises")
        meta = ops[0]
        source = meta.operands[0]
        element = source.type.element_type
        if (len(source.type.get_shape()) != rank or str(element) != row["dtype"]
                or source.type.memory_space != builtin.NoneAttr()):
            raise ValueError("volatile descriptor differs from actual memory storage")
        offset = meta.results[1]
        for i in range(rank):
            mul, add = ops[1 + i * 2:3 + i * 2]
            if mul.operands[1] != meta.results[2 + rank + i] or tuple(add.operands) != (offset, mul.results[0]):
                raise ValueError("volatile read lost an actual descriptor offset/stride")
            offset = add.results[0]
        size, displacement, base, address, word, pointer, load, yield_ = ops[1 + rank * 2:]
        if (size.properties.get("value") != builtin.IntegerAttr(max(1, element.bitwidth // 8), builtin.IndexType())
                or tuple(displacement.operands) != (offset, size.results[0]) or base.operands[0] != source
                or tuple(address.operands) != (base.results[0], displacement.results[0])
                or word.operands[0] != address.results[0] or word.results[0].type != builtin.i64
                or pointer.operands[0] != word.results[0] or load.operands[0] != pointer.results[0]
                or pointer.results[0].type != llvm.LLVMPointerType()
                or load.properties.get("volatile_") != builtin.UnitAttr()
                or load.properties.get("ordering") != builtin.IntegerAttr(0, builtin.i64)
                or load.properties.get("alignment") is not None
                or tuple(yield_.operands) != tuple(load.results) or tuple(phase.result_types) != (element,)
                or len(tuple(pointer.results[0].uses)) != 1 or len(tuple(word.results[0].uses)) != 1):
            raise ValueError("volatile read lost its actual standard LLVM load semantics")
        all_leaves.update((pointer, load))
    if {op for op in module.walk() if op.name.startswith("llvm.")} != all_leaves:
        raise ValueError("volatile payload has an escaping or unaccounted low-level memory operation")


def lower_load(op):
    """Replace one active, in-bounds memref read using its actual descriptor."""
    if not isinstance(op, memref.LoadOp) or op.parent_block() is None:
        raise ConversionError("volatile memory leaf requires an attached standard memref load")
    type_ = op.memref.type
    element = type_.element_type
    width = getattr(element, "bitwidth", None)
    if (not isinstance(type_, builtin.MemRefType) or type_.memory_space != builtin.NoneAttr()
            or not isinstance(element, (builtin.IntegerType, builtin.AnyFloat))
            or width not in (1, 8, 16, 32, 64)):
        raise ConversionError("volatile memory leaf requires a plain logical scalar storage type")
    body = Block()

    def emit(operation):
        operation.location = op.location
        body.add_op(operation)
        return operation

    metadata = emit(memref.ExtractStridedMetaDataOp(op.memref))
    offset = metadata.offset
    for index, stride in zip(op.indices, metadata.strides, strict=True):
        product = emit(arith.MuliOp(index, stride))
        offset = emit(arith.AddiOp(offset, product.result)).result
    size = emit(arith.ConstantOp.from_int_and_width(max(1, width // 8), builtin.IndexType()))
    displacement = emit(arith.MuliOp(offset, size.result))
    base = emit(memref.ExtractAlignedPointerAsIndexOp.build(operands=[op.memref], result_types=[builtin.IndexType()]))
    address = emit(arith.AddiOp(base.results[0], displacement.result))
    word = emit(arith.IndexCastOp(address.result, builtin.i64))
    pointer = emit(llvm.IntToPtrOp(word))
    load = emit(LoadOp(pointer, element))
    load.properties["volatile_"] = builtin.UnitAttr()
    emit(scf.YieldOp(load.dereferenced_value))
    phase = scf.ExecuteRegionOp([element], Region(body))
    phase.attributes.update({"no_inline": builtin.UnitAttr(),
        "prov.family": builtin.StringAttr("memory"), "prov.op": builtin.StringAttr("volatile_read")})
    phase.location = op.location
    block = op.parent_block()
    block.insert_op_before(phase, op)
    op.results[0].replace_all_uses_with(phase.results[0])
    block.erase_op(op)
    return phase

"""Finite logical-buffer selectors paired with original source-word offsets.

The Meta 17f0c234 structured pointer branch analysis motivates selecting the
base separately from its offset. Here that selection remains ordinary integer
SSA through the source-word path; no physical pointer select reaches the public
tensor ABI. Memory events stay at their original position and reuse upstream
gather/scatter lowering plus the existing executable memory admission.
"""

from collections import deque

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, TensorType
from xdsl.ir import BlockArgument, OpResult, SSAValues
from xdsl.rewriter import Rewriter

from .pointer_control import lower_offsets
from .shared_tensor import _array, opcode
from .word_events import MEMORY_EVENTS


def normalize_selected(state, source_ids):
    original = state.operations
    operands = {op: tuple(op.operands) for op in original}
    results = {op: tuple(op.results) for op in original}
    args = {block: tuple(block.args) for op in original for region in op.regions for block in region.blocks}
    tags, constants = {}, {}

    # Uniform branch/program selectors stay scalar even for tensor pointers.
    # Only lane-varying selection/concatenation seeds a tensor selector, then
    # that fact propagates over every phi/backedge to a finite fixed point.
    varying = {v: False for v in state.values}
    users = {v: [] for v in state.values}
    for value, incoming in state.edges.items():
        for source in incoming:
            users[source].append(value)
        if len(state.bases[value]) < 2 or not isinstance(state.types[value], TensorType) or isinstance(value, BlockArgument):
            continue
        op = value.owner
        if opcode(op) == "arith.select" and isinstance(operands[op][0].type, TensorType):
            varying[value] = operands[op][1] != operands[op][2]
        elif opcode(op) in ("tt.cat", "tt.join"):
            varying[value] = len(set(incoming)) > 1
    pending = deque(v for v in state.values if varying[v])
    while pending:
        value = pending.popleft()
        for user in users[value]:
            if not varying[user] and isinstance(state.types[user], TensorType) and len(state.bases[user]) > 1:
                varying[user] = True
                pending.append(user)

    def tag_type(value):
        return TensorType(builtin.i32, state.types[value].get_shape()) if varying[value] else builtin.i32

    def at(op, new):
        new.location = op.location
        op.parent.insert_op_before(new, op)
        return new.results[0]

    def raw(name, inputs, types, properties=None, attributes=None):
        return builtin.UnregisteredOp.with_name(name).create(operands=inputs, result_types=types,
            properties=properties or {}, attributes=attributes or {})

    def broadcast(value, type_, before):
        if value.type == type_:
            return value
        assert isinstance(type_, TensorType) and value.type == type_.element_type
        return at(before, raw("tt.splat", [value], [type_]))

    def root_tag(base):
        if base not in constants:
            constant = arith.ConstantOp.from_int_and_width(base.index, 32)
            constant.location = state.function.location
            state.entry.insert_op_before(constant, state.entry.first_op)
            constants[base] = constant.result
        return constants[base]

    def tag(value):
        if value not in tags:
            assert len(state.bases[value]) == 1
            tags[value] = root_tag(next(iter(state.bases[value])))
        return tags[value]

    def add_result(op, value):
        result = OpResult(tag_type(value), op, len(op.results))
        op.results = SSAValues((*op.results, result))
        tags[value] = result

    def mixed(values):
        return [i for i, value in enumerate(values) if value in state.bases and len(state.bases[value]) > 1]

    # Preallocate selector phis before traversing their cyclic definitions.
    for op in original:
        name = opcode(op)
        if name not in ("scf.for", "scf.while", "scf.if"):
            continue
        for value in results[op]:
            if value in state.bases and len(state.bases[value]) > 1:
                add_result(op, value)
        for region in op.regions:
            for block in region.blocks:
                for value in args[block]:
                    if value in state.bases and len(state.bases[value]) > 1:
                        tags[value] = block.insert_arg(tag_type(value), len(block.args))

    for op in original:
        pointer_results = [v for v in results[op] if v in state.bases and len(state.bases[v]) > 1]
        if not pointer_results or opcode(op) in ("scf.for", "scf.while", "scf.if"):
            continue
        name = opcode(op)
        if name == "tt.addptr":
            tags[pointer_results[0]] = tag(operands[op][0])
        elif name == "arith.select":
            value = pointer_results[0]
            left, right = (broadcast(tag(v), tag_type(value), op) for v in operands[op][1:])
            tags[value] = at(op, arith.SelectOp(operands[op][0], left, right))
        else:
            pointers = [v for v in operands[op] if v in state.bases]
            incoming = [tag(v) for v in pointers]
            if all(v.type == builtin.i32 for v in incoming) and len(set(incoming)) == 1 and all(tag_type(v) == builtin.i32 for v in pointer_results):
                for value in pointer_results:
                    tags[value] = incoming[0]
                continue
            actual = []
            for value in operands[op]:
                if value not in state.bases:
                    actual.append(value)
                else:
                    type_ = state.types[value]
                    type_ = TensorType(builtin.i32, type_.get_shape()) if isinstance(type_, TensorType) else builtin.i32
                    actual.append(broadcast(tag(value), type_, op))
            copied = raw(name, actual, [tag_type(v) for v in results[op]], dict(op.properties), dict(op.attributes))
            copied.location = op.location
            op.parent.insert_op_before(copied, op)
            for old, new in zip(results[op], copied.results, strict=True):
                tags[old] = new

    lower_offsets(state)

    # While input and result states can have different arities/types. Append
    # each selector on its own original edge, preserving all original slots.
    for op in original:
        name = opcode(op)
        if name == "scf.for":
            body = op.regions[0].block
            terminal = body.last_op
            slots = mixed(args[body][1:])
            op.operands = [*op.operands, *(broadcast(tag(operands[op][i + 3]), tag_type(args[body][i + 1]), op) for i in slots)]
            terminal.operands = [*terminal.operands, *(broadcast(tag(operands[terminal][i]), tag_type(args[body][i + 1]), terminal) for i in slots)]
        elif name == "scf.while":
            before, after = (region.block for region in op.regions)
            condition, terminal = before.last_op, after.last_op
            inputs, outputs = mixed(args[before]), mixed(results[op])
            op.operands = [*op.operands, *(broadcast(tag(operands[op][i]), tag_type(args[before][i]), op) for i in inputs)]
            condition.operands = [*condition.operands, *(broadcast(tag(operands[condition][i + 1]), tag_type(results[op][i]), condition) for i in outputs)]
            terminal.operands = [*terminal.operands, *(broadcast(tag(operands[terminal][i]), tag_type(args[before][i]), terminal) for i in inputs)]
        elif name == "scf.if":
            for region in op.regions:
                terminal = region.block.last_op
                terminal.operands = [*terminal.operands, *(broadcast(tag(operands[terminal][i]), tag_type(results[op][i]), terminal) for i in mixed(results[op]))]

    memory_reports = []
    for op in original:
        name = opcode(op)
        if name not in MEMORY_EVENTS:
            continue
        pointer = operands[op][0]
        type_ = state.memory_types.get(op, state.types[pointer])
        roots = sorted(state.bases[pointer], key=lambda value: value.index)
        offset = op.operands[0]

        def address(base):
            root = base
            scalar = type_.element_type if isinstance(type_, TensorType) else type_
            if scalar != base.type:
                root = at(op, raw("tt.bitcast", [base], [scalar]))
            if isinstance(type_, TensorType):
                root = at(op, raw("tt.splat", [root], [type_]))
            return at(op, raw("tt.addptr", [root, offset], [type_]))

        if len(roots) == 1:
            op.operands = [address(roots[0]), *op.operands[1:]]
            continue
        if name == "tt.atomic_cas":
            from .compare_exchange import dispatch
            ids = source_ids(op.location)
            dispatch(op, roots, offset, tag(pointer))
            memory_reports.append({"kind": "selected-pointer-memory", "source_ids": ids, "opcode": name,
                "bases": [v.index for v in roots], "selector": "per-lane" if varying[pointer] else "uniform",
                "memory_position": "original-event; conditional-CAS-in-parallel-logical-tile",
                "missing_other_zero_refinement": False})
            continue
        write = name == "tt.store"
        atomic = name == "tt.atomic_rmw"
        if write or atomic:
            original_mask = op.operands[2] if len(op.operands) > 2 else None
            other = None
        else:
            segments = _array(op, "operandSegmentSizes")
            original_mask = op.operands[1] if segments[1] else None
            other = op.operands[1 + segments[1]] if segments[2] else None
        bool_type = TensorType(builtin.i1, type_.get_shape()) if isinstance(type_, TensorType) else builtin.i1
        false = broadcast(at(op, arith.ConstantOp.from_int_and_width(0, 1)), bool_type, op)
        if not write:
            result_type = op.results[0].type
            element = result_type.element_type if isinstance(result_type, TensorType) else result_type
            constant = (arith.ConstantOp.from_int_and_width(0, element) if isinstance(element, builtin.IntegerType)
                        else arith.ConstantOp(builtin.FloatAttr(0.0, element)))
            internal_zero = broadcast(at(op, constant), result_type, op)
            if other is None:
                other = internal_zero
        merged = other
        for base in roots:
            selector = tag(pointer)
            reference = broadcast(root_tag(base), selector.type, op)
            compared_type = TensorType(builtin.i1, selector.type.get_shape()) if isinstance(selector.type, TensorType) else builtin.i1
            comparison = arith.CmpiOp.create(operands=[selector, reference], result_types=[compared_type],
                                            properties={"predicate": IntegerAttr(0, builtin.i64)})
            match = broadcast(at(op, comparison), bool_type, op)
            mask = match if original_mask is None else at(op, arith.SelectOp(original_mask, match, false))
            properties = dict(op.properties)
            attributes = dict(op.attributes)
            attributes["prov.pointer_base_ordinal"] = IntegerAttr(base.index, builtin.i32)
            if write:
                selected = raw(name, [address(base), op.operands[1], mask], [], properties, attributes)
            elif atomic:
                selected = raw(name, [address(base), op.operands[1], mask], [op.results[0].type], properties, attributes)
            else:
                properties["operandSegmentSizes"] = DenseArrayBase.from_list(builtin.i32, [1, 1, 1])
                attributes.pop("operandSegmentSizes", None)
                selected = raw(name, [address(base), mask, internal_zero], [op.results[0].type], properties, attributes)
            selected.location = op.location
            op.parent.insert_op_before(selected, op)
            if not write:
                merged = at(op, arith.SelectOp(mask, selected.results[0], merged))
        memory_reports.append({"kind": "selected-pointer-memory", "source_ids": source_ids(op.location),
            "opcode": name, "bases": [v.index for v in roots], "selector": "per-lane" if varying[pointer] else "uniform",
            "memory_position": "original-event; no-hoisting-across-effects",
            "missing_other_zero_refinement": not write and not atomic and bool(_array(op, "operandSegmentSizes")[1]) and not bool(_array(op, "operandSegmentSizes")[2])})
        Rewriter.replace_op(op, [], [] if write else [merged])

    reports = [{"kind": "selected-pointer-control", "source_ids": source_ids(op.location), "opcode": opcode(op),
        "pointer_offset_bits": 64, "selector_bits": 32, "selector_domain": "source-entry-buffer-argument-ordinals",
        "base_contract": "finite-source-entry-buffers; unchanged-element-type",
        "source_integer_arithmetic": "unchanged-before-signed-extension", "control_flow": "original-regions-and-edge-order"}
        for op in state.controls]
    return reports + memory_reports

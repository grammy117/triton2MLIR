"""Expose pointer state as logical bases and signed i64 element offsets.

Meta triton-shared 17f0c234's TritonToUnstructuredPass.cpp provides the
base/offset representation and offset-bit-width policy. Its use traversal
handles scf.for but leaves pointer-valued scf.if/while unresolved. This narrow
adapter supplies those region edges and finite selectors before running the unchanged upstream
converter. It does not choose a physical address, memory space or grid mapping.
"""

from collections import deque
from dataclasses import dataclass

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import BlockArgument
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


@dataclass
class PointerState:
    operations: tuple
    controls: tuple
    entry: object
    function: object
    values: tuple
    types: dict
    bases: dict
    edges: dict


def offset_type(type_):
    return TensorType(builtin.i64, type_.get_shape()) if isinstance(type_, TensorType) else builtin.i64


def lower_offsets(state):
    """Shared source-word conversion for single- and selected-base state."""
    from .shared_tensor import opcode
    offsets = {}
    for value in state.values:
        if isinstance(value, BlockArgument) and value.block is state.entry:
            zero = arith.ConstantOp.from_int_and_width(0, 64)
            zero.location = state.function.location
            state.entry.insert_op_before(zero, state.entry.first_op)
            offsets[value] = zero.result
        else:
            offsets[value] = Rewriter.replace_value_with_new_type(value, offset_type(value.type))
    for op in state.operations:
        op.operands = [offsets.get(v, v) for v in op.operands]
    for op in state.operations:
        if opcode(op) != "tt.addptr":
            continue
        lhs, rhs = op.operands
        added = []
        if rhs.type != lhs.type:
            extended = arith.ExtSIOp(rhs, lhs.type)
            added.append(extended)
            rhs = extended.result
        result = arith.AddiOp(lhs, rhs)
        added.append(result)
        for emitted in added:
            emitted.location = op.location
        Rewriter.replace_op(op, added)


def normalize_pointer_control(module, source_ids):
    from .shared_tensor import _pointer_type, opcode

    original = list(module.walk())
    controls = [op for op in original if opcode(op) in ("scf.if", "scf.while", "scf.for", "arith.select",
                                                     "tt.cat", "tt.join", "tt.split", "tt.trans", "tt.reshape", "tt.gather", "tt.unsplat")
                and any(_pointer_type(v.type) for v in (*op.results,
                    *(a for region in op.regions for block in region.blocks for a in block.args)))]
    if not controls:
        return []

    def refuse(op, message):
        raise ConversionError(message, opcode=opcode(op),
                              source_id=next(iter(source_ids(op.location)), None))

    functions = [op for op in original if opcode(op) == "tt.func"]
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        refuse(controls[0], "pointer control needs one fully inlined entry block")
    entry = functions[0].regions[0].block
    values = [v for op in original for v in (*op.results,
              *(a for region in op.regions for block in region.blocks for a in block.args))
              if _pointer_type(v.type)]
    types = {v: v.type for v in values}
    bases = {v: {v} if isinstance(v, BlockArgument) and v.block is entry else set() for v in values}
    edges = {}
    shape_ops = {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans",
                 "tt.cat", "tt.join", "tt.split", "tt.gather", "tt.unsplat"}

    def block(op, index):
        if index >= len(op.regions) or len(op.regions[index].blocks) != 1:
            refuse(op, "pointer control requires single-block structured regions")
        return op.regions[index].block

    def terminator(op, region, name):
        result = block(op, region).last_op
        if result is None or opcode(result) != name:
            refuse(op, "pointer control has an unexpected region terminator")
        return result

    for value in values:
        if isinstance(value, BlockArgument):
            if value.block is entry:
                edges[value] = ()
                continue
            op = value.block.parent_op()
            if opcode(op) == "scf.for" and value.index > 0:
                index = value.index - 1
                inputs = (op.operands[index + 3], terminator(op, 0, "scf.yield").operands[index])
            elif opcode(op) == "scf.while":
                if value.block is block(op, 0):
                    inputs = (op.operands[value.index], terminator(op, 1, "scf.yield").operands[value.index])
                else:
                    inputs = (terminator(op, 0, "scf.condition").operands[value.index + 1],)
            else:
                refuse(op, "pointer block arguments need a qualified structured control edge")
        else:
            op, index = value.owner, value.index
            name = opcode(op)
            if name == "tt.addptr" or name in shape_ops:
                inputs = tuple(v for v in op.operands if _pointer_type(v.type))
            elif name == "arith.select":
                inputs = tuple(op.operands[1:])
            elif name == "scf.if":
                inputs = tuple(terminator(op, r, "scf.yield").operands[index] for r in (0, 1))
            elif name == "scf.for":
                inputs = (op.operands[index + 3], terminator(op, 0, "scf.yield").operands[index])
            elif name == "scf.while":
                inputs = (terminator(op, 0, "scf.condition").operands[index + 1],)
            else:
                refuse(op, "pointer control has an unqualified pointer producer")
        if not inputs or any(v not in bases for v in inputs):
            refuse(op, "pointer control has an unknown logical base")
        edges[value] = inputs

    # Monotone sets over the finite source graph: cycles keep their entry base,
    # and every alternate/backedge base participates in the same fixed point.
    users = {v: [] for v in values}
    for value, inputs in edges.items():
        for incoming in inputs:
            users[incoming].append(value)
    pending = deque(v for v in values if bases[v])
    while pending:
        value = pending.popleft()
        for user in users[value]:
            previous = len(bases[user])
            bases[user].update(bases[value])
            if len(bases[user]) != previous:
                pending.append(user)
    for value in values:
        owner = value.block.parent_op() if isinstance(value, BlockArgument) else value.owner
        if not bases[value]:
            refuse(owner, "pointer control has no established source-entry base")
        element = value.type.element_type if isinstance(value.type, TensorType) else value.type
        if any(element != base.type for base in bases[value]):
            refuse(owner, "pointer control cannot change the logical buffer element type")
        for use in value.uses:
            op, position = use.operation, use.index
            name = opcode(op)
            allowed = (name == "tt.addptr" and position == 0 or name in shape_ops
                       or name == "arith.select" and position in (1, 2)
                       or name == "scf.for" and position >= 3 or name == "scf.while"
                       or name == "scf.yield" or name == "scf.condition" and position >= 1
                       or name in ("tt.load", "tt.store") and position == 0)
            if not allowed:
                refuse(op, "pointer control cannot observe or escape a logical buffer address")

    # Validate arithmetic before mutating anything. Only addptr's operand is
    # extended; computations producing it keep their original source widths.
    for op in original:
        if opcode(op) == "tt.addptr":
            rhs = op.operands[1].type
            rhs = rhs.element_type if isinstance(rhs, TensorType) else rhs
            if not isinstance(rhs, IntegerType) or not 1 <= rhs.width.data <= 64:
                refuse(op, "pointer control requires integer element offsets no wider than i64")

    state = PointerState(tuple(original), tuple(controls), entry, functions[0], tuple(values), types, bases, edges)
    if any(len(roots) > 1 for roots in bases.values()):
        from .pointer_select import normalize_selected
        return normalize_selected(state, source_ids)
    # The official converter already handles same-base for state. Leave that
    # supported case upstream when no other pointer control needs this adapter.
    if all(opcode(op) == "scf.for" for op in controls):
        return []
    memory = {op: (types[op.operands[0]], next(iter(bases[op.operands[0]])))
              for op in original if opcode(op) in ("tt.load", "tt.store")}
    lower_offsets(state)

    # Reconstruct pointers only at memory events. Upstream now sees local
    # base+offset uses and no pointer-valued structured region edge.
    for op, (type_, base) in memory.items():
        pointer = base
        emitted = []
        if isinstance(type_, TensorType):
            splat = builtin.UnregisteredOp.with_name("tt.splat").create(operands=[base], result_types=[type_])
            emitted.append(splat)
            pointer = splat.results[0]
        address = builtin.UnregisteredOp.with_name("tt.addptr").create(
            operands=[pointer, op.operands[0]], result_types=[type_])
        emitted.append(address)
        for item in emitted:
            item.location = op.location
        op.parent.insert_ops_before(emitted, op)
        op.operands = [address.results[0], *op.operands[1:]]

    return [{"kind": "same-base-pointer-control", "source_ids": source_ids(op.location),
             "opcode": opcode(op), "pointer_offset_bits": 64,
             "base_contract": "single-source-entry-buffer-on-all-edges",
             "source_integer_arithmetic": "unchanged-before-signed-extension",
             "control_flow": "original-regions-and-edge-order"} for op in controls]

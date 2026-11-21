"""Actual scalar dependencies determine logical-program uniformity.

Loop invariance and interval equality do not imply uniformity across programs.
Follow SCF tuple edges, including their original controls, without evaluating
or speculating any source operation. Unknown memory/effects remain nonuniform.
"""

from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import BlockArgument

from .invariant_recurrences import SCALAR_ARITH


def is_grid_uniform(value, inputs, opcode):
    pending, visited = [value], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > 512 or not isinstance(current.type, (IntegerType, IndexType)):
            return False
        if current in inputs:
            continue
        op = current.owner
        if isinstance(current, BlockArgument):
            op = current.block.parent_op()
            if op is None or opcode(op) != "scf.for":
                return False
            pending.extend(op.operands[:3])
            if current.index:
                pending.extend((op.iter_args[current.index-1], op.body.block.last_op.operands[current.index-1]))
            continue
        kind = opcode(op)
        if kind == "scf.for":
            pending.extend((*op.operands[:3], op.iter_args[current.index], op.body.block.last_op.operands[current.index]))
        elif kind in ("scf.if", "scf.execute_region"):
            if kind == "scf.if":
                pending.append(op.operands[0])
            if any(len(region.blocks) != 1 or region.block.args for region in op.regions):
                return False
            for region in op.regions:
                end = region.block.last_op
                if end is None or opcode(end) != "scf.yield" or len(end.operands) != len(op.results):
                    return False
                pending.append(end.operands[current.index])
        elif kind == "tt.get_num_programs":
            continue
        elif kind in SCALAR_ARITH and not op.regions:
            pending.extend(op.operands)
        else:
            return False
    return True

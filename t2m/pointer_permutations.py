"""Carry scalar table indices for closed permutations of immutable selector tiles.

Only exact pointer forwarding through addptr/bitcast qualifies. Source offsets
retain their ordinary recurrences; computed selectors use the general path.
"""

from dataclasses import dataclass

from xdsl.dialects import builtin

from .shared_tensor import opcode


@dataclass(frozen=True)
class SelectorTable:
    loop: object
    initials: tuple


def selector_tables(state, varying, operands, args, results):
    encoded, reports = {}, []
    def forwarded(value):
        while hasattr(value.owner, "name") and opcode(value.owner) in ("tt.addptr", "tt.bitcast"):
            value = operands[value.owner][0]
        return value
    for loop in state.operations:
        name = opcode(loop)
        if name == "scf.for":
            block = loop.regions[0].block
            carried, initial = args[block][1:], operands[loop][3:]
        elif name == "scf.while":
            before, block = (region.block for region in loop.regions)
            if operands[before.last_op][1:] != args[before] or len(results[loop]) != len(args[before]):
                continue
            carried, initial = args[block], operands[loop]
        else:
            continue
        previous = {v: i for i, v in enumerate(carried)}
        terminal = operands[block.last_op]
        mapping = {i: previous[forwarded(value)] for i, value in enumerate(terminal)
                   if carried[i] in varying and varying[carried[i]]
                   and forwarded(value) in previous}
        visited = set()
        for start in mapping:
            path, seen, current = [], {}, start
            while current in mapping and current not in seen and current not in visited:
                seen[current] = len(path)
                path.append(current)
                current = mapping[current]
            visited.update(path)
            cycle = path[seen[current]:] if current in seen else []
            if (len(cycle) < 2 or any(state.types[carried[i]] != state.types[carried[cycle[0]]] for i in cycle)
                    or any(forwarded(initial[i]) in encoded for i in cycle)):
                continue
            table = SelectorTable(loop, tuple(initial[i] for i in cycle))
            for i in cycle:
                encoded[carried[i]] = encoded[results[loop][i]] = table
                if name == "scf.while":
                    encoded[args[before][i]] = table
            reports.append((loop, cycle))
    # Addptr changes only offsets; an equal-width bitcast changes only the view.
    # Neither changes the immutable selector table or its scalar selection.
    for op in state.operations:
        if opcode(op) not in ("tt.addptr", "tt.bitcast") or not results[op]:
            continue
        source = operands[op][0]
        value = results[op][0]
        if source in encoded and value in state.types and isinstance(state.types[value], builtin.TensorType):
            encoded[value] = encoded[source]
    return encoded, reports

"""Definite output initialization for the structured, unconditional memory path.

Fresh tensor results may omit incoming storage only after executable source
stores cover every returned element and every observed load is initialized.
This analysis does not consume caller provenance as a proof.
"""

from math import prod

from .diagnostics import ConversionError


class DefiniteWrites:
    def __init__(self, abi):
        self.grid = abi.grid
        self.fresh = {b.name: b for b in abi.buffers if b.effect == "write"}
        self.writes = {name: [] for name in self.fresh}

    def load(self, buffer, offsets, sizes, *, source_id=None, opcode=None):
        if buffer.name not in self.fresh or not all(sizes):
            return
        for written, extent in self.writes[buffer.name]:
            differences = [read + start.scale(-1) for read, start in zip(offsets, written, strict=True)]
            if all(delta.bounds(self.grid)[0] >= 0 and delta.bounds(self.grid)[1] + size <= bound
                   for delta, size, bound in zip(differences, sizes, extent, strict=True)):
                return
        raise ConversionError(f"fresh output {buffer.name} is read before its observed tile is initialized",
                              source_id=source_id, opcode=opcode)

    def store(self, buffer, offsets, sizes):
        if buffer.name in self.fresh and all(sizes):
            self.writes[buffer.name].append((offsets, sizes))

    def snapshot(self):
        return {name: list(values) for name, values in self.writes.items()}

    def restore(self, state):
        self.writes = {name: list(values) for name, values in state.items()}

    def merge_branches(self, first, second):
        # A store in only one branch cannot initialize the other branch. Keep
        # boxes established on both paths; unions/intersections of differently
        # partitioned boxes need a separate coverage rule.
        self.writes = {name: [box for box in boxes if box in second[name]]
                       for name, boxes in first.items()}

    def finish(self, *, source_id=None):
        evidence = {}
        for name, buffer in self.fresh.items():
            for offsets, sizes in self.writes[name]:
                if any(off.loops for off in offsets):
                    continue  # Union coverage over changing loop slices is a later rule.
                if all(off.bounds(self.grid)[0] == 0 and off.bounds(self.grid)[1] + size == extent
                       for off, size, extent in zip(offsets, sizes, buffer.shape, strict=True)):
                    # Parallel disjointness is established independently by the
                    # functionalizer; equal cardinality then proves no holes.
                    if prod(self.grid) * prod(sizes) == prod(buffer.shape):
                        evidence[name] = {"initial_value_required": False,
                                          "coverage": "disjoint-bounded-tiles-with-equal-total-cardinality",
                                          "reads": "dominated-by-covering-source-stores"}
                        break
            else:
                raise ConversionError(f"fresh output {name} has no proved complete overwrite", source_id=source_id)
        return evidence

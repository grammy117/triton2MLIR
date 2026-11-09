"""Definite output initialization for the structured, unconditional memory path.

Fresh tensor results may omit incoming storage only after executable source
stores cover every returned element and every observed load is initialized.
This analysis does not consume caller provenance as a proof.
"""

from math import prod

from .diagnostics import ConversionError
from .tail_masks import TailExtent, maximum
from .dynamic_abi import GridExtent
from .shared_indexing import Linear


class DefiniteWrites:
    def __init__(self, abi):
        self.abi = abi
        self.grid = abi.grid_bounds
        self.varying_axes = {i for i, g in enumerate(abi.grid) if isinstance(g, GridExtent)
                             and (next(s for s in abi.runtime_scalars if s.name == g.dimension).bounds[0] + g.divisor - 1) // g.divisor != self.grid[i]}
        self.fresh = {b.name: b for b in abi.buffers if b.effect == "write"}
        self.writes = {name: [] for name in self.fresh}

    def load(self, buffer, offsets, sizes, *, source_id=None, opcode=None):
        if buffer.name not in self.fresh or not all(maximum(n) for n in sizes):
            return
        for written, extent in self.writes[buffer.name]:
            differences = [read + start.scale(-1) for read, start in zip(offsets, written, strict=True)]
            if all((delta.constant == 0 and not delta.dynamic and size == bound) or
                   (not isinstance(bound, TailExtent) and delta.bounds(self.grid)[0] >= 0
                    and delta.bounds(self.grid)[1] + maximum(size) <= bound)
                   for delta, size, bound in zip(differences, sizes, extent, strict=True)):
                return
        raise ConversionError(f"fresh output {buffer.name} is read before its observed tile is initialized",
                              source_id=source_id, opcode=opcode)

    def store(self, buffer, offsets, sizes):
        if buffer.name in self.fresh and all(maximum(n) for n in sizes):
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
            shape = tuple(self.abi.dimension(n) for n in buffer.shape)
            if any(isinstance(n, Linear) and n.bounds(self.grid) == (0, 0) for n in shape):
                evidence[name] = {"initial_value_required": False, "coverage": "zero-element-result", "reads": "separately-checked"}
                continue
            for offsets, sizes in self.writes[name]:
                if any(off.loops for off in offsets):
                    continue  # Union coverage over changing loop slices is a later rule.
                if any(isinstance(size, TailExtent) for size in sizes):
                    if self._clipped_partition(offsets, sizes, shape):
                        evidence[name] = {"initial_value_required": False,
                                          "coverage": "disjoint-clipped-affine-prefix-partition",
                                          "reads": "dominated-by-covering-source-stores"}
                        break
                    continue
                if self.varying_axes or any(isinstance(n, Linear) for n in shape):
                    if self._exact_runtime_partition(offsets, sizes, shape):
                        evidence[name] = {"initial_value_required": False,
                                          "coverage": "disjoint-exact-runtime-grid-partition",
                                          "reads": "dominated-by-covering-source-stores"}
                        break
                    continue
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

    def _exact_runtime_partition(self, offsets, sizes, shape):
        """Complete unclipped tiles, independently of access/ownership checks."""
        used = set()
        for off, size, dimension in zip(offsets, sizes, shape, strict=True):
            if not isinstance(size, int) or size <= 0 or off.constant or off.loops or off.parameters:
                return False
            axes = [i for i, coefficient in enumerate(off.programs)
                    if coefficient and i < len(self.grid) and self.grid[i] > 1]
            axes.extend(i for i in self.varying_axes if self.grid[i] == 1
                        and self.abi.dimension(self.abi.grid[i].dimension) == dimension
                        and self.abi.grid[i].divisor == size)
            if len(axes) > 1:
                return False
            programs = self.grid[axes[0]] if axes else 1
            axis = axes[0] if axes else None
            if axis is not None:
                if axis in used or self.grid[axis] > 1 and off.programs[axis] != size:
                    return False
                used.add(axis)
            if not self._exact_axis(dimension, size, axis, programs):
                return False
        return used == {i for i, count in enumerate(self.grid) if count > 1} | self.varying_axes

    def _exact_axis(self, dimension, size, axis, programs):
        if isinstance(dimension, Linear):
            bounds = dimension.bounds(self.grid)
            launch = self.abi.grid[axis] if axis is not None else None
            if (isinstance(launch, GridExtent) and self.abi.dimension(launch.dimension) == dimension
                    and launch.divisor == size):
                # A contiguous guarded interval establishes divisibility only
                # for unit tiles or a fixed, aligned dimension value.
                return size == 1 or bounds[0] == bounds[1] and bounds[0] % size == 0
            return axis not in self.varying_axes and bounds == (programs * size, programs * size)
        return axis not in self.varying_axes and programs * size == dimension

    def _clipped_partition(self, offsets, sizes, shape):
        used = set()
        for off, size, dimension in zip(offsets, sizes, shape, strict=True):
            axes = [i for i, coefficient in enumerate(off.programs)
                    if coefficient and i < len(self.grid) and self.grid[i] > 1]
            axes.extend(i for i in self.varying_axes if self.grid[i] == 1
                        and self.abi.dimension(self.abi.grid[i].dimension) == dimension
                        and self.abi.grid[i].divisor <= maximum(size))
            if off.constant != 0 or off.loops or off.parameters or len(axes) > 1:
                return False
            programs = self.grid[axes[0]] if axes else 1
            cap = maximum(size)
            if axes:
                axis = axes[0]
                if axis in used or self.grid[axis] > 1 and off.programs[axis] != cap:
                    return False
                if axis in self.varying_axes:
                    launch = self.abi.grid[axis]
                    if self.abi.dimension(launch.dimension) != dimension or launch.divisor > cap:
                        return False
                used.add(axis)
            if isinstance(size, TailExtent):
                limit = size.limit if isinstance(size.limit, Linear) else Linear(size.limit)
                bound = dimension.bounds(self.grid)[1] if isinstance(dimension, Linear) else dimension
                if not size.fits(off, dimension) or limit != (dimension if isinstance(dimension, Linear) else Linear(dimension)) or programs * cap < bound:
                    return False
            elif not self._exact_axis(dimension, size, axes[0] if axes else None, programs):
                return False
        return used == {i for i, count in enumerate(self.grid) if count > 1} | self.varying_axes

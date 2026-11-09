"""Recover logical tensor axis permutations from upstream structured strides."""

from .diagnostics import ConversionError
from .tail_masks import TailExtent, maximum
from .shared_indexing import Linear


def axis_permutation(buffer, strides, offsets, active, grid, *, physical_offsets=False, dimensions=None, proven_axes=None):
    rank = len(buffer.shape)
    if len(strides) != rank or len(offsets) != rank:
        raise ConversionError("structured tensor view does not fit declared rank/strides/bounds")
    empty = not all(maximum(n) for n in active)
    dimensions = buffer.shape if dimensions is None else dimensions

    def matches(source, destination):
        low, high = offsets[destination if physical_offsets else source].bounds(grid)
        size = active[source]
        offset, dimension = offsets[destination if physical_offsets else source], dimensions[destination]
        bounded = size.fits_access(offset, dimension, grid) if isinstance(size, TailExtent) else (
            (offset + Linear(size) + dimension.scale(-1)).bounds(grid)[1] <= 0 if isinstance(dimension, Linear)
            else high + size <= dimension)
        matched_stride = proven_axes is not None and proven_axes[source] == destination or (
            proven_axes is None and (strides[source] == buffer.strides[destination] or strides[source] == 0 and maximum(size) <= 1))
        return matched_stride and (empty or low >= 0 and bounded)

    if proven_axes is not None:
        if sorted(proven_axes) != list(range(rank)) or not all(matches(i, j) for i, j in enumerate(proven_axes)):
            raise ConversionError("active structured access exceeds the declared tensor bounds")
        return proven_axes
    if all(matches(i, i) for i in range(rank)):
        return tuple(range(rank))
    if rank > 8:
        raise ConversionError("nonidentity tensor axis recovery exceeds the eight-axis analysis budget")
    # Degenerate dimensions can share a physical stride. Bounds, not an
    # arbitrary stride sort, determine which axis can carry nonunit extents.
    choices = [[j for j in range(rank) if matches(i, j)] for i in range(rank)]
    order = sorted(range(rank), key=lambda i: len(choices[i]))
    result, attempts = [None] * rank, 0

    def solve(position, used):
        nonlocal attempts
        attempts += 1
        if attempts > 256:
            raise ConversionError("tensor axis recovery exhausted its bounded search")
        if position == rank:
            return True
        source = order[position]
        for destination in choices[source]:
            if destination not in used:
                result[source] = destination
                if solve(position + 1, used | {destination}):
                    return True
        return False

    if not solve(0, set()):
        if tuple(strides) == buffer.strides:
            raise ConversionError("active structured access exceeds the declared tensor bounds")
        raise ConversionError("structured tensor view does not fit declared rank/strides/bounds")
    return tuple(result)


def inverse_permutation(axes):
    return tuple(axes.index(i) for i in range(len(axes)))

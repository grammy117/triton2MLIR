"""Recover logical tensor axis permutations from upstream structured strides."""

from .diagnostics import ConversionError


def axis_permutation(buffer, strides, offsets, active, grid, *, physical_offsets=False):
    rank = len(buffer.shape)
    if len(strides) != rank or len(offsets) != rank:
        raise ConversionError("structured tensor view does not fit declared rank/strides/bounds")
    empty = not all(active)

    def matches(source, destination):
        low, high = offsets[destination if physical_offsets else source].bounds(grid)
        return strides[source] == buffer.strides[destination] and (
            empty or low >= 0 and high + active[source] <= buffer.shape[destination])

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

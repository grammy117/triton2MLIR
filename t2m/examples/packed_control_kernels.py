"""Packed scalar control retains branches and per-element loop recurrences."""

import triton
import triton.language as tl

if __package__:
    from .elementwise_map_kernels import _map_scalar, _map_scalar_while
else:
    # Source capture runs in the model2MLIR environment without importing the
    # frontend package and its separate xDSL version.
    from elementwise_map_kernels import _map_scalar, _map_scalar_while


@triton.jit
def _pair(a0, a1, b0, b1, n0, n1, WHILE: tl.constexpr):
    if WHILE:
        q0, r0 = _map_scalar_while(a0, b0, n0)
        q1, r1 = _map_scalar_while(a1, b1, n1)
    else:
        q0, r0 = _map_scalar(a0, b0, n0)
        q1, r1 = _map_scalar(a1, b1, n1)
    return q0, q1, r0, r1


@triton.jit
def _for2(a0, a1, b0, b1, n0, n1):
    return _pair(a0, a1, b0, b1, n0, n1, False)


@triton.jit
def _while2(a0, a1, b0, b1, n0, n1):
    return _pair(a0, a1, b0, b1, n0, n1, True)


@triton.jit
def _for4(a0, a1, a2, a3, b0, b1, b2, b3, n0, n1, n2, n3):
    q0, q1, r0, r1 = _pair(a0, a1, b0, b1, n0, n1, False)
    q2, q3, r2, r3 = _pair(a2, a3, b2, b3, n2, n3, False)
    return q0, q1, q2, q3, r0, r1, r2, r3


@triton.jit
def _while4(a0, a1, a2, a3, b0, b1, b2, b3, n0, n1, n2, n3):
    q0, q1, r0, r1 = _pair(a0, a1, b0, b1, n0, n1, True)
    q2, q3, r2, r3 = _pair(a2, a3, b2, b3, n2, n3, True)
    return q0, q1, q2, q3, r0, r1, r2, r3


@triton.jit
def packed_control(x, y, counts, out_a, out_b, n, iterations, B: tl.constexpr,
                   PACK: tl.constexpr, WHILE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if PACK == 4:
        lane = tl.reshape(lane, (4, B // 4))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    trips = tl.load(counts + lane, lane < n, other=0) + iterations
    if PACK == 2:
        if WHILE:
            first, second = tl.map_elementwise(_while2, a, b, trips, pack=PACK)
        else:
            first, second = tl.map_elementwise(_for2, a, b, trips, pack=PACK)
    else:
        if WHILE:
            first, second = tl.map_elementwise(_while4, a, b, trips, pack=PACK)
        else:
            first, second = tl.map_elementwise(_for4, a, b, trips, pack=PACK)
    tl.store(out_a + lane, first, lane < n)
    tl.store(out_b + lane, second, lane < n)

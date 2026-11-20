"""Pure source region algorithms compose with partial tile memory and loops."""

import triton
import triton.language as tl

from t2m.examples.elementwise_map_kernels import _map_scalar_while
from t2m.examples.reduction_kernels import _compose
from t2m.examples.scan_control_kernels import _segmented_sum


@triton.jit
def region_partition(x, bias, out, seen, n, steps, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    valid = lane < n
    word = tl.load(x + lane, valid, other=0) ^ tl.load(bias)
    positive = word > 0
    tl.store(out + lane, word ^ 1, valid & positive)
    if MODE == "block":
        earlier = tl.load(out + lane, valid & positive, other=-17)
        total = tl.sum(tl.where(valid, word, 0), 0)
        tl.store(out + lane, word ^ total ^ earlier, valid & ~positive)
        snapshot = tl.load(out + lane, valid, other=0)
    else:
        snapshot = tl.full((B,), 0, tl.int64)
        for iteration in range(steps):
            earlier = tl.load(out + lane, valid & positive, other=-17)
            value = word ^ (iteration.to(tl.int64) + 2) ^ earlier
            tl.store(out + lane, value, valid & ~positive)
            observed = tl.load(out + lane, valid, other=0)
            if MODE == "tuple":
                a, b = tl.reduce((observed, tl.where(valid, word, 0)), 0, _compose)
                mixed = a ^ b
            elif MODE == "scan" or MODE == "reverse":
                scanned, heads = tl.associative_scan((observed, (valid & positive).to(tl.int32)),
                    0, _segmented_sum, reverse=MODE == "reverse")
                mixed = scanned ^ heads.to(tl.int64)
            elif MODE == "map":
                first, second = tl.map_elementwise(_map_scalar_while, observed,
                    tl.full((B,), 3, tl.int64), iteration)
                mixed = first ^ second
            else:
                mixed = tl.sum(observed, 0)
            snapshot ^= observed ^ mixed
            tl.store(out + lane, observed ^ 8, valid & positive)
        if MODE == "zero-reduce":  # noqa: SIM102 -- specialize before runtime control.
            if steps <= 0:
                tl.store(out + lane, word ^ 16, valid & ~positive)
    tl.store(seen + lane, snapshot, valid)

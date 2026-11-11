"""Observe every native by-value scalar width without floating arithmetic."""

import triton
import triton.language as tl


@triton.jit
def scalar_words(out, s8, s16, s32, s64, f32, f64, choose):
    tl.store(out + 0, s8.to(tl.int64))
    tl.store(out + 1, s16.to(tl.int64))
    tl.store(out + 2, s32.to(tl.int64))
    tl.store(out + 3, s64)
    tl.store(out + 4, f32.to(tl.uint32, bitcast=True).to(tl.int64))
    tl.store(out + 5, f64.to(tl.int64, bitcast=True))
    tl.store(out + 6, choose.to(tl.int64))


@triton.jit
def copy_words(x, out):
    lane = tl.arange(0, 8)
    tl.store(out + lane, tl.load(x + lane))


@triton.jit
def constant_words(out):
    lane = tl.arange(0, 8)
    tl.store(out + lane, lane.to(tl.float32) + 1.25)

# FP8 atomic addition through the public Merlin interface

The `typed-atomics`, `guarded-buffer`, `source-words` route admits
`tt.atomic_rmw fadd` on `f8E4M3FN`, `f8E4M3FNUZ`, `f8E5M2` and
`f8E5M2FNUZ`. This is generic numerical computation, even when a particular GPU
has no matching native instruction. The named contract is
`ieee-<dtype>-rne-add-quiet-nans`; the format name determines its actual
zero, subnormal, overflow, infinity and NaN rules. The `ieee` prefix names the
rounding/addition policy; it does not give FN/FNUZ an IEEE infinity encoding.

Each event preserves the original old storage word exactly, including signed
zero, signaling NaNs and NaN payloads. Both operands extend exactly to f64,
addition occurs there and the replacement rounds once to the original FP8
format using nearest-even. Every finite FP8 sum is exactly representable in
f64, including the largest-to-smallest exponent span. This retains each
event's FP8 rounding, rather than accumulating in f64 until the end. Arithmetic
NaNs are quiet; their payload/sign is unspecified. FN/FNUZ have no signaling
NaN encoding. Applying an IEEE quiet-bit mask to FNUZ's `0x80` NaN would
corrupt the value and is deliberately avoided.

Portable packages contain standard `memref.generic_atomic_rmw`, `arith.extf`,
`arith.addf`, `arith.truncf` and, for E5M2, explicit result quieting. Masks,
source order/scope, exact returned old values, logical grid/tiles, bounds,
initialization and the exactly-once effect boundary stay executable. Provenance
records these choices; it does not implement the effect or rounding.

The native atomic-word feature validates the complete closed update and
reuses the qualified FP8 cast expansions inside an integer byte load/CAS retry
loop. FP8 helper arguments and old results use i8, avoiding a native FP8
by-value ABI. An outside f64 constant folded from a widened operand retains
its actual f64 value and ABI; it is not narrowed before the add. Rounding stays
inside every retry. Ownership/lifetime analysis sees only
the independently reconstructed allocation-free synchronous borrowing body;
address arithmetic, memory order, retry condition and returned old value are
all checked. Early cast passes leave atomic regions to this separate owner.
An altered address, retry or return cannot acquire borrowing permission from
the helper's name or tags.

This adapts Meta/shared's element-typed masked event and old-value converter,
not a device atomic instruction. The reviewed source and tests are
[`UnstructuredAtomicRMWConverter`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp)
and its [atomic fixtures](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/UnstructuredToMemref/atomic_rmw.mlir)
at `17f0c234ce474bee334af36e351c62eca7852836`. The remaining work here is
explicit per-event numeric semantics, functional tensor/result transport,
provenance and checked native ownership legalization.

## Focused qualification

`tests/test_fp8_atomic_add.py` contains four tiled/scalar public programs, four
native/public signed-zero programs and four whole-helper validation programs.
The public programs retain both tiled masked updates and an ordered scalar
recurrence. Three mask rotations exercise every one of the 65,536 input pairs
per format, including all original signaling/quiet NaN and zero words.
Independent rational rounding checks finite results; computed NaNs require the
right class and quietness. Original tiled old words and unchanged storage are
checked byte-for-byte. Scalar histories retain intermediate update rounding.
Collisions check complete valid old-value multisets instead of a lane order.
Changed, ragged, single and empty inputs reuse the same library; invalid active
indices abort through the original guard. Caller arrays stay unchanged and
compiler staging excludes evaluator data.

E4M3FNUZ selects execution-owned results; the other formats select caller-owned
results. Each tiled/scalar program has eight successful reusable-library calls
and a separately rejected bounds invocation. Native and public constant
`+0`/`-0` updates cover all 256 initial words and exact original old results.
These add one public call per format, for 36 successful public invocations.
Helper checks accept the complete body and reject independently altered
address, retry, return and arithmetic operations for integer-transport and
wide-constant additions, plus the address/retry/return checks for identity
forms. All 56 helper-validation decisions use actual operations and types.
These focused checks extend the evidence without modifying the frozen
128-case selection or its 35 fixture files. They do not prove every possible
composition or physical accelerator execution.

With the existing environments selected in `.scratch/goal-tools.env`:

```sh
python -m pytest -q tests/test_fp8_atomic_add.py
```

## When this matters and what it costs

FP8 atomic addition can appear in custom low-precision state, accumulators or
imported/generated TTIR. Most ordinary Triton kernels accumulate in f16/f32;
their wider-state source has a different rounding/storage contract. It is an
explicit source alternative, not a replacement that this frontend silently
chooses.

The native fallback may execute many more scalar instructions and retries
than a hardware atomic, especially under contention. It preserves logical
tiling and values/effects, but claims no GPU throughput, lane order or
hardware-specific arithmetic equivalence. Physical scheduling remains a
consumer decision. Private invocation state excludes external participants,
cross-device publication and race-sensitive completion protocols; those need
an actual execution contract/provider. Automatic integer RMW/CAS through
BF16/FP8 owning roots remains separately unqualified. This addition does not
provide low-precision by-value entry bindings or infer packed storage.

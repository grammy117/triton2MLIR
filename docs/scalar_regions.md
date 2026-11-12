# Scalar TTIR computation inside maps, reductions and scans

The same scalar operation now composes inside a checked scalar region before
that region becomes standard Linalg. Previously, map/reduction/scan admission
ran before some existing casts, unsigned high products and shared arithmetic
mappings, so valid nested computation could fail despite a working outer-kernel
implementation. This is a transformation-order and executable-lowering change;
the public package and Merlin interfaces retain their existing schemas.

## Reuse and transformation

The adapter follows Meta/shared's
[`ConversionPatterns.hpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp):
`ClampConverter`, `PreciseSqrtConverter`, `PreciseDivConverter`, `FpToFpConverter`
and the external-elementwise arithmetic table. The archived Microsoft
[`TritonToLinalg.cpp`](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToLinalg/TritonToLinalg.cpp)
registers those same patterns. Existing native tensor
conversion stays upstream; only the missing scalar composition is made explicit
before scalar-region checks. Copyright and MIT notices remain with the adapter.

1. Existing exact FP8/ordinary float cast and unsigned high-product expansions
   run before scalar-container admission. RTZ, equal-width formats and double
   rounding retain their established standard expansions.
2. Checked scalar precise division/square root become `arith.divf`/`math.sqrt`.
   Clamp uses maximum/minimum for ALL and maxnum/minnum for NONE, following
   the original typed NaN policy. Ordinary unequal-width RTNE casts and numeric
   word bitcasts retain source types. Existing checked pure external prototypes
   become their registered standard math operations.
3. Full map/reduction/scan type, capture, control and effect checks still apply
   to the expanded body. Unknown operations gain no name-based purity grant;
   custom external libraries, flags/properties and incomplete prototypes refuse.
   Pointer bitcasts retain the separate address/effect analysis.
4. Source locations/provenance and `source-scalar-builtin` transformation
   records bind each original operation to its standard expansion. Operations
   stay in their original control position. Nothing is hoisted or speculated;
   a source guard or per-call recurrence still governs its arithmetic.

BF16-to-f64 widening additionally uses an exact BF16-to-f32-to-f64 bridge.
Every finite BF16 value, signed zero and infinity is exactly representable in
both formats. Existing standard BF16/F32 legalization then removes the remaining
BF16 arithmetic before LLVM. Computed NaN payloads retain the declared separate
numeric policy; no original storage word is reinterpreted as arithmetic.
Both TTIR casts and neutral standard `arith.extf` use this bridge, including
tensor forms. Neutral-flag and complete-candidate checks remain mandatory.

## Evidence boundary

`tests/test_scalar_builtins.py` checks the complete existing default external
prototype table, scalar NaN policies and all-candidate refusals before edits.
`tests/test_scalar_builtin_regions.py` uses original native-verified typed TTIR:
floating maps, reductions and scans combine precise operations, pure external
math, RTZ and equal-format bridges with runtime tensor recurrences. Integer maps
retain actual unsigned high products on loaded i32/i64 words. Independent
numeric/word oracles remain outside compilation. Each memory profile follows
the same source-derived path, with no kernel-name or example-data dispatch.

The 66 direct scalar contract cases and 24 complete kernel cases pass. Two
additional native scalar-bitcast maps retain original f32/f64 signaling NaN,
payload, signed-zero and other exceptional words exactly. The
kernel selection contains eighteen floating cases (f32/f64 across map/reduce/
scan and three memory profiles) and six i32/i64 unsigned-product maps. All
24 native/Merlin-host libraries and actual Spike images compare their complete
result; floating results use the IEEE policy, while integer results compare
exact words. Four further runtime domains per library retain the unchanged host
library across empty/tail/zero-step and repeated-step invocations, for 96 replays.
`out/scalar-regions/summary.json` reopens those 24 target receipts. The combined
`out/scalar-region-inventory/coverage.json` reopens 213 packages, 168 standalone
qualifications and fourteen maintained product cases. All fifty pinned source
names remain observed; these counts do not establish arbitrary composition or
per-operation execution coverage.

The 335-check cast/external/reduction/scan/map/packed-map/original-effect/shared/
maintained-corpus regression selection passes, including the complete BF16-word
widening check and another current-package target invocation. This overlaps
earlier selections and does not qualify every builtin, composition or target.

General effectful combiners, remaining scalar builtin families and tensor
captures, external participants, alias/runtime/source-version combinations and
accelerator arithmetic/performance remain part of the full completion goal.
The pinned source verifier still forbids map writes, atomics and volatile loads.

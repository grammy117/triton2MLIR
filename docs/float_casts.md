# Floating conversion and FP8 delivery

The complete numeric workstream includes every pinned `TT_Float` format:
f16, BF16, f32, f64 and the four FP8 types `f8E4M3FN`, `f8E4M3FNUZ`,
`f8E5M2`, `f8E5M2FNUZ`. Reader support, byte transport, numeric conversion,
arithmetic and target execution are separate obligations.

## Implementation order

1. Preserve each type and all constant bytes through frontend/Merlin readers.
2. Carry FP8 tensors through an explicit one-byte archive/host ABI, retaining
   the portable float types and all uncomputed source words. Qualify masks,
   loops, runtime dimensions and actual reused Merlin libraries.
3. Expand numeric casts with explicit rounding, subnormal, zero, NaN and
   overflow semantics. Qualify equal-width format changes and RTNE/RTZ;
   preserve each source narrowing boundary through loops and contractions.
4. Extend mixed/scaled contractions and remaining numeric/effect profiles,
   then qualify applicable staged targets. Host checks alone do not close
   the [full completion plan](completion_plan.md).

## Current evidence

The frontend and Merlin now bind all four FP8 types to xDSL, including scalar
and dense constants. The optional `fp8` installation extra supplies
`ml-dtypes>=0.6,<0.7`; type-only intake does not require it. Validation uses
ml_dtypes 0.6.0, frontend xDSL 0.68 and native-SDK xDSL 0.65.

Every one of the 256 encodings in each format survives codec unpack/pack,
scalar typed hex constants, dense typed lists, raw dense bytes and serialized
reparse. Raw words remain authoritative across NaNs and signed zeros;
decimal scalar/dense constants round to the declared format. Complete float
constraints are extended without admitting integers or widening restricted
float subsets. Both frontend/Merlin import orders share canonical types.
Native MLIR parse/print also retains the complete dense byte inventory.

This removes a reader prerequisite. It does not yet add FP8 buffer arguments
or grant successful numeric/operation coverage; those gates remain open.
No package ABI revision or existing floating comparison policy changes here.

## Upstream boundary

The shared converter at Meta revision
`17f0c234ce474bee334af36e351c62eca7852836`,
`include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp`,
`FpToFpConverter`, asserts on RTZ and chooses `arith.extf` for all
non-narrowing widths. Equal-width BF16/f16 and FP8 format changes therefore
need an adapter; emitting an invalid extension is not coverage. The archived
Microsoft source uses the same converter family. The surveyed arithmetic
tests do not independently establish FP8 or explicit-rounding behavior.

Triton revision `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`,
`lib/Dialect/Triton/IR/Ops.cpp::FpToFpOp::verify`, requires a rounding attribute
for width-narrowing casts. Its `python/test/unit/language/test_conversions.py`
has independent bit-oriented rounding references but excludes/sanitizes some
nonfinite/out-of-range inputs. Its NVIDIA and AMD
`Triton*GPUToLLVM/ElementwiseOpToLLVM.cpp` implementations include saturating
FP8 conversions. A generic IEEE `arith.truncf` cannot silently stand in for
every backend's overflow behavior. Resolve that executable source profile
before claiming those casts are portable; provenance alone cannot select it.

Use existing upstream conversion/expansion passes where their semantic
contracts agree. Keep source format, rounding, saturation/NaN policy and
source-operation IDs explicit in both portable MLIR and lowering evidence.

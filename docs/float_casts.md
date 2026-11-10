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

The tensor ABI now additionally accepts each FP8 format with raw `uint8`
NumPy archive carriers. Portable MLIR retains the exact float element type;
numeric comparisons decode it only under that declared type. Incorrect
numeric/word archive dtypes refuse. No by-value FP8 calling convention is
implied. Existing package schemas and comparison policies remain unchanged;
compatibility metadata explicitly names the raw byte archive difference.

`tests/test_fp8_transport.py` qualifies real-source masked loads/stores,
runtime value selection and zero/nonzero source loops for all four formats.
Tensor, buffer and guarded-buffer profiles retain source types and control;
the guarded profile also qualifies scalar memory events. Every encoding
survives the native JIT and a reused Merlin library over empty, singleton,
ragged and full domains. Numeric equality alone does not prove NaN or zero
bits: the tests additionally compare the complete original byte arrays.

The existing native toolchain lowers ordinary FP8 transport to byte storage.
No custom transport pass, changed public tensor signature, or skipped
ownership/lifetime gate is needed. Merlin's loop destination and generation
checks now recognize the four FP8 value types as non-handle elements, while
unknown calls and invalid/double releases still refuse. Its C-runtime input
arrays and footprint accounting use one byte per element; compiled C storage
checks retain all 256 words. Generated C storage is a separate check from
target compilation/execution and numeric output interpretation.

```bash
python -m t2m shared-demo --kernel fp8-transport \
  --out out/fp8-transport --python /path/to/native-python
python -m t2m qualify-merlin out/fp8-transport \
  --out out/fp8-transport-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

The default demo uses E4M3FN. `t2m.fp8_demo.demo(dtype=...)` also accepts the
other three formats. The saved v2 reference selects exact word comparison,
and compiler staging excludes its reference bytes and digests. The ordinary
native and Merlin observers each bind one actual invocation to that result.

`capture_ttir(..., source_target=(backend, arch, warp_size))` can select an
official AST frontend descriptor. The demo uses HIP/gfx942/64 because its
official frontend admits all four types; the historical CUDA/80/32 default
remains. This is AST-to-TTIR capture only, without GPU compilation, launch or
an accelerator execution claim. Typed zero bitcasts supply inactive values
without requesting an unsupported direct integer-to-FP8 numeric cast.

Numeric conversion remains distinct: the existing LLVM converter leaves
`arith.extf` from these FP8 formats unresolved in direct probes. RTZ,
equal-width casts, FP8 narrowing/saturation, arithmetic, scaled dot and
target materialization remain required work. Transport is not numeric closure.

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

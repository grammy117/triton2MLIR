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

## Executable widening

FP8 widening now reaches actual native and Merlin execution for all four
formats and f16/BF16/f32/f64 destinations. Portable MLIR retains standard
`arith.extf`, original FP8 tensors, tile structure, source loops and provenance.
The existing LLVM converter leaves these extensions unresolved by default;
the explicitly selected Merlin `lower_fp8_widening` feature implements their
scalar native form before ordinary lowering. Storage, alias/lifetime and
complete-LLVM checks remain active.

The native expansion decodes sign/exponent/fraction fields through standard
operations. All finite FP8 values fit exactly in each destination, including
their subnormals. f32 intermediate arithmetic is exact and normal, so it adds
no FTZ dependence. The finite-only top binades, E5M2 infinities, signed zeros
and FNUZ NaN sentinel stay distinct. Computed NaN class has no payload/sign
selection; the adjacent uncomputed FP8 result retains every original word.
Narrowing/arithmetic/calls, nonneutral flags and shaped native casts remain
separate obligations and refuse before this feature edits candidates.

`tests/test_fp8_widening.py` qualifies 48 format/destination/memory-profile
combinations over all 256 source words, then reuses each actual Merlin library
over empty, singleton and ragged domains and zero/nonzero source loop counts.
Independent rational decoding supplies expected values; 16 additional cases
cross-check that oracle against the standard codec. Native feature/refusal and
BF16 interaction checks pass 48 cases, including every runner callback and
unchanged storage/ownership boundaries.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel fp8-widen \
  --out out/fp8-widen --python /path/to/native-python
python -m t2m qualify-merlin out/fp8-widen \
  --out out/fp8-widen-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

This demo's first result uses the explicit IEEE policy and its second result
uses exact words. Reference bytes/digests remain excluded from compilation.
The API also accepts `demo(dtype=..., target=...)` for all qualified pairs.
The saved `out/fp8-widen` package and `out/fp8-widen-merlin` qualification
verify 26 and 45 artifacts respectively. The explicit 25-package observation
report at `out/fp8-widen-coverage/coverage.json` records 29/50 definitions
observed and 25 present in host-qualified captures; these are composition
observations rather than whole-opcode closure.
FP8 narrowing/saturation, equal-width FP8 formats, arithmetic, scaled dot,
broader model bindings and target materialization remain required work.

## Rounding and standard-format conversion

The four standard formats f16/BF16/f32/f64 now additionally support RTNE/RTZ
across these conversion pairs: f32→f16/BF16, f64→f16/BF16/f32 and
f16↔BF16. Original source rounding stays in the native-verified TTIR inventory.
Triton's pinned generic rounding attribute is an i32 enum (`0` for RTZ,
`1` for RTNE), rather than a custom `#tt.rounding` attribute.

The adapter leaves ordinary supported RTNE conversions to shared. Missing
forms receive a deterministic standard MLIR expansion, within one elementwise
`linalg.generic` for a tensor tile. Its composite annotations retain source
and destination formats, rounding, locations and original source IDs. There
is no GPU mapping, inline assembly, opaque helper or changed tensor ABI.

RTZ first narrows with the standard nearest cast, extends the result exactly
and compares magnitudes. An overshooting result moves one destination word
toward zero. This handles ties, subnormals and finite overflow while retaining
source infinities/zeros and computed NaN class. Equal-width f16/BF16 changes
use an exact f32 bridge before the actual narrowing boundary.

F64→16-bit conversions need more care. A plain f32 intermediate can double
round, and the native JIT lacks a direct f64→f16 runtime helper. The RTZ route
uses two exact toward-zero boundaries; RTNE first truncates to f32 toward zero,
jams an inexact intermediate to odd and performs the final nearest conversion.
The precision gap preserves the single source rounding decision, including
values immediately beside a destination midpoint. No intermediate reference
or evaluator values supply compiler behavior.

This expands selected scalar cast internals below the granularity of a single
cast operation. The source cast intent remains an explicit composite. A
consumer with faithful native cast support may choose that implementation
after proving the numeric contract; annotations alone grant no rewrite.
Generated compatibility metadata records this representation difference.

Fourteen native arithmetic cases cover every f16/BF16 word, all wider source
exponents, both signs and adjacent midpoint/overflow/underflow values. All 42
pair/rounding/memory-profile cases execute through actual Merlin libraries,
including exact adjacent source words and reused libraries over empty/ragged
runtime dimensions. Unknown rounding values/types, missing narrowing modes,
unqualified properties and mixed invalid candidates refuse before edits.
Six further real-source loop cases check per-iteration rounding with changing
runtime trip counts and by-value f32 increments through one reused library.
For example, repeatedly adding less than one destination ULP must round each
iteration; accumulating in f32 until the final store gives a different result.

```bash
python -m t2m shared-demo --kernel rounded-cast \
  --out out/rounded-cast --python /path/to/native-python
python -m t2m qualify-merlin out/rounded-cast \
  --out out/rounded-cast-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

The public demo captures a real f32→f16 RTZ Triton source kernel and uses an
independent rational reference with the explicit IEEE comparison policy.
Original uncomputed source words are checked separately in the typed fixtures.

Caller `prov.family`, `prov.op` and `prov.region_id` StringAttrs are retained
through shared rewriting and final emission. `prov.intake_intent` records the
source-ID-to-tags mapping; agreed fields are projected back to their original
names, while conflicting merged intents retain the complete mapping. The IDs
belong to the explicitly named intake graph (original or normalized). This
metadata supplies intent only; all executable memory/numeric checks still run.

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

# FP8 arithmetic and low-precision scaled-dot results

This extension keeps FP8 in portable standard MLIR and supplies its missing
native host legalization. It covers f8E4M3FN, f8E4M3FNUZ, f8E5M2 and
f8E5M2FNUZ. Source tiles, reused loads, original control and every FP8 rounding
boundary stay executable. Focused public qualification is recorded below;
the final frozen release selection and semantic audit remain separate gates.

## Arithmetic and source intent

The existing frontend already converts clamp, precise division/sqrt and
ordinary arithmetic into typed `arith`/`math` operations inside Linalg/SCF.
Those operations remain in exported `model.mlir`. Consumer selection now
derives `emulate_fp8_elementwise` from their actual types and operations.
Provenance alone cannot select arithmetic or supply its semantics.

Merlin reuses the official `arith-emulate-unsupported-floats` and
`math-extend-to-supported-types` passes with f64 intermediates. The latter
deliberately leaves FMA untouched, so the adapter widens that operation too.
Each original result narrows back to its source FP8 format before the next
operation or loop iteration. The existing exact scalar FP8 narrowing/widening
features implement those casts, including subnormals, signed/unsigned zeros,
overflow and NaN classes. The native and public workers use the same helpers.

The official passes introduce casts marked `contract`. The adapter neutralizes
only the new cast flags, preserving the source rounding boundaries. Source
arithmetic fast-math flags permit relaxed results; this native route chooses
the strict refinement and retains nonneutral source permissions in
`prov.fp8_native_source_fastmath`. Portable source keeps its original fields.
A source contraction-permission case checks that a large sum rounds before a
following subtraction. An annotation never replaces the actual narrowing.

Every finite FP8 product fits exactly in f64. The exponent/significand range
also lets f64 represent a finite FP8 product plus an FP8 addend exactly. This
avoids prematurely rounding the product of an original FMA. Exhaustive binary
input pairs and an independent rational palette oracle check add/sub/mul/div;
sqrt uses exact squared-midpoint comparisons. Computed arithmetic NaN payloads
remain unspecified, while exact copied words remain exact.

## Operations that observe words

Negation, absolute value and copy-sign use exact integer bit operations in the
native consumer. They do not canonicalize an E5M2 NaN payload or turn a
signaling encoding into another word. FNUZ's zero and unique NaN cannot change
sign, matching the pinned APFloat format semantics. Copy-sign still observes
the second operand's sign representation. Every pair of input/sign-source
words is qualified, including the unique FNUZ NaN.

The independent NumPy/ml_dtypes oracle supplies word results for ordinary
negation/absolute value/copy-sign. Its raw FNUZ copy-sign loop can turn zero
into NaN or NaN into zero; the reference explicitly applies APFloat's
unchangeable-zero/NaN rule instead. This exception is documented in the test,
rather than silently treating the different numeric behavior as acceptable.

## Scaled-dot C and recurrence

The existing [logical MX profile](scaled_dot.md), `logical-mx-f64-v1`, now
admits FP8 C/result types. It still extends the original operands and C before
the logical f64 contraction and narrows once at that original dot result.
It does not quantize f16 operands to the result format before multiplying.
An enclosing loop therefore carries an actual FP8 value into each subsequent
dot, rather than carrying a wider accumulator and truncating only at exit.

This extends the existing logical profile; it does not assert bit agreement
with every GPU tensor-core accumulation order or FTZ mode. Source precision,
packing, scales and original C still require their existing checked contracts.

## Public interface and qualification

No tensor/scalar ABI, receipt or package schema changes. Existing uint8 FP8
archive carriers retain the real FP8 MLIR types. Generated compatibility
metadata states the added consumer requirements and the software expansion.
Older consumers may refuse the new numeric forms. The expansion costs f64
and integer operations on this host; it is not a GPU performance measurement.
Consumers can choose conforming native arithmetic downstream.

The public qualification consists of thirteen complete source programs:

- Four arithmetic programs cover all 65,536 input-word pairs, both clamp NaN
  modes, FMA, exact copies and atomic exchange old/final words. Six invocations
  per library include changed pairs, multi-iteration source control, tails,
  disabled/zero-trip control and empty storage. Three use caller-owned results;
  E4M3FNUZ uses execution-owned results.
- Four scaled-dot programs start from all 256 C words and execute two retained
  contractions per source loop iteration. Five replays per library cover changed
  operands, zero trips, disabled control and empty outputs. Independent checks
  distinguish premature input quantization and rounding only at loop exit.
- Four word-math programs cover all 65,536 value/sign-source pairs and five
  changed/ragged/single/empty replays per library, comparing exact bytes.
- One source fast-math contraction-permission program checks all four results
  across five replays, including source NaNs and signed zeros.

These are 69 successful public library invocations, plus separate native
execution, source verification and artifact checks. Caller inputs remain
unchanged, libraries are reused, and evaluator arrays are absent from compiler
staging. Existing source/profile, cast and scaled-dot regressions remain
separate evidence from the final 128-case release qualification.

The execution-owned check exposed an existing interface bug: a named Boolean
control was treated as an integer dimension. The corrected validator accepts
ordinary scalar controls while requiring actual non-Boolean integer dimension
sources. Malformed Boolean storage and inconsistent input dimensions fail
before invocation; a Boolean dimension declaration also fails independent
consumer validation. This does not add by-value f16/BF16/FP8 entry bindings.

## Source and remaining boundaries

The implementation reuses LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`:
`Arith/Transforms/EmulateUnsupportedFloats.cpp` and its tests,
`Math/Transforms/ExtendToSupportedTypes.cpp` and its tests, plus APFloat's
`changeSign`/`copySign` and NaN-as-negative-zero rules. This is a reviewed source
reference; the installed native SDK/tool identities are recorded independently
and are not inferred from that checkout. No new LLVM build was required.

The admitted source remains verified pre-layout Triton 3.7.1. Newer directed
binary-rounding syntax is rejected by its actual native parser; its probe is
retained separately. Existing explicit FP8 cast rounding/overflow profiles
remain available. Unknown semantic fields, external implementations and
post-layout vector/device forms retain their own admission requirements.
[FP8 atomic addition](fp8_atomics.md) now has its separate qualified numeric
and effect profile. [Entry value ports](entry_values.md) now separately carry numeric scalar/tile
arguments and results through typed storage. Scalar arithmetic alone supplies
no native by-value entry ABI.

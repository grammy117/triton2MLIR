# Per-cast FP8 source contracts

`convert_shared(..., fp8_overflow="per-cast")` selects the explicit
`t2m.per-cast-fp8.v1` software source profile. Each nonidentity `tt.fp_to_fp`
whose result is FP8 must carry a semantic StringAttr:

```mlir
%rounded = "tt.fp_to_fp"(%value) {
  rounding = 1 : i32,
  t2m.fp8_overflow = "satfinite"
} : (tensor<32xf32>) -> tensor<32xf8E5M2>
```

The three existing [conversion policies](float_casts.md) remain unchanged:
`ieee`, `satfinite` and `satfinite-nan-inf`. They can now coexist in one
kernel, with separate original RTNE/RTZ rounding attributes. Both source
formats and FP8 destination formats retain their ordinary typed meaning.
This is an explicit source numeric contract, not a guess about GPU backend
behavior. `prov.*` records intent and never supplies a policy.

## Transformation and preserved algorithm

All source casts are validated before editing. With this profile, each cast
expands before helper inlining, symbol deletion or greedy folding can discard
its policy. Unused helpers still require valid contracts. Missing, unknown,
wrongly typed or conflicting selections refuse. A source semantic attribute
without explicit per-cast selection refuses; a global policy cannot override
it. Policies on identity casts, non-FP8 results or already-standard
`arith.truncf` also refuse. A standard cast retains its own numeric semantics.

The implementation reuses the existing FP8 expansion: one parallel
`linalg.generic` per source tile contains executable standard clamps,
selections, nearest casts and exact RTZ word correction. Source scalar casts
retain scalar computation. No opaque helper, ISA instruction or physical lane
mapping implements the policy. Logical grids, loop-carried states, masks and
each original narrowing boundary remain executable. Reports retain the actual
source IDs, formats, mode, selected policy and source profile. Conflicting
provenance tags cannot change the computation.

Original source bytes and operation inventory include the semantic attributes.
The ordinary package receipt binds those bytes and the emitted program.
Merlin consumes the standard payload; it never reconstructs the choice from
provenance. Existing compiler-only staging, readonly inputs, result accounting,
native ownership and lifetime gates remain active.

The source API and `shared-tensor` / `bind-model` CLI accept `--fp8-overflow
per-cast`. Per-binding JSON can use `"fp8_overflow": "per-cast"`. Public
tensor/scalar ABI and receipt schemas stay unchanged. Compatibility metadata
names the additional software source profile. Existing global selections retain
their original behavior; unannotated casts under per-cast selection refuse.

## Upstream and consumer boundary

The pinned Meta/shared `FpToFpConverter` still chooses truncation versus
extension by bit width and rejects RTZ. The archived Microsoft converter
registers the same implementation family. Neither supplies these mixed
overflow/infinity contracts. The existing adapter supplies the missing
executable arithmetic while reusing upstream inlining, pointer/mask analyses,
tensor conversion and ordinary native MLIR lowering. Source references:

- [Meta converter, revision 17f0c234](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp).
- [Meta unary arithmetic fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonArithToLinalg/convert_1d_elemwise_arith_unary.mlir).
- [Archived Microsoft registration, revision 050b93ae](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToLinalg/TritonToLinalg.cpp).

Actual target execution exposed a Merlin composition error: sorted feature
edits placed widening before narrowing, while the native runner requires
narrowing first. Widening now inserts after an existing initial narrowing
marker. Both application orders produce the same pipeline; duplicates and
moved markers still refuse. Source callbacks and loop destination planning
retain their original seams and ownership checks. This changes no cast math.

## Measured scope and remaining work

The typed TTIR fixture in `t2m/examples/mixed_fp8.py` retains an original private
helper and a three-state loop. All three policies run simultaneously, then
widen each result to F32 before the next iteration. A fourth result preserves
every original F32 input word. Independent exact-rational FP8 palettes supply
rounding references; computed NaNs use class equality, while other IEEE words
and the uncomputed result have their stricter checks.

Native cases cover all four FP8 formats and both modes. Eight actual
native/Merlin/Spike kernel cases cover every format/mode pair across selected
tensor, buffer and guarded/source-word memory routes. They retain 32 complete
target results. Each host library additionally replays empty, singleton and
ragged domains with changed loop counts and scalar increments: 24 additional
invocations reuse their original library bytes. The E5M2FNUZ case also exercises
the public CLI. This is a finite selection, not the complete format/mode/memory
cross-product or accelerator performance qualification.
The producer numeric/regression selection passes 149 checks, including the
source-contract refusals before unused helper deletion. All eight additional
kernel cases pass. Merlin's native FP8 and runner selection passes 67 checks;
structure, no-regex and target-independence gates pass.

The separate target receipts and runtime replay records are reopened in
`out/spike-mixed-fp8/summary.json`; they do not change the older 106-package
observation selection. Source-profile/type/attribute/composition and applicable
target closure remain in the full completion denominator.

For now this profile requires the installed shared source parser to accept the
original TTIR before transformations. Compositions requiring an earlier
descriptor/block-pointer/version normalization or composite F32 dot expansion
refuse before a pass could fold source casts without their semantics. They
need a protected numeric expansion before that native normalization seam;
refusal is incomplete coverage. Broader model captures, source backend
correspondence and accelerator execution also remain open.

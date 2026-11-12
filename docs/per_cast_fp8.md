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

The implementation reuses the existing FP8 scalar recipe for executable
standard clamps, selections, nearest casts and exact RTZ word correction.
Before native TTIR passes, shaped casts use standard elementwise tensor
`arith`/`math`: those passes load these dialects without loading Linalg.
The later shared arithmetic conversion supplies parallel Linalg. The ordinary
late expansion can still build a parallel `linalg.generic` directly. Source
scalar casts retain scalar computation. No opaque helper, ISA instruction or physical lane
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

## Protected composite precision

When the source-only parser accepts the original kernel, per-cast expansion
now precedes the official F32 dot normalization too. The complete typed
preflight graph is snapshotted before expansion, including original casts,
semantic attributes, source intent, region/block edges and the original dot.
Its matching SSA/types/fields are checked before copying the inventory.
Native normalization sees executable standard arithmetic. Its partial
contractions then receive a separate preflight with `normalized.ttir.op.*`
IDs; accumulator repair uses those normalized contracts, rather than adding
original C to every partial dot. The v2 trace retains both graphs. Original
contraction intent also reaches the function and named matmuls.

The integration selection combines independently selected FP8 casts on A, B
and loop-carried C with residual F32 additions, a ragged runtime K loop, and
BF16x3, BF16x6, logical TF32 or TF32x3. Residuals retain bits beyond BF16, so
these cases exercise the composite precision boundary. Independent exact-rational
FP8 palettes and ordered partial-dot references evaluate each original narrowing
and loop step. Tensor, buffer and guarded/source-word routes exercise native,
Merlin host and functional RISC-V/Spike execution. All twelve precision/profile
cases pass, retaining twelve complete target results and 36 additional host
invocations. Empty M, zero K and changed K domains replay the unchanged host
library separately. The receipt selection and final producer byte replays live
in `out/spike-protected-fp8-precision/summary.json`. This separate selection does
not change the older 106-package inventory.

Target qualification exposed two generic consumer gaps: the named-operation
builder's captured float union omitted FP8, and attributed `tensor.yield`
custom printing used an incompatible attribute order. Merlin extends the real
float union and prints attributed tensor yields generically, preserving all
source fields. Source-exact scalar contraction additionally introduces nested
tensor-update loops; destination planning can copy their complete values after
checking all nested operations, including unused/void effects. Original loop
order, zero-trip behavior, ownership and lifetime gates remain intact.
The additional source/carrier and existing cast/dot/reader regression selection
passes 75 checks. Three negative cases prevent malformed original policies from
reaching native normalization. Merlin's focused reader suite passes 46 checks;
its destination-reuse suite passes 111 with all lifetime checks retained.

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

The [complete native source reader](native_source_print.md) now handles original
block-pointer and descriptor type spellings rejected by shared's source parser.
It prints the entire native-verified graph without normalization, preserving
every attribute before the ordinary cast expansion. Official pointer/descriptor
rewriting then runs before shared inlining; normalized preflight and accumulator
repair retain their own graph. Six pointer/descriptor and memory-profile cases
qualify through native/Merlin host/Spike with mixed RTNE/RTZ source policies.
Two ragged GEMMs additionally retain source pointer tiles, K reuse and BF16x3/
TF32x3 through those execution boundaries. All eight target receipts and the
public descriptor CLI capture reopen in `out/spike-native-source-fp8/summary.json`.
Original external descriptor entry parameters additionally qualify for selected
rank-one/two F32 loads with three policies and mixed modes through native/Merlin
host/Spike. Their initial source-only audit grants no ABI admission: the real
expanded field binding and normalized ABI preflight remain mandatory. Both
host libraries replay three changed field/domain cases. Malformed original
policies refuse before normalization, and wrong expanded widths/bounds refuse
after it. See [the direct field contract](external_descriptors.md#original-numeric-contracts-before-field-expansion).
[Allocation-root bindings](descriptor_state.md#protected-contracts-before-allocation-normalization)
now protect the same source choices before their initial normalization too.
Source capsule v2 retains complete original generic IR, native printer identity,
the typed original graph and executable protected TTIR. Pointer/descriptor
recurrences pass native/Merlin host/Spike and eight host-library replays; a
complete learned-state model preserves three calls, dependent updates and all four
results through those execution boundaries, with its original snapshot checked
as exact words. The separate receipt selection is
`out/spike-protected-fp8-allocations/summary.json`. Broader source versions,
binding/numeric/effect compositions, source backend correspondence and
accelerator execution remain in the completion denominator.

## Allocated tiled contractions

Actual pointer and external-descriptor kernels now combine three independent
FP8 contracts with BF16x3, BF16x6 or logical TF32x3 inside a runtime K loop.
The source private helper retains residual F32 additions on both inputs and
the loop-carried accumulator before its dot. Shared C/output allocation state,
16x16 output tiles, 32-element K reuse, masks and runtime signed strides remain
executable. Official composite-dot passes supply the ordered partials; original
C is added at its declared boundary, rather than once per partial.

All six source/precision cases pass native, Merlin host and actual Spike
execution with exact complete allocation-word references. Twenty-four further
invocations reuse unchanged host libraries across reversed input strides,
multi-program M/N tails, zero K and empty M. Invalid active C origins abort
without a success receipt; invalid original cast policies refuse before native
normalization. The eight focused checks pass. Independent rational cast palettes
and ordered partial-dot arithmetic supply the references, with actual NumPy
owning-buffer views establishing the signed addresses. These finite cases do
not qualify unspecified GPU cast/Tensor Core behavior or accelerator performance.

Descriptor normalization may expand the dot before allocation binding. New
`prov.allocation_source_contracts` and `prov.allocation_original_dot_precisions`
annotations retain the pre-wrapper hash, exact selected policies, original
precision, operand types and caller intent separately from the bound graph.
Entry and named contractions carry that source context without claiming a
partial-to-original bijection or granting arithmetic semantics. ABI and existing
capsule/receipt schemas stay unchanged.

Two complete model2MLIR captures additionally qualify descriptor/BF16x3 and
pointer/TF32x3 compositions. Each retains learned padded input roots, logical
17x5x37 GEMM, two dependent full-K calls, a zero-K call and a live original C
snapshot. All four results per model match exact words through native, Merlin
host and actual Spike execution. Parent source/data/weights remain unchanged.
Original typed precision-code and descriptor-choice arguments must match the
selected call specialization; wrong choices refuse. Two distinct scalar
specializations serve the three call sites, preserving the reusable source
program rather than requiring one binding per call ordinal.

Padding belongs to the example's owning inputs, making its original Python
descriptor construction valid while retaining ragged logical K/N dimensions.
The portable entry keeps logical signed strides; the standalone runtime cases
also qualify nonaligned stride values under the explicit logical-field profile.
No physical GPU descriptor ABI/alignment or target performance is inferred.
All eight target images and fourteen complete results, the 24 host replays and
a public original-source allocation CLI capture reopen separately in
`out/spike-protected-allocation-contractions/summary.json`. The older 106-package
inventory is unchanged; arbitrary type/attribute/effect/model and accelerator
closure remain open.

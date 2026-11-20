# Actual iteration ordinals for affine pointer streams

Pointer streams now reach tensor MLIR when their increment is
`a * actual_loop_step + b`, with source-proved integer constants `a` and `b`.
The new constant term advances once per actual execution, independently of
the source step. This closes the structured refusal saved in
`.scratch/runtime-pointer-independent-step1/`; the original source passes all
four profiles and sixteen native comparisons in
`.scratch/runtime-pointer-independent-step2/probe.log`.

The source loop, tiling, masks, i64 word state, memory order and final pointer
reads remain. There is no replacement loop with a maximum trip count, no
opaque kernel call and no GPU thread, warp, memory-space or launch ABI binding.

## Exact recurrence and source proof

The existing source affine matcher now retains both coefficients, including
negative streams and chains of pointer additions. Source SSA identity and
uniform checked arithmetic establish the relation; unrelated invariant runtime
scalars and conditional pointer updates decline this optional normalization.
Equal bounds and provenance do not establish equal values.

The body pointer is
`initial + a * (actual_IV - original_lower) + b * actual_ordinal`.
The final pointer uses the actual exit IV and returned execution count. The
ordinal is an additional standard scalar SCF carrier initialized to zero and
incremented by one on each executed body. The original loop keeps its actual
lower, upper and positive-step SSA. A zero trip retains the complete initial
tuple and returns count zero.

For `range(1, 8, 3)`, count 3 and exit IV 10 have different meanings. A unit-row
pointer stream starting at row 1 finishes at row 4. Neither the upper bound 8,
the exit IV 10 nor the maximum allowed count 8 is a substitute. Mixed streams
retain both terms and their actual scalar sources.

The existing SCF initial/yield recognizer now also recognizes a literal-zero,
own-carried-argument-plus-literal-one scalar recurrence. Header ranges and
positive step still require an independent signed source-width proof. Counts
must fit the original carrier width; an i8 header may be valid while its
0..255 mathematical count cannot be represented by an i8 ordinal. The new
source adapter uses i64 only when its complete count and intermediate offset
enclosures fit. Original and newly introduced arithmetic are audited separately.

Body ordinals bind their own actual scalar, separately from the source IV.
Their facts expire with the actual loop, including initialization facts that
depend on those coordinates. Returned counts bind actual result SSA in both
memory functionalization and pure numeric cloning. Separate loops and tuple
slots retain separate identities. Bounds grant no complete-write image,
ownership, alias, effect or numeric permission; prior zero-trip and fresh-output
checks remain authoritative.

## Upstream reuse and representation

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` retain original SCF control while
exposing carried pointer offsets in `PtrAnalysis::rewriteForOp`. Reviewed
`tensor_indices_loop_iterargs_nested.mlir` and `scalar_store_nested_loop.mlir`
fixtures retain their actual index updates. The extension reuses the existing
source adapter, constant-recurrence range checks and native structured-pointer,
mask, arithmetic and SCF conversions; it does not add an arithmetic converter.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s SCF initial/yield and
successor interfaces, plus its range-analysis loop fixtures, remain the reviewed
loop-state boundary. Source revisions do not attest the installed binary.
No native compiler build or Triton CPU backend is introduced.

`source-ordinal-pointer-recurrences` records original loop/update IDs and each
stream's `step_coefficient` and `iteration_increment`. The existing step-only
record remains for streams needing no ordinal. `actual_iteration_ordinal_arguments`
and `actual_iteration_ordinal_results` identify independently recognized body
and result edges. Exact original TTIR and complete inventories remain retained.
Generated compatibility names both source normalization and coordinate facts.
Public tensor/scalar ABI, receipts and artifact rosters are unchanged; buffer
and guarded-memory profiles retain their separate contracts.

## Real kernels and complete models

`ordinal_pointer_mix` in `t2m/examples/runtime_header_kernels.py` contains three
actual source variants over row/column tiles:

- `memory`: forward/reverse pointers advance one row per execution, with body
  loads feeding a nonzero i64 recurrence and final loads remaining observed.
- `pure`: a preloaded tile is reused through word arithmetic inside the loop;
  pointer loads occur afterward and use the actual execution count.
- `mixed`: pointer increments combine the runtime step and a constant term,
  retaining both original IV state and actual count in their coordinates.

All four memory/address profiles, i32/i64 controls, both storage orientations,
ragged lanes, i64 extreme words, nonunit steps and both zero-trip forms use
independent NumPy references. Tensor output retains standard slices and an
unmapped logical grid. Out-of-root final domains still refuse.

`scripts/capture_pointer_step_model.py --ordinal-mode memory|pure|mixed`
captures each variant as two dependent PyTorch calls. Live singleton controls,
learned matrix state, surrounding addition, earlier observations and all ten
model results remain. Independent Torch references are outside compiler staging.
The two original body-load/final-only models remain separate regression cases.

## Qualification and remaining scope

The source/domain/state selection passes 84 checks in
`.scratch/ordinal-pointer-proof2.log`; subsequent sequential-counter coverage
is included in the final regression selection. The kernel selection passes
75 in `.scratch/ordinal-pointer-final1.log`: twelve reusable Merlin libraries
handle six changed invocations each, and 24 actual serial images retain all
results. The model selection passes twenty in
`.scratch/ordinal-pointer-models1.log`: twelve checks qualify the three new
models and eight cover the two existing models. Each model retains ten initial,
thirty replay and twenty exact-word serial results across two images, plus an
actual invalid-step abort without published output.

`out/ordinal-pointer-streams/summary.json` reopens only the new variants:
twelve kernel libraries and three model libraries, 54 initial results, 234
replay results and 108 exact-word serial results across thirty actual images,
plus three actual model-domain aborts. Every original callsite, complete parent
archive, source header SSA, package compatibility and actual target receipt
remains checked. These counts are separate from the older model regressions.
Selected source hashes are not a full build dependency closure.

The final reopening log is `.scratch/ordinal-pointer-report2.log`. Fourteen
selected producer implementation/test/helper files match commit
`b55fc70d05d37fd840ad9bc0155cedf26679251a`, separately from observed reopening
revisions. The unchanged companion pins are model2MLIR
`f3f0de1ae130faebbd27bd36374d11e1e211daa2` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`.

The related source/control/memory selection passes 137 checks in
`.scratch/ordinal-pointer-regression1.log`, including the final independent
sequential-counter cases. New/changed helper/test Ruff and diff checks pass.
The final width/ordinal selection separately passes fifteen in
`.scratch/ordinal-counter-width1.log`, including a valid small carrier with
negative header IVs and a count fitting its signed range. Fourteen cases overlap.
The standalone literal-header probe `.scratch/ordinal-literal-headers1/`
additionally exercises a source loop with constant lower/step and runtime upper;
all four profiles pass sixteen native comparisons, separately from the retained
kernel/model packages above.

Reproduce with existing source/native/model/Merlin/serial tools from the README:

```bash
python -m pytest -q tests/test_iteration_ordinals.py \
  tests/test_ordinal_pointer_streams.py tests/test_pointer_step_model.py \
  --basetemp=.scratch/pytest-ordinal-pointer-streams
```

General unrelated invariant runtime strides, conditional/while pointer streams,
alias ownership, variable-step complete-write images, dynamic-root projection,
shared/asynchronous participation, remaining source/numeric compositions and
applicable accelerator correctness/performance remain completion work. This
does not change the historical TTIR name denominator or establish performance.

## Next reproduced boundary

`.scratch/runtime-pointer-runtime-stride1/` replaces the literal pointer
increment with `memory_stride * row_stride`, where `memory_stride` is a
different bounded runtime scalar from the loop step. All declared source
combinations remain in-bounds. Guarded source-word conversion passes four
native cases; the three structured selections retain the native i64 boxing
refusal because this separate SSA value supplies no affine step/constant match.
Continue source-proved invariant runtime increments and exact carried logical
offsets, retaining tile geometry and final reads. Interval equality is not a
substitute for source identity. This probe has no Merlin host or serial
qualification yet.

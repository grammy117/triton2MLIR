# Rank-reducing source views through the Merlin interface

Runtime source-view binding now also retains standard rank-reducing slices,
including retained unit dimensions and scalar views. The explicit
`runtime-logical-coordinates-v1` policy and execution view profile v3 remain
required. Source coordinates, original allocation pointers, every model result
and the complete original SSA remain independently checked by Merlin.

## Coordinate and package contract

Only statically unit slice sizes may disappear. Producer and consumer separately
reconstruct the standard reverse axis map from mixed sizes and result types.
Full source offsets and bounds are checked before projecting shape and strides.
The additive `retained_axes` step field appears only for rank-reducing slices;
old rank-preserving records keep their meaning. Coordinate schema v1 and view
profile v3 retain their names; consumers lacking this form reject it. Parent
archives, public tensor types, receipt v2 and signature v6 are unchanged.

Model2MLIR's literal `aten.select.int` lowering now retains actual dynamic
dimensions, signed index bounds and scalar results. Invalid static indices
refuse; runtime invalid indices abort before extraction. Other unit axes use
actual dynamic constant-one sizes plus a compatible result cast, so the selected
axis remains distinct. The standard empty-slice refinement retains zero-sized
unselected dimensions. Reference review uses LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`, TensorOps.cpp `getDroppedDims`
and `tensor/canonicalize.mlir`'s rank-reduction preservation fixture.

A scalar model result uses the existing checked runtime dense adapter from a
one-element ranked kernel result. Exact count guards remain executable. This is
an explicit internal representation difference: the public model result remains
rank zero. The Triton allocation ABI still uses ranked buffers. Logical source
relationships grant no public physical alias or mutation permission.

## Real program and evidence

`scripts/capture_rank_selected_view_model.py` captures a real model with input
shape `(1, rows, cols)`, learned parameters and four actual Triton calls:

```python
initial = x + self.weight
first, observed = update(initial, initial[:, 1, :], ...)
second, later = update(first, first[:, :, -1], ...)
third, number = update_scalar(second, second[0, 1, -1], ...)
empty, none = update(third, third[:, 1, :0], ...)
```

The full return preserves thirteen results, including earlier root states,
source views, scalar observations, empty views, the original input and final
surrounding arithmetic. The retained leading unit axis has logical stride
`rows * cols`; confusing it with the selected axis would give `cols` and fail
the actual coordinate check. The kernel keeps its original 8x4 tiles, logical
grid, masked repeated loads/stores and source loop.

Both i64 and f32 complete models retain all twenty-six initial results through
Merlin host qualification and package reopening. Six independently changed
shape/parameter invocations retain all seventy-eight results with unchanged
libraries. Six native layout checks retain wide i64 and nonfinite f32 words,
scalar outputs and empty retained dimensions. Three source index failures
actually abort without publishing results. Model2MLIR's selection/slice suite
passes 51 checks; three separate type/lowering comparisons agree and explicitly
report their noncanonical standard-operation form. Merlin's independent rank/
coordinate suite passes 22 checks and the existing view/correspondence suite
passes 40. Counts overlap earlier selections and do not change the historical
TTIR inventory.

Eight rehashed rank/guard mutations refuse in both producer and independent
Merlin staging: retained-axis changes, scalar-rank changes and removal of either
coordinate or original source-index assertions. The separate producer regression
selection passes 27 checks. Original parent archives and complete SSA remain
mandatory; declarations cannot authorize a missing check.

Two actual Spike images retain all twenty-six outputs for independently changed
i64/f32 invocations. Retained qualification reopens source staging, images,
observed shapes and all reference words. The report at
`out/rank-reduced-source-views/summary.json` additionally reopens original host
results, every replay, all eight rehashed refusals and three actual source-index
aborts. Prototype logs remain separate from this final leading-unit selection.

Reproduction uses the existing environments documented in the README:

```bash
python -m pytest -q tests/test_tensor_selection_native.py \
  tests/test_rank_selected_view_models.py --basetemp=.scratch/pytest-rank-views
```

General reshape/stride-generating views, negative slices, physical/public
mutation, broader ownership/effect/numeric/source-version combinations and
applicable accelerator correctness/performance remain open.

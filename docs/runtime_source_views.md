# Runtime source views through the Merlin interface

[Rank-reducing source views](rank_reduced_source_views.md) additionally retain
selected unit axes and scalar views through the same explicit profile. The
rank-preserving selection below retains its original evidence and meaning.

Explicit `source_view_policy="runtime-logical-coordinates-v1"` connects
runtime logical source coordinates to execution-owned model binding. It selects
`t2m.execution-checked-call-views.v3` within binding v4. Static v1 and runtime
dense v2 policies retain their existing meanings. Older readers reject the
new policy and fields. Public model tensor types, parent archives, receipt v2
and Merlin signature v6 retain their existing schemas.

## Preserved computation and coordinates

The binder follows actual parent `linalg.transpose`, positive rank-preserving
`tensor.extract_slice`, compatible `tensor.cast` and canonical empty-slice
conditionals back to the declared owning tensor. Runtime dimensions supply
dense logical root strides. Mixed slice operands and permutations derive the
selected logical shape, strides and offset. Checked index products and additions
retain nonnegativity and representability. Every coordinate must equal its
actual checked kernel scalar; every view must retain its original executable
allocation-pointer root/offset binding. A two-axis kernel ABI keeps both axes
explicit, without inferring them from a flattened element count.

Empty slices retain exact `scf.if` extraction/yield and empty/yield branches,
common sizes and dominating coordinates. The actual condition must equal all
sizes being positive. Model2MLIR emits this refinement because the selected
LLVM verifier rejects offsets at the source boundary even for empty extracts.
Both branches, the original source guards and all surrounding model computation
remain executable. Unknown branches, effects and coordinate paths refuse.

Records use `t2m.runtime-source-coordinates.v1`; cloned kernel bodies retain
`t2m.runtime-source-tensor-views.v1`. Merlin independently reconstructs the
coordinate guards and complete parent/kernel SSA, types and source locations.
It imports no producer analysis. Compiler staging excludes invocation arrays
and references. These are logical value relationships; original source-pointer
bindings establish internal kernel sharing. No public physical alias or mutation
permission is added. Bounds, initialization, ownership and lifetime gates remain.

## Real source program

`scripts/capture_dynamic_source_view_model.py` captures three actual Triton
calls with independent exported matrix axes, learned parameters and snapshots:

```python
initial = x + self.weight
first_view = initial.t()[1::2, :]
first, observed = update(initial, first_view, ...)
second, later = update(first, first[::2, 2::3], ...)
empty, none = update(first, first.t()[:0, :], ...)
full = second + self.bias
return full, first, observed, later, empty, none, first_view, initial, x
```

The actual kernel retains its logical two-axis grid, 8x4 tiles, repeated masked
loads/stores and runtime loop. Source roots are functional results; earlier
snapshots and the original input remain unchanged. The model declares rows
2..9 and columns 2..17; a derived slice is empty at column count two.
Torch's scoped experimental `backed_size_oblivious` mode keeps uncertain
contiguity metadata conservative during this capture. It does not disable the
exported range or source-value assertions.

Both i64 and f32 complete models pass Merlin host qualification and producer
reopening. All eighteen initial results match original PyTorch words. Eight
unchanged-library shape/parameter replays check all seventy-two results,
including empty boundary and ragged views. Twelve rehashed view/profile/pointer
claims refuse in producer verification and independent evaluator-free staging.
The separate native layout checks preserve wide i64 and nonfinite f32 words
across zero, singleton and independent matrix dimensions.

Merlin's independent coordinate/view/correspondence selection passes 48 checks;
five additional profile checks refuse downgrades and unknown policies. Native
coordinate and conditional mismatches actually abort before returning outputs.
Model2MLIR's 69-check dynamic-layout selection and two separate lowering/type
oracle comparisons pass. These are separate scopes and overlapping counts,
without any increase to the historical TTIR opcode inventory.
The existing source-view/call-guard/runtime-dense selection passes 29 checks;
three target checks are excluded there and have their own qualification scope.

Two actual serial Spike images retain all eighteen results for an empty-boundary
i64 invocation and a ragged f32 invocation. The retained report at
`out/runtime-source-views/summary.json` reopens original package results, all
replay outputs, target qualifications and actual native guard failures.

Reproduce with the existing tool environments documented in the README:

```bash
python -m pytest -q tests/test_dynamic_permutation_native.py \
  tests/test_dynamic_source_view_models.py --basetemp=.scratch/pytest-runtime-source
```

## Remaining scope

Negative slices, arbitrary stride-generating operations, physical/mutating
public aliases, ambiguous result-axis recovery, general ownership/effect/numeric
compositions and applicable accelerators remain open. The source algorithm and
tiling remain represented; no GPU correspondence or performance verdict follows.

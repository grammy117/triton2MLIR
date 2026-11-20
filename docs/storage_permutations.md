# Dense mutable storage permutations

Complete mutable roots may retain a proved dense axis permutation instead of
requiring row-major source strides. The actual source examples are:

```python
state = x.T.clone()
left, right = state[:, 1], state[:, -1]
```

and the rank-three source with matrix views:

```python
state = x.permute(1, 2, 0).clone()
left, right = state[:, 1, :], state[:, -1, :]
before = state.clone()
a, b = update(left, right, bias)
middle = state.clone()
c, d = update(left, right, bias)
return before, a, b, middle, c, d, state, bias
```

The registered operations use real Triton implementations in
`t2m/examples/mutating_views.py`. Both observe the original arguments, update
left with the captured bias, then update right by multiplication. The matrix
implementation retains source quotient/remainder address arithmetic and one
reused bias load outside its loops. Minimum selected dimensions make left and
right identical, so update order and earlier snapshots are observable.

## Source proof and standard representation

Torch 2.10.0 at `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` implements
`auto_functionalized_v2_dense` with `clone_preserve_strides`. Its original
shape/stride/origin must survive; treating every cloned root as a contiguous
logical tensor changes its regenerated views. Relevant selected upstream
sources are [`auto_functionalize.py`](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/_higher_order_ops/auto_functionalize.py),
[`clone_preserve_strides`](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/_prims_common/__init__.py)
and the actual [view/stride tests](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/test/inductor/test_auto_functionalize.py).

Model2MLIR's `storage_layouts.dense_axes` proves a zero origin and exact dense
axis order using `statically_known_true`. It does not materialize sample hints
or use guarded comparisons to decide layout. Unit axes remain distinct, and
all claimed strides must equal their exact preceding dimension products.
Padded, overlapping, nonzero-origin and unproved layouts still refuse. The
proof deliberately admits less than Torch's general density predicate: empty
or unit dimensions with arbitrary otherwise-irrelevant strides are not granted
a new hidden-storage contract.

The importer reuses the existing standard `ir/permutations.py` expansion:

1. Transpose the complete logical source root into its canonical storage order.
2. Pass that tensor once to the existing functional storage call. All aliased
   arguments retain their actual size/stride/offset coordinates into this root.
3. Apply the exact inverse transpose to the updated base result. Ordinary
   results retain their original slots and logical types.

These are real `linalg.transpose`, `tensor.dim` and `tensor.empty` operations.
`prov.storage_normalization = "dense-permutation-v1"` records source intent on
the transposes; it grants no missing arithmetic, effect or physical alias.
Post-call adapters require complete inverse permutations of their corresponding
incoming roots, with no changed ordinary result or unaccounted payload.

Torch may hoist regenerated dimension/stride queries to the input before its
clone/permutation chain. Capture correspondence checks actual source schemas,
axis maps, typed geometry and scalar arithmetic before accounting for these
nodes. An unrelated dimension owner or unchanged symbolic tag is insufficient.
Actual `sym_stride` reads lower to checked standard products of source tensor
dimensions, with logical i64/index representability checks. They describe
source coordinates, not a query of a downstream physical tensor layout.

## Merlin interface and granularity

The normalized call still selects `m2m.functional-kernel-storage.v1`, executed
binding v4 / `t2m.functional-kernel-binding.v2`, receipt v2 and signature v6.
Both existing readers continue checking the actual readwrite allocation,
complete updated-base ties, pointer ordinals and coordinate operands. No new
public argument, receipt schema, source-word profile or effect permission is
introduced. Original archives remain byte-identical parent artifacts.

Both roots are internal clones, so the examples have eight user results and
no public-input state port. Input tensors remain borrowed; physical public
aliases and automatic caller copyback remain separate work.

Automatic source-only memory selection admits guarded structured-address memory
for the transpose/column example. The rank-three matrix-view source selects
guarded source-word memory, retaining its original division/remainder and access
order. This is an explicit granularity difference from pure model2MLIR tensor
payloads, with executable memory admission overhead. Stronger tensor/structured
raising must prove these accesses; it cannot simply erase the checks. Source
loops and reuse remain, with no GPU thread/warp/address-space mapping.

## Validation scope

`tests/test_mutating_view_models.py` adds both complete two-call models, with
original PyTorch references, changed-shape/parameter host replays, damaged
source/profile/pointer checks, original-domain failures and actual serial
Spike execution. All original returned values and snapshot shapes are checked.
Model2MLIR additionally checks dynamic rank-three permutations, real unit axes,
source schema/owner/axis tampering, malformed result adapters and the continued
hidden/overlapping-storage refusals.

The final new-model selection passes ten checks. Both complete models retain
all sixteen initial, forty-eight changed-invocation and sixteen actual serial
target results as exact words. Ten rehashed profile/pointer/guard changes refuse
in both readers, and two original public-domain violations abort without
publishing results. The original row-major matrix integration separately passes.
Model2MLIR's storage/shape/kernel regression passes 84; the revised two actual
unit-axis cases pass separately. Merlin's independent storage/state contracts
pass 27. Counts overlap and are not a universal coverage measure.

`out/storage-permutations/summary.json` reopens complete source packages, parent
archives, host replay results, both actual target qualifications, all original
reference words, refusals and abort outcomes. It pins actual sources and retained
logs. The original dense-layout and correspondence failures, plus the first
rank-three stride-import and representation-expectation failures, remain in
their original `.scratch` logs. The final model log is
`.scratch/permuted-view2.log`; the reopening log is
`.scratch/storage-permutation-report1.log`.

General base layouts/hidden storage, metadata-changing or optional/list state,
additional symbolic-coordinate/effect/numeric compositions, physical aliases,
caller copyback and accelerator performance remain full-goal work. This adds
no arbitrary-kernel completion percentage or historical opcode inventory count.

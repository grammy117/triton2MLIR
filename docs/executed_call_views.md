# Source views with execution-sized model results

[Runtime dense views](runtime_tensor_views.md) separately select profile v2
with checked runtime dimensions and count/overflow guards. [Runtime source
views](runtime_source_views.md) additionally select profile v3 for checked
positive transpose/slice coordinates. The static v1 scope below retains its meaning.

The execution-owned model binder now connects its existing dense value adapters
and static source-view coordinate proofs. The original parent graph still
computes results and allocation sizes. Views supply no public physical alias or
mutation permission, and provenance cannot replace executable computation.

## Algorithm and independent reconstruction

Dense adapters require the explicit `dense-row-major-v1` policy. Compatible
same-rank tensor casts retain their existing runtime dimension checks. A
rank-changing adapter additionally requires concrete dimensions from original
captured kernel scalars, exact element types, unencoded tensors and equal
element counts. It emits ordinary collapse/expand operations, with scalar
extract/pack for upstream's rank-zero case. Dynamic-to-static narrowing retains
actual dimension assertions before the reshape. Empty tensors retain their
shapes and execute no element accesses.

Source views reuse the established parent-SSA transpose/slice/root recovery and
original allocation-pointer contract. Every captured shape, stride and offset
must agree with both the parent computation and the source pointer binding.
All original schema operands remain accounted for. Kernel bodies keep their
logical grids, tile accesses, loops, source IDs and view provenance.

Merlin independently reconstructs these adapters with its own standard builders
and existing source-coordinate checker. Complete model comparison includes the
original surrounding SSA, every call/body/guard, result type and source location.
Rehashed records cannot authorize false coordinates, dropped operands or a
different adapter. The compiler projection excludes references and invocation
arrays. Kernel/source conversion correctness and runtime lifetime checks remain
separate requirements.

## Explicit package extension

The existing binding schema v4 additionally selects
`execution_view_binding_profile: t2m.execution-checked-call-views.v1` exactly when
calls contain `tensor_views` or `source_tensor_views`. The former records the
known dense policy; the latter retains the existing complete coordinate/root
records. Unknown or missing profiles and incomplete per-call records refuse.
Older execution-owned readers reject the added fields. Bindings without views
keep their existing bytes and meaning. Parent archives, model tensor ABI,
receipt v2 and Merlin signature v6 retain their existing schemas.

Compiler fingerprints additionally bind the independent view and coordinate
implementations. Producer package reopening rebuilds the complete source with
the original view configurations; Merlin staging does not import that producer.

## Real model composition

`scripts/capture_source_view_model.py --execution-sizes --dtype f32` combines
three actual view-update kernels with actual count and scan/scatter kernels:

```python
initial = x + self.weight
first, observed = update(initial, initial.t()[1::2, :], ...)
second, later = update(first, first[::2, 2::3], ...)
empty, none = update(first, first.t()[:0, :], ...)
full = second + self.bias
count = count_positive(full.reshape(-1), 96)
n = count.item()
torch._check_is_size(n)
torch._check(n <= 96)
packed = compact(full.reshape(-1), n, 96)
```

All nine original results include the full state, earlier snapshots, empty
views, packed output and count. Packing retains the 128-lane parallel scan and
masked stores, using explicit automatic memory selection. The source example's
matrix/tile constants do not select compiler behavior.

## Remaining scope

Four complete packages qualify: the original i64/f32/BF16 view models and the
combined f32 view/compaction model. All thirty initial host results and thirty
actual Spike results retain exact source words. Three changed-input domains
reuse the combined model's library and retain all twenty-seven results, including
empty, full and partial compaction. Six rehashed view/profile/pointer mutations
refuse in both producer verification and independent evaluator-free staging.
`out/executed-call-views/summary.json` reopens these separate source, model,
library, raw output and target receipts.

The consumer selection passes 59 checks; argument-word tests pass 31 and existing
call-guard/source-view contracts pass 23. A separate existing compaction target
check retains all seven results. These selections overlap. The initial full
model selection retained three passes and a BF16 projection failure; the
source-frozen BF16 recheck passes after restoring receipt-v2 parent metadata.
The generic owned-result pipeline also now places source legalization before
grid normalization, retaining actual per-operation BF16 rounding. Earlier
failed probes remain in their original logs rather than becoming successful
evidence retroactively.

This connects checked static source coordinates and captured-size dense adapters
to execution-owned results. Fully runtime reshape coordinates, rank-reducing or
negative source slices, encoded/mixed-width public views, physical alias/mutation
ABIs and arbitrary ownership/effect/numeric compositions still need separate
implementations. Captured source-layout options now have the separate
[execution-owned binding](executed_source_layouts.md). Host/serial target correctness
does not qualify accelerator placement or preserved performance. Historical
opcode inventory counts do not change.

# Composed private atomic and volatile memory effects

`--effect-profile atomic-volatile` selects the existing typed atomic numeric
contract together with private volatile observations. It requires guarded
source-word memory, selected directly or through `--memory-profile auto`.
An f32 atomic addition still requires explicit `--f32-atomic-add ieee|ftz`.
This composition does not introduce another arithmetic policy.
`bind-model` carries that global policy into conversion and permits each
binding's `f32_atomic_add` field to override it explicitly.

## Two components, one executable memory contract

Admission `t2m.memory-admission.v6` retains both independent fields:

```json
{
  "effect_profile": "private-invocation-typed-atomics-v2",
  "volatile_profile": "private-invocation-volatile-loads-v1"
}
```

Atomic participant ownership and its thirty-bit fragments remain selected;
adding volatility cannot replace them with the ordinary guard layout. Atomic
old values, operation/order/scope, numeric policy and actual standard atomic
payload remain bound. Each active volatile source read retains an actual LLVM
volatile descriptor load, including unused observations. Existing private
identity anchors keep values live until those effects have been restored, then
disappear from exported computation.

Source-word stages v4 retain the exact nondestructive compute/memory pipelines
and both source inventories. Complete executable tensor tiles, loops, values
reused across iterations and unmapped logical grids remain. An absent source
component creates no synthetic operation; the combined selection also admits
actual atomic-only or volatile-only kernels under the same checked permissions.
Legacy default, atomic and volatile selections keep their prior meanings.
Selecting only one effect still refuses source requiring the other.

Actual synchronous barriers additionally select
`t2m.synchronous-program-barrier.v4` when both effects occur. Producer replay and
the independent Merlin reader reconstruct both original primitive contracts,
protected flags/types, phase scope/prefix/live values and memory-event order.
Both phase selections must match the bound executable memory components.
Deleting one effect cannot be justified by the other effect's provenance.

## Real kernels and upstream boundary

`t2m/examples/atomic_volatile_kernels.py` retains three source volatile sites
and source atomic old values:

```python
value = tl.load(x + lane, active, other=0, volatile=True)
unused = tl.load(x + lane, active, other=0, volatile=True)
for _ in range(steps):
    before = tl.atomic_add(state + slot, value, active,
                           sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    after = tl.load(state + slot, active, other=-3, volatile=True)
    tl.store(old + lane, before, active)
    tl.store(seen + lane, after, active)
    tl.debug_barrier()
```

The synchronized form has two colliding lanes per bin, with bins private to
each logical program. The unsynchronized form instead uses distinct lane
addresses. Both keep the input value outside the runtime loop. A separate CAS
kernel retains actual unmasked compare-exchange over padded state; masking the
returned observations cannot erase an inactive lane's original state update.

The implementation reuses Meta/shared
`17f0c234ce474bee334af36e351c62eca7852836` unstructured pointer/atomic conversion,
including `UnstructuredAtomicRMWConverter`, its `atomic_rmw.mlir` fixture and
`TritonToStructured/atomics.mlir`. The latter includes returned-old-value and
CAS cases. Archived Microsoft `050b93ae9c737111f1ca425d793419b32427a971` records
the earlier pointer/gather boundary. Their conversion APIs supply standard
atomic computation; the load converter's omitted volatile property and the
additional private ABI/protected phase requirements are documented in
[volatile_reads.md](volatile_reads.md) and
[atomic_program_phases.md](atomic_program_phases.md). No compiler rebuild or
Triton CPU backend is selected.

## Interface and remaining scope

Public tensor/scalar ABI, original allocation-source capsules, model capture
receipts and functional result ports retain their existing meanings. Guarded
standard memory, private admission state and localized LLVM volatile leaves
remain explicit granularity differences from pure model2MLIR tensor payloads.
Compatibility metadata names admission v6, word stages v4 and barrier v4.
Earlier readers refuse those versions rather than reinterpreting them.

The private invocation ABI excludes external participants. Ordinary bounds,
initialization, sticky cross-program conflicts and native ownership/lifetime
checks remain required. A volatile read is not an atomic load or publication
permission, and a CTA phase cannot order other programs' ordinary accesses.
Serial atomic qualification uses a source-valid lexical modification order;
it does not establish deterministic GPU lane scheduling or device-bit equality.

General shared/external/asynchronous participation, byte-backed volatile views,
broader numeric/storage/control compositions and accelerator correctness/
performance remain completion work. Actual primitive selection in legacy word
envelopes also needs its own source-driven audit; historical custody-only stage
evidence is not a source-equivalence proof. The arbitrary-kernel goal stays open.

```bash
python -m pytest -q tests/test_atomic_volatile.py \
  --basetemp=.scratch/pytest-atomic-volatile
```

## Standalone qualification

Native RMW cases exercise i32/i64/f32/f64 with and without barriers over empty,
singleton, ragged and multi-program domains and zero/runtime loops. All three
static volatile sites survive LLVM legalization. The i64/f32 protected-source
cases and the i64 unmasked CAS case retain three reusable Merlin libraries,
36 results across twelve unchanged-library invocations and nine results across
three actual serial Spike images, all as exact words. CAS preserves its padded
lane update even when the ordinary observations are masked.

Twenty-eight changed memory/source/phase claims refuse in both readers, and a
real cross-program ordinary conflict aborts without outputs. The new core
selection passes 21 checks; two additional absent-component checks pass. The
existing atomic/volatile/automatic/CAS selection passes 155. Merlin's independent
source selection passes 467 and its separate standalone interface selection
passes 206. These are overlapping regression scopes, not new opcode counts.

## Complete models and the public binder

`scripts/capture_atomic_phase_model.py --volatile --dtype i64|f32` captures
actual PyTorch/Triton models with learned bias and a 32-element initial state:

```python
original = x + self.bias
first, old1, seen1 = exchange(original, self.state, x.shape[0], 2)
second, old2, seen2 = exchange(original, first, x.shape[0], 3)
empty, old0, seen0 = exchange(x[:0], second, 0, 1)
return second, first, old1, seen1, old2, seen2, original, x, empty, old0, seen0, self.state, self.bias
```

The original wrapper clones incoming state. Both dependent calls retain actual
RMW old values and volatile observations across the original phases. The empty
call preserves incoming state and returns empty observation tensors. Every
earlier snapshot, parameter word and original parent archive remains intact.
Existing executed-call binding v4, receipt v2 and Merlin signature v6 retain all
thirteen results. Updated private state remains functional SSA; no public
physical mutation or caller copyback is introduced.

The two models retain 26 initial host results, 78 results across six
unchanged-library invocations and 26 results across two actual serial Spike
images, all as exact words. Twelve rehashed component/source/stage claims refuse
in both readers; two original public shape-domain violations abort without
outputs. All ten model pytest checks pass. The default nonvolatile script path
also passes both existing complete-model regressions.

A separate retained-fixture check invokes the real `bind-model` CLI twice on
the f32 model: once with global IEEE policy, once overriding global FTZ with
per-kernel IEEE selection. Both complete host packages retain all thirteen
original results, for 26 additional comparisons. This closes the missing public
numeric-policy forwarding rather than granting an implicit default. These CLI
packages have host qualification; their selected API counterparts have the
separate target qualification above. The test is part of
`tests/test_atomic_volatile_models.py` for fresh reproduction; the retained run
has ten pytest checks plus this additional two-case invocation.

`out/atomic-volatile-effects/summary.json` reopens all packages, arrays, original
archives, library/input identities, refusals and aborts, plus the five new actual
target images. It records current independent readers and keeps the standalone,
complete API-model and public-CLI scopes separate. The historical fifty-name
inventory is unchanged; general source/effect/storage and target closure remain.

```bash
python -m pytest -q tests/test_atomic_volatile_models.py \
  --basetemp=.scratch/pytest-atomic-volatile-models
```

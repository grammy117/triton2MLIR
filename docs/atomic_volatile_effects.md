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

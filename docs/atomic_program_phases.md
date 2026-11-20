# Atomic updates across synchronous program phases

The explicit `t2m.synchronous-program-barrier.v2` contract composes pre-layout,
single-CTA program barriers with the existing private-invocation atomic effect
profiles. It does not introduce a new atomic arithmetic policy. Ordinary source
without actual atomics retains barrier v1, even if an atomic profile is selected.

```python
for _ in range(steps):
    previous = tl.atomic_add(state + slot, value, active,
                             sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    after = tl.load(state + slot, active, other=-3)
    tl.store(old + lane, previous, active)
    tl.store(seen + lane, after, active)
    tl.debug_barrier()
```

`t2m/examples/atomic_barrier_kernels.py` uses eight lanes and four bins per
logical program. Two lanes collide in each full bin. The original value load
stays outside the loop and is reused. Returned atomic old values cross the
first phase boundary; subsequent ordinary reads observe the completed updates.
The second boundary completes observations before the next iteration. Masks,
unused state bins, initial outputs and zero-trip behavior remain executable.

## Representation and source checks

The existing Meta/shared unstructured atomic conversion supplies parallel
Linalg payloads and returned old values. Existing integer or typed atomic
adapters retain standard `memref.atomic_rmw` or generic atomic computation;
standard exactly-once SCF regions retain the phase boundaries. Logical grids
remain unmapped. No atomic or rendezvous becomes an annotation-only effect.

Source review uses Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`:
`UnstructuredAtomicRMWConverter` in
`lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp`,
`test/Conversion/UnstructuredToMemref/atomic_rmw.mlir` and
`test/Conversion/TritonToStructured/atomics.mlir`. The archived Microsoft
revision `050b93ae9c737111f1ca425d793419b32427a971` supplies the historical
comparison. Neither implementation supplies this source/model phase contract.

Original source inventory precedes normalization. Protected allocation capsules
retain both original atomic primitive fields and v2's selected effect profile/
f32 addition policy. Producer replay and Merlin's independent reader reopen
actual source, reconstruct operation/type/order/scope, check protected atomic
fields and preserve source memory-event order. The phase selection must equal
the bound executable memory profile. Deleted effects, changed primitives,
downgraded contracts and contradictory selected policies refuse.

## Execution scope

`tests/test_atomic_barriers.py` exercises real captured i64 and f32 addition,
direct and allocation-bound conversion, native execution and reusable Merlin
host libraries. Changing lengths, values and loop counts retain complete state,
old values and synchronized observations, including tails, empty grids and zero
trips. The independent serial oracle uses a source-valid lexical lane
modification order; this does not establish deterministic GPU atomic scheduling.
Merlin additionally reconstructs the pinned integer RMW/CAS and typed floating
primitive policies independently in its source-reader tests.

The existing private-buffer ABI excludes external participants. Source order
and scope stay recorded; supported operations retain the existing stronger
standard acq_rel implementation. Cross-program ordinary/atomic conflicts still
abort through the actual compiled guard. A CTA barrier neither resets that
guard nor grants grid-wide publication. Shared/external/asynchronous effects,
multi-CTA participation, broader numeric compositions and accelerator performance
remain open. The complete model/serial-target selection below has its own evidence.

The public tensor/scalar ABI, receipt schemas and source capsule artifact roster
are unchanged. Compatibility explicitly names v2; readers restricted to v1
refuse it. Guarded standard memory and private admission state remain a declared
difference from pure model2MLIR tensor payloads.

## Complete models and actual serial targets

`scripts/capture_atomic_phase_model.py --dtype i64|f32` captures real
`torch.library.triton_op` calls with parameter state and bias:

```python
original = x + self.bias
first, old1, seen1 = exchange(original, self.state, x.shape[0], 2)
second, old2, seen2 = exchange(original, first, x.shape[0], 3)
empty, old0, seen0 = exchange(x[:0], second, 0, 1)
return second, first, old1, seen1, old2, seen2, original, x, empty, old0, seen0, self.state, self.bias
```

The original wrapper clones its incoming state and returns three functional
values. Both dependent calls retain the original tile/atomic/phase algorithm;
the explicit empty call preserves the incoming state and returns empty old/
observation tensors. All earlier snapshots and parameter words remain intact.
Independent Torch CPU implementations supply the original thirteen references.
Changed-shape/parameter replays and target inputs use a separate NumPy oracle.

Existing allocation-source capsules, executed-call binding v4, receipt v2 and
Merlin signature v6 carry both complete models. Each original archive and every
result remain retained. Actual descriptors supply result sizes, and borrowed
public inputs remain readonly. The selected guarded standard-memory payload and
v2 barrier contract remain explicit compatibility differences.

The two models retain 26 initial host results, 78 results across six additional
unchanged-library invocations, and 26 results across two actual RISC-V/Spike
images, all as exact words for the selected finite integer/dyadic inputs.
Six rehashed model claims refuse in both readers; two original source-domain
violations abort without outputs. The standalone selection additionally retains
sixteen changed source/effect claims refused by both readers and an actual
cross-program-conflict abort. These are separate scopes, not additive unique
kernel coverage.

The frontend regression selection passes 158 checks; the final focused atomic
phase selection passes 16. Complete model execution and model-refusal/domain
selections pass six and four checks respectively. Merlin's independent reader
selection passes 423. Counts overlap earlier selections. Retained source,
package, unchanged-library, full-result, refusal and target records are linked
from `out/atomic-program-phases/summary.json`. Retained verification reopens both
packages and actual target images with current independent-reader fingerprints.

```bash
python -m pytest -q tests/test_atomic_barriers.py tests/test_atomic_phase_models.py \
  --basetemp=.scratch/pytest-atomic-program-phases
```

Use the existing source/model/native/Merlin/serial-target environments from the
README. This qualifies the selected private synchronous effect composition;
general shared/async/external participation, broader source/numeric/ownership
compositions and accelerator correctness/performance remain completion work.

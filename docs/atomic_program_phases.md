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
remain open. Complete model/target qualification is a separate next gate.

The public tensor/scalar ABI, receipt schemas and source capsule artifact roster
are unchanged. Compatibility explicitly names v2; readers restricted to v1
refuse it. Guarded standard memory and private admission state remain a declared
difference from pure model2MLIR tensor payloads.

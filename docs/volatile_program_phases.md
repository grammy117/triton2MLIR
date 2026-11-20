# Volatile observations across synchronous program phases

`t2m.synchronous-program-barrier.v3` binds actual private volatile reads to
the existing pre-layout, single-CTA phase representation. Select the existing
`--effect-profile volatile --memory-profile guarded-buffer --address-policy
source-words`; actual source barriers additionally require the v3 contract.
Source without volatile reads retains ordinary barrier v1. Atomic phases retain
their separate v2 contract and numeric policy.

## Original algorithm and executable representation

`t2m/examples/volatile_barrier_kernels.py` performs a real unused volatile read
before the loop, then repeated volatile read/update/transpose-observe phases:

```python
unused = tl.load(state + pointer, active, other=0, volatile=True)
for step in range(steps):
    before = tl.load(state + pointer, active, other=0, volatile=True)
    tl.store(state + pointer, before + step + 1, active)
    tl.debug_barrier()
    after = tl.load(state + transposed_pointer, transposed_active,
                    other=-3, volatile=True)
    tl.store(out + pointer, after, active)
    tl.debug_barrier()
```

The source uses four-by-four tiles. The transposed read observes values written
by other lanes in the same logical program; the second boundary finishes those
observations before the next update. Complete tensor computation and standard
SCF phases retain this order. The grid stays unmapped, original runtime loops
remain, and each actual active source read uses the existing descriptor-derived
standard LLVM volatile leaf. The unused read survives even when the loop has no
iterations. No source observation or rendezvous becomes annotation-only.

Reuse remains at the existing pinned boundary: Meta/shared
`17f0c234ce474bee334af36e351c62eca7852836` supplies isolated unstructured pointer
events and arithmetic/memory conversion; archived Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` records the earlier gather conversion.
Their load converters omit the volatile property, as recorded with the source
and `gather_no_loop.mlir` fixture in [volatile_reads.md](volatile_reads.md).
Triton `142586f90aac9b917af1081e4697b085027e13b2` defines barrier address-space
participation. Neither shared converter supplies the protected source/model
phase contract. Existing defined identity anchors prevent premature removal of
observations until executable volatile loads have been restored.

## Independent contracts and package differences

Both source readers reconstruct every original barrier's participation, mask,
phase prefix, live results and complete event order. V3 additionally records
`volatile_effects.profile = private-invocation-volatile-loads-v1`. Every original
volatile primitive retains its actual boolean and result type in protected
source, including unused observations. Missing flags, invented observations,
changed primitive claims, removed/reordered phases and downgraded selections
refuse. Original allocation capsules keep the pre-wrapper graph separate from
the eventual bound graph. Its phase profile must equal the bound memory profile.

The standard memory payload, private executable admission state and localized
LLVM leaves remain explicit differences from pure model2MLIR tensor payloads.
Public tensor/scalar types, allocation capsules, capture receipts and complete
model binding interfaces retain their existing meanings. Older barrier readers
refuse v3; compatibility metadata names the new selection explicitly.

Bounds, complete initialization, sticky cross-program conflicts and native
ownership/lifetime checks remain authoritative. The private invocation ABI
excludes external participants. A program barrier cannot authorize another
program's ordinary accesses or grid-wide publication. Selected
[atomic/volatile composition](atomic_volatile_effects.md) has a separate v4
contract. Shared/external/asynchronous effects, physical observations, broader numeric/
storage compositions and accelerator correctness/performance remain completion
work. The full arbitrary-kernel objective remains open.

## Reproduction

The existing environments in the README supply native, source, model2MLIR,
Merlin and serial-target tools; no Triton CPU backend or new compiler build is
selected.

```bash
python -m pytest -q tests/test_volatile_barriers.py \
  --basetemp=.scratch/pytest-volatile-phases
```

The native selection retains all three static volatile sites after LLVM
legalization for i32/i64/f32/f64 over empty, singleton, ragged and multi-program
domains and zero/runtime loop counts. Protected allocation i64/f32 cases retain
all four reusable-host invocations and four results across two actual serial
Spike images. Sixteen source/effect claims refuse in both readers; a real
cross-program conflict still aborts without outputs. The overlapping ordinary/
atomic/program-phase frontend selection passes 54 checks, with one additional
original-location identity check. Merlin's independent selection passes 459.

## Complete dependent models

`scripts/capture_barrier_model.py --volatile --dtype i64|f32` captures actual
PyTorch/Triton models with learned bias, dependent state and an empty call:

```python
original = x + self.bias
first, a = exchange(original, x.shape[0], 2)
second, b = exchange(first, x.shape[0], 3)
empty, z = exchange(x[:0], 0, 4)
return second, first, a, b, original, x, empty, z, self.bias
```

The original wrapper clones each incoming state and returns both the updated
base and transpose observations. Original CPU implementations supply every
reference before compilation. Captured rows retain the actual `[2, 19]` source
domain. Existing executed-call binding v4, receipt v2 and Merlin signature v6
retain all nine results, earlier snapshots and every parent archive byte.
Borrowed public inputs remain readonly; this introduces no caller copyback.

Both complete models qualify with 18 initial host results, 54 results across
six unchanged-library invocations and 18 results across two actual serial Spike
images, all as exact words. Twenty-two rehashed source/effect/phase claims refuse
in both readers, including removed flags, changed primitive claims and bound
effect mismatches. Two original public-domain violations actually abort without
outputs. Complete model tests pass all ten checks; two earlier ordinary model
configurations additionally pass their complete-result regressions with default
script options. These counts overlap other selections and change no historical
opcode inventory or accelerator/performance claim.

`out/volatile-program-phases/summary.json` reopens the source packages, complete
models, original archives, every host/replay/target result, actual aborts and
current independent source readers. It records the four selected new target
images separately from earlier ordinary/atomic regressions.

```bash
python -m pytest -q tests/test_volatile_barrier_models.py \
  --basetemp=.scratch/pytest-volatile-phase-models
```

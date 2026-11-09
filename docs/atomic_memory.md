# Private-invocation atomic memory through the tensor interface

Status: implementation plan with the native RMW middle layer verified. Atomics
are still refused by the product frontend. The plan extends the full objective;
it does not redefine all-kernel coverage as accepting a diagnostic or opaque call.

## Source-backed boundary

Meta `17f0c234ce474bee334af36e351c62eca7852836` already supplies:

- `TritonToUnstructuredPass.cpp`: `tt.atomic_rmw` becomes
  `tts.unstructured_atomic_rmw` with exact indices, operation, mask, memory
  semantics and scope.
- `UnstructuredToMemrefPass.cpp`, `UnstructuredAtomicRMWConverter`: a parallel
  tensor `linalg.generic` performs standard `memref.atomic_rmw` and yields its
  old value. `test/Conversion/UnstructuredToMemref/atomic_rmw.mlir` checks
  integer/floating forms, masking and returned values. Structured fixtures
  additionally check chained old-value uses.

The existing optimizer has executed this exact path for a captured Triton
integer indexed atomic-add kernel. Both native stages and standard xDSL
verification pass; source operation, scope and order survive to the TTS stage.
This probe does not yet pass the product's memory admission or Merlin execution.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`,
`MemRefToLLVM.cpp`, lowers standard RMW atomics with `acq_rel` ordering.
That ordering is at least Triton's relaxed/acquire/release/acq_rel requirement.
The first effect profile owns private invocation buffers with no external
participants: GPU scope closes over logical programs, CTA scope over one
logical program, and system scope adds no externally shared storage to this
ABI. Stronger standard atomics are a refinement; source scope/order remain
recorded for downstream analyses. No physical CTA/thread/warp mapping enters
the frontend. External publication, aliases and progress/synchronization
contracts remain additional work.

## Admission protocol

Keep the ordinary-memory profile unchanged. Explicit atomic admission uses a
versioned contract and the same public tensor/scalar interface. Preserve the
logical forall and parallel payload rather than serializing source programs.

The existing sticky ownership protocol must distinguish atomic participants
from ordinary reads/writes in both arrival orders. Use thirty-bit program
fragments paired with their complements, leaving three flags in each i64 word:
ordinary access, ordinary writer and atomic access. Accumulate them with atomic
OR. An atomic access may see foreign tags only when all previous accesses are
atomic. An ordinary read may see foreign tags only without a writer/atomic
flag. An ordinary store always requires one program owner. A same-program
ordinary observation after an atomic remains sticky and prevents a later
foreign atomic from bypassing that observation. Duplicate atomic lanes are
allowed; duplicate ordinary stores retain their existing rejection.

Atomic RMW requires an already completed/initialized element. Readwrite inputs
provide that state. Fresh state still requires an actual preceding source
write and post-grid coverage; an atomic does not invent its initial value.
Guard storage cost is derived from the new fragment width and actual shapes,
including the boundary where its word count differs from v2. Allocation and
consumer budgets must use the same declared contract.

Upstream masked RMW yields the uninitialized output iter-arg on inactive lanes.
Represent only those undefined old values by an explicit zero refinement,
with no inactive memory effect and no full-output initialization work. Retain
that choice in provenance/package differences. Active old values and every
atomic update remain executable operations.

## Delivery order and evidence

1. Implement and independently enumerate the sticky protocol's state
   transitions, including both atomic/ordinary orders and later observations.
2. Admit indexed integer RMW through the official source-word pipeline;
   instrument bounds/initialization/compatibility without lane uniqueness for
   atomics. Validate actual native payload against the declared RMW operations,
   scope/order and source inventory. Keep all existing memory guards.
3. Version admission/evidence, explicit profile selection and package
   differences. Extend standalone and derived Merlin intake, scalar ABI,
   allocation budgets and source-only staging together.
4. Execute colliding and disjoint i32/i64 add, exchange, bitwise, signed/unsigned
   min/max with masks, tails, loops, old-value chains and fresh initialization.
   Use exact references where ordering is unique; for colliding old values,
   check a valid linearization/permutation instead of imposing a GPU schedule.
   Reuse unchanged Merlin libraries over independent domains and preserve inputs.
5. Extend isolated event/pointer-control conversion to region-local atomics.
   Then implement CAS from exact compare/exchange semantics using a standard
   atomic region/consumer implementation: the current unstructured converter
   has no CAS case. Reuse official/shared patterns where available and retain
   a complete, replayable disposition for every source event.
6. Add the float/numeric, descriptor-reduction, synchronization/external-effect
   and target materialization contracts required by the universal objective.

Pure opcode appearances, source-stage success, stronger-order annotations and
host-only execution are distinct evidence. None certifies all effects, all
numeric modes, all targets or the completed all-TTIR goal.

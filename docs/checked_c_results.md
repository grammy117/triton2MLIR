# Execution-sized results through a checked C call

Merlin now separates lowering from host linking (`bb347b1de`) and supports
explicit serial allocation context/external failure reporting (`f2552dec1`).
Actual observed cross-compiler contracts produce model and bridge objects.
A standard-MLIR runtime fixture additionally links and executes on Spike,
retaining all nine results across three invocations; an allocation-failure
image exits without publishing results. This is a runtime-seam test, not the
production seven-result TTIR package driver or its readback protocol.

The final 152-check Merlin selection passes in
`out/build/owned-runtime-final1.log`. Two source-frozen image rechecks use
explicit retention in `out/build/pytest-owned-images-retained/`, with actual
source/tool/image/output identities and process outcomes. The frontend's real
TTIR model separately passes all eight C checks against the new implementation
in `.scratch/compact-c-lowering1.log`. Source fingerprints now pin the common
lowerer too. The [target plan](execution_result_targets.md) retains the remaining
staged-driver, input-transport and observed-shape readback requirements.

Merlin `0606fd4f1` adds `merlin_owned_result_invoke_checked` to the existing
official returned-descriptor bridge. A caller explicitly begins one tracked
invocation, invokes the actual source once, validates every result, copies
logical words into independent storage and releases the recorded owners.
The original Python validator remains a separately implemented path.

The C validator checks source-fixed rank/storage/static extents, actual live
owners, alignment, signed-stride footprints, shape/product overflow, total
logical output bytes and complete owner closure. Invalid results clear all
views before any copy. Empty tensors make no payload access. Duplicate results
share source owners but retain their complete logical byte charge and receive
independent caller copies. Copying rechecks live source ownership and refuses
overlapping source-owner storage. Allocation and lifetime gates remain active.

The source-typed C generator and runtime implementation/header are pinned in
saved compiler programs and now in the frontend's selected consumer-source
fingerprints. Public portable MLIR, parent archives, receipt v2, signature v6
and existing host qualification schemas retain their meanings. The additional
C checks do not establish TTIR equivalence, a separate compiler backend or
automatic target support.

## Real TTIR model evidence

`tests/test_compact_model_c.py` links a standalone C executable against the
actual qualified three-kernel [compaction model](compaction.md). Its inputs
are the learned scalar parameter and original tensor, not predicted counts or
result extents. All seven output shapes and payloads come from actual returned
descriptors after source computation. Copies remain valid after source owners
are released, and original inputs/parameters stay unchanged.

Six input sets cover empty/sparse/full masks, changed parameters and i32
overflow. Two C calls per set check all 84 complete results as exact words.
Two invalid input lengths abort through original exported-domain assertions.
One unchanged library and C executable handle all these invocations.
The final selection passes eight checks in `.scratch/compact-c-final1.log`;
`out/compact-model-c/summary.json` reopens results, source fingerprints and
actual abort evidence. The test's small binary record is an observation fixture,
not a production target readback protocol.

Merlin's separate focused selection passes 115 checks in
`out/build/owned-c-final1.log`, including corrupt descriptors in optimized and
undefined-arithmetic-trapping builds, exact scalar calling conventions, actual
compiled C/Python compaction and existing ownership/persistence regressions.

## Remaining integration

This implements the reusable checked C invocation/copy seam. The separate
[generated driver](owned_result_driver.md) and
[execution-owned target package](execution_owned_target_packages.md) now add
ordinary staged generation, observed compiler layout, serial allocator context,
actual linking/readback and Spike qualification. The current bridge retains
its selected 64-bit ABI; the C view helper requires matching size/address widths.
Broader executors/effects and accelerator/performance qualification remain open.

# Program phases under original scalar CFG control

Valid early-return kernels can now retain synchronous program barriers through
the existing official CFG-to-SCF path. Original scalar branches choose an edge
for the complete logical program. Tile updates and cross-lane observations
remain actual memory phases at their original control positions; runtime loops,
early exits, masks and the unmapped logical grid stay executable.

The reproduced kernel previously refused at its first barrier because its
containing TTIR function had multiple blocks. Allocation binding separately
required one source block. The frontend now proves scalar control participation
and derives a complete void wrapper signature from the original entry block,
retaining every source block behind the actual typed call. No source CFG is
flattened or replaced with a kernel-name implementation.

## Source and transformation contract

Phase contracts v1/v2/v3/v4 optionally add
`source_control = whole-program-scalar-cfg-v1` when their source function has
multiple blocks. Both readers reconstruct actual scalar integer conditions,
same-region successors and exact complete block-argument types. Tensor/lane
conditions, foreign edges, incomplete arguments and unknown terminators refuse.
Structured regions keep their single-block requirements. Original inventories,
effect policies, phase scope, complete payload and memory-event order remain
separately checked.

Expansion still wraps synchronous operations in standard exactly-once SCF
regions, then uses the existing [official CFG lifter](cfg_intake.md). Reviewed
LLVM `a47bddccec30255619bb8c37fa59700e661d4e66` implements scalar conditional/
switch lifting and tests original edges, block arguments and backedges. Meta
`17f0c234ce474bee334af36e351c62eca7852836`'s `early_return.mlir` retains original
TTIR branches through its arithmetic converter; it does not supply our tensor
ABI or barrier participation proof. Existing shared conversion still supplies
the enclosed computation; no native stack is rebuilt.

The independent Merlin reader imports no producer or evaluator implementation.
The additional contract grants whole-program participation only: no lane
divergence, cross-program synchronization, async progress or new arithmetic/
effect policy. Existing bounds, initialization and sticky conflict gates remain.
Multi-CTA, general shared/external/async and physical observations stay open.

Public tensor/scalar ABI, capture receipts and artifact rosters are unchanged.
Existing CFG and allocation-source capsules retain their original compiler files.
Older phase readers reject the additional contract field; generated compatibility
metadata explicitly names it. See [package differences](package.md).

## Qualified selection

`t2m/examples/cfg_barrier_kernels.py` updates a matrix tile, rendezvouses, reads
its transpose and retains original old values. A runtime mode skips alternating
logical programs via an original early return. Runtime rows/steps cover empty,
singleton, zero-trip, ragged and multiple-program domains.

Eight i64/f32/effect configurations execute natively, preserving 120 complete
results across forty invocations. Four protected allocation configurations pass
actual Merlin execution and sixteen unchanged-library replays, retaining twelve
initial and 48 replay results as exact words. Three changed participation claims
refuse both source readers. The 62-check frontend/regression selection and the
independent 336-check Merlin selection pass, with overlapping scopes.
Complete enclosing-model and serial-target qualification is tracked separately;
these kernel results alone supply no whole-model or accelerator verdict.

```bash
python -m pytest -q tests/test_cfg_barriers.py \
  --basetemp=.scratch/pytest-cfg-program-phases
```

General CFG compositions, value-returning public source entries, shared/async
participation, source-version/numeric/storage closure and applicable accelerator
correctness/performance remain part of the full arbitrary-kernel goal.

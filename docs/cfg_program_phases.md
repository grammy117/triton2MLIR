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
Complete enclosing-model and serial-target qualification is recorded below;
these kernel checks retain their separate scope.

```bash
python -m pytest -q tests/test_cfg_barriers.py \
  --basetemp=.scratch/pytest-cfg-program-phases
```

## Complete model and serial-target qualification

`scripts/capture_cfg_barrier_model.py` captures real i64 ordinary and f32
atomic/volatile models. Each call clones all three incoming states. The second
call selects the complementary programs; original old-value/observation states
survive skipped programs. An empty third call retains empty results:

```python
original = x + self.bias
old, seen = torch.full_like(original, -17), torch.full_like(original, -23)
first, a, b = exchange(original, old, seen, x.shape[0], 2, 0)
second, c, d = exchange(first, a, b, x.shape[0], 3, 1)
empty, e, f = exchange(x[:0], a[:0], b[:0], 0, 1, 1)
return second, first, a, b, c, d, original, x, empty, e, f, self.bias
```

Model intake exposed a separate `full_like` initialization gap. Model2MLIR now
passes actual source `tensor.dim` values into dynamic `tensor.splat`, retaining
existing literal conversion and explicit dtype without an opaque call or sample
extent. Its 56 selected fill/pointwise/literal checks pass. The static fill
differential oracle lowers in both frontends with matching result types; this
oracle does not supply dynamic execution evidence.

The complete ten-check model selection retains 24 initial and 72 unchanged-
library replay results, plus 24 results across two actual serial Spike images.
All twelve model outputs retain exact words and every parent archive stays
byte-identical. Six consistently rehashed model participation claims refuse in
both readers; two original public-domain violations abort without results.
The target records pin the actual independent barrier reader. Existing executed
binding v4, receipt v2 and signature v6 remain unchanged. Generated compatibility
metadata names the source-control extension separately from effect permissions.

`out/cfg-program-phases/summary.json` reopens original failure evidence, all
forty native observations, four protected libraries/replays, complete model/
target results, refusal/abort records and exact source pins. Historical opcode
inventory counts are unchanged. No accelerator or performance verdict follows.

General CFG compositions, value-returning public source entries, shared/async
participation, source-version/numeric/storage closure and applicable accelerator
correctness/performance remain part of the full arbitrary-kernel goal.

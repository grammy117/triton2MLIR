# Independent fresh tile groups preserve source dependencies

Fresh tile outputs may now retain independent write groups across interleaved
ordinary accesses to other actual ABI roots. Each source observation still
reads only preceding writes under its actual mask. Source values, input-state
updates, fills, tail masks and last-active-write ordering remain executable.
The payload stays in tensor/Linalg/SCF dataflow with an unmapped logical grid.

This closes `.scratch/interleaved-tiles-probe1/source.ttir`, whose complete
positive/negative writes to two distinct outputs previously failed at the first
masked scatter's attempted old-output read. The original failed probe remains
failed evidence; later source conversion and execution have separate records.

## Actual root and observation facts

`TensorABI`'s `write` roots are fresh functional results, omitted from public
inputs and materialized as separate initial tensor values. The source adapter
now keeps a group for each such root. Root identity comes from actual entry
arguments and typed pointer expressions, not names or provenance. Two stores
through the same pointer are never made independent by separate formal names
or caller tags. Logical program/lane ownership remains a separate proof.

The collector retains ordinary reads and writes to other actual roots at their
original positions. Fresh result storage cannot alias those input/result roots;
no independence between input roots is inferred. Readonly reloads retain their
source operations. Input-state updates and later rereads retain the original
state ordering and functional state result. Unknown pointers, changed addresses
within a pending root, incompatible properties, calls, nonordinary effects and
region control remain boundaries.

Groups are re-proved from current actual SSA when discharged. An earlier sibling
may already have reconstructed an exact same-address observation. Re-reading
predicates and coverage avoids stale candidates or facts about erased load
results. Each active read needs implication from its actual mask to preceding
writes, with its original defined fill. Neither sibling coverage nor final output
coverage grants permission to read an earlier hole.

The source-order select chain remains the same as the single-root transform.
Bounds, original source-width arithmetic, complete geometry and ownership run
independently. Existing 128-event/predicate and Boolean budgets remain; the
collector also caps 128 pending roots and 1024 ordinary events before a boundary.
Missing facts decline the optional rewrite and retain the ordinary path.

## Real source programs and enclosing model

`t2m/examples/interleaved_tile_kernels.py` includes:

```python
tl.store(a + lane, first_a, valid & p)
earlier_a = tl.load(a + lane, valid & p, other=-17)
tl.store(b + lane, earlier_a ^ value ^ 2, valid & q)
earlier_b = tl.load(b + lane, valid & q, other=-31)
tl.store(a + lane, earlier_b ^ value ^ 4, valid & ~p)
tl.store(b + lane, earlier_a ^ value ^ 8, valid & ~q)
```

Both outputs are complete, while their intermediate observations differ from
the final values. Another form derives `q` from the first output's observation,
retaining both data and control dependencies. Further forms preserve an
immutable reload or a mutable input write/reread between fresh-output stores.
The mutable input remains a borrowed public value and returns updated state;
this does not add imperative caller copyback or physical alias support.

`scripts/capture_interleaved_model.py` captures two dependent real Triton calls,
learned i64 bias, live `n-2` control and surrounding standard addition. Every one
of its eleven results remains: final outputs, first-call outputs, all intermediate
observations, original input, dependent input and learned state. The exported
length domain remains 2..31. The registered PyTorch CPU implementation supplies
an independent reference outside compiler staging; no Triton CPU backend is used.

## Upstream reuse and package boundary

The native pointer/mask passes and existing tensor adapters remain in use. The
reviewed Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` implementations and fixtures include
`MaskAnalysis.cpp`, `UnstructuredToMemrefPass.cpp` and
`gather_scatter_all_mask.mlir`. Their actual per-lane store predicates and
inactive no-ops motivate retaining exact source order and observation masks.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s
`mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.cpp` and
`mlir/test/Analysis/test-alias-analysis.mlir` distinguish new allocations from
other allocations and enclosing inputs, while input arguments may alias. The
added proof applies the already-declared functional fresh-result contract at
TTIR intake; it does not claim LLVM ran alias analysis on unlowered TTIR pointer
arguments or invent a general source alias solver. No upstream code blocks were
copied into the collector.

The existing `source-tile-write-partition` report adds actual independent-root
and retained ordinary-access correspondence. Generated compatibility metadata
explicitly records that scope. Original TTIR and complete source accounting
remain. No public ABI, result mode, receipt schema, artifact roster, numeric or
effect profile changes. Provenance supplies no allocation, alias, initialization,
effect or target permission.

## Qualification and next boundaries

The final source/kernel/model selection passes 33 checks; earlier tile,
source-partition, masked-scatter, conditional, shared-tensor and attention
regressions pass 132. Targeted Ruff and `git diff --check` pass. The first
five-form kernel run passed 25; a separate initial model run passed three
checks but its serial test lacked a helper import path. The final run adds
the predicate-dependent form and fixes that test harness. The earlier failure
is retained and does not qualify a target image.

`out/interleaved-tiles/summary.json` reopens six kernel libraries and the full
model library: all 36 initial, 158 unchanged-library replay and 72 exact-word
target results, across fourteen actual serial images. The original model
domain also actually aborts without outputs. The state kernel retains all
five results, including the updated input; the complete model retains all
eleven. Original parent archives and every result are checked. Selected source
hashes are not a full dependency closure; serial execution establishes no
accelerator correctness or performance claim.

Implementation `6716ba902f6519703133081de99d0dda19c6e884` retains
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba` on
`ttir-experiment-grammy`. All six selected implementation/test source files
match the actual producer commit bytes. The reopening report records that
match separately from its original observed revisions.

Final logs are `.scratch/interleaved-final1.log`,
`.scratch/interleaved-regression1.log` and `.scratch/interleaved-report1.log`.
Reproduction uses the existing tool environments documented in the README:

```bash
python -m pytest -q tests/test_interleaved_tiles.py tests/test_interleaved_model.py \
  --basetemp=.scratch/pytest-interleaved
```

The next reproduced tensor-admission gap at that revision was
`.scratch/branch-tile-probe2/source.ttir`: an initial positive-lane write precedes
an actual scalar branch whose two arms each complete the negative lanes with
different values. Its control admits the complete signed-i32 domain. The current
collector stopped at the region boundary, and the default tensor path refused at
the first masked scatter. [Branch-completed tiles](branch_tiles.md) now close
that form. The failed probe is not a package/target result and
does not establish that every other memory profile refuses. Region-aware tile
state remains separate from the broader shared/async and target requirements.

General pointer equivalence, overlapping address groups, region-spanning
dependencies, shared/async participation, broader numeric/source versions and
applicable accelerator correctness/performance remain full-goal work. The
historical operation-name denominator is unchanged.

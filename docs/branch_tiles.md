# Partial tile prefixes complete through actual branch successors

Fresh tile writes may now complete across scalar SCF branches while retaining
the original conditional computation. The frontend distributes an eligible
write prefix into both actual successors only after every selected source path
completes the same retained cover. Exactly one copy of the prefix executes.
Later ordinary tensor raising combines its stores in source order. Branch-local
arithmetic, input reuse, masks, fills and logical parallel tiles remain.

This closes the default tensor refusal saved in
`.scratch/branch-tile-probe2/source.ttir`. Its original failed conversion is
separate from later successful packages and execution evidence.

```python
tl.store(out + lane, first, valid & positive)
earlier = tl.load(out + lane, valid & positive, other=-17)
if control > 0:
    quotient = (word & 65535) // control
    tl.store(out + lane, quotient, valid & ~positive)
else:
    tl.store(out + lane, alternative, valid & ~positive)
```

The earlier observation is reconstructed at its original point from preceding
writes and its explicit fill. The prefix store moves into the selected arm;
the division stays in its original arm and never executes for zero control.
The frontend does not speculate branch-local arithmetic to manufacture a
whole-tensor expression or substitute an opaque operation.

## Proof and source scope

`tile_write_proof.analyze_writes` checks ordered stores and actual same-address
reads. Its Boolean facts come from real enclosing `scf.if` choices. Scope can
simplify a redundant mask inside its selected region; it cannot initialize an
earlier read before that choice. Prefix observations are proved before any
future edge is considered. Source widths, active access bounds, complete output
geometry and program/lane ownership remain independent checks.

The source planner checks each selected path through nested scalar branches.
It retains the same typed pointer expression and fresh functional result root.
Unknown operations/effects, changed same-root addresses and loop boundaries
decline this optional distribution. Distinct fresh roots remain independent;
ordinary input-state ordering retains its existing contract. The current proof
caps are 128 events/predicates per group, 128 selected paths, 32 branch levels,
1024 visited source operations per plan and 1024 distributed store copies per
function, in addition to the existing Boolean budgets. Exhaustion supplies no
optional proof. `static_copies=2` describes the two copies of the whole prefix;
`stores` records its original store count.

An omitted else is the false fallthrough. When that path's prefix itself
completes the output, an explicit false block retains the prefix effect there.
Original branch results and yields remain; no dummy input storage or
initialization bitmap is introduced. Only stores move across the choice.
Arithmetic values already computed before the branch remain before it, and
arithmetic originally inside either arm remains there.

## Existing native pointer-state prepass

The branch-observation form exposed a separate native-stage limitation: the
installed optimizer boxed an i64 data result as `tts.get_structured_state`, whose
operand verifier accepted only i32 in that binary. The earlier failure remains
under `.scratch/pytest-branch-structure1`. The surveyed Meta source now accepts
i32/i64, while the historical Microsoft source selected wider integer/index
types. Those source revisions do not attest the installed binary's build.

The existing `numeric-control-without-pointer-state` selection now admits
ordinary memory inside numeric control interfaces when actual state dependency
checks prove that results and carried block arguments stay out of pointer and
mask computation. The for induction variable is separate from carried state.
Data stores and diagnostics keep their existing checks; unknown calls/effects
or address/mask uses retain the original prepass. The supported
`skip-prepass=true` option skips only that boxing prepass. Actual pointer
conversion, complete regions, memory operations and ownership checks remain.
No region is outlined, hidden or rewritten for this option.

## Upstream reuse and package boundary

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s SCF.cpp successor and
invocation interfaces, plus `mlir/test/Dialect/SCF/ops.mlir`, establish actual
then/else and absent-else control. Existing native SCF/pointer/mask conversion
and tensor adapters remain in use. Reviewed primary shared sources include:

- [Meta TritonToStructuredPass.cpp](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToStructured/TritonToStructuredPass.cpp)
  and its `if_diff_store_load_at_every_level.mlir` / `if_multiple_ptrs.mlir` fixtures.
- [Historical Microsoft TritonToStructuredPass.cpp](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToStructured/TritonToStructuredPass.cpp)
  and `gather_scatter_ptr_1d_with_if.mlir`.

Those pipelines retain source control and pointer state; they do not by
themselves prove functional fresh-output definedness. The added distribution
uses actual source paths to bridge that gap. The numeric operation admission
reuses the existing `value_regions.tensor_computation` contract. No upstream
code blocks were copied into the planner or Boolean proof.

Original TTIR remains in package evidence. The new
`source-tile-prefix-distribution` report retains source stores, original branch
IDs, reconstructed observations, selected-copy semantics and proof scope.
Local write proofs separately record actual selected SCF edges. Generated
compatibility metadata names the transformation. Provenance grants no numeric,
range, effect, alias, allocation, ownership or target permission. Public ABI,
result modes, receipt schemas and artifact rosters are unchanged.

## Real kernels and complete model

`t2m/examples/branch_tile_kernels.py` contains real branch, pre-branch and
branch-local observation, nested choice, redundant-mask, guarded division,
absent-else and two-root forms. The negative cases retain missing-arm,
pre-branch/branch-local premature read, wrong-address/root and overlapping
program refusals. The initial `early-branch` negative fixture had an undefined
local variable; its corrected version reaches the intended actual read refusal.
That original frontend failure is not a semantic refusal qualification.

`scripts/capture_branch_model.py` captures two dependent real Triton calls,
learned i64 bias, live `n-2` control, surrounding standard addition and all seven
results. One call retains branch-local observations and i64 data results; the
second retains division only in its positive-control arm. The original model
length domain remains 2..31, including the zero-control case. Its registered
PyTorch CPU implementation is an independent reference outside compiler staging;
no Triton CPU backend executes either kernel.

The selected source/proof/kernel/model checks pass in three runs: 37 proof and
prepass checks, 38 kernel checks and four complete-model checks. These 79 checks
do not establish arbitrary branch/loop, pointer, effect or source-version
closure. Broader shared/async participation and applicable accelerator execution
and performance remain full-goal requirements. The operation-name denominator
is unchanged.

The separate earlier interleaved/tile/model/conditional/shared/while/runtime-loop/
fresh-output/mixed-effect regression selection passes 191. Targeted Ruff and
`git diff --check` pass; the structured-state module and its existing unit file
retain their prior one import-order diagnostic each. Final logs are
`.scratch/branch-math2.log`, `.scratch/branch-kernels1.log`,
`.scratch/branch-model1.log` and `.scratch/branch-regression1.log`.

`out/branch-tiles/summary.json` reopens eight kernel libraries and one complete
model library. All 23 initial, 117 unchanged-library replay and 46 exact-word
target results match across eighteen actual serial images. One original model
domain violation actually aborts without outputs. Every parent archive and
result remains checked. Saved division operations remain inside actual SCF
branches, and source store/branch/read correspondence remains independently
accounted. The reopening log is `.scratch/branch-report1.log`. Selected source
hashes do not establish a full dependency closure or accelerator/performance
qualification.

Implementation `3c09f8e6015b0334841c61b2fb12a28566c1e3b7` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. All ten selected implementation/test source files
match the actual producer commit bytes. The reopening report records that
match separately from its original observed revisions.

Reproduction uses the existing tool environments documented in the README:

```bash
python -m pytest -q tests/test_tile_write_proof.py tests/test_structured_state.py \
  tests/test_source_boolean.py tests/test_partial_initialization.py \
  tests/test_loop_write_coverage.py tests/test_branch_tiles.py tests/test_branch_model.py \
  --basetemp=.scratch/pytest-branch-tiles
```

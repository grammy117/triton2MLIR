# Partial tile memory through original source loops

Same-address fresh tile writes and observations can now become an additional
numeric tensor carrier through original `scf.for` and `scf.if` regions. Loop
bounds, induction variables, original numeric results, arithmetic placement,
logical parallel tiles and reuse remain executable. No iteration is peeled,
unrolled or mapped to hardware.

The reproduced source family includes a partial positive-lane prefix, source
reads with explicit fills, complementary writes in each loop iteration, full
post-write observations and subsequent positive-lane updates:

```python
tl.store(out + lane, word ^ 1, valid & positive)
for iteration in range(steps):
    earlier = tl.load(out + lane, valid & positive, other=-17)
    tl.store(out + lane, update(word, earlier, iteration), valid & ~positive)
    observed = tl.load(out + lane, valid, other=0)
    snapshot ^= observed
    tl.store(out + lane, observed ^ 8, valid & positive)
```

A declared positive-trip domain may complete the prefix inside the loop.
When zero or negative upper bounds remain in the entry domain, actual source
fallback writes before or after the loop must complete the nonexecuting path.
An already complete prefix also supports zero trips. A fill does not initialize
storage, and a possible future iteration cannot justify an earlier read.

## Proof and executable transformation

The planner proves the entire source span before mutation. It reuses the bounded
actual-SSA Boolean analysis and original source-width/lexical-scope range checks.
Each selected read must imply preceding initialization. Branch coverage joins
actual true and false successors. A loop body must establish a dominating,
loop-invariant coverage predicate from the retained prefix candidates; this
invariant is sufficient for later iterations without assuming that changing
body values or predicates stay constant. The zero-trip exit keeps entry coverage.
Actual signed comparisons connect execution to a source fallback condition.
Only same-type no-op integer bitcasts are removed from comparison operands.

The transformation retains original control regions and numeric tuples and adds
one tile carrier. Actual stores update that carrier using ordered selections;
loads select their original mask and fill at their original source point. A
covered store publishes the completed span. No incoming fresh allocation is read.
Ordinary accesses to other declared roots retain their operations and ordering;
fresh functional output disjointness supplies no independence between inputs.
Source widths, active bounds, whole-output geometry, program ownership and
consumer lifetimes remain separate requirements.

The optional loop-domain witness is produced by the existing source range
checker, including source-width induction-update overflow checks. Failure gives
no rewrite and does not change existing admission. Unknown effects/control,
volatile or atomic accesses, changed same-root addresses and unavailable bounds
decline this optional planner. Its current budgets are 1024 visited operations,
128 target memory events and 32 nested control levels per plan, plus the existing
Boolean node/atom/work budgets. General while-carried partial tiles, effectful
spans and arbitrary varying coverage remain further work. Selected functional
region-container compositions now qualify under [the region contract](region_tile_memory.md); existing paths retain their independent scope.

## Native reuse and source correspondence

Reviewed Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` sources include
[`PtrAnalysis::rewriteForOp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/AnalysisStructured/PtrAnalysis.cpp)
and `tensor_indices_loop_iterargs_nested.mlir`. Historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` includes the corresponding analysis
and [`scalar_store_nested_loop.mlir`](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/test/Conversion/StructuredToMemref/scalar_store_nested_loop.mlir).
The local LLVM `a47bddccec30255619bb8c37fa59700e661d4e66` SCF definitions,
loop transformation source and front-peeling fixtures retain original numeric
interfaces. The existing native pointer conversion and standard SCF/Linalg
conversion remain in use. They do not by themselves establish functional fresh
output initialization. No upstream implementation blocks were copied here.

Dependent numeric controls additionally reuse the checked `skip-prepass=true`
selection when their actual results and carried arguments have only data uses.
Independent ordinary memory remains visible inside another control consuming
such data. Pointer, mask, unknown-call and effect dependencies still retain the
original prepass. This avoids the installed optimizer's separate i64 state-box
verifier limitation without outlining memory or replacing native pointer analysis.
Installed source and binary identities remain separate; no native build was made.

`source-loop-tile-state` records actual source stores, reads and loops, retained
recurrence, source-ordered initialization and undecided physical mapping. Final
stores and reconstructed observations retain source locations. Original TTIR
remains separately inventoried. Generated compatibility metadata explicitly names
the transformation. Public tensor/scalar ABI, result modes, receipt schemas,
artifact rosters, numeric and effect profiles are unchanged. Provenance grants
no rewrite, omitted arithmetic, initialization, alias or target permission.

## Real programs and qualification

`t2m/examples/loop_tile_kernels.py` contains actual positive-trip, nonpositive
fallback-before/fallback-after, complete-prefix and nested-branch forms. Runtime
lengths include empty and ragged tiles; nonpositive upper bounds remain valid
for the appropriate forms. Independent NumPy references retain exact i64 words,
all intermediate observations and original inputs. Missing zero completion,
premature reads, changed addresses, incomplete body coverage and overlapping
programs remain refusals. The 25 kernel checks include native execution, five
unchanged Merlin host libraries with six replays each, and ten actual serial
images. These are functional RISC-V/Spike results, without accelerator or
performance qualification.

`scripts/capture_loop_model.py` captures two dependent real Triton calls,
learned i64 bias, live `n-2` bounds, surrounding addition and all seven model
results. Its length domain is 2..31, including zero trips at length two. The
registered PyTorch CPU implementation supplies an independent reference;
no Triton CPU backend executes the source kernels. All four complete-model checks pass: byte-identical parent archives, all seven
initial results, three changed-input/parameter replays, two actual serial images
and an original-domain abort without outputs. The related regression selection passes 221 checks in
`.scratch/loop-tiles-regression1.log`.

The final focused selection passes 80 proof/source/prepass checks, 25 kernel
checks and four whole-model checks in three recorded runs (109 selected checks).
Logs are `.scratch/loop-tiles-proofs2.log`, `.scratch/loop-tiles-execution1.log`
and `.scratch/loop-model1.log`. The first proof run had one invalid unit fixture:
xDSL's custom tensor comparison parser produced a scalar result type. The fixed
fixture declares its actual tensor result explicitly and all 80 checks pass;
that first parser error is not a semantic-refusal qualification. Targeted Ruff
and `git diff --check` pass. Modified established source files retain their prior
import-order diagnostics, and shared_tensor retains its prior B023 diagnostic.

## Historical source-boundary probes

`.scratch/loop-tile-next-probe1/` retains two fresh probes against this change.
With the complete-prefix form's entire `steps` domain set to `[-2, 0]`, native
intake still refuses with `memory loop has no proved reachable iteration; needs
empty-region elimination`. The zero-after partial-prefix form also refuses in
that domain. This is a source-domain/control gap: valid nonexecuting loops must
preserve entry values and numeric initial results without evaluating the body.
The qualified mixed nonpositive/positive domains above do not close it.

A positive-trip form additionally computes `tl.sum(observed, 0)` inside the
loop. Original native source verification succeeds, but partial-tile conversion
still refuses at `normalized.ttir.op.0028 (tts.scatter)` with the fresh-output
initialization diagnostic. This region-container refusal is now resolved for selected functional
compositions in [the region contract](region_tile_memory.md). General scalar/tuple reduction expansion already exists; the
next extension must admit its actual pure region without hiding source-memory
or effectful combiners. The original failed probes remain recorded in `empty.log` and `reduction.log`;
new region qualification does not relabel those failures. The wholly nonexecuting
declared loop domain now has separate [empty-edge qualification](empty_loop_memory.md),
including complete original initial tuples and guarded native continuations.
Its original failed probes remain failed historical evidence.

`out/loop-tiles/summary.json` reopens five kernel libraries and one complete
model library: 17 initial, 81 unchanged-library replay and 34 exact-word target
results across twelve actual serial images, plus one original-domain abort.
All parent archives and source store/read/loop correspondence remain checked.
The reopening log is `.scratch/loop-tiles-report1.log`.

Implementation `df4248f929bfa4bc918eaa4d69bbc8736e3e992c` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`. All eleven selected
implementation/test sources match actual committed bytes. These hashes are
recorded separately from observed revisions and are not a full dependency
closure, accelerator or performance qualification. No companion source change
was required for this capability.

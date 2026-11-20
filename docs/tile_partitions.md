# Fresh tile partitions retain parallel tensor computation

[Branch-completed tiles](branch_tiles.md) additionally distribute eligible
partial prefixes across actual scalar successors. Prefix reads keep their
pre-branch facts, branch reads use actual selected edges, and conditional
arithmetic stays in its original arm. General loop and effect scope remains open.

[Interleaved fresh groups](interleaved_tiles.md) extend the original scope below
across ordinary accesses to distinct actual roots, including input-state
updates. Same-root observations are re-proved from current SSA; unknown effects,
changed same-root addresses and control regions remain boundaries. The original
single-root qualification below retains its recorded revision and scope.

Complementary data-dependent writes to a fresh output tile now re-raise before
native pointer conversion. The actual same-address source writes become an
ordered select chain and one covered store; ordinary same-address observations
remain their exact value at the original point. No incoming output storage is
read or supplied as a dummy public argument. The logical grid, tile size, input
reuse, arithmetic and active tail predicate remain executable.

The reproduced original refusal is retained under
`.scratch/tile-partition-probe1/{False,True}/source.ttir`. Both forms failed at
the first masked scatter's attempted read of uninitialized old tile values.
Their original failed probes are not execution qualifications. The following
real source pattern is now admitted:

```python
lane = tl.program_id(0) * B + tl.arange(0, B)
valid = lane < n
value = tl.load(x + lane, valid, other=0)
positive = value > 0
tl.store(out + lane, first, valid & positive)
earlier = tl.load(out + lane, valid & positive, other=-17)
tl.store(out + lane, second, valid & ~positive)
tl.store(seen + lane, earlier, valid)
```

The earlier observation is a select of the first value and the original fill,
not the eventual complete output. The final covered store selects the last
active source write. Three-way masks and overlapping masks use the same rule;
coverage does not imply disjointness or reorder overlapping writes.

## Source proof and upstream boundary

`source_boolean.BooleanCoverage(pointwise_tensors=True)` proves Boolean
identities at each unchanged tensor coordinate. Unknown tensor observations
remain separate actual SSA atoms; no agreement between different lanes or
different loads is assumed. Scalar Boolean splats and splat constants retain
their actual meaning. The default scalar analysis retains its original API
and behavior. The decision-diagram budgets still apply.

`tile_partitions.expand_tile_partitions` requires a fresh ABI root and the same
typed source pointer expression. It traverses no more than 32 pointer-expression
levels and collects at most 128 memory events and 128 candidate predicates per
group. The group stays in its actual block; unknown operations, changed same-root
accesses, region control and volatile observations form boundaries. Intervening
value operations remain at their original points. Store properties must agree.
An intermediate read requires its actual active mask to imply a preceding
write, and retains an explicit inactive fill. A masked read without a defined
fill cannot be replaced. Only actual source predicates already available before
the final store may represent the collective cover, or a proved unconditional
cover may omit the mask.

The original source-width/address checks run before this rewrite. Upstream
pointer and mask conversion runs afterward; normal output geometry, tail bounds,
source masks and cross-program/lane ownership still decide admission. There is
no initialization tracker or blanket permission to read uninitialized tensors.
Missing alternatives, unrelated masks, different indices, early observations,
duplicate lanes and overlapping logical programs remain refusal cases.

The reviewed upstream work is Meta/shared
`17f0c234ce474bee334af36e351c62eca7852836` and historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971`: `lib/Analysis/MaskAnalysis.cpp`,
`lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp` and
`test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir`.
Their scatter expansion retains the actual per-lane conditional and inactive
no-op. A direct tensor select against old storage additionally needs that old
storage initialized. Re-raising a proved complete group removes that unnecessary
old-storage read while retaining the same active values. Existing supported
native passes and licensed adapters remain in use; no upstream code blocks
were copied into this source rewrite.

## Package and complete model

Original TTIR remains intact in package evidence. The transformation report
records every participating source store and reconstructed observation;
locations retain emission correspondence. The new
`source-tile-write-partition` report describes the actual proof and grants no
numeric, bounds, alias, ownership or effect permission. Public ABI, result modes,
receipt schemas and artifact rosters are unchanged. Generated compatibility
metadata explicitly identifies the re-raising and its independent obligations.
Guarded source-word and effect profiles retain their existing paths.

`scripts/capture_partition_model.py --tiled` captures two dependent real Triton
calls with an eight-lane logical tile, learned i64 bias, live `n-2` control,
surrounding standard addition and all seven original results. The original
model domain remains 2..31. Its PyTorch CPU implementation is an independent
reference outside compiler staging; no Triton CPU backend is used.

## Qualification and next source gap

The final focused selection passes 53 source/math/native/host/serial/model
checks. The separate earlier source-partition, scatter, fresh-output, attention,
conditional and shared-tensor regression selection passes 114. Targeted Ruff
and `git diff --check` pass; the larger shared-tensor module retains its five
existing baseline diagnostics. The Boolean constant reader checks packed i1
bytes without unpacking a Python tuple for every lane; an 8192-lane mixed/splat
test checks that nonuniform constants do not become unconditional facts.

`out/tile-partitions/summary.json` reopens five tiled kernel libraries and two
complete model libraries, the original scalar model being regression evidence.
All 24 initial, 92 unchanged-library replay and 48 exact-word target results
match. Fourteen actual serial images execute; both model-domain violations
actually abort without outputs. Original parent archives and source store/read
correspondence remain checked. Selected source hashes do not constitute a
complete dependency closure, and serial execution establishes no accelerator
correctness or performance equivalence.

Implementation `f271d7ab1b3127aebb033ce241451f7fccb23d9e` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba` on
`ttir-experiment-grammy`. All nine selected implementation/test source files
match the actual producer commit bytes; the reopening report records that
separate match alongside its original observed revisions.

Final logs are `.scratch/tile-partitions-final2.log`,
`.scratch/tile-partitions-regression1.log` and
`.scratch/tile-partitions-report3.log`. The earlier 52-check run and its successful
reopening remain recorded. The first report attempt failed while asking the
xDSL parser to identify an entry in custom-syntax raw TTIR; the report now uses
the saved generic annotated source. That reporting failure was not a compiler
or execution qualification failure.

Reproduction uses the existing tools/environments documented in the README:

```bash
python -m pytest -q tests/test_source_boolean.py tests/test_partial_initialization.py \
  tests/test_loop_write_coverage.py tests/test_tile_partitions.py \
  tests/test_partition_model.py --basetemp=.scratch/pytest-tile-partitions
```

The next reproduced source gap at that revision was
`.scratch/interleaved-tiles-probe1/source.ttir`: positive writes to distinct
fresh outputs `a` and `b` precede their complementary negative writes.
Both results have complete valid-lane coverage, but the original collector
treated the other output's store as a boundary and the tensor path refused.
The interleaved implementation above now closes this form using actual
disjoint fresh-root facts while retaining cross-root observations and unknown/
overlapping effects. The original failed probe supplies no qualification.

General store groups across regions or other memory events, different equivalent
pointer expressions, inter-lane dependencies, richer numeric predicate facts,
shared/async effects and applicable accelerator execution/performance remain
completion requirements. This extends a concrete source/memory composition;
it does not change the historical operation-name denominator or establish
arbitrary-kernel closure.

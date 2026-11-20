# Functional numeric regions inside tile memory spans

Partial tile writes can now retain pure source reductions, scans and maps
between their writes and inside their original source loops. The actual region
bodies, captures, tuple order, numeric formats, loop bounds, logical parallel
tiles and reuse remain executable. The change adds admission to existing
conversion; it introduces no replacement kernel or reduction algorithm.

This closes the source-region refusal reproduced in
`.scratch/loop-tile-next-probe1/positive/translation/`. Its unchanged source now
converts under `translation-pure1/`. The original failed capture remains failed;
new kernel/model qualification is recorded separately below.

## Actual dataflow and independent gates

A representative source span is:

```python
tl.store(out + lane, first, valid & positive)
for iteration in range(steps):
    earlier = tl.load(out + lane, valid & positive, other=-17)
    tl.store(out + lane, update(word, earlier, iteration), valid & ~positive)
    observed = tl.load(out + lane, valid, other=0)
    snapshot ^= observed ^ tl.sum(observed, 0)
    tl.store(out + lane, observed ^ 8, valid & positive)
```

The existing source write/read proof still establishes every observation from
preceding actual writes. The running tile and original snapshot remain loop
carriers. The reduction stays at its source position and uses the existing
first-element-seeded standard reduction expansion. It is neither moved outside
the loop nor replaced with a whole-kernel name. Pure containers between ordinary
same-block writes also retain their original position.

`pure_value_region` checks every descendant and every used capture. Values and
region arguments must be numeric. Source reduction/scan/map containers have
explicit field admission; structured numeric controls and existing functional
tensor operations retain their actual regions. Arithmetic uses known no-effect
traits, with the existing exact cast/remainder exceptions for xDSL's missing
traits. Addresses, source memory operations, diagnostics, assertions, calls,
unknown instructions and CFG successors do not acquire purity. A parent Pure
trait or a provenance attribute cannot hide an unused nested effect.

The bounded classifier visits at most 1024 operations and 32 nested levels per
region. Exhaustion grants no admission. It neither changes compiler pass order
nor publishes a prepared source expansion. Existing postorder region conversion
still checks source geometry, complete tuples, packing independence, precision,
first-element initialization and exact generated-guard authority. Original
reduction/scan helper-effect auditing still happens before native inlining.

The tile analysis keeps its independent source widths, masks/fills, initialization,
whole-output geometry, disjoint fresh roots, active bounds and ownership/lifetime
requirements. A numeric tensor capture is an immutable value; it grants no new
permission for a pointer read, public alias, source effect or physical mapping.

## Reuse and provenance

Reviewed Meta/shared revision `17f0c234ce474bee334af36e351c62eca7852836` includes
[`TritonArithToLinalg.cpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp),
`ConversionPatterns.hpp`'s `ReduceConverter`, and `reducesum_scalar.mlir`.
Historical Microsoft `050b93ae9c737111f1ca425d793419b32427a971` includes the
corresponding pattern source and
[`reducesum_scalar.mlir`](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/test/Conversion/TritonArithToLinalg/reducesum_scalar.mlir).
Those patterns already express reductions with standard tensor/Linalg
computation, but their selected identities and widening rules are not generic
source numeric permission. The existing bridge's complete first-element and
coupled-tuple expansions remain authoritative; no upstream source blocks were
copied for this admission check. Existing native scalar/tensor conversion and
Merlin lowering remain in use. No native build was performed.

`retained_value_region_source_ids` records actual retained region locations in
write partitions, prefix distributions and loop-state transformations. Locations
can include both a definition and its original/normalized callsite chain; those
namespaces remain separate in the retained inventories. Existing expansion
reports retain complete combiners and numeric intent. Compatibility metadata
explicitly names recursive numeric-region admission. Public tensor/scalar ABI,
result modes, receipt schemas, artifact rosters and effect/numeric profiles are
unchanged. Provenance records intent and correspondence, without supplying an
omitted effect or granting any rewrite permission.

## Real programs and measured scope

`t2m/examples/region_tile_kernels.py` reuses existing real Triton helpers for
coupled affine tuple reduction, segmented scans and mapped scalar while loops.
Seven forms cover ordinary sums, nonpositive-trip fallback, coupled tuple
composition, forward/reverse scans, scalar map loops and a reduction between
same-block partial writes. The scan retains parallel doubling stages; the map
retains actual scalar control; tuple reduction retains source component order
and every source word. Empty/tail lengths, changed bias and signed i64 extremes
use independent tile-local NumPy references. The 28 kernel checks pass native,
seven reusable Merlin host libraries with six invocations each, and fourteen
actual serial RISC-V/Spike images.

`scripts/capture_region_model.py` captures two dependent real Triton calls: a
sum-based partial-tile recurrence with live `n-2` steps and a coupled tuple
recurrence with live `n-1` steps. Learned i64 bias, surrounding addition and all
seven model results remain. Its original domain is 2..31, including first-call
zero trips at length two. A separate PyTorch CPU implementation computes its
references outside compiler staging; no Triton CPU backend executes either
kernel. All four model checks pass: initial complete results and byte-identical
parent archives, three changed-input/parameter host replays, an actual original-
domain abort without outputs, and two actual serial images with every result.

The purity/source-effect/region-contract selection passes 38 checks. The kernel
and model selections pass 28 and four in separate runs: 70 selected checks, not
one combined invocation. The first structure selection used incorrect expected
report-kind names and failed seven assertions after successful conversion.
Correcting those expectations required no compiler change; that failed run is
not execution qualification. Logs are `.scratch/region-tiles-proof1.log`,
`.scratch/region-tiles-kernels1.log` and `.scratch/region-model1.log`.
Targeted Ruff and `git diff --check` pass. The related memory/control/reduction/scan/map/tensor-region selection passes 212
checks in `.scratch/region-tiles-regression1.log`. Reopening qualification is recorded below.

The wholly nonexecuting declared trip domain now has separate
[empty-edge qualification](empty_loop_memory.md); the original failed probes
remain retained under [the earlier loop contract](loop_tiles.md#historical-source-boundary-probes).
General while-carried partial tiles, source memory inside region combiners,
shared/async participation, numeric/source-version closure and applicable target
correctness/performance remain full-goal work. This finite i64 qualification
changes no operation-name denominator and establishes no accelerator support.

Reproduction uses the existing native/Merlin environments from the README:

```bash
python -m pytest -q tests/test_pure_value_regions.py \
  tests/test_value_region_contract.py tests/test_source_combiner_effects.py \
  tests/test_region_tiles.py tests/test_region_model.py \
  --basetemp=.scratch/pytest-region-tile-memory
```

`out/region-tiles/summary.json` reopens seven kernel libraries and one complete
model library: 21 initial, 105 unchanged-library replay and 42 exact-word target
results across sixteen actual serial images, plus one original-domain abort.
Every parent archive and actual original/normalized source-region link remains
checked. Definition and callsite IDs retain their actual opcodes separately.
The reopening log is `.scratch/region-tiles-report1.log`.

Implementation `9218ad836c1a2be7c1a1be120c82cc3aa7c7092c` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`. All twelve selected
implementation/test/helper sources match committed bytes. These hashes remain
separate from observed revisions and are not a full dependency closure,
accelerator or performance qualification. No companion source change was
required for this capability.

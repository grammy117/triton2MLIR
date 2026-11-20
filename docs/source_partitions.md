# Source branch and mask partitions initialize tensor results

Scalar source writes may now collectively initialize an address through actual
branches or store masks. The tensor path retains those branches, predicates,
write order and intermediate observations. It proves local read definedness and
whole-result coverage separately, without a runtime initialization bitmap or a
replacement dense algorithm.

`t2m/examples/partition_kernels.py` contains real Triton sources. One pattern is:

```python
for i in range(n):
    value = tl.load(x + i) ^ reused_bias
    positive = value > 0
    tl.store(out + i, value ^ 1, positive)
    earlier = tl.load(out + i, positive, other=-17)
    tl.store(out + i, value ^ 2, ~positive)
    tl.store(seen + i, earlier)
```

The first read observes the first write only on its original active edge, with
the original fill otherwise. The complementary store completes the output;
replacing the earlier observation with the final output would change results.
Other sources use nested branches, three-way masks, Boolean selects and
overlapping writes. Overlap retains the original last-write behavior; coverage
does not establish disjointness. Row/column kernels combine both forms inside
an inner loop, retain outer-loop bias reuse, then observe the completed region
before the remaining output is written.

## Actual SSA and bounded Boolean coverage

`source_boolean.BooleanCoverage` builds reduced ordered decision diagrams over
actual scalar i1 SSA. It follows constants, AND/OR/XOR, Boolean select and integer
comparison complements, including operand reversal and signed/unsigned
distinctions. Integer literal keys retain actual type/word bits. Unknown
producers remain distinct SSA observations: identical names, addresses, calls or
provenance cannot make a later observation equal an earlier one.

The analysis has limits of 128 atoms, 4096 decision nodes, 16384 cached Boolean
applications and 128 source-expression levels. Missing/budget-exhausted facts
give no optional proof; unconditional source writes retain their existing path.
It enumerates neither tensor elements nor source input values. No assumption
about an unknown predicate's sampled truth value enters compilation.

The loop-store reader groups actual scalar addresses and the same written loop
domains, combining enclosing branch conditions and explicit store masks. Only
a Boolean cover of every alternative supplies an unconditional write image.
The existing exact bound-product and mixed-radix proofs still establish complete
output geometry. Different addresses, missing arms, unrelated predicates,
zero-trip body writes and parallel conflicts retain their independent refusals.

Point images may stay guarded at their real program point. A partial read needs
both an actual in-image address witness and a Boolean implication from its own
source branch/mask conditions to some preceding write. Guarded facts intersect
or combine under the actual branch alternatives, and varying loop scopes expire.
Source widths, bounds, initialization of later results and ownership remain
separate obligations. Read failures retain original source operation IDs.

Floating comparisons remain opaque Boolean observations. Their same-SSA logical
negation still covers every predicate outcome, including NaNs. The maintained
f32 example returns integer labels for positive and other inputs; IEEE word
classification independently checks both infinities, signed zeros, subnormals
and quiet/signaling NaNs. Independent ordered `x > 0` and `x <= 0` comparisons
cannot claim NaN coverage and refuse. No floating comparison is replaced or
given an invented finite-input/numeric-policy premise.

## Upstream reuse and package boundary

The reviewed Meta/shared revision is
`17f0c234ce474bee334af36e351c62eca7852836`; the historical Microsoft revision is
`050b93ae9c737111f1ca425d793419b32427a971`. `lib/Analysis/MaskAnalysis.cpp`
retains scalar/rectangle mask handling and actual conjunctions. The existing
`UnstructuredToMemrefPass.cpp` gather/scatter converters and
`test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir` retain the
per-lane `scf.if` and inactive no-op/fill behavior. Those supported native passes
and the existing licensed tensor adapters remain in use. The added Boolean
cover bridges their actual masks to functional fresh-output definedness;
memref stores alone do not supply that proof. No upstream code blocks were
copied into the decision-diagram implementation.

`scripts/capture_partition_model.py` captures two dependent real Triton calls,
a learned integer bias, ordinary surrounding addition and a live `n-2` control
derived from the original exported length. Both calls, intermediate observations,
all seven results and original model lengths 2..31 remain. Standalone kernels
separately admit empty inputs and full signed-i32 controls.

Public tensor/scalar types, result modes, receipt schemas, artifact rosters,
mutation and effect profiles are unchanged. Provenance records
`condition_coverage` on source-loop and read evidence. It describes the actual
Boolean proof and supplies no numeric, memory-effect, bounds, ownership or
target permission. Existing guarded/buffer profiles keep their own contracts.

## Qualification

The focused checks pass in three recorded runs: 66 source/math/model checks,
four additional f32 checks and one ordered-float NaN-hole refusal. The separate
259-check earlier memory/control/model regression selection also passes.
Targeted Ruff checks pass; the two larger existing modules retain their prior
one and five diagnostics respectively, without introducing another diagnostic.

`out/source-partitions/summary.json` reopens ten kernel libraries and one
complete model library: 27 initial results, 121 unchanged-library replay
results and 54 exact-word target results across 22 actual serial images.
The original exported model domain also actually aborts without outputs.
Complete parent archives, every result, the source-mask/initialization evidence
and selected source-file hashes are checked. The selected hashes do not claim
a complete dependency closure. Native execution, host execution and serial
target execution remain separate evidence; none establishes accelerator
correctness or preserved performance.

Implementation `658c2265d2531228922fe2f0aa9e0cdf399961ab` uses unchanged
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. All twelve selected source files match the actual
implementation commit bytes. The final reopening log is
`.scratch/source-partitions-report2.log`; a post-review math/state check also
passes 17 in `.scratch/partition-review1.log`. These counts overlap the
focused checks and do not expand the historical operation-name denominator.

General vector-lane partitions, richer arithmetic/floating predicate relations,
while/ownership invariants, irregular parallel dependencies, shared/async effects
and applicable accelerator execution/performance remain full-goal work. The
historical operation-name inventory is unchanged; these source/control/memory
compositions do not establish arbitrary-kernel closure.

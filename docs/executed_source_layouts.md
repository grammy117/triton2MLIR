# Source grouping through execution-owned model packages

Execution-owned binding v4 now reuses the existing
`t2m.captured-source-layout.v1` profile. Every semantic source option binds to
its exact original typed model-call scalar, including target, warp count,
threads per warp and CTA count. The argument mapping and captured binding
records must agree. The profile appears exactly when a selected kernel has a
source-layout specialization; unspecialized kernels cannot claim one.

Source geometry can determine which packed values interact. It cannot be
discarded as an optimization hint. The existing coordinate bijection and full
packed scalar body still expand into standard parallel Linalg, tensor and SCF
computation. Consumer placement remains independent. This connects established
[source grouping](source_grouping.md) and [packed memory](grouped_memory.md)
analyses to the owned-result interface; it introduces no new GPU operation,
layout inference or source-equivalence permission.

Merlin reuses its independent original-call checker and then reconstructs the
entire model/kernel/adapter SSA and locations. Missing profiles, false source
options, different mappings and incomplete records refuse after outer hashes
are repaired. Original masks, fills, snapshots, source domains and allocation/
lifetime gates remain executable. Parent archives and public model types remain
unchanged; receipt v2 and signature v6 keep their existing schemas. The added
v4 fields are explicit compatibility differences and older v4 readers refuse.

## Real combined program

`scripts/capture_grouped_memory_model.py --execution-sizes` adds actual count
and compaction kernels after the existing three grouped-state calls:

```python
first = implementation(x, self.weight, 300, 320, 3, 2, *profile)
second = implementation(first[0], self.weight, 319, 320, 1, -3, *profile)
empty = implementation(x[:0], self.weight[:0], 0, 0, 2, 5, *profile)
flat = second[0].reshape(-1)
count = count_positive(flat, 320)
n = count.item()
torch._check_is_size(n)
torch._check(n <= 320)
return *first, *second, *empty, compact(flat, n, 320), count
```

All seventeen results remain public. The source reads retain selected roots,
for/while recurrence and observations before and after state updates. The
512-lane compaction retains nine parallel scan stages and guarded scatter.
Automatic memory selection leaves count as tensor computation and packing as
guarded source-word memory. Two source profiles exercise i64/two-warp/while
and f32/four-warp/for computation; wide i64 data exceeds exact f64 transport.

## Explicit finite resource selection

The i64 model exposed an actual default-resource refusal: 10,321 small
allocations exceed the original 4,096-work limit even though live memory peaks
at 67,054 charged bytes. The default guard remains unchanged. An operator can select
a complete bounded policy through the Python API's `execution_allocation_policy`
or `bind-model --execution-allocation-policy policy.json`, with explicit
`--result-mode execution-owned`:

```json
{
  "schema": "merlin.owned-result-limits.v2",
  "max_bytes": 67108864,
  "max_allocations": 16384,
  "max_output_bytes": 67108864,
  "index_bits": 64
}
```

The work limit bounds cumulative allocations, separately from live charged bytes
and complete output bytes. Every limit is a positive signed-i64 integer and the
current transport uses 64-bit indices. The original fixed v1 profile retains
its exact limits. Producer and independent consumer both validate v2. The input
receipt, signature, saved program, generated driver and execution evidence must
agree; changing the policy cannot reuse an existing qualification. Older staging
that discards the requested policy refuses at the producer interface check.
This is resource selection, without source-specific production decisions,
predicted result shapes or disabled guards.

## Evidence and remaining scope

Both models retain all 34 initial host results. Four empty/full invocations reuse
unchanged libraries and retain 68 results. Four actual Spike images retain 68
complete results. Twelve rehashed source-layout mutations refuse in both producer
and independent consumer; four resource mutations cannot reuse existing execution
claims. The host/correspondence selection passes twenty checks and the separate
target-only recheck passes four. Merlin's resource/source/driver selection passes
87 checks; three existing default-policy model packages and the corrected
three-check API/CLI selection also pass. These selections overlap.

The retained report is `out/executed-source-layouts/summary.json`; historical
opcode counts are unchanged. Original logs preserve the default resource refusal
and the resolved CLI/target test-harness failures. The target recheck executes
actual images after correcting its script import path. No earlier failed log is
rewritten into successful evidence.

Runtime non-dense source coordinates, ambiguous result-axis reconstruction,
physical/mutating aliases, broader ownership/effect/numeric/source-version
compositions and applicable accelerators remain open. The initial source-layout
contract does not establish correspondence to every final optimized GPU layout
or preserved performance. Resource policy v2 is currently published for explicitly
selected execution-owned derived model packages; other package kinds refuse it.

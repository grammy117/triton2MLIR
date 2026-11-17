# Runtime input words controlling result shapes

The real PyTorch model in `scripts/capture_data_size_model.py` now preserves an
unbacked scalar read controlling two Triton result allocations:

```python
n = count.item()
torch._check_is_size(n)
torch._check(n <= 31)
original = x + self.weight
first = prefix(original, n, x.shape[0])
second = prefix(first, n, n)
return second, first, original, x, count, self.weight, first[::2]
```

The exported input capacity is `[2, 31]`; the source separately guards the
runtime count in `[0, 31]`. A count larger than capacity is valid: the original
kernel explicitly fills the excess result with zero. Count zero yields empty
results. The count is input data, independent of shape and the captured value 7.

## Source computation and TTIR masks

Model2MLIR [`863d797`](https://github.com/grammy117/model2MLIR/commit/863d7974bf264cd22296df22bc323e13dc54d08c)
materializes integer `aten.item`/`aten._local_scalar_dense`
as actual singleton `tensor.extract` and exact signed/unsigned extension. The
original integer constraints, comparisons and assertions stay executable.
No unbacked symbolic hint supplies a value or conversion permission.

The Triton kernel retains `(offset < n) & (offset < capacity)` for its read and
`offset < n` for its store. Meta/shared
[`MaskState::minStates`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Analysis/MaskAnalysis.cpp)
already emits the read intersection as a minimum of clipped prefix lengths.
The tensor adapter now retains that exact SSA minimum, with all independently
proved upper prefixes available to establish read bounds. It grants no complete
write/initialization coverage. Static clamps retain their original exact path;
unknown masks and overflow still refuse. The source grid, tiles, masks, fills,
ordinary model computation and complete result tuple remain executable MLIR.

## Explicit package extension

The selected interfaces are:

| Boundary | New explicit profile |
|---|---|
| Model shape facts | `t2m.symbolic-model-shapes.v3` |
| Live size scalar | `t2m.checked-word-runtime-arguments.v1` |
| Invocation shape contract | `t2m.runtime-shapes.v4` |
| Typed shape program | `logical-size-program.v2` |
| Independent Merlin signature | `merlin.compile-signature.v5` |

The new program leaf represents an actual public singleton integer tensor read
or native integer argument. `word_bindings` names its exact argument position,
indices and dtype. Storage words retain their original widths and raw bits;
casts/arithmetic keep their executable signed/unsigned meaning. V1 programs and
older runtime/model/signature profiles reject this extension. Public model
arguments/results, receipt schemas, artifact rosters and saved parent archives
are unchanged. The positional runtime input projection retains its prior schema.

Producer and independently owned consumer code reconstruct the actual read and
dominating source guards. Only injective integer extensions transfer a word
enclosure; truncated-word comparisons, late/sibling assertions and metadata
ranges supply no such proof. Existing bounded bit-vector equivalence checks
reconcile actual kernel/model dimensions. The per-call `checked-size-scalar`
record retains `integer_range` and the unchanged exact integer SSA operand.
Guarded scratch remains derived from source upper bounds.

Before allocating results, Merlin reopens the complete standard body and checks
the expression program against actual source dimensions. It then evaluates the
declared input words at invocation. Rehashed false word bindings, outputs and
canonical but incorrect programs refuse. Negative logical result sizes and
wrong storage carriers refuse before allocation; source-valid positive sizes
continue to the unchanged compiled body and its guards.

Compiler staging reads source types/operations and compiler-owned members,
without invocation arrays, references or their digests. Bundle/reference
verification separately reads only explicitly selected singleton source words
from input/extra/safetensors storage; other argument values remain unopened.
These runtime/evaluator values never become compiler facts or replace source
proof. Provenance grants no alias, mutation, numeric, effect or target permission.
The independent Merlin implementation is
[`656b1d1cf`](https://github.com/ucb-bar/merlin/commit/656b1d1cf9231502d4857760701dd19a403c1aeb).

## Qualification and remaining work

`tests/test_data_size_model.py` checks complete native/Merlin-host/Spike packages
in tensor, buffer and guarded-buffer profiles. All seven results are retained,
with 21 complete initial target results across the selection.
Unchanged host libraries additionally vary capacity/count independently,
including empty results, exact/ragged tiles, counts below/above capacity and
changed parameter words. Real source-domain failures abort without outputs;
negative allocation counts and altered word contracts refuse before invocation.
The initial target invocation uses capacity 13/count 7; host runtime replays
qualify their separately recorded values. No GPU correspondence or performance
measurement is claimed.

The final producer/model selection passes 189 checks in
`.scratch/data-size-final1.log`, including the earlier derived/input-axis models,
program contracts and standalone dynamic shapes. Forty-three final mask/word/
input-storage checks also pass after retaining all upper prefixes across a
scalar mask gate. Independent Merlin runtime/staging/source checks pass 196;
layout/access checks pass 111 with two unavailable checks skipped. Model2MLIR's
integer-item/symbolic/domain selection passes 32. Counts overlap.
`out/input-word-shapes/summary.json` reopens all three complete package/target
receipts, 21 unchanged-library host replays with 147 results, and nine actual
compiled source-domain aborts. Three further negative-count calls refuse before
allocation. The earlier `data-size-model1.log` retains its three target-reference
projection failures; final qualification follows the corrected source-word
projection rather than relabeling that run all green.

Input words currently use exact static singleton or native i8/i16/i32/i64
storage. General vector-data-dependent size computation, arbitrary scalar
arithmetic at PyTorch import, runtime slice bounds, resizing state, effectful
size observations and broader model/alias/target compositions remain open.
The existing finite shape-program/proof budgets remain unchanged. This closes
the reproduced input-count allocation boundary; full arbitrary-TTIR coverage
remains active and incomplete. The selection stays separate from the historical
352-package inventory.

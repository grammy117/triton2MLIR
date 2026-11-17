# Real Triton count, parallel scan and scatter through Merlin

`scripts/capture_compact_model.py` captures a PyTorch model with three actual
Triton kernels and a learned integer parameter:

```python
original = x + self.weight
count = count_positive(original, x.shape[0])
n = count.item()
torch._check_is_size(n)
torch._check(n <= 31)
first = compact(original, n, x.shape[0])
second = prefix(first, n, n)
return second, first, original, x, count, self.weight, first[::2]
```

The packing kernel retains the source algorithm:

```python
selected = (lane < capacity) & (value > 0)
rank = tl.cumsum(selected.to(tl.int32), 0) - 1
tl.store(out + rank, value, selected & (rank < n))
```

This filters values in source order. It differs from the earlier
[count-then-prefix example](executed_model_binding.md), which copies the first
`n` original elements regardless of sign. The demo's actual exported capacity
domain is `[2, 31]`, with a 32-lane pack tile and 8-lane dependent-copy tiles.
Those constants belong to this source example, not name-based compiler rules.

## Preserved algorithm and hardware boundary

The existing general scan expansion emits five parallel `linalg.generic`
stages for the 32-lane inclusive scan. The masked indexed store retains actual
scan ranks, selected values and lane predicates. The complete model retains all
three kernels, actual count computation, parameter words, source checks and
seven original results. No dense filter replacement or opaque custom call is
introduced. GPU physical mapping remains undecided.

The implementation reuses the existing shared/unstructured scatter conversion
and local general scan expansion. The source review uses Meta/shared revision
`17f0c234ce474bee334af36e351c62eca7852836`, including `ScatterConverter` in
`lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp`, its
`gather_scatter_all_mask.mlir` fixture and the arithmetic `cumsum.mlir` fixture.
The upstream addition-only `ttx` form is not the portable final payload; the
existing local stages retain standard Linalg and composite/source provenance.

## Explicit representation and package selection

`--memory-profile auto` chooses tensor form for count and dependent copy.
Packing currently refuses tensor and guarded logical admission because the
structured address analysis cannot prove the original scan-dependent integer
range. Its third candidate, guarded source-word memory, retains original i32
arithmetic instead of promoting an unproved expression to mathematical indices.
All prior refusal diagnostics remain in the versioned selection record.

The pack therefore contains standard memory operations and private executable
bounds/initialization/conflict state. This is an explicit granularity difference
from pure model2MLIR tensor payloads. Parallel scan structure remains present;
no numeric/effect policy is inferred. A future stronger source range proof may
admit a higher-level representation without changing the algorithm.

Binding explicitly selects `--result-mode execution-owned --merlin-host`.
The existing [receipt v2 and Merlin signature v6](executed_model_binding.md)
retain original parent archives and independently reconstruct all calls and
surrounding SSA before compiler staging. Actual execution produces `n` and
allocation sizes. Reference arrays do not select representations or enter
compiler staging. Qualification compares every original result only after
execution. The public `bind-model` CLI exercises the same mixed selection.

## Qualification and remaining scope

`tests/test_compact_model.py` verifies original sample output
`[1, 6, 2, 5, 3, 4]`, every returned value and byte-identical parent archives.
Six additional invocations reuse the same compiled package library and cover
empty selections, holes, alternating/full masks, changed parameters and signed
i32 overflow. Each invocation checks all seven results and preserved inputs.
The standalone kernel also permits valid truncated prefixes; requesting more
results than selected source values must abort through its actual compiled
complete-write-coverage guard without publishing output.

This selection qualifies the real PyTorch/Triton model on Merlin host. It does
not qualify v6 generated-C/Spike harnesses, accelerators, arbitrary filtering
sizes, cross-program scan carries, physical views or unrestricted alias/effect
compositions. The historical 352-package/50-name inventory stays unchanged.
No performance equivalence is claimed.

# While-carried integer words as portable tensor indices

Actual fixed-width `scf.while` arguments and results can now enter the tensor
memory path without pretending they are mathematical induction variables.
Original counter widths, modular updates, conditions and final false-edge values
remain executable. Dependent address arithmetic still needs its separate source
overflow, cast, divisor and active allocation checks.

## The source boundary

`t2m/examples/while_index_kernels.py` contains actual Triton sources. One folds
a runtime-length input through mutable scalar state:

```python
i = 0
while i < n:
    value = tl.load(x + i)
    tl.store(state, tl.load(state) ^ (value + i.to(tl.int64)))
    i += 1
```

Another intentionally allows signed i32/i64 state to wrap:

```python
i, count = start, 0
while count < steps:
    offset = ((i % 15) + 15) % 15
    value = tl.load(x + offset)
    tl.store(state, tl.load(state) ^ value)
    i += delta
    count += 1
```

The original final counter also indexes a post-loop observation. Modulus 15
distinguishes fixed-width wraparound from a widened mathematical recurrence;
a power-of-two modulus dividing the word modulus could hide that difference.
Start and delta keep their full signed domains, including both signed extremes.
The direct-index form retains executable active bounds; a negative body index
is harmless on zero body trips, while an active negative/oversized index aborts.
An i64 index `2**32 + 3` must not become an in-bounds index through truncation.

## Range proof and upstream reuse

`source_while_ranges.while_word_interval` first checks actual before/after,
condition/backedge and result interfaces. It supplies the full signed interval
of the actual integer storage width. Index types do not acquire an invented
target width. This enclosure proves no trip count, monotonicity, no-wrap
recurrence, memory bounds, initialization or ownership.

`check_source_addresses` applies the enclosure only at those typed SSA ports.
Other source arithmetic retains its original-definition validation. Thus
`i + 3`, division by an unproved nonzero `i`, and an unproved i64-to-i32 narrowing
still refuse mathematical-index admission. A valid original modular loop update
is not moved into that mathematical domain. Existing isolated native event
conversion retains its actual numeric interfaces and turns source memory into
standard tensor operations with active guards. No loop is replaced by a count
or a whole-operation shortcut, and no hardware grid placement is introduced.

The reviewed LLVM boundary is revision
`a47bddccec30255619bb8c37fa59700e661d4e66`:

- `mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp` widens unresolved carried
  ranges to the maximum storage-bit range.
- `mlir/test/Dialect/Arith/int-range-analysis-convergence.mlir` retains a full
  i32 carried range while keeping bounded transfer results.
- `mlir/test/Dialect/Arith/int-range-loop-iter-args.mlir` separates genuinely
  bounded recurrences from that full-domain case.
- `mlir/lib/Dialect/SCF/IR/SCF.cpp` defines actual while region/result successors.

This is a conservative typed reader, not a claim to run LLVM's native solver.
Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` source in
`lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp` supplies the
reviewed pointer-offset width/event boundary. Their pointer-state traversal
does not by itself supply our while tensor ABI or ownership proof. Existing
supported passes and the isolated control/event adapter remain in use.

## Consumer and complete PyTorch model

Two generic Merlin consumer changes close actual failures without disabling
the lifetime checker. First, actual `tensor.dim` queries close their tensor/axis
SSA as immutable value metadata. Second, a completed serial loop result may
seed a new private value copy only after matching its actual carried interface
to a fresh seed and closing every source operation. Unused calls, mutable reads
and writes cannot bypass the proof. Earlier loops, guards and snapshots remain;
the preceding result storage gains no ownership premise. Unknown/permuted or
resizing ownership still needs its own contract.

`scripts/capture_indexed_while_model.py` captures two real indexed Triton calls:

```python
state = self.counter.clone()
before = state.clone()
first = update(x, state, x.numel())
middle = state.clone()
shifted = x + self.bias
second = update(shifted, state, shifted.numel())
return before, *first, middle, *second, state, x, self.counter, self.bias, shifted
```

All eleven results, saved parameters, both dependent calls, original input
domain and earlier snapshots remain. The actual model export declares lengths
2..31; the standalone prefix kernel separately admits 0..31. Original PyTorch
provides the model reference, and changed-input replays include i64 arithmetic
wraparound. The model remains pure tensor MLIR with actual source while loops.

## Provenance, package and remaining work

`source-while-word-ranges` records original source IDs, actual typed interfaces
and the number of enclosed values. It deliberately does not call those values
"narrowed". Generated compatibility describes full word state and independent
address checks. Annotations supply no fact, source rewrite or effect permission.
Public functional tensor/scalar types, artifact rosters, receipts, mutation
contracts and Merlin signatures remain unchanged. Guarded source-word and
unguarded buffer continuations keep their independent admission requirements.

The focused checks retain exact i64 results, modular i32/i64 state, dynamic
lengths, old-state observations and actual serial target execution. Earlier
source-range, dynamic-dimension and complete-model lifetime failures remain
in their original logs. Those diagnostic probes stop before bufferization and
are not successful compiler or target qualifications.

General relational while invariants, wider source arithmetic/cast/division
compositions, partial/nested initialization, resizing/permuted owners, irregular
parallel dependencies, shared/async effects and applicable accelerator
correctness/performance remain open. The historical operation-name inventory
is unchanged; this establishes neither arbitrary-kernel closure nor performance.

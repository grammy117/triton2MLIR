# Source while memory as tensor data flow

Actual pre-layout Triton `while` regions can now carry complete tensor memory
versions through both the condition and body. Original conditions, numeric
state, tail masks, reused tiles and earlier snapshots stay executable. This
extends the standard tensor path; it does not replace the existing guarded
pointer/source-word paths or invent a finite trip count.

## Source and abstract computation

`t2m/examples/while_tensor_kernels.py` contains actual stock-Triton sources:

```python
tile = tl.load(x + lane, active, other=0)
while i < steps:
    value = tl.load(state + lane, active, other=0)
    tl.store(state + lane, value + tile + i, active)
    i += 1
```

The load outside the loop remains reused. Standard `scf.while` carries complete
tensor storage; extracts/inserts or slices retain the actual per-program tile.
The logical launch remains independent of physical threads, warps or cores.
Preinitialized fresh state and conditional pass-through updates have their own
source initialization and ownership checks. Empty grids, negative/zero body
counts and ragged tails retain their original meaning.

The second source makes condition effects visible:

```python
count = tl.load(checks)
tl.store(condition_seen, count)
tl.store(checks, count + 1)
return tl.load(state) < limit
```

That probe is the actual while condition, not a reference-only approximation.
It executes once even when the body never runs, and once after the final body.
Condition stores must therefore reach the loop result on the final false edge.
Numeric before/after interfaces may have different arities and types under SCF;
memory versions are appended to each actual interface rather than equating them.

Initialization after the loop uses facts established by the before region,
which always executes. Body-only writes cannot establish a fresh nonempty
result when a zero-trip body is reachable. Unknown loop attributes refuse the
tensor memory rewrite rather than disappearing when state slots are appended.
Descriptive provenance remains; it supplies no arithmetic or ownership proof.

## Merlin consumer and upstream boundary

Merlin's selected `reuse_loop_tensor_destinations` feature now normalizes a
computed condition tensor with typed initial-condition peeling. It executes the
first complete before body outside, then carries the actual Boolean and values.
Its new after body contains the original after body followed by the next before
evaluation. The new before region forwards its full state unchanged, allowing
the existing destination proof to run. It preserves all source evaluations and
does not substitute a trip count or grant ownership.

LLVM revision `a47bddccec30255619bb8c37fa59700e661d4e66` supplies the reviewed
SCF semantics and related initial-condition rotation in
`mlir/lib/Dialect/SCF/Transforms/{RotateWhileLoop,WrapInZeroTripCheck}.cpp`, with
`mlir/test/Dialect/SCF/wrap-while-loop-in-zero-trip-check.mlir` covering distinct
interfaces. The installed bindings do not register that test pass. The Boolean
carrier is a typed variation at Merlin's existing native seam, not a claim to
run the unavailable pass. Existing triton-shared computation conversion remains.

Completed tensor values made from scalar elements and checked immutable carried
reads can use a late identity copy at the loop edge. Direct destination
retargeting retains its stricter proof. Old-state reads, source guards, unknown
call/mutable-read refusals and native alias/allocation/deallocation lifetime
checks remain. Normalizing an unknown effect never authorizes destination reuse.
Unknown semantic loop attributes keep their original consumer control because
the added Boolean changes operand/result ordinals.

## Complete PyTorch model

`scripts/capture_while_state_model.py` captures two dependent actual mutating
Triton calls, with internal state and a learned i64 counter:

```python
state, checks = x.clamp(0, 9).clone(), self.counter.clone()
before = state.clone(), checks.clone()
first = update(state, checks, 5, 2)
middle = state.clone(), checks.clone()
second = update(state, checks, 9, 3)
return *before, *first, *middle, *second, state, checks, x, self.counter
```

The clamp is original model computation, not an inferred compiler restriction.
All fourteen results remain, including observations from both conditions and
snapshots preceding mutation. Existing functional mutation contracts map the
three ordinary observations followed by two updated bases to the translated
buffer ABI. Parent archives, original result order and learned words remain.
No public caller copyback or physical alias contract is added.

## Qualification and remaining scope

`tests/test_while_tensor_memory.py` checks actual Triton capture, distinct numeric
interfaces, source initialization and attribute refusals, exact i64/f32 native
execution, reusable Merlin host libraries and actual serial RISC-V/Spike
images. The four kernel forms exercise zero/positive trips and final condition
effects. Independent references include integer words above float precision and
i64 counter wraparound.

`tests/test_while_state_model.py` checks every original model result and parent
archive, unchanged-library replays with changed inputs/counters, and two actual
serial target images. The first model target attempt's test-driver import
failure remains in its original log; it is not a compiler or target success.

The functional tensor/scalar ABI, artifact roster, receipts, effect profiles and
Merlin signatures are unchanged. Generated compatibility metadata explicitly
describes while memory and its final false edge. Both readers still enforce
the ordinary source, numeric, initialization, ownership and lifetime contracts.

At producer `1d4e8a1d40926b5066e3238d18ad25485d831ed7` and Merlin
`b6d664a9ec5462f21e22ee0c51b9858eee315354`, the final focused selection passes
25 checks and memory/control/pointer/attention regression passes 150. The
consumer destination/deallocation selection passes 165, broader ownership 163
with two skips, and repository structure passes. These selections overlap.
Targeted Ruff passes; `shared_tensor.py` retains the same five pre-existing
diagnostics as its committed baseline.

`out/while-tensor-memory/summary.json` reopens all five host packages, every
saved reference/result archive, original model-parent bytes and ten actual
target qualifications. Four kernel libraries retain fourteen initial and
seventy replay results; the complete model retains fourteen initial and
fifty-six replay results. Eight kernel images retain twenty-eight results;
two complete-model images retain another twenty-eight. All comparisons use
exact words. Selected source hashes and actual revisions are recorded; this
does not claim a full dependency closure. Reopening passes in
`.scratch/while-memory-report1.log`, with final tests in `.scratch/while-final1.log`
and `.scratch/while-regression1.log`. Earlier failed logs retain their original
outcomes and are not relabeled as current qualification.

General loop-carried address/range proofs, partial/nested fresh initialization,
resizing or permuted owners, irregular parallel dependencies, shared/async
effects and applicable accelerator correctness/performance remain open. This
extends source/control composition rather than the historical fifty-name TTIR
inventory, and establishes no general kernel-coverage or performance percentage.

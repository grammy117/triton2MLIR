# Runtime pointer streams through tensor MLIR

Source pointer recurrences coupled to a bounded positive runtime loop step now
reach standard tensor MLIR, including reads from both final pointer positions.
The actual loop, tiling, masks, numeric state and logical grid remain. No GPU
address, physical storage placement or device launch ABI is introduced.

The real `bidirectional_sum` source in `t2m/examples/runtime_header_kernels.py`
advances one row stream and retreats another:

```python
first = x + lower * row_stride + lane * col_stride
second = x + (R - 1 - lower) * row_stride + lane * col_stride
delta = stride * row_stride
for row in range(lower, upper, stride):
    acc += tl.load(first, mask, other=0) ^ (tl.load(second, mask, other=0) * 3)
    first += delta
    second -= delta
final_a = tl.load(first, mask, other=0)
final_b = tl.load(second, mask, other=0)
```

Original pointer state already executes through the guarded source-word path.
The initial structured probe instead fails in the installed native prepass:
it boxes the i64 numeric accumulator into a structured-state operation whose
operand verifier accepts only i32. Removing that numeric-width issue alone
also leaves final source pointers unresolved in this installed tool. Those
historical failures remain in `.scratch/runtime-pointer-steps1/` through
`.scratch/runtime-pointer-steps4/`; they are not passing qualifications.

## Executable recurrence recovery

After the original source-width audit, the source adapter recognizes additive
pointer updates whose increment equals the actual loop-step SSA times an
integer constant. Identity casts, uniform broadcasts and checked arithmetic
can establish that relation. Equal intervals, example values, kernel names and
provenance do not. Unrelated increments and conditional pointer updates decline
this optional normalization and retain their existing admission requirements.

For coefficient `c`, the body pointer becomes
`initial + (actual_IV - original_lower) * c`. One additional standard scalar
loop carrier starts at the original lower and yields `actual_IV + actual_step`
after each executed body. The final pointer uses that actual exit value. It is
never computed from the maximum iteration count or replaced by the upper bound:
`range(1, 8, 3)` exits at 10, and `range(3, 1, 2)` retains 3 without executing.
All original nonpointer initial values, numeric carriers and memory positions
remain. Original source word arithmetic retains its dtype and order.

Original arithmetic is checked before rewriting; newly introduced i64 offset
arithmetic is checked again afterward. The new difference and its scaled value
must fit signed i64. Final-state range facts require actual SCF initial/yield
edges, not a marker: the initial value must be the lower bound and the yielded
update must use the actual IV and step. Tensor functionalization binds the
actual result SSA to its proved enclosure before recovering final tile axes.
Bounds, complete initialization, source masks, ownership and effects remain
independent gates. A final access outside the root still refuses.

## Upstream reuse and source boundary

Reviewed Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical
Microsoft `050b93ae9c737111f1ca425d793419b32427a971` sources supply
`AnalysisStructured/PtrAnalysis.cpp`'s `rewriteForOp`, initial/result pointer-state
handling and `tensor_indices_loop_iterargs_nested.mlir` /
`scalar_store_nested_loop.mlir` fixtures. Both retain original loop controls
while exposing pointer offsets. Their source revisions remain distinct from
the installed optimizer's binary identity.

The Meta revision's `TritonToStructuredPass.cpp` explains the integer-tensor
boxing prepass and accepts i32/i64; that does not attest the older installed
verifier. After pointer recovery, this source has numeric control interfaces.
The existing supported `skip-prepass=true` option now tracks boxed integer
tensor state separately from scalar state, and follows actual corresponding
for/if yield slots. Numeric tensor state must still have no transitive pointer
or mask dependency. Ordinary memory remains visible to the unchanged upstream
pointer, mask, arithmetic and SCF conversions. No region is hidden or outlined,
no temporary floating-point encoding is used, and no native stack is built.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s SCF implementation and
`int-range-loop-iter-args.mlir` / `int-range-interface.mlir` fixtures supply the
initial-edge, actual-header and zero-trip range boundary used by the existing
loop-domain adapter. Its exit enclosure now includes the original zero-trip
value as well as the final positive-step update.

## Provenance and interface

`source-step-coupled-pointer-recurrences` records the original source loop and
update IDs, each step coefficient, actual-exit construction and zero-trip
semantics. The exact original TTIR and complete inventory remain retained.
`actual_exit_induction_results` identifies independently recognized lowered
SCF edges. Generated compatibility explicitly names the representation.
Public tensor/scalar ABI, receipt schemas and artifact rosters are unchanged;
buffer and guarded-memory profiles retain their separate contracts. Provenance
grants no address, initialization, effect, numeric or target permission.

## Qualification and remaining scope

The focused tests exercise row/column storage, i32/i64 controls, i64 extreme
data words, nonunit steps, ragged lanes, positive trips, and both equal-bound
and reversed-bound empty loops. Independent references check both the reduction
and an asymmetric expression of the final loads. Tensor, buffer and both
guarded address policies use their original separate paths.

Four reusable Merlin tensor-kernel libraries retain every result across six
changed invocations each. Eight actual serial images exercise nonunit and
zero-trip final pointers. The native/kernel/host/serial selection initially
passes 27 checks in `.scratch/runtime-pointer-final1.log`; the subsequent
negative selection passes four in `.scratch/runtime-pointer-final-bounds1.log`,
including a final-access-only out-of-bounds domain. Three checks overlap.
Source/domain/state proofs pass 59 in `.scratch/runtime-pointer-proofs4.log`.

`scripts/capture_pointer_step_model.py` additionally captures two dependent calls
to the same source implementation. Live singleton lower/upper/step tensors,
learned matrix state, surrounding broadcast addition, original inputs and all
ten model results remain. Its independent Torch reference uses no Triton CPU
backend. The first model run passes host/result/replay/domain-abort checks but
its target harness omits the existing required execution-owned selector; that
failed target attempt is not serial qualification. Its next run also omits the
existing scripts import path. The corrected final selection passes all four
checks in `.scratch/pointer-step-model3.log`, retaining ten initial results,
thirty changed-input/parameter/control replay results, twenty exact-word serial
results from two actual images, and an actual invalid-step abort without output.
The related source/memory/control regression selection passes 159 in
`.scratch/runtime-pointer-regression2.log`. An earlier selection names two
nonexistent test files and runs nothing; it supplies no regression evidence.
New-file Ruff and `git diff --check` pass.

The subsequent eight-check kernel-package selection regenerates all four
libraries with the explicit pointer-representation compatibility difference;
host replays and all eight serial images pass again. Its log is
`.scratch/runtime-pointer-packages1.log`. `out/runtime-pointer-steps/summary.json`
reopens those four libraries and the final model: 18 initial results, 78 replay
results and 36 exact-word serial results across ten actual images, plus the
invalid-model-step abort. All parent archives, package compatibility and actual
target receipts are checked. The reopening log is
`.scratch/runtime-pointer-report5.log`; observed reopening revisions are not a
claim of complete build dependency closure. Thirteen selected implementation,
test and capture-helper files match commit
`5d9623573b54686385e83442ae9bd84d06a2334e`. The unchanged companions are
model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`.

General non-step-coupled or conditional pointer recurrences, while/alias
ownership, dynamic root projection, variable-step store-image coverage,
shared/asynchronous participation, broader numeric/source compositions and
applicable accelerator correctness/performance remain completion work. Retained
tiling and reuse are not a performance benchmark. These qualifications change
no TTIR operation-name denominator.

## Historical pure-loop boundary

`.scratch/runtime-pointer-final-only1/` removes the body loads from the same
two-stream source, retaining pointer advancement and the final reads. Source
normalization succeeds, but the native numeric loop then takes the existing
pure-value cloning path. That path does not bind its actual exit result to the
checked enclosure, so final coordinate recovery refuses with an unproved
affine expression. This is a local representation boundary, not a GPU-specific
operation to discard. Continue by qualifying actual exit-state facts through
pure numeric loop cloning, while retaining the original loop and final loads.
The saved conversion refusal is not native, host or target qualification.
[Pure loop exit coordinates](pure_loop_exits.md) now close this exact cloning
boundary and independently qualify richer word-state kernels and complete
models. General non-step-coupled and conditional pointer streams remain open.

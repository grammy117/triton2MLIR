# Wholly nonexecuting source loops

A checked source `scf.for` whose complete declared upper-bound interval is at
or below its lower bound now returns its original initial tuple. Incoming
memory versions and initialization facts remain unchanged. The tensor result
contains the original surrounding computation, logical grid and fallback
writes, without evaluating the inactive body or granting its writes credit.

This closes the two original refusals retained under
`.scratch/loop-tile-next-probe1/empty.log`: the complete-prefix and zero-after
forms with `steps` in `[-2, 0]`. Mixed domains that permit positive iterations
still require the existing body/zero-path proofs. A located refusal remains
required when a fresh result has no actual initializing writes.

## Ordering and executable premises

The empty-edge witness uses actual scalar bounds, signed comparison and source
integer width. It currently requires a constant lower bound and positive
constant step. An empty domain provides no induction-value interval. The source
address audit still checks definitions computed in live enclosing scopes;
body inactivity cannot hide an earlier divide by zero or source-width overflow.

The tile-state planner can cross this edge without changing initialization or
examining body reads as executed events. It retains the native source loop and
body until the independent tensor memory witness has validated preceding
source accesses. Otherwise an outside load used only by the inactive body
could become dead before its original initialization/bounds checks. Only the
functional witness omits the body and resolves all original result carriers to
their corresponding initial values.

The buffer profile retains its separate native memory continuation. After
source-read validation, the same actual scalar premises become standard
`arith` comparisons and `cf.assert` operations in that continuation, before
native memory lowering. Its entry ABI retains its original caller guards too.
A test removes only those outer assertions and bypasses the test runner's
scalar-range preflight: a positive count must still abort inside the compiled
continuation, with an invocation-start marker and no output archive. This
adversarial copy is not a published package or a weakened production contract.

`inactive_loop_source_ids` distinguishes crossed inactive loops from loops
that receive an additional tile carrier. Memory provenance records the original
source loop, zero reachable iterations, complete initial tuple and unchanged
entry facts. These records describe the proof; executable guards enforce its
domain. Public ABI, receipts, artifact rosters, numeric and effect profiles are
unchanged. Buffer payloads retain their existing explicit granularity difference
from pure tensor MLIR.

## Reuse and upstream boundary

LLVM revision `a47bddccec30255619bb8c37fa59700e661d4e66` implements the SCF
zero-trip edge through `ForOp::getEntrySuccessorOperands` and
`ForOp::getSuccessorRegions` in
[`SCF.cpp`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Dialect/SCF/IR/SCF.cpp).
Its `remove_zero_iteration_loop` and `remove_zero_iteration_loop_vals` fixtures
in `mlir/test/Dialect/SCF/canonicalize.mlir` check the corresponding canonical
behavior. This extension proves the runtime-domain premise at the functional
boundary; it does not replace native SCF lowering or treat a while-before region
as an inactive for-body.

Reviewed Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`
[`scalar_store_nested_loop.mlir`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/StructuredToMemref/scalar_store_nested_loop.mlir)
and historical Microsoft `050b93ae9c737111f1ca425d793419b32427a971`
[`PtrAnalysis.cpp`](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/AnalysisStructured/PtrAnalysis.cpp)
retain source loop/pointer interfaces through structured conversion. Existing
native analyses and memory lowering remain in use. No upstream source blocks
were copied and no native compiler build was performed. The installed optimizer
binary and reviewed source revisions retain their separate recorded identities.

## Complete programs and evidence

`t2m/examples/empty_loop_kernels.py` supplies four real forms: fallback before
the loop, fallback after it, complete prefix and an address expression defined
only inside the inactive body. Nonzero i64 initial carriers, extreme words,
changed bias, empty/tail/full tiles and both tensor/buffer profiles are checked.
Negative cases cover live earlier reads/division, a domain permitting positive
trips and fresh results whose only writes would occur inside the body.

`scripts/capture_empty_loop_model.py` retains two dependent Triton calls and
all seven original PyTorch model results, including earlier snapshots, input,
shifted intermediate and learned i64 bias. Both calls receive the actual live
expression `2 - n` over the original exported domain `n=2..31`. An independent
PyTorch reference supplies outputs; no Triton CPU backend runs the kernels.

That model exposed a companion model2MLIR gap: shape-derived scalar arithmetic
previously admitted only nonnegative sizes. The companion now transports signed
controls as checked i64 SSA while retaining actual dimension reads, source
arithmetic and entry-domain guards. Nonnegative sizes keep their checked index
path; intervals exceeding signed i64 still refuse. This is ordinary scalar
computation, with no negative tensor extent or new public package schema.

The recorded producer selections pass 83 source/domain/control checks, 31 kernel
checks and four complete-model checks in separate runs. The latter kernels
include native tensor/buffer execution, four reusable Merlin host libraries with
six invocations each, and eight actual serial RISC-V/Spike images. The model
retains every result through three changed-input/parameter host replays and two
actual serial images, including a zero count and a negative count. An original
exported-domain violation still aborts without outputs.

Logs are `.scratch/empty-loop-proofs1.log`, `.scratch/empty-loop-kernels4.log`
and `.scratch/empty-loop-model3.log`. Companion model2MLIR source/scalar checks
pass 32; its shape/dynamic/trace/kernel regression selection passes 88. Their
logs are `.scratch/signed-symbolic3.log` and
`.scratch/signed-symbolic-regression1.log` in that repository. Targeted Ruff and
producer `git diff --check` pass. Initial harness failures used a hardcoded
entry name, an over-specific cast expectation and the wrong missing-write
diagnostic; those runs do not count as qualification. The original model capture
failure on signed controls was a production gap, corrected in the companion.

The related memory/control selection passed 119 checks with two failures in
historical expectations. Both failures independently reproduce at committed
baseline `b271f9c`: numeric loop branches need no memory-only conditional trace
row, and final-pointer accesses may use executable bounds checks instead of a
static refusal. The corrected seven-case selection passes, including a compiled
invalid-pointer abort with no outputs; both corrected cases also pass against
that baseline. A separate loop/functional-region/model selection passes 36.
Logs are `.scratch/empty-loop-regression1.log`,
`.scratch/empty-loop-baseline1.log`, `.scratch/current-memory-contracts1.log`,
`.scratch/current-memory-contracts-baseline1.log` and
`.scratch/empty-loop-composition1.log`. The original failed selections remain
failed records, not green qualification. No production behavior was changed for
these test corrections. The two original saved empty-domain sources also
convert unchanged in `.scratch/loop-tile-next-probe1/empty-final1.log`.

Reproduce with the existing source/model/native/Merlin environments:

```bash
python -m pytest -q tests/test_loop_domains.py tests/test_empty_loops.py \
  tests/test_empty_loop_model.py --basetemp=.scratch/pytest-empty-loops
```

General while/alias and ownership relations, variable lower bounds/steps,
shared/asynchronous participation, broader source/numeric compositions and
applicable accelerator correctness/performance remain full-goal work. This
capability changes no TTIR opcode denominator.

`out/empty-loops/summary.json` reopens four kernel libraries and one complete
model library: 15 initial, 69 unchanged-library replay and 30 exact-word target
results across ten actual serial images, plus the original model-domain abort
and the independently guarded native-continuation abort. Every parent archive
and original source-loop link remains checked. Original failed regression runs
remain explicit records beside their baseline reproduction and corrected checks.
The reopening log is `.scratch/empty-loop-report1.log`.

Implementation `e670c8aadfb74a7da26737bd5361b30a59a502dd` retains companion
model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2` and unchanged Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`. All eleven selected
producer sources and two selected companion sources match actual committed
bytes. These matches remain separate from observed reopening revisions and
are not a full dependency closure, accelerator or performance qualification.

## Next concrete representation boundary

`.scratch/runtime-loop-headers1/` retains a real selected-row sum:

```python
lane = tl.program_id(0) * B + tl.arange(0, B)
acc = tl.full((B,), 0, tl.int64)
for row in range(lower, upper, stride):
    acc += tl.load(x + row * N + lane, lane < N, other=0)
tl.store(out + lane, acc, lane < N)
```

Its input is `8x17xi64`, output is `17xi64`, `N=17`, `B=8`, and logical grid
is three. Runtime i32 controls retain declared bounds `lower=0..3`, `upper=0..8`
and `stride=1..3`. The tensor, buffer and structured-address guarded profiles
all refuse at the existing constant-lower/constant-step domain requirement.
The guarded source-word profile already preserves the original SSA loop and
actual addresses: four native cases, one initial Merlin host invocation, four
unchanged-library replays and two actual serial images pass. Cases include
strided selection, reversed endpoint ordering and equal endpoints. Artifacts
are `source.ttir`, `abi.json`, `capture/`, `merlin/`, `invoke-*` and `target-*`;
logs are `probe-structured.log`, `probe.log` and `qualify.log` in that directory.
The first probe's three `logical` policy spelling errors are harness errors,
not compiler refusals; `probe-structured.log` contains the actual source gates.

This establishes a finite existing-interface route, not a new shipped frontend
capability or full runtime-loop coverage. Its memory checks and offset tensors
remain an explicit granularity difference from tensor MLIR. Next prove reachable
IVs and the final source-width induction update for bounded variable lower/step
operands, then reuse those facts for readonly row-tile extraction while retaining
the actual SSA bounds/step. Initialization, aliases and conflicting writes keep
independent gates; a trip-count interval must not replace the original loop or
supply write coverage. General pointer carriers and variable-step write images
remain separate from that readonly conversion.

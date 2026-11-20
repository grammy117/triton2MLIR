# Variable source loop headers and rank-reduced tile reads

Bounded runtime lower, upper and positive-step operands now reach tensor MLIR
without replacing the source loop. Readonly row/column tiles retain standard
tensor slices; partial-tile recurrences retain their numeric state and actual
zero-trip completions. The logical grid stays unmapped. Existing buffer and
guarded source-word representations keep their independent contracts.

This closes the tensor refusal recorded by the
[original selected-row probe](empty_loop_memory.md#next-concrete-representation-boundary).
Its first structured paths required constant lower/step values. After the loop
enclosure extension, its vector load exposed the separate rank-two-root versus
rank-one-tile gate; the rank-reducing slice adaptation closes that gate too.
Neither failure requires per-lane gather expansion or a dense replacement
algorithm. The historical failed source and logs remain separate evidence.

## Actual source domains and memory facts

`enclosed_loop` reuses the tighter existing domain when lower/step are constant.
For variable headers, it checks all scalar intervals in their original signed
width, requires a proved positive step, encloses the actual reachable induction
values, and checks the update after the last reachable body execution. Empty
domains provide no induction-value fact. Actual scalar assertions remain at the
functional entry; annotations supply no range or no-wrap permission.

The emitted `scf.for` retains all three original SSA operands. Its coordinate
facts refer to the source IV itself. An interval of iteration counts supplies
neither an iteration ordinal nor an exact stride/congruence. Variable headers
seed only invariant scalar address recurrences; general pointer carriers and
nonzero affine address recurrences retain their own gates. Numeric tensor
carriers remain executable, including source word arithmetic.

The existing memory versioning and selected-path initialization proofs still
apply. A possible zero trip keeps entry facts, and only actual fallback writes
complete a fresh result. The separate rectangular-store-image analyses receive
no fabricated constant bounds/step for a varying header. Invariant tile writes
can use their existing proofs; general variable-step write-image coverage remains
separate. Source scope facts still validate definitions before narrowing uses.

## Restoring readonly source tile axes

A structured source pointer can describe an eight-element vector within a
rank-two logical root. The adapter reuses affine coordinate recovery to split
actual element offsets into root coordinates, including source IVs and aligned
runtime scalar terms. A bounded injective stride/bounds match identifies its
retained axes. Every omitted axis also needs an in-bounds unit slice.

Standard `tensor.extract_slice` drops only those unit axes. A standard transpose
restores the original tile-axis order when necessary, and existing crop/pad
handling retains the source mask and fill. Retained unit axes use actual
constant-one mixed-size operands before a checked tensor cast, keeping them
distinct from dropped static units. The public root shape remains intact.
This adds no per-lane gather, physical alias or memory effect permission.

This projection currently applies to direct loads of positive static-shape
readonly roots. Dynamic roots, escaping descriptors and writable observations
retain their existing analyses or located refusals. The axis search keeps the
existing eight-axis and 256-attempt limits. A failed proof grants no conversion.

## Reuse and version boundary

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`
[`IntegerRangeAnalysis.cpp`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp)
reads lower/upper/step ranges through the actual loop interface. Reviewed
`int-range-loop-iter-args.mlir` and `int-range-interface.mlir` cover bounded
carriers, empty loops and runtime loop-interface values. This source boundary
also requires explicit step positivity and a checked final source-width update;
an undiscoverable positive step is not assumed.

The same revision's
[`TensorOps.cpp`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp)
implements `getDroppedDims`; its `rank_reducing_slice_preserve_dropped_dims`
fixture checks retained-unit behavior. The companion model2MLIR's
`ir/tensor_selection.py` already uses constant-one mixed sizes to preserve that
boundary. Existing transpose, mask and padding builders remain in use.

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` supply the reviewed
`AnalysisStructured/PtrAnalysis.cpp` loop-interface patterns and
`scalar_store_nested_loop.mlir` fixtures. Native pointer/arithmetic conversion
and SCF lowering remain upstream operations. No native compiler build or copied
upstream source block is introduced. Reviewed source and installed optimizer
binary identities remain separately recorded.

## Provenance and package contract

Memory provenance records `lower_interval`, `upper_interval`, `step_interval`
and `source-IV-enclosure; not-an-iteration-ordinal`. `source_projected_reads`
retains actual source load IDs, original tile sizes, root rank and retained axes.
Source/emitted correspondence and complete original TTIR remain available.
Generated compatibility explicitly names both adaptations. Public tensor/scalar
ABI, receipts, artifact rosters and numeric/effect profiles are unchanged.
The separate buffer payload retains its existing granularity difference.

## Real programs and measured evidence

`t2m/examples/runtime_header_kernels.py` includes selected-row reduction in both
storage orientations and a partial-tile write/read recurrence with nonzero
initial observations. Runtime i32/i64 controls, signed i64 extremes, independent
header combinations, negative initial IVs, tails and zero trips use independent
NumPy references. Tensor and buffer payloads retain actual source controls.
Six reusable tensor-kernel libraries and twelve actual serial images preserve
their complete results; native checks also exercise the buffer form.

`scripts/capture_runtime_header_model.py` retains two dependent calls to one
actual source implementation. First-call controls are `(n//4, n-2, n//8+1)`;
second-call controls are `(n//8, n, n//4+1)`. Learned i64 bias, surrounding
addition and all seven original model results remain. The original exported
domain is `n=2..31`, including first-call zero trips. Independent Torch reference
execution uses no Triton CPU backend. Three changed-input/parameter host
invocations and two actual serial images retain every result and source guard.

The recorded runs pass 103 source/domain/projection checks, 45 kernel/model
checks, three compiled invalid-header guard checks and one additional native
multiaxis projection check. These are separate runs. Logs are
`.scratch/runtime-headers-proofs2.log`, `.scratch/runtime-headers-final1.log`,
`.scratch/runtime-header-guards2.log` and `.scratch/runtime-header-projection2.log`.
The related memory/control regression selection passes 133 in
`.scratch/runtime-headers-regression2.log`. Targeted new-file Ruff and
`git diff --check` pass; established source files retain their prior lint issues.

The first proof run exposed a real coordinate-recovery omission of runtime
scalar terms; retaining aligned parameter terms fixes it. Initial structure
and guard harness failures used tuple/list expectations and a nonexistent
dynamic shape contract for static ports. Those failed runs are not execution
qualification. Two historical preinitialization trace assertions independently
fail at `f55f180`; corrected cases pass there and in the final regression run.
Their original/baseline logs remain `.scratch/runtime-headers-regression1.log`
and `.scratch/runtime-header-baseline1.log`; corrected baseline execution is in
`.scratch/runtime-header-corrected-baseline1.log`.

Reproduce with the existing source/native/model/Merlin environments:

```bash
python -m pytest -q tests/test_loop_domains.py tests/test_projected_reads.py \
  tests/test_runtime_headers.py tests/test_runtime_header_model.py \
  --basetemp=.scratch/pytest-runtime-headers
```

General pointer/while and alias ownership, variable-step store images, dynamic
root projection, shared/asynchronous participation, remaining numeric/source
compositions and accelerator correctness/performance remain full-goal work.
These finite qualifications change no TTIR operation-name denominator.

`out/runtime-headers/summary.json` reopens six kernel libraries and one complete
model library: 15 initial, 69 unchanged-library replay and 30 exact-word target
results across fourteen actual serial images, plus the original model-domain
abort and three compiled invalid-header aborts. Both original model callsites,
every parent archive and actual source-load/loop links remain checked. The
reopening log is `.scratch/runtime-headers-report2.log`.

Implementation `182eafa780ea2f9d70b175f41f388b86b54f2a1a` retains unchanged
model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both companions on
`ttir-experiment-grammy`; the producer remains on `main`. All thirteen selected
producer implementation/test/helper sources match actual committed bytes.
The report keeps that match separate from the observed reopening revisions.
These pins are not a full dependency closure, accelerator or performance
qualification. No companion source change was needed for this capability.

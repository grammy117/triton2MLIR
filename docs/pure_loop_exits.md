# Pure numeric loops retain actual exit coordinates

The tensor adapter now retains checked exit-state facts when it clones an
ordinary numeric `scf.for`. This closes the final-pointer-only refusal saved in
`.scratch/runtime-pointer-final-only1/`. Source pointer normalization already
made that loop numeric; the missing step was binding its actual result SSA to
the existing exit enclosure after cloning. The original source loop, final
loads and complete numeric tuple remain executable.

## Actual edges and scope

The same `bind_exit_induction` helper now serves cloned numeric loops and
functionalized memory loops. It recognizes only the actual initial/yield
relation: the selected result starts at the source lower bound and its body
yields the actual source IV plus the actual step. The signed source header must
already have a checked positive-step domain. Its final update is checked in the
source width, and the enclosure includes the original lower value on zero trips.
The helper attaches that fact to the actual cloned result, with a distinct
local binding for each loop and result slot.

Equal intervals do not identify different SSA operands or tuple slots. An
unrelated initial value, different update operand, unsigned comparison,
unknown header range or nonpositive step supplies no exit-address proof.
Advisory `prov.exit_induction` markers grant nothing. Tests retain a decoy slot
with identical declared operand ranges and two sequential pure loops with
different initial edges. Actual native reads distinguish their exit values.
Ordinary numeric loops needing no address fact keep their existing admission.

Nothing substitutes the upper bound or a worst-case iteration count for the
actual result. `range(1, 8, 3)` produces exit 10; `range(3, 1, 2)` retains 3.
Loop arithmetic, nonzero numeric initial state, selected masks, preloaded tile
reuse and subsequent observations retain their original execution. Bounds,
initialization, ownership, effects and numeric profiles remain independent.

## Real kernel and complete models

`final_pointer_mix` in `t2m/examples/runtime_header_kernels.py` loads one row
tile before the loop and reuses it as i64 word state. Both pointer streams
advance without memory accesses inside the loop:

```python
state = tl.load(x + lane * col_stride, lane < N, other=0)
for row in range(lower, upper, stride):
    state = (state ^ (row.to(tl.int64) + 2)) * 3
    first += delta
    second -= delta
final_a = tl.load(first, lane < N, other=0)
final_b = tl.load(second, lane < N, other=0)
```

The two outputs combine the carried word state and final loads. `MIX=False`
additionally exercises a pointer-only loop with the original preloaded tile
retained. Row/column orientations, i32/i64 controls, signed i64 extremes,
nonunit steps, ragged lanes and both zero-trip forms use independent NumPy
references. All four tensor/buffer/guarded address selections retain their
separate paths. Tensor output keeps the logical grid unmapped and introduces
no memory operation inside the original pure loop.

`scripts/capture_pointer_step_model.py --final-only` captures two dependent
calls, original singleton control tensors, learned matrix state and surrounding
broadcast addition. All ten original results remain, including earlier final
observations, original input, intermediate tensor, weights and controls. An
independent Torch CPU implementation supplies references without a Triton CPU
backend. The original body-load model remains a separate regression selection.

## Upstream boundary and package representation

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s `SCF.cpp` checks matching
initial/carried/result types and describes both the body and zero-trip parent
successors. Its `int-range-loop-iter-args.mlir` and `int-range-interface.mlir`
fixtures cover bounded carried state and runtime loop headers. This change
reuses the existing source-width domain and exit-edge recognizer rather than
introducing another pointer or iteration analysis.

Reviewed Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`'s
`PtrAnalysis::rewriteForOp` separately tracks loop arguments and loop-result
pointer state; `tensor_indices_loop_iterargs_nested.mlir` retains source
controls while exposing index carriers. Those upstream pointer, arithmetic and
SCF conversions remain unchanged. Source pins remain distinct from the
installed optimizer binary. No native compiler build is introduced.

`actual_exit_induction_results` now also records independently recognized
edges under `memory_effects="no-buffer-accesses"`. Generated compatibility
explicitly names this representation. Exact original TTIR, original/emitted
source correspondence, public tensor/scalar ABI, receipts and artifact rosters
remain. No new effect, physical alias, target mapping or numerical permission
comes from those annotations. Both companion repositories retain their existing
`ttir-experiment-grammy` branches; this capability requires no companion edit.

## Measured evidence

The first focused selection passes 39 kernel/source/native/host/serial checks
in `.scratch/pure-pointer-final1.log`. This includes four reusable kernel
libraries, six host replay domains per library and eight actual serial images.
The extended source-coordinate selection separately passes twelve in
`.scratch/value-exit-proofs3.log`, including actual native observations from
two sequential loops and six negative edge/domain cases. Ten source checks
overlap the earlier selection.

The complete-model selection passes eight in `.scratch/pointer-pure-model1.log`:
four checks for the original body-load model and four for the new pure-loop
model. Each model retains all ten initial results, thirty results across three
changed-input/weight/control invocations, twenty exact-word serial results
across two actual images and an actual invalid-step abort without output.
Original parent archives remain byte-identical and the host library is reused.

The first related regression selection has 117 passes and two test-parser
failures: the new sequential-loop fixture omitted parentheses required by the
local xDSL custom parser. The corrected twelve-case source selection passes;
the original failed log remains `.scratch/pure-exit-regression1.log`.
Earlier standalone fixture syntax failures remain separately retained and do
not count as conversion refusals or successful qualification.
The corrected related control/memory/source selection passes all 119 checks in
`.scratch/pure-exit-regression2.log`. New/changed test and capture-helper Ruff
and `git diff --check` pass.

`out/pure-loop-exits/summary.json` reopens the new four kernel libraries and
pure-loop model: 18 initial, 78 replay and 36 exact-word serial results across
ten actual images, plus the invalid-model-step abort. Package compatibility,
complete parent archives and actual target receipts are checked. The original
body-load model remains separate regression evidence. Source/reopening pins
are not a full build dependency closure.

General conditional/non-step-coupled pointer streams, while/alias ownership,
variable-step store images, dynamic root projection, shared/asynchronous
participation, broader source/numeric compositions and applicable accelerator
correctness/performance remain completion work. This finite qualification
changes no TTIR operation-name denominator and establishes no performance claim.

Reproduce using the existing source, native, model2MLIR, Merlin and target tools
documented in the README:

```bash
python -m pytest -q tests/test_value_exit_coordinates.py \
  tests/test_final_pointer_values.py tests/test_pointer_step_model.py \
  --basetemp=.scratch/pytest-pure-loop-exits
```

## Next reproduced boundary

`.scratch/runtime-pointer-independent-step1/` changes the original body-load
kernel's pointer increment from `stride * row_stride` to `row_stride`. The
pointer therefore advances once per actual iteration, independently of the
runtime loop step. Guarded source-word conversion passes four independent
native cases, including zero trips and final reads. All three structured
selections still fail in the installed native i64 structured-state prepass.
The current source normalization deliberately grants no step-coupling proof
for this different actual SSA expression. Continue with a proved actual
iteration ordinal or equivalent exact recurrence representation; never replace
it with a maximum trip count or discard the pointer/state observations. This
probe has no Merlin host or serial qualification yet.

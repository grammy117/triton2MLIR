# Independent runtime memory strides through tensor MLIR

Bounded invariant integer memory increments can now remain independent of source
loop controls while reaching the functional tensor interface. The source loop,
tile shape, masks, numeric state, reuse and final pointer observations remain.
The logical grid receives no thread, warp, core or physical memory assignment.

The real `invariant_stride` kernel in
`t2m/examples/runtime_stride_kernels.py` exercises four source variants:

```python
first = x + (16 + lower) * row_stride + lane * col_stride
second = x + (R - 17 - lower) * row_stride + lane * col_stride
delta = memory_stride * row_stride
for row in range(lower, upper, stride):
    state += load(first) ^ (load(second) * 3)
    first += delta
    second -= delta
final_a, final_b = load(first), load(second)
```

`memory` retains these loads and an i64 recurrence. `pure` reuses a preloaded
tile through original word arithmetic and observes pointers after the loop.
`mixed` uses `(stride + memory_stride) * row_stride`. `late` computes
`(1 // upper) * row_stride` inside the body: `upper == 0` remains a valid
zero-trip entry and must never cause speculative division. Positive, negative
and zero memory increments retain their actual source values.

## Representation and proof

The existing step/constant pointer adapter remains the first normalization.
The new optional continuation factors uniform pointer increments into an
actual-loop-step term and a linear expression of invariant scalar SSA values.
Source integer constants supply a greatest-common-divisor factor, keeping the
progress carrier in logical axis units where possible. Each pointer stream
has its own zero-initialized i64 progress state. The body pointer becomes:

`initial + step_coefficient * (actual_IV - original_lower) + axis_factor * progress`

The original body computes the actual increment and updates progress at its
yield edge. Final reads use actual returned progress and exit-IV values.
Maximum iteration bounds do not replace execution; zero trips retain every
original initial value. Original numeric carriers keep their original order,
types, operations and values.

Loop invariance requires an actual dependency closure: outside SSA samples or
local recognized scalar integer operations whose operands are invariant.
An IV, another varying carrier, a body load, an unknown operation or a region
cannot acquire invariance from a range or annotation. This is not LICM: body
arithmetic stays in its original body and receives no speculation permission.

Original source widths are audited before normalization. Every introduced
multiply/partial sum, body/final progress enclosure and scaled offset must fit
signed i64. The existing source-scope analysis supplies definition-local range
facts, including positivity available only in an executed body. Facts from a
later body cannot validate an earlier division. Actual additive scalar SCF
initial/yield edges also provide checked body/result coordinates directly;
the loop header and carrier keep their independent original width checks.

Each varying coordinate has its own actual scalar identity and expires at its
original loop boundary, including initialization facts depending on it.
Returned states bind actual result SSA. Bounds grant no complete-write,
ownership, alias or effect permission. Source masks, fills, active address
bounds, output initialization and parallel ownership retain independent checks.

## Upstream boundary and provenance

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical
Microsoft `050b93ae9c737111f1ca425d793419b32427a971` retain source controls
while exposing carried pointer offsets in `PtrAnalysis::rewriteForOp`.
Their `tensor_indices_loop_iterargs_nested.mlir` and
`scalar_store_nested_loop.mlir` fixtures supply the reviewed loop-state boundary.
The adapter continues to use native shared pointer, mask, arithmetic and SCF
conversion. Source pins do not attest the installed optimizer binary.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`'s LICM utility and
`loop-invariant-code-motion.mlir` distinguish outside definitions and pure
dependency closures from permission to move operations across unknown trip
counts. The local adapter uses the dependency distinction without moving code.
Existing SCF initial/yield and range-analysis contracts retain the zero edge.
No native build or Triton CPU target backend is introduced.

`source-invariant-runtime-pointer-recurrences` records original loop IDs,
stream positions, actual-step coefficients, logical factors and retained-body
evaluation. `actual_invariant_recurrence_arguments` and
`actual_invariant_recurrence_results` identify independently recognized actual
SCF edges. Original TTIR, operation inventories and source correspondence remain.
Generated compatibility names both representation extensions. Public ABI,
receipt schemas and artifact rosters retain their existing meanings.

## Qualification and remaining work

`tests/test_invariant_recurrences.py` checks actual scalar identity, refusal of
IV/body-load dependencies, retained local division, narrow carriers and introduced
offset overflow. `tests/test_invariant_stride_kernels.py` uses independent NumPy
references for row/column tiles, i32/i64 controls, ragged lanes, large i64 words,
negative/zero progress, nonunit steps, final reads and both zero-trip forms.
Tensor, buffer and both guarded address policies keep their separate admission.

`scripts/capture_invariant_stride_model.py` captures two dependent calls with
learned matrix state and live singleton lower/upper/loop-step/memory-step tensors.
All eleven original results remain, including earlier observations, surrounding
broadcast addition, inputs, parameters and controls. Independent Torch references
remain outside compiler staging. Model tests qualify unchanged-library replays,
actual source-domain aborts and complete serial target results.

The kernel selection passes 51 in `.scratch/invariant-stride-kernels1.log`,
including eight reusable libraries, seven changed invocations per library and
sixteen actual serial images. The combined complete-model/source selection
passes 62 in `.scratch/invariant-stride-models1.log`: all four models preserve
eleven initial, thirty-three replay and twenty-two exact-word serial results
each, plus an actual original-memory-stride-domain abort without outputs.
The earlier source/control selection passes 159 in
`.scratch/invariant-stride-regression1.log`. The final ten invariant/width checks
pass in `.scratch/invariant-stride-width3.log`, including independently typed
i32 headers near 200 and safe/overflowing i8 memory carriers. Counts overlap;
they must not be added into a unique-coverage total.

The initial source-proof run retains two obsolete refusal expectations for
now-valid additive states. The corrected tests require actual invariant-state
facts, no ordinal claim, and independent native results. The first final-width
fixture also omits the existing required result/effect-only ABI selector; the
corrected fixture supplies it. Those failed runs remain in
`.scratch/invariant-stride-proof1.log` and `.scratch/invariant-stride-width1.log`;
neither is passing qualification. New helper/test/source Ruff and diff checks
pass. Pre-existing style diagnostics in the large shared modules are outside
this change.

`out/invariant-runtime-strides/summary.json` reopens all twelve libraries,
sixty initial results, 244 replay results and 120 exact-word target results
across twenty-four actual serial images, plus four actual domain aborts.
Original parent archives, both callsites, actual header SSA, generated
compatibility and target receipts remain checked. The first reopening log is
`.scratch/invariant-stride-report1.log`. Selected source hashes do not establish
a complete build dependency closure.

General conditional/while pointer state, varying increments, nested relational
invariants, alias ownership, dynamic-root projections, variable-step complete
write images, shared/asynchronous participation, broader numeric/source versions
and applicable accelerator correctness/performance remain completion work.
Retained tiling and reuse supply no performance benchmark. This extension changes
no historical TTIR operation-name denominator or arbitrary-kernel completion claim.

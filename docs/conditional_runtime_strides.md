# Conditional memory progress through tensor MLIR

Actual scalar branches can now select per-iteration memory steps without losing
logical tiles, source control, reuse or intermediate/final observations. This
extends the [invariant runtime stride](invariant_runtime_strides.md) path with
checked varying progress. A bounded increment is not an invariant value or a
unit execution ordinal.

`t2m/examples/conditional_stride_kernels.py` contains real source variants:

```python
for row in range(lower, upper, stride):
    state += load(first) ^ (load(second) * 3)
    if row < 3:
        first += memory_stride * row_stride
        second -= memory_stride * row_stride
    else:
        first += (1 // (row - 2)) * row_stride
        second -= (1 // (row - 2)) * row_stride
final_a, final_b = load(first), load(second)
```

`scalar` selects an integer increment before pointer updates. `pointer` yields
the actual updated pointers. `pure` reuses a preloaded tile through original
i64 arithmetic and observes the final pointers. `observe` additionally reloads
the updated first pointer inside the body. `nested` includes another selected
pointer update in the second arm. `bit` advances by the actual zero-extended
comparison `(row < 3)`. Negative, positive and zero memory strides,
nonunit loop steps, ragged lanes and both zero-trip forms keep their meaning.
Division stays in its original selected arm; neither low rows nor zero trips
evaluate `1 // (row - 2)`.

## Reused representation and source proof

The original source-width audit now joins ranges from both actual typed SCF
yield regions, using each arithmetic definition's own context. Original entry
definitions still require their own checks before narrower branch-use facts.
A later or unrelated condition cannot rescue a zero divisor or earlier overflow.
Only actual source SSA and established ranges supply the enclosure.

Signed i1 comparisons use their actual `-1/0` word enclosure, with operand
definitions checked independently. Boolean zero extension retains the actual
`0/1` value; sign extension retains `-1/0`. This admits normalized Boolean
increments without promoting an unknown larger unsigned word to a mathematical
index or changing executable arithmetic.

The optional source branch adapter reuses the existing checked uniform forms.
For integer results it exposes a common positive source constant factor by exact
division inside each original arm and a restoring multiply after the join. The
factor is proved from actual arithmetic, rather than guessed from sampled values.
For pointer results, both actual yields must retain the same dominating pointer
SSA base. Their selected scalar displacement becomes the branch result; a
standard `tt.addptr` reconstructs the original typed pointer after the join.
Common outside pointer expressions remain reused. Different bases or unknown
nonuniform offsets decline this optional rewrite and keep their existing profile
requirements.

Original region blocks, operations, memory observations, conditions and numeric
tuple slots remain. Every introduced expression must fit its actual integer
width. The source is audited again before native shared pointer/mask/arithmetic/
SCF conversion. There is no branch speculation, hidden region, target instruction,
physical address or kernel-name selection.

The existing progress adapter then retains each actual selected increment in a
zero-initialized standard scalar loop carrier. Bounds enclose every reachable
body/final value even when increments vary; they supply no closed-form summation
or replacement loop. Source constant axis factors retain row/column slices.
Actual returned progress defines final reads, including original zero-trip
initials. Original and introduced arithmetic have separate width checks.

Actual additive scalar SCF edges similarly obtain checked coordinate enclosures.
Invariant and varying states remain separately identified. Actual integer-if
results may carry checked union ranges as local executable coordinates, allowing
reads immediately after a selected update. Such coordinates are not uniform ABI
parameters: they cannot establish parallel write ownership. They and dependent
initialization facts expire with their defining block; carried coordinates also
expire at the actual loop boundary. Complete-write, initialization, bounds,
aliasing and effect checks remain independent.

Loop invariance does not imply grid uniformity. Returned states become uniform
coordinate parameters only after an actual scalar dependency closure through
original SCF controls, initial/yield edges and pure operations reaches uniform
by-value inputs/constants. Program IDs, memory observations and unknown operations
retain scoped varying coordinates instead. Actual conflicting-write examples
refuse in tensor and unguarded-buffer profiles, both directly after a branch and
after invariant per-program steps cross a numeric loop.

## Upstream and package boundary

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` implements
`PtrAnalysis::getIfPtrState`, `reconcileIfPtrState` and `rewriteIfOp`, retaining
selected offset/stride/base state. Reviewed `if_inside_loop.mlir`,
`if_preincrement.mlir` and nested conditional fixtures retain actual branches and
their pointer geometry. Historical Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` recognizes scalar if-produced
pointers as source bases; its staged pointer-loop work remains reused through
the existing shared adapter. These are source pins, separately from the installed
optimizer's binary identity. The local work supplies logical tensor axes, original
source-width/scope proofs and the functional interface, using existing shared
converters afterward. No native compiler stack or Triton CPU backend is built.

`source-branch-logical-offsets` records actual selected results and source factors.
`source-bounded-runtime-pointer-recurrences` distinguishes varying progress from
the invariant profile; its term count is `bounded_scalar_terms`. Earlier local
qualification packets retain the prototype diagnostic count name, which grants
no invariance. `actual_bounded_recurrence_arguments/results` record real
scalar SCF edges; `actual_bounded_integer_results` records checked integer branch
results. Original TTIR, source inventories and source/emission correspondence
remain. Generated compatibility names these representations and their limits.
Public tensor/scalar ABI, receipt schemas and artifact rosters are unchanged.

## Qualification and completion scope

Revalidation after the repository-layout merge passes **144 tests, no skips** in
`.scratch/resume-conditional-current1.log` against Merlin `dd7b62801` and
model2MLIR `f3f0de1`. The selection includes grid/branch/increment proofs,
all conditional kernel profiles and the complete ten-mode invariant/conditional
model suite, including host reuse, original-domain refusals and actual serial
images. This validates the new consumer owners without changing historical
counts into arbitrary-kernel coverage. The recorded source and target policies
remain independently selected.

The source proof tests exercise actual region identity, branch-local division,
checked divisibility, different-base refusal, absent facts, independent narrow
carrier/header widths and unchanged original unsafe-definition refusals. Kernel
tests use independent NumPy references across all four memory/address profiles,
row/column storage and i32/i64 controls. Every original result remains observed.

`scripts/capture_invariant_stride_model.py --conditional-mode
scalar|pointer|pure|observe|nested|bit` captures two dependent real Triton calls with
learned matrix state and four live scalar-tensor controls. Independent Torch
references retain all eleven model results, original parent archives, both
callsites and changed-invocation state. The same model driver keeps the original
invariant variants as separate regressions. Host, target and source-domain
refusals are measured separately; serial execution supplies no accelerator or
performance verdict.

The first five-mode kernel selection passes 65 in
`.scratch/conditional-stride-kernels1.log`: ten libraries, seven changed
invocations each and twenty actual serial images. The combined model selection
passes 24 in `.scratch/conditional-stride-models1.log`, qualifying five new
models and the original invariant `late` regression separately. Each model
retains all eleven results, three changed invocations, two serial images and
an actual original-domain abort without outputs.

The Boolean selection passes twelve in `.scratch/conditional-stride-bit1.log`:
eight all-profile native checks and four complete-model checks. Four separate
checks in `.scratch/conditional-stride-bit-libraries1.log` qualify its two
standalone libraries, changed invocations and four serial images. The earlier
27-check final safety selection passes in `.scratch/conditional-stride-safety4.log`.
The final extended source/control/memory/attention selection passes 188 in
`.scratch/conditional-stride-regression2.log`. Counts overlap and must not be
added into a unique-coverage total.

After the final grid-uniformity and Boolean refinements, fresh conversion in
`.scratch/conditional-stride-grid-equivalence2.log` reproduces all eighteen
qualified kernel/model implementation payloads byte-for-byte. The unchanged
payloads retain their native/host/serial evidence; the new conflicting-write
cases separately exercise the stronger admission gate. Selected source hashes
and byte equality are not a complete build dependency closure.

The initial proof selection retains two obsolete refusal expectations for
bounded varying increments. Corrected tests still reject unproved ranges and
unbounded i64 samples, and require a varying-state classification for a checked
source IV. The first through-loop ownership fixture reaches an earlier unsupported
comparison-to-integer range case; that case is now independently implemented and
tested, alongside a min-based fixture that reaches the intended ownership gate.
Original failed probes/logs remain separately from passing qualification.

`out/conditional-runtime-strides/summary.json` reopens the twelve kernel and
six model libraries: ninety initial results, 366 replay results and 180 exact-word
target results across thirty-six actual serial images, plus six actual model-domain
aborts. Original parent archives, both callsites, actual source header SSA,
generated compatibility and target receipts remain checked. The extended reopening
log is `.scratch/conditional-stride-report2.log`; the earlier five-mode reopening
in `.scratch/conditional-stride-report1.log` retains its separate original scope.
New helper/test/source Ruff and diff checks pass. Existing style findings in the
large shared modules remain outside this change.

Reproduce with the existing source/native/model/Merlin/serial tool variables
documented in the README:

```bash
python -m pytest -q tests/test_grid_uniformity.py tests/test_source_branch_offsets.py \
  tests/test_invariant_recurrences.py tests/test_conditional_stride_kernels.py \
  tests/test_invariant_stride_model.py \
  --basetemp=.scratch/pytest-conditional-runtime-strides
```

General pointer-base selection, while-carried and nonlinear address relations,
broader loaded/varying state and alias compositions, dynamic-root projections,
variable-step complete writes, shared/asynchronous participation, broader numeric/
source versions and applicable accelerator correctness/performance remain open.
This extension changes no historical opcode-name denominator or arbitrary-kernel
completion claim.

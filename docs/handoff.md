# Review handoff — updated 2026-11-20

[Independent runtime memory strides](invariant_runtime_strides.md) now close
the separate-runtime-increment refusal as tensor computation. Four modes retain
body memory, preloaded reuse, mixed step/progress and body-local division with
valid zero-trip entries. Eight reusable kernel libraries and four complete
two-call PyTorch models qualify host replays and twenty-four serial images,
retaining every result and source-domain refusal. Kernel checks pass 51;
combined model/source checks pass 62; earlier source/control checks pass 159;
the final invariant/width selection passes ten, with overlap. General pointer/
alias/dynamic-root/write-image, shared/async, numeric/source and accelerator
closure remains open. See the linked contract for the explicit representation.

[Actual iteration ordinals](iteration_ordinals.md) now retain pointer
increments of the form `a * actual_step + b`, including actual final positions.
Twelve reusable kernel libraries and three new dependent-call PyTorch models
qualify through host replays and thirty actual serial images. All ten results
per model survive. `out/ordinal-pointer-streams/summary.json` reopens 54 initial,
234 replay and 108 exact-word serial results, plus three invalid-step aborts.
Kernel checks pass 75, model checks twenty (including two original models), and
related source/control/memory regressions 137. No companion source edit is needed.
General runtime-stride/conditional/while/alias, dynamic-root, shared/async and
applicable source/numeric/target closure remains open.
The final reopening log is `.scratch/ordinal-pointer-report2.log`; fourteen
selected producer files match `b55fc70d05d37fd840ad9bc0155cedf26679251a`.
Unchanged companion pins remain model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2`
and Merlin `53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. This is not a full dependency closure. The next
reproduced boundary is `.scratch/runtime-pointer-runtime-stride1/`: a separate
bounded runtime memory stride. Source words pass four native cases; structured
profiles retain the upstream boxing refusal. Preserve the real invariant scalar
and exact carried logical offsets when extending that representation.

[Pure loop exit coordinates](pure_loop_exits.md) now close the saved
final-pointer-only refusal by binding actual cloned result SSA through the
existing checked initial/yield relation. Original loops, preloaded tile reuse,
word state and final observations remain. Four new reusable kernel libraries
and a complete dependent-call PyTorch model preserve all results through host
replays and ten actual serial images, including all ten model results and an
invalid-step abort. The focused selection passes 39 and the extended source
selection twelve with overlap; eight model checks qualify new and original
forms separately. `out/pure-loop-exits/summary.json` reopens 18 initial, 78
replay and 36 exact-word serial results. No companion source change is needed;
the general pointer/ownership/effect/source/target endpoint remains open.
The final reopening log is `.scratch/pure-loop-exit-report2.log`; fifteen
selected sources match `7a4fb747093d9a5c05fb8e787e22c764b4a7b6fe`. The
unchanged companion pins remain model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2`
and Merlin `53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. This is not a full dependency closure.
The corrected related source/control/memory regression selection passes 119
in `.scratch/pure-exit-regression2.log`. The next actual probe is
`.scratch/runtime-pointer-independent-step1/`: constant pointer advancement
independent of the runtime loop step. Guarded source words pass four native
cases; structured selections retain their upstream boxing refusal. Preserve
actual iteration-count/state semantics while extending that representation.

[Runtime pointer streams](runtime_pointer_steps.md) now preserve actual
source-step coupling, forward/reverse row/column traversal and final pointer
observations through tensor MLIR. Four reusable kernel libraries and a complete
dependent two-call PyTorch model retain all results through host replays and ten
actual serial images. All ten model results survive; an invalid live step
actually aborts without output. The focused selections pass 59 source proofs,
27 kernel checks, four negative checks (three overlapping) and four model checks;
the related regression selection passes 159. New-file Ruff and diff checks pass.
Logs and historical failed source/harness probes remain in the linked contract.
`out/runtime-pointer-steps/summary.json` reopens 18 initial, 78 replay and 36
exact-word serial results across ten images, plus the model abort. Its kernel
packages come from the subsequent eight-check compatibility/host/target run;
the reopening log is `.scratch/runtime-pointer-report5.log`. Thirteen selected
producer sources match implementation `5d9623573b54686385e83442ae9bd84d06a2334e`;
unchanged companion pins are model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2`
and Merlin `53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. This is not a full dependency closure.
No companion source change is required. Continue conditional/non-step-coupled
pointer state, while/alias ownership, dynamic roots, variable-step store images,
shared/async and applicable numeric/source/target closure; the full goal is open.
The next actual refusal is `.scratch/runtime-pointer-final-only1/`: removing
body loads leaves a pure numeric loop whose actual exit result currently lacks
coordinate facts after cloning. Preserve that loop and its final reads while
extending the same executable exit-edge proof; the linked contract records it.

[Variable source loop headers](runtime_headers.md) now retain actual
lower/upper/positive-step SSA, row/column tiles and partial-tile recurrences as
tensor MLIR. Six kernel libraries and a dependent two-call PyTorch model pass
host replays and fourteen actual serial images, retaining all seven model
results. General pointer/ownership, effects, source/numeric and accelerator
closure remains open.

`out/runtime-headers/summary.json` reopens six kernel libraries and one complete
model library: 15 initial, 69 unchanged-library replay and 30 exact-word target
results across fourteen actual serial images. Both original model callsites,
all parent archives, the original domain abort and three actual compiled header
aborts remain checked. Final logs are `.scratch/runtime-headers-proofs2.log`
(103), `.scratch/runtime-headers-final1.log` (45),
`.scratch/runtime-header-guards2.log` (three),
`.scratch/runtime-header-projection2.log` (one), and
`.scratch/runtime-headers-regression2.log` (133). The reopening log is
`.scratch/runtime-headers-report2.log`; original failed regression/baseline runs
remain separate historical evidence.

Producer `182eafa780ea2f9d70b175f41f388b86b54f2a1a` retains unchanged
model2MLIR `f3f0de1ae130faebbd27bd36374d11e1e211daa2` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`. Thirteen selected producer sources match committed
bytes, separately from observed reopening revisions. This is not a full
dependency closure or accelerator/performance qualification. Continue general
pointer recurrences, while/alias ownership, variable-step store images,
dynamic root projection and shared/async/source/numeric compositions.

[Wholly nonexecuting source loops](empty_loop_memory.md) now retain
complete original initial tuples and incoming tensor versions without body
read/write credit. Four kernel libraries and a dependent two-call PyTorch model
pass native/host checks and ten actual serial images, retaining all seven model
results. Signed shape-derived controls such as `2 - n` now cross model2MLIR as
checked i64 SSA. General memory/effect/source and accelerator closure remains open.

`out/empty-loops/summary.json` reopens four kernel libraries and one complete
model library: 15 initial, 69 unchanged-library replay and 30 exact-word target
results across ten actual serial images, plus the original model-domain abort
and independently guarded native-continuation abort. All parent archives and
actual source-loop links remain checked. The reopening log is
`.scratch/empty-loop-report1.log`. The source/domain, kernel and model selections
pass 83 + 31 + 4; the separate composition regression passes 36. Two historical
memory expectations reproduce at the old baseline; corrected checks pass in
both source states, and the original failed logs remain explicit.

Producer `e670c8aadfb74a7da26737bd5361b30a59a502dd` retains model2MLIR
`f3f0de1ae130faebbd27bd36374d11e1e211daa2` and unchanged Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both companions on
`ttir-experiment-grammy`. Eleven selected producer and two companion sources
match committed bytes. These pins are not a full dependency closure or
accelerator/performance qualification. General while/alias ownership, pointer
recurrences, variable-step store images, shared/async and source/numeric composition remain.

The historical boundary probe is the
[selected-row sum with runtime loop headers](empty_loop_memory.md#next-concrete-representation-boundary)
under `.scratch/runtime-loop-headers1/`. Guarded source-word conversion passes
four native cases, one initial host invocation, four unchanged-library replays
and two actual serial images. Its original tensor/structured-address refusals
are now closed by [the runtime-header implementation](runtime_headers.md).
The original finite prototype artifacts remain separate from that new shipped
capability and its qualification; neither establishes arbitrary-loop coverage.

[Functional regions in tile memory](region_tile_memory.md) now retain reductions,
coupled tuples, forward/reverse scans and scalar map loops inside actual partial-tile
dataflow. Seven kernel libraries and a complete dependent-call PyTorch model pass
native/host and sixteen actual serial images, retaining all seven model results. The
focused selections pass 38 + 28 + 4; related regressions pass 212. General
memory/effect/source and accelerator closure remains open.

`out/region-tiles/summary.json` reopens seven kernel libraries and one complete
model library: 21 initial, 105 unchanged-library replay and 42 exact-word target
results across sixteen actual serial images, plus one original-domain abort.
Every parent archive and actual original/normalized source-region link remains
checked. Definition and callsite IDs retain their actual opcodes separately.
The reopening log is `.scratch/region-tiles-report1.log`.

Implementation `9218ad836c1a2be7c1a1be120c82cc3aa7c7092c` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`. All twelve selected
implementation/test/helper sources match committed bytes. These hashes remain
separate from observed revisions and are not a full dependency closure,
accelerator or performance qualification. No companion source change was
required for this capability.

[Loop-carried partial tiles](loop_tiles.md) now retain original loop bounds, numeric
recurrences and selected-path observations as tensor computation. Actual source
fallback writes preserve nonpositive-trip paths without narrowing entry domains. Five
kernel libraries and a complete two-call model qualify native/host execution and
twelve actual serial images, retaining all seven model results. The related regression
selection passes 221 checks. General memory/effect/source and accelerator closure
remains open.

`out/loop-tiles/summary.json` reopens five kernel libraries and one complete
model library: 17 initial, 81 unchanged-library replay and 34 exact-word target
results across twelve actual serial images, plus one original-domain abort.
All parent archives and source store/read/loop correspondence remain checked.
The reopening log is `.scratch/loop-tiles-report1.log`.

Implementation `df4248f929bfa4bc918eaa4d69bbc8736e3e992c` retains companion
model2MLIR `a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba`, both on
`ttir-experiment-grammy`; the producer remains on `main`. All eleven selected
implementation/test sources match actual committed bytes. These hashes are
recorded separately from observed revisions and are not a full dependency
closure, accelerator or performance qualification. No companion source change
was required for this capability.

[Branch-completed tiles](branch_tiles.md) now close the scalar branch-boundary
tensor refusal. Prefix observations, nested/absent-else choices, two fresh roots
and branch-local division retain original execution. I64 data results use the
existing checked prepass option while ordinary memory remains visible; carried
pointer/mask state retains the original path. The recorded focused runs pass
37 + 38 + 4; earlier source/tile/while/effect regressions pass 191.

`out/branch-tiles/summary.json` reopens eight kernel and one model libraries,
23 initial, 117 replay and 46 exact target results across eighteen actual
serial images, plus an original-domain abort. All seven complete-model results
and parent archives remain. Final logs are `.scratch/branch-math2.log`,
`.scratch/branch-kernels1.log`, `.scratch/branch-model1.log`,
`.scratch/branch-regression1.log` and `.scratch/branch-report1.log`. The earlier
structured-state verifier failure and invalid negative fixture remain recorded;
neither is relabeled as a successful source/target qualification. General
pointer/loop/alias, shared/async, numeric/source-version and applicable target
requirements remain open.

Producer `3c09f8e6015b0334841c61b2fb12a28566c1e3b7` retains companion
model2MLIR `a3ce8ed` and Merlin `53e7b0115` on `ttir-experiment-grammy`.
All ten selected implementation/test sources match committed bytes; the
reopening report records that match separately from observed revisions.
These hashes are not a full dependency closure. The producer remains on
`main`, and no companion source change was required for this capability.

[Interleaved fresh tiles](interleaved_tiles.md) now close the distinct-output
refusal while retaining cross-output observations and predicates, immutable
reloads and original input-state updates. The 33-check focused selection and
132 earlier source/tile/memory/attention regressions pass. Reopening
`out/interleaved-tiles/summary.json` retains 36 initial, 158 replay and 72 exact
target results, fourteen actual serial images, and an original-domain abort.
All eleven complete-model results and five state-kernel results remain.

Final logs are `.scratch/interleaved-final1.log`,
`.scratch/interleaved-regression1.log` and `.scratch/interleaved-report1.log`.
The first model attempt's missing serial helper import remains recorded; the
final run fixes the test harness and actually executes both model images.
The original tensor-only refusal in `.scratch/branch-tile-probe2/source.ttir`
crossed a scalar branch boundary. The branch-completed implementation above now
closes that form. Its original failed source remains separate from successful
packages and does not measure every other profile. General pointer/loop,
shared/async, source/numeric and target requirements remain open.

Producer `6716ba902f6519703133081de99d0dda19c6e884` retains companion
model2MLIR `a3ce8ed` and Merlin `53e7b0115`, both on
`ttir-experiment-grammy`. All six selected source files match committed bytes;
the report keeps that verification separate from its observed reopening
revisions. These hashes are not a full dependency closure. The producer stays
on `main`; companion source changes were unnecessary for this capability.

[Fresh tile partitions](tile_partitions.md) now close the reproduced fresh
masked-tile refusal while retaining parallel structure, overlap order and
intermediate observations. A complete two-call model preserves every result
through host/replays and actual serial images. General groups across memory
events/regions, dependencies, shared/async and accelerator closure remain open.

The final focused tile/source/model selection passes 53; the separate earlier
source/scatter/fresh/attention/conditional selection passes 114. Reopening
`out/tile-partitions/summary.json` checks 24 initial, 92 replay and 48 exact-word
target results across fourteen actual serial images, including both model
domain aborts. One scalar model is regression evidence; the other retains the
new parallel tiled calls. Final logs are `.scratch/tile-partitions-final2.log`,
`.scratch/tile-partitions-regression1.log` and
`.scratch/tile-partitions-report3.log`. Next extend actual disjoint-root facts
through `.scratch/interleaved-tiles-probe1/source.ttir`, whose complete writes
to two fresh outputs still encounter the collector's memory-event boundary.
That failed source supplies no successful package or target qualification.

Tile implementation `f271d7ab1b3127aebb033ce241451f7fccb23d9e` retains
model2MLIR `a3ce8ed` and Merlin `53e7b0115` on `ttir-experiment-grammy`.
All nine selected implementation/test sources match actual committed bytes;
the report records that check separately from the original reopening revisions.
This is selected evidence, not a full dependency closure. The producer stays
on `main`; no companion source change or branch rename was required.

[Source branch/mask partitions](source_partitions.md) now close the actual
both-arm final-coverage and complementary-mask local-read refusals. Source
choices, order, fills, reuse and observations survive complete model host/serial
execution. All seven original model results remain. Continue vector-lane,
numeric predicate, while/ownership, shared/async and applicable target closure;
the full goal stays incomplete.

Producer `658c2265d2531228922fe2f0aa9e0cdf399961ab` retains companion revisions
model2MLIR `a3ce8ed` and Merlin `53e7b0115`, both on
`ttir-experiment-grammy`. The focused recorded runs pass 66 + 4 + 1; the
earlier memory/control/model regression selection passes 259. Reopening
`out/source-partitions/summary.json` checks all 27 initial, 121 replay and
54 target results, 22 actual serial images, one original-domain abort and
twelve selected source files against committed bytes. No full dependency
closure or accelerator/performance qualification is claimed. Original logs
are `.scratch/source-partitions-{final1,float1,float-gap1,regression1}.log`;
the reopening log is `.scratch/source-partitions-report2.log`.

[Partial source initialization](partial_initialization.md) now closes completed
row reads and preserves strided column/plane observations, point updates and
reread recurrences through standard tensor MLIR. All nine complete-model results
qualify through host/replays and actual serial images. The 59-check source/model
selection and eleven math/state checks pass with overlap. General vector/masked/
conditional initialization, ownership/effects and accelerator closure remain open.

Implementation `b077a2b` additionally passes 209 earlier memory/control checks;
the final package selection passes 24 after the compatibility note. The report
`out/partial-initialization/summary.json` reopens 39 initial, 177 replay and 78
exact target results across 22 actual serial images and an original-domain
abort. Eleven selected files match committed bytes. The original source
refusals remain in `.scratch/partition-initialization-probe1/False/source.ttir`
(both branch arms initialize each element but final coverage refused) and its
`True` sibling (complementary masked stores lacked local read credit). The
source partition implementation above now closes those two forms. Their
original failed probes supply no successful package or target qualification.

[Rectangular source-loop coverage](nested_output_coverage.md) now retains fresh
results from nested/permuted traversals and complete branch arms. The original
two-call PyTorch model, runtime parameter slices and all seven results qualify
through host/replays and actual serial images. Static empty ports retain exact
types and active-access guards. Continue partial row-local/masked/conditional
initialization, while/ownership relations, irregular parallel dependencies,
shared/async effects and applicable targets; full coverage remains incomplete.

Implementation revisions are producer `7c9f0a9`, model2MLIR `a3ce8ed` and
Merlin `53e7b0115`. The 51 focused checks plus one original-domain check,
188 overlapping producer regressions, 150 consumer checks with two skips and
68 model2MLIR checks pass. The reopened report is
`out/nested-output-coverage/summary.json`: 25 initial, 91 replay and 46 exact
target results, eighteen actual serial images, six active-access aborts and
two model-domain aborts. Both companion repositories retain branch
`ttir-experiment-grammy`; the producer remains on `main`.

[While-carried word indices](while_word_ranges.md) now admit actual i32/i64
ports by their complete word domain, retaining original modular updates and
post-loop values. Other address arithmetic still needs its own source checks.
The 24 focused checks cover four native/host/serial kernel forms, five actual
active-index aborts, and a complete two-call model with all eleven results.
Modulus 15 distinguishes original wraparound from widened induction; the i64
`2**32 + 3` refusal distinguishes full-width active bounds from truncation.

Merlin `9dbb83df744fb655936b2179ed84163087fde399` closes actual `tensor.dim`
dependencies. `83dc7e3a002c9c596de105900c7f3fd1c6113a8e` permits an independent
private seed from a closed completed serial loop value. Actual carried edges,
fresh seeds and every operation are checked; unused calls, mutable reads/writes
still refuse. The native destination/ownership/access selection passes 337 with
two skips and repository structure passes. Earlier source-range, dynamic-prefix
and complete-model lifetime failures remain recorded. Diagnostic probes stop
before bufferization and publish no qualification. Continue relational range/
arithmetic proofs, partial/nested initialization, resizing/permuted owners,
parallel dependencies, shared/async effects and applicable accelerator closure.

Producer `c5c64009876e2fd3c38551763b1a4637587c7d05` passes 24 focused and 166
overlapping source/memory/control checks against the current consumer. The
source-pinned reopening report is `out/while-word-ranges/summary.json`: five
host packages retain 23 initial and 93 replay results, and ten actual serial
images retain 46 results. Five native bounds violations actually abort without
outputs. All original model-parent bytes and every result are checked. Final
logs are `.scratch/while-word-final1.log`, `.scratch/while-word-regression2.log`
and `.scratch/while-word-report1.log`. Selected hashes are not a full dependency
closure; the historical operation-name denominator and performance claims remain
unchanged. Full arbitrary-kernel coverage remains incomplete.

[Source while memory](while_tensor_memory.md) now preserves source conditions,
both tensor-state regions and final false-edge effects. Real tiled i64/f32
kernels retain reuse, masks, empty/zero trips and earlier snapshots. The actual
condition-memory kernel retains its counter update and final observation.
A complete two-call PyTorch model retains every one of its fourteen results,
learned counter words, original parent archives and intermediate snapshots
through host replays and two actual serial images.

Merlin `b6d664a9ec5462f21e22ee0c51b9858eee315354` adds typed initial-condition
peeling and late completed-value copies. Unknown semantic loop attributes keep
their original control; normalization grants no ownership or effect permission.
The native destination/deallocation selection passes 165; broader ownership
passes 163 with two skips and repository structure passes. The earlier printer
assertion failure and test-driver import failure remain in their original logs;
neither indicates missing source effects or qualifies as a successful target run.
Next extend while-carried source ranges, partial/nested fresh initialization,
resizing/permuted owners and broader parallel/effect/target composition. The
full goal remains incomplete; public ABI, receipts and signatures are unchanged.

Producer `1d4e8a1d40926b5066e3238d18ad25485d831ed7` passes 25 focused and 150
memory/control/pointer/attention checks. `out/while-tensor-memory/summary.json`
reopens five host packages, all original model-parent bytes, 28 initial and
126 replay results, plus 56 results from ten actual serial images. The report
records selected source hashes and exact revisions, not a full dependency
closure. Final logs are `.scratch/while-final1.log`, `.scratch/while-regression1.log`
and `.scratch/while-memory-report1.log`. The source/consumer attribute refusals
and fresh zero-trip initialization refusal remain explicit checks. Selected
Ruff passes; shared tensor code retains its five existing baseline diagnostics.

[Branch-local source ranges](branch_source_ranges.md) at `60f3bad` now retain
actual scalar signed/unsigned comparison constraints and necessary boolean
implications. Original definitions still precede narrower use facts. Source
operations, conditions and both state observations remain; no annotations
grant facts. Six source forms retain 48 direct native results. Four unchanged
Merlin libraries retain 48 replay results, and six actual serial images retain
twelve results. Two configurations admit the full signed i32 divisor domain
and unrestricted allow words, including negative extremes and zero.

The focused selection passes 66, plus one advisory-tag refusal and two full
signed-domain native/host/serial qualifications. Memory/control passes 109 and
existing pointer/events 23, with overlap. Two public domain checks reject before
execution; two separate native runs abort at serialized executable scalar
guards. Earlier incorrect test expectations are retained, not relabeled.
The original native structured attempt remains recorded. Only actual residual
source loads/stores select the existing event adapter; successful structured
tiles keep their previous conversion and proofs. Public ABI, receipts, artifact
rosters, effect profiles and Merlin signatures are unchanged.

Final logs are `.scratch/branch-ranges4.log`, `.scratch/branch-memory2.log`,
`.scratch/branch-event-regression2.log` and `.scratch/branch-full-signed1.log`.
The additional advisory check is `.scratch/branch-event-regression1.log`.
Reopening passes in `.scratch/branch-range-report1.log`; evidence is
`out/branch-source-ranges/summary.json`. Continue varying-
value relations, disjunctive/while constraints, partial/nested initialization,
irregular parallel ownership and broader numeric/storage/effect and applicable
target coverage. This is no new complete PyTorch model or accelerator/
performance qualification. No Merlin/model2MLIR source change is needed; the
foreign model2MLIR docstring edit remains. Full arbitrary-kernel coverage is open.

[Source loop-body ranges](source_scope_ranges.md) at `fe65631` now close the
rank-three shared matrix-view form's tensor refusal. Original definitions are
validated before narrower use facts; actual signed product-bound reachability
proves local positivity without excluding zero at entry. Original division,
remainder, loops, update order and reused bias remain. Public types, receipts,
artifact rosters, effect/ownership profiles and Merlin signatures are unchanged.
LLVM remainder inference attribution and distribution licenses are retained.

The source/domain selection passes 39, memory regression 91 and complete-model
selection 30, with overlap. All six models select tensors, retaining 52 initial,
156 replay and 52 actual serial target results. Thirty rehashed claims refuse
in both readers and six original domains abort without results. A nontrivial
i64 matrix computation retains six exact native results and six results from
one reused Merlin library, including three zero-sized shapes. Final logs are
`.scratch/source-scopes9.log`, `.scratch/scoped-memory-pinned1.log` and
`.scratch/scoped-models-pinned1.log`; reopening evidence is
`out/source-scope-ranges/summary.json`. The corrected replay-test API error and
earlier test-driver failures remain in their original logs.
Reopening passes in `.scratch/source-scope-report2.log`. Original rank-three
typed source operations remain identical after excluding locations; its
emitted memref count drops from 51 to zero with four tensor insertions.

Next extend actual conditional/while constraints and partial/nested fresh
initialization, then irregular parallel ownership and general storage/effect
composition. Reuse `source_index_ranges`' checked comparison/dominance work
where applicable; sequential assertions need instruction-position facts rather
than indiscriminate block-wide assumptions. Nested range-audit positives do not
qualify nested output ownership. Accelerator correctness/performance and full
arbitrary-kernel coverage remain open. No Merlin/model2MLIR source change is
needed for this capability; the foreign model2MLIR docstring edit remains.

[Serial scalar tensor memory](serial_tensor_memory.md) at `1ceb878` now closes
the five shared-view forms' guarded-memory requirement. Source masks, ordering,
loops, bias reuse, initialization and native lifetime checks remain. Actual
typed source operations are unchanged in the retained matrix comparison, while
emitted memref operations drop from 49 to zero. Tensor runtime checks cannot
authorize the separate unguarded buffer continuation; compatibility says so.

Source-frozen memory/model selections pass 91/30, with overlap. Five tensor
models retain 44 initial, 132 replay and 44 actual target results; 25 changed
claims refuse in both readers and five original domains abort. The sixth,
rank-three guarded source-word model retains its own 8/24/8 results, five
refusals and domain abort. Native primitives retain 49 results over thirteen
invocations and three actual bounds aborts, including exact nonfinite copies.
`out/serial-tensor-memory/summary.json` reopens all source-pinned evidence.
No Merlin/model2MLIR source change is needed; the foreign m2m docstring edit
remains. The subsequent source-loop proof above closes the concrete local
range/reachability gap without strengthening valid empty entry domains.
Continue broader irregular parallel ownership/initialization,
storage/effect composition and applicable accelerator qualification. Full
arbitrary-kernel coverage and performance remain open.

[Dense mutable storage permutations](storage_permutations.md) now close the
reported `x.T.clone()` refusal and the rank-three source-stride importer gap.
Model2MLIR `3ebbe89` proves exact zero-origin dense order, reuses standard
permutation lowering, restores complete updated bases and checks actual source
shape ancestry. Existing functional storage/package/consumer profiles remain.
Two complete models retain sixteen initial, forty-eight replay and sixteen
actual serial results. Ten rehashed claims refuse in both readers and two
original domains abort. The final new-model selection passes ten, legacy matrix
one, model2MLIR regression 84 plus two revised unit-axis cases, and independent
Merlin contracts 27, with overlap. `out/storage-permutations/summary.json`
reopens source-pinned packages, archives, outputs, targets, refusals and aborts.
The foreign model2MLIR docstring whitespace remains outside the commit.
Continue hidden/general storage, metadata/list state, public aliases/copyback,
broader source/numeric/effect composition and applicable accelerator coverage;
full arbitrary-kernel coverage stays open.

[CFG program phases](cfg_program_phases.md) now close the reproduced original
early-return/barrier and complete-void allocation-wrapper refusals. Both readers
derive scalar whole-program participation from actual CFG edges and block
argument types; original phase payload and memory order remain. Core changes
are pushed: triton2MLIR `b5b0385`, Merlin `5cb916ef2` and model2MLIR `67abffb`.
The latter closes surrounding dynamic `full_like` initialization with actual
source dimension operands and existing literal conversion. Remote tips and
GitHub author/committer attribution were verified for each.

Final model selection passes ten. Complete i64/f32 models retain 24 initial,
72 replay and 24 actual serial-target results, with all twelve results and every
parent archive intact. Six rehashed model claims refuse and two original public
domains abort. Native/direct/protected regression passes 62, independent Merlin
336 and model2MLIR fill/pointwise/literal regression 56, with overlap. The static
fill differential oracle also lowers with matching result types; dynamic model
execution is separately checked. `out/cfg-program-phases/summary.json` reopens
source failure/positive evidence, complete results, refusals/aborts and current
reader pins. The pre-existing model2MLIR docstring whitespace edit is preserved.
Continue broader CFG/type/control/effect composition, general shared/async
participation and applicable accelerator qualification; full coverage stays open.

[Actual word-source effect admission](word_source_effects.md) now closes the
legacy early-return gap in both complete memory gates. Producer selection
passes 26 and independent Merlin passes 861. Six consistently rehashed real
source changes refuse in both readers; six full staging cases publish nothing.
The separate historical audit admits 293 actual word contexts from 352 packages.
Complete mixed-effect models retain 26 initial, 78 replay and 26 fresh serial
results, with all original parent archives intact. Public CLI binding retains
26 further results; twelve changed model claims refuse and two source domains
abort. Eleven model checks pass with one expected skip. Record reopening found
the separate execution-owned builder's omitted reader pin; both actual target
test bodies additionally pass against newly built images with the exact pin.
Earlier images remain retained in their original scope, not relabeled.
`out/word-source-effects/summary.json` reopens the before/after probes, finite
audit and current source-pinned model/target evidence. Continue shared/async
participation, broader composition and applicable accelerator work.

[Composed atomic/volatile effects](atomic_volatile_effects.md) now preserve
both original effects through protected sources and complete i64/f32 models.
All 26 initial, 78 replay and 26 actual serial model results retain exact words;
twelve changed model claims refuse and two source domains abort. The actual
public CLI global/override paths retain 26 further host results. The source
selection passes 21, two additional component checks pass, legacy frontend
selection 155, complete models ten, default model regressions two and independent
Merlin selections 467/206, with overlapping scope. Five new target images and
all evidence reopen in `out/atomic-volatile-effects/summary.json`.
Continue remaining memory/numeric/storage composition and general
shared/asynchronous participation.
Full arbitrary-kernel and accelerator closure remain open.

[Volatile program phases](volatile_program_phases.md) now preserve actual
private observations through original allocation source and complete i64/f32
models. All 18 initial, 54 replay and 18 actual serial model results retain exact
words; 22 changed source/effect/phase claims refuse in both readers and two
original shape domains abort. The source/ordinary/atomic selection passes 54,
new complete models ten, ordinary model regressions two and independent Merlin
selection 459, with overlap. `out/volatile-program-phases/summary.json` reopens
the four selected new images and full evidence. Continue mixed atomic and broader
numeric/storage/effect composition, general shared/async participation and
applicable accelerator qualification. Full arbitrary-kernel coverage stays open.

[Program barriers](program_barriers.md) now retain source-verified CTA phases
through ordinary shared conversion. A real runtime-loop mutable tile transpose
qualifies in tensor and guarded forms through native, reusable Merlin host and
two serial Spike images. Original scopes, zero trips, tails and conflict gates
remain. Protected allocation source and complete three-call models now retain
original phase custody, independent source/payload/order checks and every result.
Continue with multi-CTA participation and broader shared/async/atomic effects;
the historical inventory is unchanged.

[Shared mutable views](mutating_views.md) now retain slice/as-strided aliasing,
complete bases, actual coordinates and original TTIR pointer offsets through
four two-call models. Source-only automatic selection chooses guarded structured
memory; original loops/reuse/snapshots remain. Final qualification evidence is
under `out/mutating-views/summary.json`: 36 initial, 108 replay and 36 target
results, twenty rehashed refusals and four actual domain aborts. Continue with general storage layouts,
optional/list/metadata state, public alias/copyback and broader effect/target scope.

[Functional mutation](functional_mutation.md) now has three actual two-call
Triton models, complete initial/reference state archives and explicit updated
state ports. Producer and independent Merlin checks bind each mutation to the
same source readwrite buffer. All twenty initial, sixty replay and twenty actual
Spike results retain exact words; twelve rehashed claims refuse and two source
domains abort. `out/functional-mutation/summary.json` reopens that finite evidence.
Next implement nontrivial mutated views/shared bases and public invocation
copyback/alias semantics; these are not supplied by state-result annotations.

[Rank-reducing views](rank_reduced_source_views.md) now retain selected unit
axes and scalar observations through complete four-call i64/f32 models. Every
snapshot, pointer root and original loop remains independently checked. The
same libraries retain all seventy-eight changed-invocation results. General
view/ownership/effect and applicable accelerator scope remains open. Two actual
Spike images retain all twenty-six results; eight rehashed rank/guard claims
refuse in both readers. See `out/rank-reduced-source-views/summary.json`.

[Runtime source views](runtime_source_views.md) now connect complete i64/f32
dynamic-matrix models through explicit view profile v3. All eighteen initial
and seventy-two replay results retain exact source words; twelve rehashed
coordinate/profile/pointer claims refuse in both readers. The source algorithm,
root ABI, snapshots and guards remain. Two actual Spike images retain all eighteen
results. `out/runtime-source-views/summary.json` retains the separate evidence.
General view/effect/target scope stays open.

[Executed source grouping](executed_source_layouts.md) connects captured source
options to execution-owned models. The combined grouped-state/compaction example
retains seventeen results and explicitly selects finite resource policy v2. The
full arbitrary-kernel goal remains open; keep default resource and source guards.

[Runtime dense views](runtime_tensor_views.md) now retain actual dimension
operands, checked index products and standard reshapes through execution-owned
models. A real dynamic-matrix count/scan/scatter model preserves all seven
results across host replays and three actual target images. Runtime non-dense
coordinates, remaining result-axis/source-layout bindings and broader effects
and accelerators remain open.

[Executed call views](executed_call_views.md) now connect selected dense/source
adapters to the execution-owned package path. Original coordinates, root pointer
contracts, every schema operand and complete emitted SSA remain checked.
The combined five-kernel source retains all nine results, including prior
snapshots and content-sized scan/scatter output. Fully runtime coordinates,
physical mutation, source-layout options and broader effects/targets remain.

[Runtime returned-loop ownership](runtime_loop_ownership.md) now closes the
fresh-generation runtime/zero-trip refusal through the ordinary independent
checker. Initial uses, ownership flags, actual zero-trip copies and tuple result
coordinates remain checked. Native/saved/C and actual target execution preserve
changing/empty results and source guards; independent i32/f64 buffers keep exact
nonfinite words. General branch/while/permuted ownership, dynamic steps, retained
aliases, other final consumers and broader view/effect/target work remain open.

[Execution-owned target packages](execution_owned_target_packages.md) now
connect the real three-kernel model to source-only signature-v6 staging, the
ordinary Merlin builder and actual observed-shape Spike execution. Four qualified
images check all 28 original/changed-invocation results. Source-domain and wrong
reference cases cannot publish success; retained verification independently
replays staging and complete input/driver/wire/reference checks. The source
package and original archives remain unchanged. Broader view/resizing/numeric/
effect/executor/accelerator scope remains open; this is no full-coverage claim.

[Production generated owned-result invocation/readback](owned_result_driver.md)
now executes the real three-kernel model LLVM with separate runtime input bytes.
Three input capacities reuse their generated executables across four content/
parameter patterns each; every one of the 84 results matches exact words.
Two invalid capacities still abort in original source guards before publication.
Merlin's 63-check driver/readback/legacy transport selection passes. The generic
controller also checks cleanup after descriptor/allocation/observer failures and
preserves another caller's busy context. The selected serial target package
integration now has separate whole-model evidence above.

Merlin `bb347b1de` separates owned-result lowering from host linking, and
`f2552dec1` adds explicit serial context/external failure reporting. Both are
pushed on `ttir-experiment-grammy`, with tips and attribution verified. The final
152 runtime/allocator/cross/native/persistence checks pass. Two source-frozen
Spike image rechecks are retained under `out/build/pytest-owned-images-retained/`:
one reports nine exact standard-MLIR results over three calls, and one exits on
allocation exhaustion without results. Source/tool/image/output digests reopen.
These are historical runtime-seam fixtures; the production v6 TTIR driver/readback
now has separate selected whole-model qualification above.
The frontend now pins the common lowerer and its real-TTIR host/C regression
selection passes eight in `.scratch/compact-c-lowering1.log`. Follow the
[target integration plan](execution_result_targets.md) for the actual next work.

`out/owned-result-target-seams/summary.json` reopens the two evidence scopes,
including all frontend results/aborts and actual retained runtime image/tool/
output identities. Successful pytest temporaries are normally removed by Merlin;
the image recheck explicitly uses `-o tmp_path_retention_policy=all` under the
project-owned output root. No unowned scratch path is needed.

Merlin `0606fd4f1` now adds [checked generated-C result calls](checked_c_results.md)
on `ttir-experiment-grammy`; its remote tip and `grammy117` author/committer
attribution are verified. Its final 115 C/allocator/native/persistence checks
pass, as do regenerated documentation and structure gates. The frontend now
pins the new generator and C runtime/header in selected consumer fingerprints.
The real TTIR compaction model additionally passes eight standalone-C checks:
six input sets/twelve calls check all 84 results, and two source-domain failures
abort. `out/compact-model-c/summary.json` reopens this evidence. Staged driver
generation, target allocator context/linking and production shape readback
remain the next actual v6 target work; no Spike execution is claimed.

Published follow-through: triton2MLIR `59237b7` adds explicit automatic memory
selection and `b77da69` qualifies real count/scan/scatter compaction. Both are
on `main`; remote tips and GitHub author/committer attribution were checked.
That selection used Merlin `e943745a3`; its checked C successor is above.
Model2MLIR remains `863d797` on `ttir-experiment-grammy`, with the pre-existing
whitespace edit preserved.
The initial [target boundary](execution_result_targets.md) probe recorded the
legacy Spike driver's receipt-v2 refusal before staging or image creation.
That historical probe is superseded by the explicit execution-owned serial
target path above; default legacy receipt requirements remain unchanged.

[Real TTIR compaction](compaction.md) now reaches the existing qualified
execution-sized package and Merlin host interface from Python and public CLI.
All three kernels and seven original model results remain. Count/copy select
tensors; pack retains five parallel Linalg scan stages and masked indexed
stores in guarded source-word memory, with original structured-range refusals
recorded by [explicit automatic selection](automatic_memory.md).

The final four compaction integration checks pass in
`.scratch/compact-final5.log`. `out/compact-model/summary.json` reopens two
packages/fourteen initial results, six library replay input sets/42 results,
three valid standalone sizes and one actual unwritten-result abort. The
automatic-selection/native/legacy memory selection passes 46 checks in
`.scratch/auto-final1.log`; the earlier overlapping policy/model selection
passes 24. Source/memory/effect/numeric/ownership checks remain authoritative.
V6 generated-C/serial Spike now have the separate qualification above. Stronger
scan range proofs, general views/resizing/effects and arbitrary source/target
coverage remain open. Historical inventory counts
are unchanged. This adds no GPU or performance equivalence claim.

The source-to-package-to-Merlin pipeline works for the qualified selections.
**Full arbitrary-Triton-kernel coverage remains incomplete.** This is a review
boundary requested by the user, not a completion claim.

## Follow-through since the November 12 review boundary

[Internal computed sizes](executed_model_binding.md) now reach qualified
packages and the source-only Merlin interface. Receipt v2 retains v4 call
contracts and located kernels; independent complete-SSA/result-type/location
reconstruction precedes signature v6 staging. All 21 initial PyTorch results
and 105 results across fifteen published-library replays match in the three
memory profiles. Six actual source-domain failures abort without outputs.
Staging succeeds with evaluator members removed; a contradicted original
reference produces actual host results but leaves no published package.

The final new/legacy producer selection passes 49 checks in
`.scratch/executed-package-final5.log`. Independent Merlin staging/correspondence
passes 56 in `out/build/execution-model-staging-final.log`. Selections overlap.
`out/executed-model-packages/summary.json` reopens the complete package/result,
replay, abort, source-only stage and failed-publication evidence and verifies
current consumer source hashes. The historic 352-package inventory is unchanged.
Qualification explicitly names Merlin host with the original PyTorch oracle;
the separate native evaluator, generated C/target harnesses, v4 views/layouts,
runtime resizing ownership and broader source/numeric/effect compositions remain.

[Execution-sized results](execution_owned_results.md) now connect explicit
Merlin signature v6 staging to complete model/TTIR host invocation in all three
memory profiles. Saved compiler programs reopen fixed checked artifacts without
compilation. All 21 original PyTorch results match; eighteen further host replays
check 126 results with independent lengths/counts/parameters. Twelve actual
source-domain failures abort without outputs and three false rehashed result
declarations refuse. Standard-MLIR compaction also executes after staging and
reopening, with dimensions discovered from actual vector contents.

`out/execution-result-models/summary.json` reopens every selected initial result,
replay and abort. `.scratch/execution-model-final1.log` passes 47 with one optional
feature import skipped; Merlin's `out/build/execution-staging-final1.log` passes
355 and `execution-staging-final2.log` passes ten with overlap. The native stage
fragment composition fix is `54b5a5c22`. Original capture/package receipts and
default signatures remain unchanged. Arbitrary internally computed producer/
binding shapes, normal qualification receipts, native/generated-C/target
evaluators and runtime/zero-trip generation proofs remain further work.
The staged/runtime implementation is Merlin
`99c96b1066f0e3f974da753f9d6decab2b121fad` on `ttir-experiment-grammy`.
The final `.scratch/execution-result-model-final2.log` passes all twelve selected
model checks after source freeze. The summary reopens that selection and checks
every interface consumer-source hash against the current implementation.

[Runtime input-word shapes](input_word_shapes.md) retain real scalar input data
and source guards through Triton result allocation, dependent calls and a
stepped slice. Model2MLIR `863d797` supplies executable integer reads/guards;
source-only shape proofs and explicit Merlin signature v5 size runtime outputs.
Read-mask intersections retain Meta/shared's exact SSA minimum and independent
bounds without write-coverage permission. General data-dependent computation,
runtime bounds/views, resizing, effects and targets remain completion work.

[Derived model dimensions](derived_model_shapes.md) now qualify a real shared-axis
broadcast/stepped-slice model and two dependent Triton calls through native,
Merlin host and Spike in all three memory profiles. Typed shape programs and
independent source proofs supply signature v4 result allocation. All seven
results survive; twelve library replays check 84 results and nine invalid
source-domain invocations abort without outputs. Model2MLIR `4b0e7bd` retains
actual symbolic arithmetic SSA. Broader dynamic views, resizing state,
data-dependent results, effects and targets remain active completion work.

[Dynamic surrounding arithmetic and slicing](dynamic_surrounding.md) now closes
the three reproduced integration failures recorded below. Model2MLIR `9e2b646`
emits actual runtime broadcasts and clipped positive-step slices. The real
addition/two-dependent-copies/empty-call model retains all five results through
native, Merlin host and actual Spike in all three memory profiles. Twelve
unchanged-library host replays check sixty complete results with changed lengths
and parameter words; six out-of-domain calls abort through original guards.
The final seventeen primitive emitter checks additionally execute sixty native
results and two incompatible-broadcast aborts, including signed-i64 extreme
slice bounds/step. An initially missing wide-division runtime helper is removed
by proved bounded length arithmetic; earlier failing logs remain retained.
The final focused selections pass 17 primitive cases and three complete models
with seven unrelated cases deselected. `out/dynamic-surrounding/summary.json`
reopens all target/native/replay/guard receipts and original parent archives.
The earlier broad selection retains 160 passes, one skip and the now-fixed
extreme-slice failure. It is not relabeled an all-green run.
Model2MLIR's 274 broader passes with sixteen skips and 46 final symbolic/domain
passes overlap. General conditional/derived shape signatures, runtime view/bound
bindings, effects and targets remain open. Source word/shape facts still come
from actual computation and independent consumer reconstruction.

Complete [dynamic model packages](symbolic_model_packages.md) now extend the
native-only binding below. Copy/weighted-XOR models qualify all three memory
profiles through native, Merlin host and actual Spike execution, with all 21
complete initial results checked at each boundary. Twenty-four further host
invocations vary lengths and parameter words without changing libraries or
programs; six invalid lengths abort through the actual source guards.
`merlin.compile-signature.v3` explicitly binds source-proved public input axes.
Merlin independently reopens original parent calls, kernel destinations,
complete inlined bodies and scratch upper bounds. Runtime generation preserves
captured parameter bytes; reusable host calls accept independent parameters.
False rehashed result sizes refuse before allocation/library invocation.
The final 88 focused checks pass across `.scratch/pytest-symbolic-package-final4/`
and its source-frozen tensor-copy Spike recheck. The initial in-flight source
edit was correctly refused and remains recorded. Reopened receipts and complete
word comparisons are in `out/symbolic-model-packages/summary.json`. Merlin's
matching implementation is `2b32726086`. Its 355 frontend/runtime, 71 final
runtime and 146 layout/access passes overlap; two unavailable environment
checks skip. Documentation and structure gates pass.
That earlier broader producer selection retained 141 passed, three surrounding-
arithmetic/slice failures and one unselected optional target case skipped in
`.scratch/symbolic-package-final3.log`; it remains a historical failing run.
Those three failures are now closed by the source computation above.
Broader surrounding dynamic computation, derived/data-dependent dimensions,
mutation/effects and applicable targets remain incomplete. These packages are
not yet folded into the historical 352-package inventory.

[Symbolic tensor binding](symbolic_tensor_bindings.md) now proves actual kernel
input/destination/result relationships across a real two-call exported copy
model. All three memory profiles retain 36 complete exact native results at
independent empty/singleton/tail lengths, with three source-domain guard aborts.
The original and complete inlined shape contracts agree; sampled lengths supply
no compiler fact. The final selection passes 32 tests, including the existing
i32 model's native/Merlin-host/Spike regression; the broader overlapping producer/
whole-model selection passes 163. `out/symbolic-model-binding-native/summary.json`
reopens that earlier source/native evidence. That selection added no qualified
package or dynamic Merlin library. The earlier full-writer refusal retained in
`.scratch/symbolic-model-package-probe1/` is superseded by the complete package
work above; the surrounding-arithmetic integration is now qualified separately.

[Checked index arguments](checked_index_arguments.md) now recover unsigned
enclosures from original dominating standard assertions and checked round trips,
then emit a representable `arith.index_castui` without choosing index width.
Producer verification and Merlin reconstruct the exact profile/adapter/range
record; source guards remain executable. The fresh probe under
`.scratch/symbolic-model-boundary3/` retained the prior narrowing-planning success
and symbolic tensor-dimension refusal, now superseded by the binding above. Public dynamic model allocation/signatures,
reusable execution and broader semantic/target closure remain unfinished. These
source-proof cases add no TTIR inventory or complete model qualification.
The final producer selection passes 100 tests in
`.scratch/checked-index-final3.log`; the independent Merlin selection passes
425 with two unavailable environment checks skipped. Native checks retain three
valid controls and four actual guard aborts; compiler protocol fixtures retain
evaluator exclusion and profile/range tamper refusals. Counts overlap.
Merlin's implementation is `50873d7cc`; the original i32 runtime-control model
also passes native/Merlin host/Spike, retaining all five results and four
unchanged-library host replays under its existing live profile.

Executable [entry shape domains](symbolic_models.md) now preserve actual source
ranges, shared axes and affine inverse/divisibility relations using standard
unsigned casts and interval-sized integer checks. Dynamic channel sums retain
their Linalg bodies and static retained axes. Five real configurations pass
fifteen valid native runs and fifteen actual guard aborts; the final focused
source-domain/axis/native selection passes 41 tests. The broader source selections and final
seventeen dimension contracts pass separately with overlap retained. Merlin's
reader/staging/layout/access selection passes 410 with two unavailable checks
skipped. That guarded capture in `.scratch/symbolic-model-boundary2/` retained
the earlier index-narrowing refusal, superseded by the source proof above. Public model
allocation/signatures and reusable consumer/target execution remain unfinished.
No new TTIR inventory or bound-model qualification is claimed by the native
domain selection.
The source-domain producer is model2MLIR `088496f`; the scalar cast reader is
Merlin `4ca31cc24`. Native receipts reopen in `out/shape-domain-native-axis/summary.json`.
The explicit runtime-shape v2 axis profile also executes dynamic-result identity
calls without extra scalar inputs. It retains v1 checks and exact typed caller
validation; deriving an independent model compiler signature remains required.

Model2MLIR `f85f949` now preserves backed symbolic dimensions through shared
literal-aware shape helpers. Its relevant selection passes 134 tests with 26
unrelated cases deselected; the final twelve dimension contracts pass. Only
the intended importer hunk was committed, preserving the unrelated whitespace
edit. The [symbolic model contract](symbolic_models.md) records the next actual
boundary: the fresh real Triton copy model now reaches the writer's dynamic
argument-materialization requirement after checked binding. The existing tensor-profile shape
integration is an open failing
integration, not a passing refusal case. Source-domain checks, symbolic binding,
runtime allocation and independent Merlin signatures/execution remain required.
Earlier 51-case live-argument evidence used model2MLIR `5a681ea` and static
public types; its retained packages do not qualify the new symbolic capture path.

[Live runtime arguments](live_runtime_arguments.md) now retain original scalar
SSA and singleton control tensors alongside captured geometry. Eleven complete
models pass native/Merlin host/Spike with 57 exact-word results, 32 further host
invocations and two compiled range aborts. The final 51-case producer/model
selection has no skips; the separate 174-case model/view/bundle selection passes.
Merlin's independent staging/layout/access checks pass 446 with two unavailable
bubblewrap/release-wheel cases skipped. These counts overlap. Both frontends
reopen original parent facts; no prior kernel inlining, sampled input or
provenance grants scalar/shape permission. Public model types stay static.
Remaining symbolic/mutating model ABIs and applicable target scope remain in
the completion plan. Generated receipts reopen under `out/live-runtime-models/`.
The latest `out/live-runtime-model-inventory/coverage.json` reopens 352
packages, 278 standalone qualifications and fourteen product cases, retaining
all fifty names under selected native/Merlin-host scopes.

[Region-local pointer seeds](integer_address_regions.md#pointer-seeds-defined-inside-a-region)
now recover actual outer roots and recompute source-width byte/divisibility
facts. Independent per-seed retention preserves unaligned typed pointers while
other required recoveries stay enabled. Ten native-verified saved-TTIR libraries
and actual Spike images retain fourteen exact results, 48 further host invocations
and four actual bounds aborts. `out/pointer-address-seeds/summary.json` reopens
the receipts; `out/pointer-address-seed-inventory/coverage.json` reopens 341
packages and 278 standalone qualifications. Broader root/region/alias/effect,
dynamic model/source and applicable target closure remains open.
The final 52-case contract/address-region/complete-model selection passes with
no skips and twelve intentionally deselected region-width cases. It includes
all four existing complete address-state models. The retained/recovered seed
receipts use the final compiler source; earlier selections remain separately
scoped. No Merlin/model2MLIR source edits were needed for this extension.

[Integer-address region state](integer_address_regions.md) now closes the
reproduced integer-address for-loop refusal. Actual logical pointer fields pair
with modular byte state across for/while/if interfaces, preserving original
numeric/control/effect computation. Sixteen source captures pass native/Merlin
host/Spike with all 32 results exact and eighty further host replays. Unknown
physical, mixed-state, root/alias/effect and applicable target forms stay open.
One native-verified saved-TTIR while fixture additionally executes independent
input/result ports and actual loaded condition state. The complete selection
has 17 libraries/Spike images, 33 exact results, 83 further host invocations and
four compiled bounds aborts. `out/integer-address-regions/summary.json` reopens
the receipts; `out/integer-address-region-inventory/coverage.json` reopens 327
packages and 268 standalone qualifications. The 57-case address/model regression
selection and updated 23 focused contracts pass; their counts overlap.
The compiler implementation is `534ecedd1d997c54a85c1c216bb5a05dcb6b4c81`,
pushed on `main` with verified GitHub author/committer attribution. Four complete
PyTorch model configurations subsequently retain all 44 original results through
native/Merlin host/Spike. Source wrappers clone incoming state, preserving the
functional public interface and earlier snapshots. Parent archives and static
public model ABI stay unchanged. `out/integer-address-models/summary.json`
reopens those separate target receipts.
`out/integer-address-model-inventory/coverage.json` reopens 331 packages,
268 standalone qualifications and fourteen product cases; all fifty source names
remain observed under selected native/Merlin-host scopes. This is composition
evidence, not arbitrary-kernel or accelerator completion.

[Same-base pointer differences](pointer_addresses.md#same-base-byte-differences)
now retain cancelling SSA roots, source byte units and runtime memory/control.
Seven native/Merlin-host/Spike libraries retain eleven exact-word results, with
32 further host invocations and four actual bounds aborts. That initial
`out/pointer-difference-inventory/coverage.json` reopens 304 packages and 247
standalone qualifications. Twelve focused difference contracts and the 27-case
address selection pass. Unproved absolute/root/region/effect and applicable
target forms remain in the completion plan.
The core change is `288586781e2ed71b374bacebbbbc89a5a52a1f00`. Four further
width cases retain overflowing i32 increments before byte scaling, with four
exact results and sixteen host replays. Two complete relative-pointer models
also retain all twenty original results through native/Merlin host/Spike.
The latest `out/pointer-difference-width-inventory/coverage.json` reopens 310
packages and 251 standalone qualifications. The reproduced integer-address
`scf.for` refusal is now closed by the paired region-state work above; see the
[address contract](pointer_addresses.md#complete-model-qualification-and-next-region-boundary).

[Local tensor regions](tensor_regions.md) and complete learned-state models
now retain functional tensor intermediates, for/while state and mixed nested
containers through native/Merlin host/Spike. Merlin's
`56b938a9196728bc249fd51b4f5efe1971e1cadc` protects captured writable destinations
at its actual native seam. Triton2MLIR commits `05cda56`, `7dd011b` and `dbbfdf4`
retain the frontend and complete-model qualification; the original 42 model
results match exact words across six configurations.

[Numeric region builtins](numeric_regions.md) additionally close mapped
histogram/dot refusals through the existing numeric converters and contracts.
Uniform logical queries reach the actual enclosing runtime grid without
choosing physical placement or moving surrounding effects. Thirty standalone
configurations pass native/Merlin host/Spike: 24 exact-word and six
tolerance-qualified math results, with 120 further host replays. The frontend
change is `2aca8b92b56ed2abaf4881e5ee3a8f7418adbc34`. Six complete numeric-region
models additionally retain all 39 original results as exact words through
native/Merlin host/Spike, with parameters, dependent calls and snapshots intact.
That earlier `out/numeric-region-model-inventory/coverage.json` reopens 297 packages, 240 standalone
qualifications and fourteen product cases. The 32 focused contract and 187
runtime/grid/external/original-effect regression selections pass.

Current branches remain `main` for triton2MLIR and `ttir-experiment-grammy` for
Merlin/model2MLIR. Remaining tensor/type/effect/runtime/shared-state/model and
applicable target work follows [the completion plan](completion_plan.md).
The sections below retain the original November 12 slice and its evidence;
their counts and commit table are historical.

## Repositories and publication

| Repository | Branch | Relevant implementation |
|---|---|---|
| `/scratch/grammy/blast/triton2MLIR` | `main` | `b72caed1e800c024811d57be2f4ca1831d178930` — located kernel retention; the following package commit adds receipt v2 creation/verification, actual qualified host execution, CLI/coverage support and this handoff update. |
| `/scratch/grammy/blast/merlin-graph-proof-intake` | `ttir-experiment-grammy` | `e943745a3dacac22f7d9071b1772527551c9635a` — independent v4 package staging; `2963c543e757a2afad3e3788b34970e7b974a049` — complete-SSA/result-type/location reconstruction. |
| `/scratch/grammy/blast/model2MLIR` | `ttir-experiment-grammy` | `863d797` — runtime integer item reads and guards; no new source edits in the executed-call slice. Preserve the unrelated pre-existing whitespace edit in `m2m/ir/import_fx.py`. |

The unrelated dirty `model2MLIR/m2m/ir/import_fx.py` is preserved. Its SHA-256 is
`8d4000b64f18802086bcf55325f21e268f795df4ccaefea40266e7d23a7a1379`.
Do not reset or include it in these changes. Continue the verified Graham Anderson
Git identity and ordinary incremental commits; no co-author trailers or history rewrites.

## What changed

- `t2m/scalar_tile_reads.py` reuses shared's singleton projection and the existing
  guarded gather lowering. Captured tensor values become standard element reads.
  Shared/conditional projections retain one lookup at the original gather position.
- Reduction/scan checks admit exact typed reads and literals. Only guard operation
  identities created by that transform receive generated-guard admission. Tags,
  locations and serialized copies cannot grant it. Original assertions and hidden
  pointer-memory effects retain their source audit.
- Lookup family, original source IDs, logical shape/axis and source layout hints
  remain annotations. Bounds, values and control are executable without them.
- Merlin constructs an independent initial-state copy only after closing its
  producer/capture proof. Closed padding supplies a readable value, not ownership.
  Guards, zero-trip behavior and native alias/lifetime gates remain enabled.
- Public package ABI, artifact rosters and receipt schemas are unchanged.
  [Package differences](package.md) and generated compatibility metadata state
  the representation. [The design/evidence contract](scalar_tile_reads.md)
  describes the hardware/software boundary.

## Evidence to inspect

The combined inventory reopens **237 packages, 192 standalone qualifications
and fourteen product cases**. All fifty pinned TTIR definition names occur in
selected native/Merlin-host-qualified captures. This measures opcode breadth;
it establishes neither universal operation support nor a completion percentage.

The new state selection has **24 libraries and 24 actual Spike images**, with
**48 complete exact-word results and 96 additional host runtime invocations**.
Each kernel loads a table, mutates its source storage, then uses the old snapshot
in runtime reduction/scan recurrence. Both returned results are checked: the
changed table and output. Cases cover i32/i64, both table axes, three memory
profiles, empty/tail inputs, zero steps, overflow and different runtime tables.

VM-local generated evidence, intentionally ignored by Git:

- `out/tile-lookup-state/summary.json` — target receipts and runtime counts.
- `out/tile-lookup-inventory/coverage.json` — explicit combined package selection.
- `.scratch/pytest-tile-lookup-state2/` — the 24 complete captures/libraries/images.
- `.scratch/tile-read-regressions3.log` — **177 passed**, one upstream deprecation warning.
- `.scratch/tile-read-final.log` — **21 passed**, covering guard identity, annotation
  independence, a current-package target case and actual bounds aborts inside both
  combiners in an existing Merlin library.

Merlin's destination selection passed **135 tests** after the padding fix;
structure and documentation gates passed. Earlier ownership/deallocation checks
also passed, separately. Test selections overlap; do not add them as unique coverage.
There are no known unresolved failures in the listed final selections.

## Reproduction on this VM

Use the existing environments and a fresh project-local pytest basetemp.
No toolchain rebuild, install, GPU or Triton-CPU dependency is required.

```bash
cd /scratch/grammy/blast/triton2MLIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_SHARED_OPT="$PWD/.scratch/toolchain/triton-shared-opt"
export T2M_NATIVE_PYTHON=/scratch/grammy/blast/model2MLIR/.venv/bin/python
export T2M_MERLIN_ROOT=/scratch/grammy/blast/merlin-graph-proof-intake
export T2M_MERLIN_PYTHON=/scratch/grammy/blast/merlin/.venv/bin/python
export T2M_MERLIN_CLANG=/scratch/grammy/blast/merlin/third_party/llvm-build/bin/clang-23
export MERLIN_COMPILER_PYTHON="$T2M_NATIVE_PYTHON"
export MERLIN_CLANG="$T2M_MERLIN_CLANG"
export MERLIN_MLIR_INSTALL=/scratch/grammy/blast/merlin/third_party/llvm-build
export MERLIN_RISCV_GCC=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc
export MERLIN_NM=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm
export MERLIN_SPIKE=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/spike

"$T2M_MERLIN_PYTHON" -m pytest -q tests/test_scalar_tile_read_contract.py \
  tests/test_scalar_tile_reads.py --basetemp=.scratch/pytest-review-contracts
"$T2M_MERLIN_PYTHON" -m pytest -q tests/test_tile_lookup_state.py \
  --basetemp=.scratch/pytest-review-table-state
```

Keep execution serial on this shared VM. Missing environment variables can skip
integration tests, so inspect the final pass/skip summary. The 24-image selection
took about four minutes with the existing tools.

## Limits and next work after review

These are native-verified **typed TTIR fixtures**, not new Python AST captures,
GPU execution/correspondence evidence or a performance benchmark. No new whole-model
capture was added for this lookup family; existing model integration evidence
remains separately scoped. Floating/other-width lookup compositions and general
tensor intermediates are not qualified by this selection.

Keep the full [completion plan](completion_plan.md) active after resumption:
close remaining builtin/region and typed capture forms; give direct memory,
shared/external effects and synchronization concrete semantics; extend physical/
mutating views, runtime/alias/CFG/value-returning interfaces; cover supported source
versions and applicable accelerator materialization, correctness and performance.
Do not erase difficult source operations or count opaque fallbacks as completion.

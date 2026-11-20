# Roadmap: Triton kernel to a Merlin capture package

[Loop-carried partial tiles](docs/loop_tiles.md) now retain original loop bounds,
numeric recurrences and selected-path observations as tensor computation. Actual
source fallback writes preserve nonpositive-trip paths without narrowing entry
domains. Five kernel libraries and a complete two-call model qualify native/host
execution and twelve actual serial images, retaining all seven model results. The
related regression selection passes 221 checks. General memory/effect/source and
accelerator closure remains open.

[Branch-completed tiles](docs/branch_tiles.md) now close the reproduced scalar
branch-boundary refusal. Actual selected paths retain prefix observations,
conditional arithmetic and output coverage; ordinary memory remains visible
when numeric data interfaces use the checked native prepass option. Continue
general pointer/loop/effect dependencies and applicable target qualification.

[Interleaved fresh tiles](docs/interleaved_tiles.md) now close the reproduced
two-output refusal. Actual fresh-root disjointness permits independent groups
while current-SSA proofs retain cross-output data/control dependencies and
ordinary input-state ordering. Continue general address and region/effect
dependencies, shared/async participation and applicable targets.

[Fresh tile partitions](docs/tile_partitions.md) now re-raise complementary,
three-way and overlapping same-address writes into source-ordered selects with
their actual covered store. Intermediate observations retain their masks/fills;
logical parallel tiles and reuse remain. Native pointer conversion and ordinary
geometry/ownership checks stay active. Continue region-spanning memory groups,
general dependencies/effects and applicable target qualification.

[Source branch/mask partitions](docs/source_partitions.md) now close the
reproduced both-arm whole-output and complementary-mask read refusals. Actual
SSA Boolean covers retain nested alternatives, overlap order and early masked
observations. Row/column reuse and all seven complete-model results reach host
and serial execution. Continue vector-lane and numeric predicate relations,
while/ownership invariants, irregular parallel dependencies, shared/async effects
and applicable targets; the arbitrary-kernel endpoint remains incomplete.

[Partial source initialization](docs/partial_initialization.md) now closes the
reproduced completed-row read refusal, plus strided column/plane, immediate
read-after-write and reread-loop forms. Each active scalar read needs a source
image membership witness; complete output and parallel ownership remain separate.
All nine results of the two-call model qualify through host and serial execution.
Continue vector/masked/conditional partitions, while/ownership relations,
irregular parallel dependencies, shared/async effects and applicable targets.

[Rectangular source-loop coverage](docs/nested_output_coverage.md) now extends
fresh-output initialization through nested/permuted traversals, unit/empty axes
and independently complete branch arms. A complete two-call model retains
runtime parameter slices, reuse and every result through host and actual serial
execution. Continue partial row-local/masked/conditional initialization,
while/ownership relations, irregular parallel dependencies, shared/async effects
and applicable target closure; the arbitrary-kernel endpoint remains incomplete.

[While-carried word indices](docs/while_word_ranges.md) now use actual full
storage-width enclosures while retaining original modular state and final
counter values. Standard tensor memory handles dynamic prefix loops and
wrapped/direct lookups; source arithmetic, active bounds and lifetime checks
remain. A complete dynamic two-call model preserves every result. Continue
relational while facts, richer arithmetic/casts, partial/nested initialization,
resizing/permuted ownership, irregular parallel dependencies and broader effects
and applicable targets. This does not close arbitrary-kernel coverage.

[Source while memory](docs/while_tensor_memory.md) now preserves tensor state
through original conditions and bodies, including zero trips and final false
effects. Reused tiles, conditional updates, distinct numeric interfaces and
all fourteen results of a complete two-call model survive host/serial execution.
Continue loop-carried address/range proofs, partial/nested initialization,
resizing/permuted ownership, irregular parallel dependencies and broader effects
and targets; this does not close arbitrary-kernel coverage.

[Branch-local source ranges](docs/branch_source_ranges.md) now qualify selected
signed/unsigned conditions, true conjunctions and false disjunctions through
tensor MLIR and native/host/serial execution. Original-definition checks and
complete state observations remain; full signed i32 entry domains qualify.
Native residual memory uses the existing event adapter after the structured
path, retaining successful tile/reuse proofs. Continue varying-value relations,
disjunctive and while-carried domains, partial/nested initialization, irregular
parallel ownership and broader source/effect/target coverage.

[Source loop-body ranges](docs/source_scope_ranges.md) now close the rank-three
matrix-view divisor refusal without narrowing valid zero-sized entry domains.
Definition-scoped checks precede body-only positivity facts; source arithmetic,
loops and state reuse remain. Continue general conditional/while constraints,
partial/nested initialization, irregular parallel ownership and broader
source/numeric/effect/consumer coverage.

[Serial scalar tensor memory](docs/serial_tensor_memory.md) now raises selected
ordered state updates and complete fresh scalar loops into standard tensor
versions. Preserve this path's source proofs and native lifetime checks.
Product-bounded signed-loop reachability now has the separate proof above.
Continue broader irregular parallel ownership and partial initialization
without discarding source loops or reuse.

[Dense mutable storage permutations](docs/storage_permutations.md) now preserve
zero-origin transpose/rank-three roots and shared matrix views through complete
models. Standard normalization/inverse operations retain logical results and
snapshots. Continue general layouts/hidden storage, optional/list and metadata
state, public copyback/aliases and broader source/effect/target composition.

[CFG program phases](docs/cfg_program_phases.md) now retain original scalar
early exits, runtime loops and actual synchronized tile state through native
and Merlin execution. Complete void allocation entries retain their CFG behind
the typed wrapper. Continue broader control/effect and consumer/target closure.
Complete i64/f32 models now retain all twelve results through host replays and
actual serial images. General shared/async and accelerator qualification remain.

[Actual word-source effect admission](docs/word_source_effects.md) now closes
the legacy-stage primitive-selection gap in both readers. Continue general
shared/asynchronous participation, memory/numeric/storage composition and
applicable consumer/target materialization; arbitrary coverage remains open.

[Composed atomic/volatile effects](docs/atomic_volatile_effects.md) now retain
both original effects through protected sources and complete i64/f32 models.
All thirteen model results survive host replays and actual serial images;
public CLI binding carries global/per-kernel f32 policy. Continue general
memory/numeric/storage composition and
shared/asynchronous participation before applicable accelerator qualification.

[Volatile program phases](docs/volatile_program_phases.md) now compose private
observations with synchronous barriers through original allocation sources and
complete i64/f32 models. All nine results, runtime shapes and earlier snapshots
survive host replays and actual serial images. Continue mixed effect/numeric/
storage composition, shared/async participation and applicable accelerators.

[Volatile reads](docs/volatile_reads.md) now have an explicit private-memory
profile with actual standard LLVM effects and source-preserving normalization.
Native i32/i64/f32/f64 and selected host/serial cases retain unused observations.
Original allocation intake and complete i64/f32 models now preserve all fourteen
results per model through host replays and actual serial images. Continue
atomic/barrier, source/numeric/storage/effect compositions, general external/
shared/async participation and applicable accelerator qualification.

Selected [atomic program phases](docs/atomic_program_phases.md) now retain
original old values and synchronized observations through native, reusable
Merlin host and complete i64/f32 models on serial Spike. Independent source and
model refusal checks retain all thirteen results per model and actual source
guards. Continue general shared/async effects, participation scopes and remaining
source/consumer compositions; arbitrary-kernel coverage remains incomplete.

[Synchronous program barriers](docs/program_barriers.md) now retain source
address-space masks and actual phase dependencies through a real runtime-loop
tile exchange in tensor and guarded-memory forms. Native/host/serial-target
checks pass. Protected allocation source and complete three-call models now use
the same phases with independent typed-source/payload/order checks. Multi-CTA
and general shared/async/atomic compositions remain completion work.

[Shared mutable views](docs/mutating_views.md) now retain complete base state,
actual coordinates and original TTIR pointer relationships through four real
two-call models. Preserve overlapping update order; continue with general base
layouts, optional/list state, public alias/copyback and broader effect/target scope.

[Functional mutation](docs/functional_mutation.md) now retains ordinary returns,
complete updated bases and public input/buffer state ports through actual Triton
model packages, Merlin host replay and three Spike images. Continue with general
mutated views/shared bases, physical aliases/caller copyback and broader effects.
The original borrowed-input and lifetime gates remain active.

[Rank-reducing views](docs/rank_reduced_source_views.md) now retain actual
selected axes, distinct unit strides and scalar results through complete
i64/f32 learned-state models. Original pointer roots, masks, loops and all
thirteen results remain. Continue with broader view/ownership/effect and target scope.

[Runtime source views](docs/runtime_source_views.md) now retain actual dynamic
transpose/slice coordinates, executable scalar agreement and canonical empty
branches through complete learned-state models. Independent Merlin reconstruction
keeps original pointer roots, snapshots and all results. General view/ownership/
effect/target closure remains open.

[Executed source grouping](docs/executed_source_layouts.md) now reuses original typed
source-option proofs under v4 and keeps consumer placement independent. The combined
grouped-state/compaction model additionally selects an explicit finite resource profile;
arbitrary view/effect/target closure remains open.

[Runtime dense views](docs/runtime_tensor_views.md) now retain actual dimension
operands, checked index products and standard reshapes through execution-owned
models. A real dynamic-matrix count/scan/scatter model preserves all seven
results across host replays and three actual target images. Runtime non-dense
coordinates, remaining result-axis/source-layout bindings and broader effects
and accelerators remain open.

[Executed call views](docs/executed_call_views.md) now connect existing checked
dense/source adapters to execution-owned results. Complete view-update models
and a combined count/scan/scatter model retain original snapshots and internally
computed sizes. Fully runtime reshape/source coordinates, physical/mutating
views, source-layout options and broader ownership/effect/target closure remain.

[Runtime returned-loop ownership](docs/runtime_loop_ownership.md) now closes
the fresh-generation runtime/zero-trip refusal. Native/saved/C and actual target
checks preserve changing/empty results and source guards; tuple i32/f64 owners
retain independent sizes and exact storage. General branch/while/permuted owners,
dynamic steps, aliases/views/effects and other final consumers remain work.

The [execution-owned target package](docs/execution_owned_target_packages.md)
now connects source-only v6 staging, the ordinary Merlin builder and observed
shape readback to actual whole-model Spike execution. Four qualified images
retain all 28 results; source-domain and reference failures publish no success.
Retained verification replays independent staging and complete source/input/wire
checks. Broader resizing/views, numeric/effect compositions and targets remain.

The [source-typed owned-result driver](docs/owned_result_driver.md) now closes
production hosted invocation and observed-shape transport for the real TTIR
compaction model. Three unchanged drivers retain all 84 results across twelve
calls; two original source-domain failures publish no results. The 63-check
Merlin driver/readback/legacy transport selection passes. Its selected serial
target integration now has the separate complete-model evidence above.

[Target runtime seams](docs/execution_result_targets.md) now lower returned
descriptors independently of host linking and explicitly select serial context
and failure reporting. A real Spike fixture retains nine content-derived results
and an allocation-failure image refuses. Production v6 package staging, generated
invocation and complete readback now have separate serial target qualification.

[Checked C result transport](docs/checked_c_results.md) now implements the first
execution-sized target prerequisite: actual source execution, complete returned
descriptor/owner validation and independent copies. Real TTIR model results pass
from C; selected serial target generation/linking/readback now have separate evidence.

[Real compaction](docs/compaction.md) now extends execution-sized packages to
actual parallel scan/indexed-store filtering with dependent consumption and
all original model results. The mixed tensor/guarded payload and original host
qualification are explicit; selected v6 target execution is now qualified too.
General effect/shape/target closure remains.

[Explicit automatic memory selection](docs/automatic_memory.md) now chooses
the first admitted tensor, guarded logical or guarded source-word conversion.
Source/ABI admission and recorded diagnostics decide representation; numeric,
effect, ownership and target requirements remain unchanged.

The [overall status](docs/status.md) separates established end-to-end paths,
remaining architectural work and evidence limits. Full arbitrary-kernel coverage
is active and incomplete; observing all pinned opcode names is not completion.

[Internal computed sizes](docs/executed_model_binding.md) now qualify the real
count-producing model through receipt v2, independently reconstructed v4 call
binding and source-only signature v6 staging in all three memory profiles.
Every result survives published-package reopening and changed-content library
replay. Failed original reference comparison prevents publication. Continue
with separate native/target evaluators, v4 views/layouts, resizing ownership
and the complete numeric/memory/effect/source scope below.

[Execution-sized results](docs/execution_owned_results.md) now have explicit
signature v6 staging and reusable Merlin host invocation of complete two-call
models in all three memory profiles. Original source/input constraints, guarded
allocation products and complete returned values survive. Arbitrary internally
computed producer/binding shapes, native/target evaluator and qualification
receipt extensions, and runtime/zero-trip generation proofs remain open.

[Runtime input-word shapes](docs/input_word_shapes.md) now retain actual scalar
input reads and source guards for Triton result allocation, under model profile
v3/runtime v4/program v2 and Merlin signature v5. Exact intersected read masks
keep original SSA and independent bounds without write-coverage credit. Continue
with general data-dependent computation, runtime views/bounds, resizing state,
effects and applicable targets.

[Derived model dimensions](docs/derived_model_shapes.md) now reach the same
complete consumer path in all three memory profiles. Shared axes, conditional
broadcast sizes and clipped slice counts retain actual typed shape programs;
Merlin independently proves source relationships and publishes signature v4.
Unchanged libraries handle new lengths/weights, and invalid source domains
abort. Continue with runtime bounds/views, data-dependent sizes, resizing state,
arbitrary effects/compositions and applicable targets.

[Dynamic surrounding computation](docs/dynamic_surrounding.md) now closes the
reproduced addition/slice failures in all three memory profiles. Standard
source-derived extents retain complete native/Merlin-host/Spike results,
unchanged-library runtime shapes and real source guards. Continue with general
conditional/derived shape signatures, runtime bounds/views, effects and targets.

[Symbolic model intake](docs/symbolic_models.md) now preserves actual `?` public
dimensions and executable entry-domain checks after the model2MLIR correction.
Five dynamic reduction configurations retain fifteen native results and fifteen
compiled guard aborts. [Checked index arguments](docs/checked_index_arguments.md)
now recover representable unsigned ranges from dominating source assertions;
Merlin reconstructs the exact scalar proof independently.
[Symbolic tensor binding](docs/symbolic_tensor_bindings.md) now proves actual
destination/result relationships across the two-call copy model in all three
memory profiles. Twelve native invocations retain all 36 results, with three
source-domain aborts. Complete [dynamic model packages](docs/symbolic_model_packages.md)
now additionally qualify copy/weighted variants through native, Merlin host and
Spike. Explicit input-axis compiler signatures, source-bound scratch, complete
result allocation and unchanged-library invocation work for this selection.
Broader derived dimensions, dynamic surrounding computation/views, mutation/
effects and applicable targets remain active work; earlier static qualification
keeps its original scope.

[Live runtime model arguments](docs/live_runtime_arguments.md) now retain actual
scalar SSA and singleton tensors, including exported `tensor.dim` operands and
data-controlled address loops. Eleven complete models pass native/Merlin host/
Spike with 57 exact results, 32 further host invocations and two actual range
abort outcomes. Independent original-parent checks grant no sampled shape facts;
general symbolic public models, mutation/effects and applicable targets remain.
The current inventory reopens 352 packages and 278 standalone qualifications,
with all fifty pinned names observed under selected native/Merlin-host profiles.

[Region-local pointer seeds](docs/integer_address_regions.md#pointer-seeds-defined-inside-a-region)
now expose actual outer roots and per-addptr byte recipes before paired address
analysis. Source narrow wrap and original region/effect order stay executable.
Only actual contributing unaligned seeds retain their original exact pointer
representation; other required recoveries remain enabled. Ten libraries/Spike
images retain fourteen exact results, 48 host replays and four bounds aborts.
The combined ledger now reopens 341 packages and 278 standalone qualifications.
Broader root/alias/effect/source, dynamic model and applicable target scope remains.

[Integer-address region state](docs/integer_address_regions.md) closes the
reproduced carried-address refusal using actual logical pointer bases paired
with modular byte offsets. Original for/while/if interfaces, numeric state,
effects and undecided grid placement remain executable. Sixteen source captures
pass native/Merlin host/Spike with 32 exact-word results and eighty further host
replays. General root equivalence, other region interfaces, broader aliases,
external effects, model/runtime/source compositions and applicable targets
remain completion requirements.

An additional saved-TTIR while fixture qualifies different paired input/result
arities and actual loaded condition state. The complete new selection reopens
17 libraries/Spike images, 33 exact results, 83 host replays and four real bounds
aborts. The combined ledger now contains 327 packages and 268 standalone
qualifications; opcode-name breadth remains separate from arbitrary coverage.
Four complete i32/i64 address-state models additionally retain all 44 original
results through native/Merlin host/Spike, with saved parameters, dependent calls,
earlier snapshots and empty/zero-trip cases. The public model ABI stays static;
general mutation/runtime/effect and accelerator scope remains required.
That earlier combined inventory reopens 331 packages and 268 standalone qualifications.

[Same-base pointer differences](docs/pointer_addresses.md#same-base-byte-differences)
now expose ordinary byte deltas only after actual SSA proves the shared base
and lane correspondence. Source-width arithmetic, strided reads, selected
pointers and runtime loops remain executable. Seven native/Merlin-host/Spike
libraries retain eleven exact-word results and 32 further host replays; four
actual bounds failures stay refused. Four further libraries retain narrow i32
wrap before byte scaling, with four exact results and sixteen host replays.
Two complete parameterized models retain all twenty original results as exact
words through native/Merlin host/Spike. That earlier inventory reopens 310 packages
and 251 standalone qualifications. General region/alias/effect/source and
applicable target scope remain completion requirements.

[Numeric region builtins](docs/numeric_regions.md) now reuse the existing
histogram, contraction, local-gather and math converters inside checked scalar
regions. Nested program-ID/count queries capture values from their enclosing
logical invocation; original control/effects stay in place and mapping stays
undecided. Thirty native/Merlin-host/Spike configurations retain 24 exact-word
and six tolerance-qualified results, with 120 unchanged-library host replays.
Six complete parameterized models additionally retain all 39 original results
as exact words through those execution boundaries. That earlier inventory reopens
297 packages and 240 standalone qualifications.
Broader region/type/effect/runtime/source and applicable target work remains.

[Local tensor regions](docs/tensor_regions.md) extend scalar container bodies
with functional tensor intermediates/state and one transactional postorder for
mixed nested maps/reductions/scans. Real source bitset kernels retain the actual
chunk loops and inner reductions. Merlin's native destination preparation keeps
captured values independent without copying ordinary readonly captures. Eighteen
libraries/Spike images retain 18 exact-word results and 72 additional host replays.
Six complete i32/i64 model configurations also retain all 42 original results
through native/Merlin host/Spike, including dependent calls and original snapshots.
That earlier inventory reopens 261 packages and 210 standalone qualifications. Broader
tensor/numeric/effect/model and applicable target scope remain required.

[Captured tensor lookups](docs/scalar_tile_reads.md) now retain original table
values across source writes and runtime reduction/scan recurrence. Twenty-four
libraries pass native/Merlin host/Spike with 48 complete exact-word results and
96 additional unchanged-library invocations. That earlier inventory reopens
237 packages and 192 standalone qualifications. The [review handoff](docs/handoff.md)
records the implementation, evidence and remaining full-coverage requirements.

[Scalar region composition](docs/scalar_regions.md) now applies checked shared
scalar mappings and exact rounding/high-product expansions before region lifting.
Twenty-four further libraries pass native/Merlin host/Spike with 24 complete
results and 96 further host invocations. That earlier combined inventory reopens
213 packages and 168 standalone qualifications. General regions/effects,
runtime/alias/source-version combinations and applicable targets remain required.

[General reductions](docs/reductions.md) now retain coupled scalar state,
source widths/order and actual first-element initialization in standard Linalg.
Twenty-four libraries and four complete models pass native/Merlin host/Spike:
80 exact-word results and 96 additional unchanged-library host invocations.
That earlier inventory reopens 189 packages and 144 standalone qualifications.
Remaining scalar builtin/region compositions, memory/external effects, runtime
and alias forms, source versions and applicable targets remain required.

[Exact IEEE source literals](docs/ieee_literals.md) now retain original
f16/BF16/f32/f64 NaN words through standard integer constants and bitcasts,
including protected allocations and complete enclosing models. Sixteen libraries
and four models pass native/Merlin host/Spike with 56 exact-word results and
48 additional host runtime replays. That earlier inventory reopens 161 packages
and 120 standalone qualifications. Broader numeric/effect/shared-state and
accelerator closure remains required.

[Structured scan combiners](docs/scans.md#structured-combiner-qualification)
now preserve scalar branches, actual for/while state, tuple order and uniform
captures inside parallel stages. Ten libraries and two complete models retain
thirty exact-word results through native/Merlin host/Spike, with forty further
host runtime replays. The combined inventory reopens 141 packages and 104
standalone qualifications. Effectful combiners, broader numeric/alias/source
versions and accelerator compositions remain completion requirements.

[IEEE f16/BF16 atomic addition](docs/half_atomics.md) now preserves subnormals,
per-event rounding, quiet computed NaNs and exact old words through standard
atomic regions and actual native/Merlin host/Spike. Four libraries replay sixteen
further runtime domains; two complete models retain dependent calls, original
snapshots and all twelve model results as exact words. Six target images compare
twenty complete results. The combined inventory reopens 129 packages and 94
standalone qualifications. Broader numeric/effect/shared-state and accelerator
compositions remain completion requirements.

[Original CFG intake](docs/cfg_intake.md) now retains early returns, successor
edges, block arguments and numeric backedges through official CFG-to-SCF
normalization. Current pointer types normalize before the older shared parser;
original source auditing precedes all rewriting. Native/Merlin host/Spike
qualification and a complete learned-state model retain the actual computation.
General value-returning entries, remaining CFG/alias/effect forms and applicable
targets remain completion requirements.

[Closed pointer-integer displacement expressions](docs/pointer_addresses.md)
now re-raise modular address arithmetic into actual logical pointer bases and
proved element offsets, preserving selections, tiling and source casts. Four
i8/i16/i32/i64 kernels pass native/Merlin host/Spike. Three scalar-memory libraries
also retain masks/state in all profiles, with 25 total additional host replays.
The combined inventory reopens 113 packages and 81 standalone qualifications.
Escaping addresses and general region/alias/effect/target closure remain open.

[Decoded descriptor arithmetic](docs/source_assembly.md#native-normalization-and-the-arithmetic-carrier)
now composes selected source word programs with original descriptor parameters,
allocation-root recurrences and complete i32/i64 learned-state models. Six
kernels and two models retain all fourteen exact-word results through native,
Merlin host and actual Spike, with 26 additional unchanged-library host replays.
Original source auditing precedes every normalization fallback, including
default/unselected effects and unused helpers. Broader instructions, packing,
shared/external effects and applicable targets remain completion requirements.

[Mixed FP8 source policies](docs/per_cast_fp8.md) now retain independent
per-cast overflow/infinity semantics in one kernel. Standard expansions precede
helper inlining/folding; eight format/mode cases execute through Merlin host
and Spike with all 32 results and 24 additional host runtime replays. The
consumer source-feature ordering fix preserves narrowing before widening.
Protected composite-dot normalization retains the original cast/dot graph
and a separately inventoried normalized graph. BF16x3/x6 and logical TF32/x3
compose with three source cast policies through the memory profiles and actual
native/Merlin host/Spike execution: twelve target results and 36 runtime
host-library replays. The [native source reader](docs/native_source_print.md)
now preserves original pointer/descriptor types and complete cast fields before
normalization. Six mixed-policy memory-profile cases and two ragged composite
GEMMs also pass native/Merlin/Spike, retaining source K reuse and partial dots.
Direct external descriptor parameters now also retain mixed policies through
native/Merlin host/Spike, with full expanded-ABI validation after source auditing.
Protected source capsule v2 additionally carries original contracts through
allocation wrappers and a complete enclosing model. Two pointer/descriptor
libraries pass native/Merlin host/Spike plus eight runtime host replays; the
learned-state model retains all four results and its original snapshot through
those execution boundaries. Broader source versions, binding/numeric/effect
compositions and accelerator qualification remain open.
[Allocated composite contractions](docs/per_cast_fp8.md#allocated-tiled-contractions)
now additionally preserve BF16x3/x6 and TF32x3 across six pointer/descriptor
libraries, actual Spike results and 24 further host runtime domains. Two
complete models retain padded owning inputs, ragged logical GEMM, dependent/
zero-K calls, a live original C snapshot and all eight results as exact words
through native/Merlin host/Spike. Source numeric intent stays explicit after
descriptor expansion; broader numeric/effect/alias/packing and target closure
remains required.

[Source model tensor views](docs/model_source_views.md) now recover static
transpose/stepped-slice coordinates from parent SSA and bind them to existing
executable owning-root pointers. Three real i64/f32/BF16 models retain learned
state, dependent updates, original snapshots and all 21 results through native,
Merlin host and RISC-V/Spike execution. Binding v3 makes the additional source
view contract explicit. General physical/mutating views, dynamic public model
ABIs and arbitrary kernel/target closure remain open.

[Observable target effects](docs/target_effects.md) now extend ordinary Merlin
target execution to void entries, initially failing tensor entries and mixed
diagnostic/result kernels. Calls, source bytes, independent outcomes, all
results and readonly inputs are checked separately. Physical/external effects
and accelerator qualification remain open.

[Standalone runtime target ABI](docs/standalone_target.md) now carries dynamic
tensors and all seven native scalar types through actual Merlin/Spike images.
Ten target invocations retain eighteen complete exact-word results. Changed
shapes/scalars preserve the kernel program/object; runtime metadata and each
image harness bind their particular invocation separately. This closes the
previous static-model-only target entry gap for the qualified forms; general
model views, effects, typed compositions and target coverage remain open.

[Functional target execution](docs/target_execution.md) now qualifies selected
complete models through actual Merlin RISC-V images and Spike. The public
qualification driver retains every result, source parameter and runtime input
observation. Empty tensor results and mixed BF16/F32 contractions exposed general
consumer reader/readback fixes; broader target coverage remains open.
Selected descriptor/map state and mixed/byte-backed RMW/CAS models now also
execute through the ordinary target pipeline. All 51 results in eight state
models pass exact-word comparison. Retained execute-region provenance and
boundaries preserve the source snapshot and native lifetime checks.
Together with the integer tuple and BF16 contraction cases, eleven complete
models now qualify all 64 results bit for bit on this functional CPU path.
Public BF16 runtime inputs additionally preserve exact original words through a
checked runtime-only archive projection. Three complete model cases retain all
thirteen results, including canonical legacy NaNs and finite legacy recovery;
computed NaN payloads keep their separately declared numeric policy.

[Source assembly implementations](docs/source_assembly.md) additionally provide
an explicit optional source-language boundary. Checked integer snippets become
standard scalar/tensor computation with retained source/provider identities;
hardware observations, broader instructions/effects/packing and applicable
targets remain completion requirements.

**Updated 2026-11-15.** The architecture in [plan.md](plan.md) is accepted as the starting point:
tile-first conversion, a single semantic emission path, and original intent carried through
annotations/provenance while computations lower into standard MLIR. This roadmap tracks delivery;
the design document explains the transformations. Milestones below are capability gates, not dates.
The active [completion plan](docs/completion_plan.md) orders the remaining frontend,
compiler-staging, payload-materialization and coverage work with explicit evidence gates.
The [maintained product corpus](docs/product_corpus.md) now records real refusals
and execution against all 50 pinned definitions. The combined selection observes
all 50 definition names in source and native/Merlin-qualified captures under
selected profiles, including the optional source implementation above. Proved
[slot-independent packed computation](docs/packed_maps.md) now also reaches Merlin
while retaining packing intent. Fourteen additional libraries preserve scalar
branches and for/while state with per-element loaded trip counts and runtime
scalars; the same libraries pass 56 further runtime domains.
Two real i64/f32 model captures also retain learned weights, dependent calls and
all eight results through native/Merlin host and actual Spike execution with
exact-word comparison. The refreshed explicit selection reopens 88 packages,
61 standalone qualifications and fourteen product cases.
Coupled/asymmetric packed maps additionally have an explicit
[initial source-layout specialization](docs/source_grouping.md) through seven
Merlin libraries and 24 further runtime domains. Its native-derived grouping
becomes logical tensor dataflow, leaving consumer placement undecided. The
current inventory reopens 106 packages and 74 standalone qualifications.
Three complete source-specialized models additionally preserve all twelve
results through actual native/Merlin host and RISC-V/Spike execution. Their
semantic grouping options are checked against original typed model-call
attributes by both the package verifier and consumer stager.
[Packed source reads and pointers](docs/grouped_memory.md) additionally qualify
through six actual libraries and five runtime domains each, preserving original
read regions, per-slot root tags and mutation snapshots. Two actual learned-state
models additionally retain three calls and all thirty results through native/
Merlin host and actual Spike execution, including live original states and empty
results. Returned-pointer model compositions and broader targets remain open.
Replicated layouts, broader packed memory, final GPU correspondence, broader mapped
forms, source assembly, typed compositions and targets stay open.
[Byte-backed integer word atomics](docs/byte_atomic_words.md) now reach complete
TTIR packages and actual Merlin libraries, retaining byte order, old values,
whole-word bounds/alignment and per-byte ownership. Source templates, executable
payload recipes and independent consumer declarations are checked separately.
Selected-root for/while state, old-value tensor recurrences, different-width
typed roots and complete seven-result models now also qualify on this path.
Floating policies and applicable targets still need their own qualifications;
full arbitrary-kernel coverage remains open.
[Indexed descriptors](docs/indexed_descriptors.md) now reuse the installed upstream
pointer expansion for gather/scatter/reducing stores, preserving loaded indices,
source padding, state and atomic kinds through the existing checked memory path.
Original descriptor graphs and actual normalizer identity remain explicit;
physical descriptor byte interfaces and applicable target materialization stay open.
[External logical descriptor arguments](docs/external_descriptors.md) now bind
the actual upstream one-to-many field interface through Merlin, including
runtime strides/flags, block/indexed state and two-descriptor tiled f32/BF16 GEMM.
[Descriptor allocation state](docs/descriptor_state.md) now retains nonzero
runtime origins and shared writable roots, including a complete model2MLIR
capture and independently checked portable original-source custody. Physical
descriptor bytes and applicable target handling remain open.
[Typed allocation aliases](docs/allocation_word_views.md) now retain equal-width
integer/floating source arguments, descriptors and model calls on one owning
root through Merlin. [Different-width source parameters](docs/typed_parameters.md) now bind original
pointer and logical descriptor arguments to typed owning roots. Both source
and root element units, independent signed strides and shared word state reach
Merlin; real model captures retain all results through API and CLI binding.
[Typed-root storage transport](docs/typed_storage.md) now retains public owning
shapes/dtypes through standard byte packing, with per-result guard costs checked
independently by Merlin. Both byte orders, dynamic matrix roots, disjoint byte
updates and complete typed model captures qualify through actual host execution.
Byte-backed atomics, broader effects and targets remain.
[Different-width byte-buffer views](docs/byte_aliases.md)
now preserve pointee-scaled addresses, word events and byte-level guards,
including selected roots, source fills, typed descriptor parameters and complete
model calls. Byte-backed atomics and target work remain open.
[Runtime source-allocation strides](docs/strided_memory.md) now qualify padded,
reversed, column-major and broadcast reads in tiled f32/f16/BF16 GEMM through
Merlin. Original K reuse, typed accumulators and active bounds remain intact;
[strided writable output state](docs/strided_outputs.md) also preserves every
untouched output word and rejects conflicting stores. [Readonly aliases](docs/readonly_aliases.md)
now prepare independent logical input values, and aliased scalar carriers retain
their by-value semantics through Merlin. [Allocation-state bindings](docs/allocation_state.md)
now express writable source aliases with runtime offsets and one returned root,
including shared C/output GEMM. [Dense model value adapters](docs/model_tensor_views.md)
now bind those roots to actual matrix-shaped model calls with independently
checked v2 shape/count records. Mixed-type aliases, physical model views,
independently strided C, mutation and target handling remain open.
[Pure external math](docs/external_math.md) now reuses shared conversions for
52 f32/f64 default-library declarations, with typed/purity/library checks and
standard math through Merlin. Arbitrary libraries and effects remain separate.
[Scaled contractions](docs/scaled_dot.md) now preserve FP4 K/M/N packing,
optional E8M0/E4M3FN scales, mixed operand types and original C through standard
tiled MLIR and real Merlin packages. The internal-precision profile is explicit;
runtime/ragged FP4 source loops also replay through all three memory profiles.
Broader scaled compositions and applicable target materialization remain.
[F32 BF16x3/x6 and TF32x3 precision](docs/dot_precision.md) now reuses the actual Triton
pass on layout-free TTIR, retaining ordered partial contractions, correction
NaN clearing and original C through Merlin. Direct/composite TF32 use an
explicit logical E8M10 profile and standard arithmetic quantization. Broader
numeric compositions and target qualification remain.
[Runtime model bindings](docs/runtime_model_arguments.md) now use captured
typed scalar attributes and checked shape casts to retain the source kernel
runtime ABI within model2MLIR's static public model interface. Merlin derives
per-call guarded scratch from the original typed parent call independently.
[Source assertions](docs/source_assertions.md) now retain scalar/tensor checks,
messages and failure effects through upstream standard lowering and real Merlin
execution, including a complete model2MLIR state/tuple capture.
[Observable-outcome packages](docs/effect_abi.md) now also admit void and
initially failing entries through real Merlin compilation/invocation, preserving
actual signatures and exact independent outcome references.
[Length-aware literal diagnostics](docs/literal_diagnostics.md) now retain
embedded NUL and escaped UTF-8 through native and Merlin execution, including
BF16/CAS compositions. Broader print forms and target handling remain effect gates.
[Logical print events](docs/logical_prints.md) now retain numeric values,
word/decimal formatting, source order and logical coordinates in standard
MLIR, with physical thread replicas abstracted explicitly.
[Mixed returning tensors and diagnostics](docs/mixed_effects.md) now share a
single observed call, complete per-result references and reusable Merlin
library. Numeric control retains original tensor widths and recurrence;
functional inputs remain readonly. Broader effect/target gates remain open.
[Clamp and precise scalar math](docs/scalar_math.md) now qualify f16/BF16/f32/f64
typed TTIR and a real public f32 kernel through Merlin, retaining source
operations and exact finite/signed-zero behavior. The BF16 native feature is
explicit; the portable payload and storage types stay BF16. A real model2MLIR
capture preserves learned bounds and all eight results.
The [package observation registry](docs/package_coverage.md) now joins verified
package evidence to the complete pinned 50-op inventory, preserving unmeasured
definitions, exact typed forms and separate native/Merlin statuses.
[BF16 storage and casts](docs/bfloat_precision.md) now reach the actual Merlin
interface in all three memory profiles, retaining typed contractions, original
F32 accumulators and exact uncomputed words. [BF16 model captures](docs/bfloat_models.md)
now additionally bind original source words, weights and all tuple results,
with legacy archive recovery limited to proved unique non-NaN embeddings.
Archive differences, native transport and remaining model/target gates are explicit.
[Lifted source constants](docs/lifted_constants.md) now also reach Merlin with
exact safetensors bytes and complete source/argument/type bindings. Real models
qualify nine tensor carriers and original BF16 contractions in all three memory
profiles. [Logical Boolean memory](docs/boolean_memory.md) retains masks, fills
and source correspondence while abstracting the checked byte-slot convention.
Recent additions retain finite selected-buffer pointer state, cat/split/tile
gather, and [mapped scalar regions](docs/elementwise_map.md) through actual Merlin
execution. The selected-memory model demo binds two real model2MLIR kernel calls
and learned weights; per-kernel package differences are explicit.
Tile-local [histogram](docs/histogram.md) also retains composite intent with a
linear indexed-state expansion and exact source masks. The visible
`shared-demo --kernel histogram` route supports all three memory profiles.
Actual logical-grid bounds now admit unmasked dense unit rows with independent
runtime output coverage. Source-word memory inside structured regions also
uses typed native event templates, retaining branch/loop placement and exact
fills. The explicit v2 evidence chain preserves both native stage reports;
see [source-word memory](docs/source_words.md).
Unsigned i32/i64 high products and Triton's integer Philox now also execute
through Merlin, preserving source word arithmetic and runtime seed/counter
inputs in standard MLIR; see [the numeric contract](docs/integer_products.md).
[Integer RMW atomics](docs/atomic_memory.md) now retain collisions, old values,
masks and source order/scope through actual Merlin execution under an explicit
private-invocation profile. A real atomic model also preserves every tuple
result and original weight through native and Merlin execution, using complete
receipt-bound references. Integer CAS now also qualifies i16/i32/i64, selected
buffers, rank-two tiles, recurrences and original old values through actual
Merlin execution. Float policies, broader effects and accelerator materialization
remain completion requirements; native CAS toolchain requirements are explicit.
Branch/loop-local atomics now also retain old-value recurrences and final
same/selected-buffer pointers, with pure accumulator destination reuse through
Merlin and all native lifetime checks retained.

## The product contract

The input is a Triton source/launch specialization or saved pre-layout TTIR plus a semantic ABI.
The output is a self-contained package a Merlin ingest adapter can consume: executable MLIR,
constants/weights, argument bindings, example inputs, independent reference results, source
provenance, and a receipt binding the actual artifact bytes.

Keep four different kinds of information distinguishable:

| Information | Representation | What it permits |
|---|---|---|
| Observed source | TTIR operations, stable IDs, typed SSA edges; original source/locations when available | Exact attribution and reconstruction of the captured graph. |
| Declared intent | Kernel/region descriptions and caller annotations, labeled with their origin | Explanation and candidate grouping; never permission to change math/effects. |
| Recovered structure | `prov.family`, `prov.op`, `prov.region_id`, source-to-emitted IDs, logical tile/grid facts | Matching lowered computations without repeatedly rediscovering their origin. |
| Established facts | Bounds, alias/effect checks, numeric permissions and transformation evidence, with preconditions | The particular rewrite whose conditions were checked. An annotation alone is not a proof. |

Every stage preserves executable value/effect semantics. Provenance can survive lowering to scalar
operations, fusion and decomposition, but cannot replace missing operations. Printed MLIR must
retain annotations: the MVP uses generic assembly to avoid xDSL custom-printer losses. An exact
hash-linked sidecar preserves the original graph and correspondence independently of printer form.

## Reuse-first implementation track

triton-shared already implements substantial parts of R1/R2/R3/R6. Those milestone descriptions
are acceptance requirements for this product, not instructions to reimplement all upstream passes.
Use the [source-backed reuse map and stage adapter](docs/triton_shared.md) before adding converters.

The shared tensor frontend now runs the actual upstream passes, functionalizes structured
memory, and preserves source intent through native name/callsite locations. A content-pinned
standalone optimizer is available without a native build or installing a target backend.
The complete pinned Meta corpus was measured: 310/339 tensor-stage and 314/339 full-stage
successes, with stage-specific/negative fixtures retained. These are middle-layer observations,
not product or universal opcode coverage; see [the measured scope](docs/coverage.md).

The [functional tensor adapter](docs/shared_tensor.md) now supports static contiguous tensors,
affine multi-axis grids with unmapped forall/ownership checks, bounded K loops and source
accumulators, nested tensor memory versions, masked softmax, helper inlining, and multiple results.
It reuses upstream computation conversion rather than enumerating local arithmetic handlers.
Complete example packages execute serialized standard MLIR through native LLVM and compare
all results against supplied independent references. [Package v2 differences](docs/package.md)
remain explicit. The stock-wheel reference frontend remains useful for its rank-one symbolic
mask/strided-view guarantees and regression cases.

The [model-call binder](docs/model_integration.md) now resolves model2MLIR's typed `prov.kernel`
calls using schema operands, scalar specializations and result ordering. Fresh outputs omit
incoming buffers only after initialization/coverage checks. Derived model bundles retain actual
weights, manifests, runtime inputs, constants and PyTorch goldens, with parent capture evidence
kept separate from changed-IR provenance. Native tests cover a real `triton_op` model and the
original `mlp_basic` / `mlp_basic_triton` pair. Merlin's actual reader and C runtime generator
accept both model interfaces. Compiler-only staging and actual native Merlin
library execution now have separate qualification; accelerator staged execution remains open.
Standalone shared tensor packages also have direct compiler staging and native
Merlin library execution through `qualify-merlin`, with a separate receipt and
explicit entry-symbol adaptation; no surrounding Torch model is required.
Loaded-index read-only gathers now preserve parallel tile dataflow, repeated
indices, per-lane masks and nonzero fills, with active bounds checks and actual
Merlin compiler/host qualification. The [indexed memory contract](docs/indexed_memory.md)
keeps general writes, writable-buffer reads and undefined lanes explicitly open.
Nonrectangular masks on independently owned affine store tiles now also have
native/Merlin execution in both profiles, with exact inactive-state preservation
and runtime tails. The buffer form retains upstream predicated stores; general
indirect scatter remains open. See [masked writes](docs/masked_writes.md).
Two-axis runtime masked stores also qualify transposed storage and traversal
without losing mask/address correspondence. `shared-demo --kernel masked-store`
and `--kernel masked-matrix` provide visible package routes.
Half tensor storage now also executes through both Merlin profiles, retaining
f32 intermediates/narrowing, nonzero f32 contraction accumulators and masked
runtime matrix state. [The measured half profile](docs/half_precision.md) keeps
other low-precision formats and rounding policies explicit.
The [runtime ABI](docs/runtime_abi.md) now carries typed by-value scalar inputs,
including interleaved TTIR arguments, into native and Merlin host execution.
Scalar payload/control values are supported independently of the remaining
runtime shape/stride/grid and address-domain work.
ABI v4 now adds guarded runtime memory prefixes, including empty/tail domains,
negative/all-inactive limits and preserved readwrite outputs. One Merlin library
handles different domains without recompilation. Logical tensor capacities are
still static; actual dynamic shapes/results and launch extents remain a separate gate.
ABI v5 now provides actual named dense runtime dimensions, executable shape
checks, dynamic result allocation and runtime ceil-divided logical grids. Both
memory profiles qualify independent shapes, including zero, through one unchanged
Merlin library. [The runtime dimension contract](docs/dynamic_shapes.md) keeps
non-dense strides, indexed access and broader shape expressions open. Exact
symbolic dense-stride cancellation now preserves two-axis tiled matrix addition
through Merlin over changing matrix shapes and empty/tail domains, in both
memory profiles, without runtime division by a shape dimension.
Runtime M/N/K GEMM now retains tile contractions and its original K loop through
both profiles and actual Merlin libraries, including transposed RHS storage,
empty M/N, K=0 and ragged tails. `shared-demo --kernel dynamic-matmul` emits a
symbolic package with independent numerical validation.
Runtime read-only indexed gathers now also qualify multidimensional allocation
shapes and independent index/result counts through both profiles, including
i32/i64 indices and empty inactive domains. The tensor expansion checks active
indices against actual storage; general indexed writes remain open.
Runtime value-only loops now preserve source SSA bounds and tile recurrences
through the same Merlin library for zero/nonzero counts in both memory profiles.
The checked consumer retains fusion and native lifetime/alias analysis; source
read loops now prove all reachable IV/address ranges, preserve row-local masks
over flattened storage, and qualify i32/i64 counts through the same library.
Runtime write loops also carry conditional state, preserve incoming values on
empty/false paths, and retain only definite initialization facts. Both profiles
qualify repeated updates and preinitialized/positive-trip fresh results.
The same checked model/package path now also covers ragged M/N/K GEMM;
`shared-demo --kernel ragged-matmul` and `examples/model_ragged_binding.json`
provide the reproducible CLI route.

The explicit `--memory-profile buffer` now preserves the full upstream standard
memory payload behind the same tensor/scalar interface. Standalone and actual
model bundles qualify through Merlin's compiler/library path with byte-bound
memory evidence. The initial profile still requires the existing tensor source
memory proof; this representation milestone does not expand opcode coverage.
See [the supported boundary and remaining work](docs/full_memory_bridge.md).

Permuted loads/stores, data-dependent `scf.if` memory merges and affine pointer-to-index
recurrences and clipped affine tail masks now have native checks. Ragged M/N/K GEMM retains
its K loop and tile sizes, including transposed weights. The next boundaries are runtime
scalars/shapes, arbitrary masks, grouped or
nonlinear indexing, block/irregular descriptors, general while/pointer state and numeric profiles.
Legacy `make_block_ptr` / `advance` now have an official pre-layout normalization
path with native ragged GEMM checks. Original and normalized source inventories
remain explicit in provenance v2. Kernel-local tensor descriptor creation/load/store
also use an official normalization pass and have native ragged GEMM/model-interface
checks. External descriptor arguments, descriptor gather/scatter/reduce and general
irregular memory are still separate capabilities.

An upstream stage is not a complete frontend result: tensor/memref boundaries, per-program launcher
arguments and missing source correspondence must remain explicit. Native toolchain builds require
approval under the shared-VM rules; no build or CPU backend is started implicitly by the adapter.

## R0 — minimum visible demo: now implemented

**Demo:** real `@triton.jit` vector add → native TTIR → symbolic access recovery → tiled
`scf.forall` / `tensor.extract_slice` / `linalg.generic` / `tensor.parallel_insert_slice` → package.
The source kernels ship in `t2m/examples/`; source capture needs no GPU/driver or triton-cpu.

Implemented scope:

- Static rank-one contiguous f32 inputs, independent allocations, one completely written output.
- Exact unmasked tiles, one logical grid dimension, add/subtract/multiply payload builders.
- A second source kernel, `x*y+x`, exercises multiple payload operations and real dataflow.
- Saved-TTIR parsing/replay; backward liveness; symbolic bounds, i32-overflow and exact-partition checks.
- Typed source SSA edges, recovered access descriptors, centrally stamped provenance, caller intent
  explicitly distinguished from operation semantics, and exact source/spec/output hashes.
- A standard-op reference interpreter executes the serialized result with all provenance ignored
  for semantics. Bundle writing compares it to an independently supplied golden before publication.
- Core model2MLIR filenames/array conventions, real empty safetensors, artifact integrity checks,
  a distinct Triton receipt/trace, and machine-readable compatibility differences.

Acceptance: source-to-output numerical comparison, multiply-add variant, replay determinism,
annotation round trip/removal, unmasked-tail refusal, unsupported-op attribution, effect/order
mismatch refusals, overflow refusal, proof on a >2²²-element shape without enumeration, and
corrupted/stale bundle refusal. See [the tests](tests/test_mvp.py) and [MVP evidence](docs/mvp.md).

**R0 boundaries (historical):** masked tails, source region/loop translation, dynamic/scalar arguments,
named-composite output, learned constants, native LLVM execution, or staged accelerator execution.
The adapter refuses nested regions explicitly; it does not flatten them. This is a working first
slice of the earlier M0, not a claim that all original M0 requirements have shipped.

## R1 — useful tensor-memory frontend

Deliver exact masked loads/stores, nonzero `other`, low/high boundary padding, transposed/strided
views, nonzero storage offsets, immutable scalar parameters, and local multiple-store sequencing.
Add preserved-output/readwrite tensor results with incoming values, while keeping model integration
mutation disabled until its caller ABI supports it. Track mixed address/data uses correctly.

**Implemented in the current R1 slice:** rectangular signed-comparison/conjunction masks; exact
low/high padding and source fills; all-inactive tiles; pointwise undefined-lane tracking; positive
strides/storage offsets re-raised into logical tensors; one partial readwrite update with owned-tile
reads; identical-read reuse; advisory source cache/eviction policies; and optional strict native
verification plus complete LLVM-dialect legalization. Exported artifacts remain tiled tensor MLIR.
See [docs/r1_memory.md](docs/r1_memory.md) and the memory/native tests.

**R1 remains open:** runtime scalar parameters, rank-two/transposed views, arbitrary/data-dependent
mask handling, multiple-store/local memory-version sequencing, richer source-region/attribute
snapshots and the typed rule registry. LLVM legalization is tested, but native host execution and
staged consumer execution are not yet qualified.

That list describes the small reference frontend. The shared path now adds rank-N contiguous
logical tensors, specialized integer scalars, ordered multiple stores/results and bounded nested
memory loops, with a separate tensor ABI and native execution checks. Runtime scalars, permuted
storage and aliasing remain open across that broader profile; staged execution remains unqualified.

Expand the native adapter's region/block/attribute snapshot and version probes before translating
control flow. Refactor the small MVP dispatcher into a typed rule registry, preserving its
failures as regression cases. Each rule declares supported types, semantic prerequisites,
source-op accounting and a defined expansion. Unknown flags and effectful operations stay refused.

Gate: vector add n=1000/B=256; nonzero fill; negative/empty active slice cases; valid permutations;
data-dependent predicates without speculative out-of-bounds extraction; preserved partial outputs;
and explicit refusal of unproved aliases or conflicting writes. Add a stock MLIR verifier and
native host execution job using an existing/pinned toolchain, separate from the IR interpreter.

## R2 — tiled GEMM and source control flow

**Implemented slice:** real multi-program 2-D tiled GEMM, a retained bounded K loop, IEEE source
accumulator restoration, affine logical-grid decoding, symbolic bounds/source-width checks and
disjoint owned output tiles. Serialized IR passes native host numerical checks. Bounded nested
loops carry functional buffer versions. Permuted/transposed weights, invariant affine pointer
increments, post-loop pointer descriptors and `scf.if` memory merges are now covered. Ragged
M/N/K contractions, dynamic prefix cropping/padding and complete clipped output partitions
also pass native checks. Nonlinear/live pointer state, grouped traversal, general while and mixed-precision
profiles remain open.

Translate `scf.for`/`if` and their block/yield arguments, pointer recurrences, reduction loops and
the source accumulator. Add multiaxis/flattened grids, batch dimensions, constant div/rem and
grouped traversal patterns. Keep tile reuse and fusion visible. Support f16/bf16/integer paths
only with their actual casts, signedness, accumulation and rounding semantics.

Gate: multiprogram ragged M/N/K GEMM, nonzero accumulator, transposed weights, batched products,
bias/activation epilogues and repeated loads from one buffer. Shared reads may overlap; output
contributions and conflicting input/output accesses must satisfy the dependence checks. Preserve
K-loop recurrence without element-count enumeration. Whole-tensor recovery remains optional.

Numeric profiles beyond directly representable source arithmetic are individual capabilities:
TF32/mixed precision cannot be legalized by dropping a precision attribute into provenance.

## R3 — reductions, scans and composite intent

**Implemented slice:** masked f32 softmax with native inlining of private reduction/combiner
helpers, exact negative-infinity fill, standard linalg reductions/math, and native numerical
checks. General pure scalar associative scans now have a parallel tensor expansion,
including reverse traversal, tuple results and retained combiner/algorithm provenance.
Real sum/product and tuple-result model2MLIR calls also pass complete-model native
comparison and Merlin interface/runtime generation with unchanged weights/goldens.
Numeric edge policies and optional composite recognition remain
capability work; no dense attention replacement or universal reduction coverage is claimed.
See [the scan expansion and qualifications](docs/scans.md).

Translate combiner regions, multi-result reductions, arg-reduction tie rules and ordered scans.
Add fused softmax/norm kernels with tails and source-defined NaN/empty-row behavior. Recognize
supported composites only after checking axes, masking, casts and the allowed numeric changes.

Deliver the second output level through an optional model2MLIR dialect adapter: `high-level`
composites deterministically expand to the same standard form. Preserve source group identity
through expansion. Do not introduce another independent TTIR emitter or assume that proposed
model2MLIR extension ops already exist.

Gate: explicit-region versus composite-expanded numerical agreement; identity-correct padding;
stable correspondence after a one-to-many decomposition; installed-package tests with the supported
xDSL versions. General whole-tensor/composite recognition failures leave valid tiled output.

## R4 — online-softmax fused attention

Preserve Q reuse, source key traversal, QK/PV contractions and loop-carried `(acc,m,l)` state.
Support causal and sequence masks, probability casts, rescaling order and source exp/exp2 contracts.
Handle source-defined all-masked rows; do not silently replace the algorithm with dense SDPA.

Gate: multiple query tiles/key iterations and ragged sequence lengths; numerical comparison to
source interpretation where supported and a separate dense reference under the declared contract;
bounded tile state and no N×N score materialization. Report algorithm retention separately from
performance, since a new target may select different physical storage and tile sizes.

## R5 — model2MLIR/Merlin package and model integration

**Implemented slice:** typed `prov.kernel` body binding, fresh-output initialization checks,
learned-weight/manifest preservation, unchanged model ABI, complete native comparison against
the parent capture, and actual Merlin interface/runtime generation. The original plain/Triton
MLP pair passes with identical source weights and eager goldens. Derived receipts now
cross Merlin's compiler-only staging boundary. Optional actual Merlin host compilation
and repeated invocation preserve input bytes and compare unchanged goldens. General
mutation schemas and accelerator staged/device execution remain separate gates.
Flat tensor tuple results now retain every independent parent reference in the
additive `goldens.npz` contract; `golden.npy` remains the original first result.

Converge the packaging schema with the consumer rather than merely matching filenames. Add external
parameters/constants, multiple results, precise dtype/storage metadata, optional trace adapters,
and version negotiation. Keep a compatibility document and machine-readable list for every
remaining difference. Do not label a t2m receipt as an m2m receipt without its required identities.

Bind model2MLIR's `prov.kernel` calls to concrete translated bodies. Verify schema argument order,
constexpr/default/scalar binding, shapes/strides, returned output pointers and mutation/aliasing.
Use a consumer-supported outlining/inlining path: the surveyed staged materializer accepts only
one function and does not generally materialize arbitrary tiled SCF.

Gate: plain `mlp_basic` versus `mlp_basic_triton` with identical inputs/weights; equivalent normalized
semantics and results through the two frontends; native host execution; explicit generic Merlin
capability changes needed for tiled loops. Add staged target evidence only when the consumer can
represent every payload/effect. Keep capture, parse, host execution and device execution statuses
separate. The frontend must not gain per-target lowering cases to make these gates pass.

## R6 — irregular and dynamic kernels

Add dynamic tensor extents/grids, runtime guards, arbitrary structured branches/loops/while,
loaded indices, gather/scatter, duplicate-index rules, views and supported aliasing. Replace the
initial independent-allocation restriction only where tensorization or an effectful ABI preserves
the original storage relationships. Extend symbolic proofs incrementally with time budgets and
explicit `not proved` outcomes; keep a capability report by opcode, type, shape and composition.

Gate: irregular indexing and dynamic-shape suites, including conflicting scatter, negative-stride
views where implemented, partial-write preservation, and shape/alias precondition enforcement.
An indexed kernel does not become complete by retaining an unexplained opaque call.

## R7 — synchronization, memory effects and descriptor kernels

Add semantically modeled descriptor operations, local staging, async copy/wait and barriers.
Recover pure SSA dataflow where ordering/storage are implementation details and a dedicated analysis
establishes equivalence. Preserve effects otherwise. Atomics need scope, ordering, returned-old-value,
race/conflict and numeric semantics; a reduction idiom is only one recoverable case.

This may require an **explicit effectful package profile**, with memory/token operations or a small
semantic extension and a checked consumer legalization. That is a documented change from the
default pure tensor contract. Do not send TTIR pipeline tokens directly to Merlin's macro-scheduling
launch/await tokens: they describe different work and synchronization boundaries.

Gate: pure versus effectful profile classification, staging/async dependency equivalence,
descriptor bounds/padding/rounding checks and atomic behavior tests. Unsupported synchronization
must be diagnosed; deleting it and annotating its former existence is not a conversion.

## R8 — arbitrary-kernel intake and maintained semantic coverage

The destination is one versioned package envelope for any input kernel, with honest status and a
declared semantic profile. Grow a corpus of real kernels and a versioned operation/type/attribute
registry; test compositions, not only isolated opcodes. Add external math implementations, RNG state,
calls/regions, mixed-precision expansions and validated adapters as separate capabilities.

There is a necessary boundary on the phrase **“any arbitrary Triton kernel”**: a kernel observing
physical device IDs, executing arbitrary PTX, accessing device-specific state, or relying on an
undefined race cannot always have equivalent target-independent pure tensor semantics. It is still
an input we can inventory and package diagnostically; that is not successful portable conversion.

Every package therefore declares one of these outcomes:

| Outcome | Meaning |
|---|---|
| Complete portable tensor | All observable behavior represented in the default standard tensor profile. |
| Complete portable effectful | All behavior represented in the declared extended effect/ABI profile; consumer support required and explicit. |
| Backend-bound | Residual device semantics/dependencies are named; cannot claim backend-independent execution. |
| Refused / incomplete | Located unsupported/unproved behavior and worklist; no successful runnable-package receipt. |

An opaque fallback may preserve an interface for diagnostics or an external dependency, but it
does not count toward complete portable coverage. Never hide a backend binary in a package and
call that standard linalg conversion.

Gate: every supported Triton version has a reproducible feature/composition matrix; complete
packages pass artifact/ABI/provenance checks, independent numerical/effect checks and applicable
consumer tests. New versions and opcodes cannot silently inherit support. Broadening support must
retain earlier kernels' algorithmic structure, source attribution and refusal guarantees.

## Immediate next work

Numeric closure now has four-format FP8 reader/constants and actual buffer
transport through native and Merlin execution. All 256 encodings per format,
both import orders, masked tails, runtime loops and reused libraries qualify
in tensor/buffer/guarded profiles. Widening to f16/BF16/f32/f64 now also
qualifies every encoding through 48 format/destination/profile combinations,
with portable casts retained and a separately selected native implementation.
Standard-format RTNE/RTZ narrowing, equal-width f16/BF16 changes and
double-rounding-safe f64→16-bit conversions now also execute through 42
pair/mode/profile combinations. FP8 narrowing and distinct-format changes now
retain three explicit overflow/infinity contracts, exact source rounding and
per-step state through actual Merlin execution. The 127-case slice covers
standard sources, all distinct FP8 pairs and three memory profiles; heterogeneous
per-cast policies and broader FP8 arithmetic remain work. Mixed FP8→f32
contractions now retain named tiles, original C and runtime K loops through
actual Merlin execution, including non-f32 precision applicability and signed
zero/downstream-add checks; see [the measured contract](docs/fp8_contractions.md).
Scaled dot, wider accumulator profiles and target materialization remain work; see
[the delivery order and upstream cast gaps](docs/float_casts.md). This work
does not imply whole-opcode or all-target coverage.

The explicit `guarded-buffer` profile now admits loaded-index stores through the
full shared memory pipeline without requiring the tensor functionalizer as a
witness. Executable active bounds, program ownership, per-store lane uniqueness
and fresh-output coverage checks preserve the original payload and unmapped
logical grid. Native tests and one reused actual Merlin library cover changing
runtime shapes, ordered repeated writes and masked/empty cases. Metadata costs
16 bytes per writable element for usual grids (24/32 for wider program IDs).
V2 also preserves writable observations before/after source stores using an
effect-preserving gather adapter before CSE. Sticky owner tags allow shared
readers and reject foreign readers/writers in either order; completion cells
check initialization. Runtime scalar/tiled recurrences qualify through one
Merlin library over empty/tail shapes and zero/nonzero trips.
Static model2MLIR bindings also qualify ordinary
and ragged tiled GEMM, block pointers and tensor descriptors through Merlin,
retaining original parent data and accounting guard scratch per inlined call.
Aliasing, pointer-carried state, general descriptor forms and dynamic
bound-model scratch remain open; see [the contract and evidence](docs/guarded_memory.md).
Scalar gathers/scatters now lift to one-element tensors and reuse the same
upstream converters/checks, retaining their source types, masks and fills.

The explicit `source-words` address policy now uses upstream's supported
`offset-bit-width=64` option before the guarded memory continuation. Original
integer computations retain their word types; pointer increments accumulate
separately as signed i64 offsets. Scalar/tiled wraparound, pointer cancellation,
runtime pointer loops/final values and nonlinear grid permutations are checked
natively and through the Merlin interface. Source-word stage bytes/policy and
access intent are retained in the distinct package profile; static real-model
bindings preserve contractions and loops. Same-base pointer-valued branches
and runtime while state now qualify through the actual Merlin interface,
including zero trips, empty/tail dimensions and final pointer observations.
A fixed-point base analysis checks every branch/backedge; original word math,
regions and source IDs remain. Merlin retains complete while destinations
when the condition forwards state unchanged, with ordinary alias/lifetime
checks. Nested branch-selected pointer state and conditional reductions now
also qualify, with original init reads, helper inlining and changing choices/counts.
Conditional destination planning retains branches, uses complete value copies
where needed and leaves native alias/lifetime checks enabled.
Finite mixed-base joins, lane selection and changing loop bases now also
qualify through logical selectors paired with source-word offsets. Accesses stay
at their original position, retaining intervening writes and source masks/fills.
Uniform selectors stay scalar; source shape operations keep selector/offset
fields paired. Broader while ownership, aliases and broader effects still need
work. See [the selector contract](docs/selected_pointers.md).

Tiled online attention now also qualifies through Merlin: two contractions,
query-tile reuse and `(maximum, normalization, accumulator)` remain inside the
key-tile loop. Consumer destination timing preserves old-state reads before
updates, with native alias/lifetime checks. One library handles independently
changing query/key dimensions, empty keys and ragged tails. The finite f32
profile and independent reference tolerance are explicit in
[the attention contract](docs/attention.md); this is not general attention,
all numeric policies or accelerator execution.
Tensor, standard buffer and guarded source-word emission each qualify through
their own unchanged Merlin library. Tensor emission keeps an additional output
carrier invariant alongside the three changing online states.

Expand the existing shared tensor path from its tested static-grid/bounded-loop profile to
nonlinear/live pointer state, block and irregular descriptors, guarded runtime shapes/masks,
general `while`, scans and explicitly qualified mixed precision. Preserve earlier mask,
ownership, source accounting and native numerical regressions. Inventory all remaining failures
against the [measured corpus](docs/coverage.md), not a claimed opcode percentage.

The binder now preserves model2MLIR's actual model ABI and learned parameters, and Merlin's
reader/runtime-generation interface is checked. Compiler-only derived-receipt admission
and native whole-body materialization are implemented. Direct standalone TTIR staging,
general mutation/model-result schemas and accelerator materialization remain consumer gates.
Backend-observing operations still need the effectful/backend-bound profiles defined above;
they cannot silently become pure tensor math.

### Typed atomic follow-through

The separately selected typed atomic profile now extends integer RMW/CAS with
bitwise f32/f64 exchange through standard MLIR, native execution, standalone
Merlin packaging and actual model2MLIR tuple-result binding. V4 memory admission
retains v3 ownership/initialization costs and binds the floating bit contract.
This is additional qualified coverage, not full arbitrary-kernel completion.
Atomic updates on same-width logical word views now retain all nine i32/i64
RMWs on floating storage and Triton min/max through native/Merlin execution,
including real model2MLIR tuples. Branches, for/while recurrences, scalar/vector
events, lane/uniform buffer selection and repeated inlined helpers now also
qualify for both word widths. Original primitive contracts remain shared while
per-event storage views support calls on integer and floating storage. IEEE
f64 addition now uses upstream standard
atomic conversion through native/Merlin execution, source regions, colliding
old-value histories and actual model2MLIR tuples. F32 now explicitly selects
IEEE or sign-preserving input/result FTZ, retaining executable numeric policy
through standard MLIR, native/Merlin regions and real model binding. Next:
other floating numeric contracts and the remaining universal intake gaps.
F64 now also retains executable NaN quieting through constant normalization,
under the versioned `ieee-f64-rne-add-quiet-nans` contract. Legacy packages keep
their original standard-addf contract and remain readable.
See [atomic memory](docs/atomic_memory.md).

The ordinary-memory foundation for same-width logical pointer views is also
implemented: loads/stores retain actual storage types and reinterpret values,
including exact integer fill bits, finite root selection and pointer recurrences.
Native/Merlin tests cover f16/i16, f32/i32 and f64/i64 with explicit provenance.
Atomic RMW views now have a separate region/legalization extension preserving
the source integer min/max algorithm. Integer-word CAS now additionally retains
i16/i32/i64 semantics on f16/f32/f64 storage through standard atomic regions,
native/Merlin execution and three genuine model2MLIR tuple-result captures.
The 42 semantic compositions, 36 producer evidence checks, three demos and 15
model cases pass. Public storage types are unchanged; the additive stage-v2
typed-template boundary is explicit in [package differences](docs/package.md).
Ordinary [mixed storage selectors](docs/mixed_storage.md) now qualify all three
word widths through actual Merlin execution and genuine model2MLIR captures.
Closed selector permutations carry scalar indices into immutable selector tiles,
preserving source loops and offset recurrences without native lifetime exemptions.
Selected integer atomics now also retain mixed integer/floating storage: all
nine i32/i64 RMWs and i16/i32/i64 CAS, with complete source context/storage
bindings, old values and original regions. The 76 execution compositions,
six profile refusals and five three-result model captures pass. Broader numeric
formats, floating operations through integer views, external effects and target
materialization remain open. See [mixed storage](docs/mixed_storage.md#selected-integer-atomics).
See [logical views](docs/pointer_views.md).

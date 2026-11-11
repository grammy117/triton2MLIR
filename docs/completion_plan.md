# Completion plan: TTIR kernels through the Merlin compiler interface

Goal: all valid kernels in each pinned supported TTIR version reach a working
Merlin compiler interface with their observable computation/effects represented,
the useful algorithm retained, and applicable execution evidence. The existing
199-case regression suite and interface/runtime-generator demos establish a
baseline, not completion. Previous implementation turns made concrete progress;
this goal begins with remaining requirements still open.

## Completion evidence

The latest effect/package slice qualifies all nine integer RMW operations under
an explicit private-invocation profile, preserving masks, old values and source
order/scope. A real model2MLIR capture now keeps both atomic model results,
learned weights and complete tuple references through native and Merlin host
execution. The 32 model regressions and 23 compiler-staging tests pass.
Region-local RMW now also preserves branches, loop recurrences and finite
selected-buffer pointer state through native and Merlin execution: 17 new
producer cases, 93 staging cases and 90 native consumer loop/lifetime cases
pass. See [atomic memory](atomic_memory.md). Floating atomics, broader
effects and external synchronization still require executable implementations.
The following 145-case pointer/event/atomic/model regression also passes;
`out/atomic-rmw/` and its separate Merlin qualification provide a visible,
fully verified host-executed package. Accelerator coverage remains open.
Integer CAS now additionally qualifies all public Triton integer widths
(i16/i32/i64), scalar/tiled/rank-two dataflow, selected bases, branch/loop state
and original old values. Its 56 focused cases and visible-demo test pass, as do
ten RMW/CAS model cases and 207 Merlin compiler/staging/lifetime cases. Portable
MLIR keeps a standard atomic region; the native ownership compatibility path
and toolchain dependency are explicit in the package contract. Float policies,
external synchronization and accelerator materialization remain required work.
Final CAS validation also passes 167 existing producer regressions and 215
Merlin compiler/staging/lifetime cases. The visible kernel qualification and
whole-model package retain actual internal native helpers and pass all 45/80
artifact checks respectively; see the atomic contract for exact evidence scope.

For each operation/type/attribute/composition, track these separately:

1. Original native-verified source inventory and disposition, including helper
   regions and exact source bytes; normalized IDs remain a separate graph.
2. Executable tensor or explicitly declared effectful MLIR expansion. Unknown
   behavior cannot become a successful opaque fallback.
3. ABI/package compatibility: tensors, shapes, runtime scalars, parameters,
   alias/mutation/result order, dependencies and explicit schema differences.
4. Compiler-only Merlin staging with byte-bound source correspondence and no
   evaluator inputs/goldens supplied to compilation.
5. Payload-preserving consumer lowering with every retained operation accounted
   for. Reading a signature or generating a C harness does not prove this gate.
6. Native and applicable target materialization/execution against independent
   numeric/effect references, with original algorithm structure and provenance
   checked independently of numerical agreement.

An operation appearing in a passing example is observed composition coverage;
it is not universal opcode coverage. Refused and backend-dependent cases remain
in the denominator. The goal stays active until all applicable gates are proved.

## Execution order

| Workstream | Concrete implementation | Required checks |
|---|---|---|
| A. Compiler boundary | Admit the explicitly derived model receipt/source graph through Merlin's compiler-only staging; qualify its generic whole-body lowering. Preserve the existing Torch receipt path. | Real bound GEMM/scan packages, compiler input exclusion, corrupted/stale correspondence refusal, no dropped payload, full-result execution. |
| B. Irregular memory | Reuse upstream gather/scatter normalization; add tensor indexed reads, exact per-lane masks/fills and bounds checks. Preserve writes through functional memory versions; prove disjointness or model effects/conflicts explicitly. | Loaded/repeated/negative indices, arbitrary inactive out-of-bounds lanes, nonzero fill, multidimensional gathers, partial outputs, conflicting scatter, initialized reads and logical-grid dependencies. |
| C. Runtime ABI | Runtime scalar values and shape/stride/grid bindings as declared ABI data; dynamic slices/allocations and guards; supported alias groups and mutation. | One compiled program over multiple shapes/scalars, empty/tail domains, overflow, mismatched shape/stride bindings, alias preservation and actual consumer invocation. |
| D. Control/index structure | General logical div/rem/grouped traversal, pointer state, branches and while loops with checked memory effects; preserve contraction/online-softmax recurrences. | Bijective and conflicting grids, source while/pointer results, ragged attention, no dense score replacement, zero/nonzero accumulator and reuse retention. |
| E. Numeric closure | Type/rounding/signedness-aware casts, mixed-precision dot/scaled dot, reductions/arg rules, math/external functions and RNG state. Reuse official/shared expansions where their contracts agree. | f16/bf16/integer/fp8 profiles as supported, rounding/ties/NaN/zero/overflow, contraction accumulator order, external/RNG dependencies and independent references. |
| F. Effects and hardware boundary | Explicit portable effect ABI for atomics, returned-old values, scopes/order, synchronization and diagnostics; normalize implementation-only physical storage after dependency proofs. | Collisions, publication/order, async wait/barrier lifetimes, volatile observations, no erased effects. Backend-observing instructions retain concrete dependencies and require an actual supported consumer implementation. |
| G. Maintained coverage | Pin the complete TTIR operation definitions and real-kernel corpus; record opcode/type/attribute/composition and every gate above in machine-readable reports. | Reproducible corpus runs, no unknown operations inheriting support, complete source-to-consumer accounting, regression and package integrity checks. |

These are dependencies, not permission to postpone ordinary compilation until
the entire registry is built. Implement and qualify coherent vertical slices,
commit each, and expand the matrix alongside them. Do not redefine the goal as
"all kernels accepted", a diagnostic package, a host-only subset, or a backend
binary hidden behind an interface. Hardware-observing assembly cannot be erased
to make a pure-tensor claim; its semantics/dependency must be explicitly handled
and qualified at the consumer boundary.

## Starting evidence and constraints

At the start of this goal, the consumer check called Merlin's `forward_signature` and
`c_runtime.generate`; staged compilation/target execution are unperformed.
`merlin.frontends.compile_inputs.stage_compile_inputs` admitted only
`m2m.capture-receipt.v1` plus its matching FX trace. The schedule-to-interface
materializer has a narrower payload vocabulary than general tiled tensor MLIR.
These consumer gates require work alongside frontend coverage.

Use existing pinned shared/Triton/torch-mlir tools. Keep CPU workers/native
threads bounded and avoid implicit LLVM/Triton builds. Follow each repository's
local contribution/identity rules; publish tested milestones on the user-authorized
branches recorded in the README. Update this plan with
verified evidence and unresolved requirements, not just intentions.

## Execution evidence and next handoff

Explicit dense model value views now bridge allocation roots to original
matrix/scalar/empty caller values with standard reshapes and no dtype change.
Four actual i64 and f32/f16/BF16 model captures retain shared updates, learned
weights, zero K and every tuple result through native and Merlin execution.
The API and CLI use v2 binding records; Merlin independently rederives original
parent shapes/counts, scalar bounds and scratch, including static kernels.
Thirty-two focused cases, 71 legacy model/binding checks and 267 consumer
staging checks pass. See [the exact logical view boundary](model_tensor_views.md).
Continue with physical/non-dense and mutating model views, mixed-type aliases,
independently strided C, portable pre-wrapper custody and the broader E/F/G
and applicable target gates. Logical reshaping supplies no alias/layout proof.

Explicit allocation-state entries now bind aliased source pointers through real
TTIR calls before upstream inlining. Seven actual Merlin libraries retain shared
updates, runtime offsets, loops and full-allocation words; f32/f16/BF16 tiled GEMM
binds dense C and strided output into one root. Twenty-three focused cases pass,
including the generic CLI, 24 successful library replays and two compiled guard
refusals. The discovered consumer load/store fusion bug has a generic correctness
fix with 109 Merlin regressions passing. See [allocation state](allocation_state.md).
Continue B/C with mixed-type/byte alias groups, model view adapters, independently
strided C and complete portable pre-wrapper source custody; broader E/F and
applicable target/corpus gates remain open. The bound source and public signature
differences are explicit; alias annotations grant no lowering permission.

Readonly tensor aliases now pass both typed package writers as exact independent
logical values. Runtime scalar carriers sharing tensor storage are captured by
value before invocation. Nine actual Merlin libraries preserve original inputs
and complete results through 33 additional invocations; writable tensor overlap
still refuses before any copies or native work. Twenty preparation checks and
all nine integration cases pass, alongside 100 relevant regressions. See
[the input and state boundary](readonly_aliases.md). The explicit root entry
above handles declared writable source sharing; unrelated writable input tensors
still require their actual shared-state mapping.

Strided writable output allocations now additionally qualify f32/f16/BF16
tiled GEMM through actual Merlin libraries. The public functional result
retains all incoming prefix/suffix/padding words, while active signed-stride
stores retain original C and the source K loop. Three added typed packages
replay 27 valid layouts/domains and reject nine lane/grid/bounds conflicts;
the combined six-case read/output matrix passes. See [strided output state](strided_outputs.md).
Continue B/C with writable tensor aliases, output-C read/modify/write and
logical-view model compositions, then the broader E/F and applicable target
gates below. The functional allocation result is an explicit interface choice,
not an implicit dense logical view or an accelerator performance claim.

The existing guarded source-word path now has real runtime non-dense allocation
qualification for f32/f16/BF16 tiled GEMM: padded, column-major, signed/reversed
and repeated readonly addresses retain original masks, source K reuse and C.
Three Merlin libraries replay changing/empty/ragged domains and reject active
out-of-bounds observations. Owning source-allocation tensors and runtime
origin/stride/capacity parameters use ABI v5; no dense logical repacking or new
layout schema is implied. Common emission preserves caller intent on rebuilt
standard-memory functions. See [the exact boundary](strided_memory.md).
Continue B/C with output-C read/modify/write and cross-allocation aliases,
then learned model compositions, broader numeric/effect cases and applicable
target materialization. This corrects an overly broad non-dense-stride gap:
the pure dense witness remains narrower, while source-allocation execution
already supplies a more general checked path.

Pure default-library external math now reaches standard MLIR and Merlin for
52 declarations: 26 function families in f32/f64. Actual shared conversions
are reused with only the missing double tanh/atanh table entries completed.
Original library/purity/prototype/source contracts remain explicit; runtime
recurrence, branches and all 26 returning values stay executable. Upstream's
targeted reciprocal-root expansion and libm conversion resolve actual native
materialization dependencies. The exact extra source-word pass has independent
producer/consumer custody checks while retaining legacy evidence. See
[the numeric and dependency boundary](external_math.md). Continue with broader
external implementations/rounding/RNG, B/C alias/stride and applicable target
materialization; the selected library mapping does not close those requirements.

F32 BF16x3/x6 now reuse the installed pinned-version Triton F32DotTC pass on
pre-layout TTIR. Six actual runtime packages preserve three/six partial
contractions, residual reuse, NaN correction and source C through all three
Merlin memory profiles and reused libraries. Original contracts/effects are
checked before normalization; source intent and normalized graphs stay distinct.
The normalizer binary and output bytes are hash-bound separately from the
source reference revision. See [the numeric boundary](dot_precision.md).
TF32/TF32x3 additionally retain explicit E8M10 dot-input quantization,
pass-generated RNA high parts, ordered cross terms and source C placement.
Six additional runtime packages cover the three memory profiles through
reused Merlin libraries. The logical numeric profile and accelerator
limitations are explicit. Continue with broader E/F external/effect cases,
batched/numeric compositions and B/C alias/stride/corpus/target work below.
No hardware-specific operation is discarded to pass admission.

Uniform conditional reads now retain shared's selected rectangular extents
through the pure tensor proof as read-only subsets. Bounds do not become
fresh-output coverage. Sixteen proof cases and three actual Merlin libraries
cover gate values, fills, tails and empty/runtime shapes; see
[the exact scope](conditional_reads.md). Broader alias/conditional-store and
numeric/effect/target work above remains open.

Scaled contractions now have a working standard-MLIR/Merlin vertical slice.
The 65-case suite covers the five public data formats, FP4 K/M/N packing,
optional/mixed E8M0/E4M3FN scales, batched tiles, accumulator/result formats,
all E8M0 codes, transactional refusals and four real packages. The visible
M/N-packed demo also executes through Merlin. Numeric precision is the
explicit `logical-mx-f64-v1` choice allowed by MX's implementation-defined
boundary; target-dependent emulation and accelerator equivalence are separate
profiles. See [the exact contract](scaled_dot.md).
Runtime FP4 scaled recurrences now additionally qualify changing M/N/K,
packed/group extents, odd-nibble tails, empty/zero-trip domains and two source
assertion failures through the same libraries in all three memory profiles.
The native consumer's checked independent seed copy resolves imported private
loop state; 160 destination/ownership cases pass without relaxing lifetime checks.
Three actual scaled model packages now preserve learned weights, scale buffers,
packed parameters and complete tuple references through native and Merlin host
execution in tensor/buffer/guarded profiles. See [model integration](model_integration.md).
Runtime scalar/shape binding now also materializes only captured typed call
attributes, checks every tensor/result extent, and preserves the static model
ABI. Three actual ragged scaled model packages qualify through Merlin; its
compiler stager independently derives each call's guarded scratch from parent
IR without evaluator data. See [runtime model arguments](runtime_model_arguments.md).
Continue with broader numeric/format/packing compositions,
then the remaining E numeric/external/RNG profiles, B/C irregular alias/stride
cases and F/G effects/corpus/target gates above. Reserved FP6 source packing
remains unresolved rather than being guessed. All A–G workstreams remain open.

Merlin now accepts the derived tensor receipt through compiler-only staging,
registers standard parallel tensor operations, and keeps original FX/TTIR
lineage separate. The actual whole-body LLVM/codegen/HostModel path is wired
into optional qualification. Repeated-call checks exposed writable function
arguments; the staged functional profile now marks them read-only explicitly.
See [the compiler/execution contract](merlin_consumer.md).

Standalone shared tensor packages now also stage directly and execute through
Merlin, including tuple-scan results, descriptor GEMM and sequenced readwrite
memory versions. The explicit standalone profile preserves original source and
records its symbol-only `forward` adaptation without manufacturing a Torch trace.
Workstream A remains open for broader runtime/effectful signature admission and
accelerator staged materialization.

Workstream B now has loaded signed i32/i64 indexed reads from immutable logical
tensors, including both upstream gather representations in the qualified slice.
Native and actual Merlin host checks cover exact inactive-lane behavior, explicit
fills, repeated indices, multidimensional allocation coordinates and active bounds.
See [the executable contract and remaining limits](indexed_memory.md). General
scatter/writable gathers, undefined-lane propagation, multidimensional runtime tensor ABI
and broader numeric/effect profiles remain open. Continue with dynamic tensor ABI
and independently owned/effectful indexed writes; do not substitute an opaque
wrapper or treat native host cases as all-target closure.
Workstream B now also admits nonrectangular masks on proved independent affine
store tiles. Necessary mask inequalities establish tails; inactive values require
incoming or source-proved initialized state. Both profiles qualify periodic and
data-dependent predicates, source loops and changing runtime shapes through
Merlin. The full buffer body retains its original predicated stores. General
indirect scatter and writable indexed reads remain separate work.

Workstream C now has typed by-value runtime scalar inputs, source-position
bindings for interleaved pointer/scalar signatures, and actual native/Merlin
checks for payload and source branch uses. ABI v3 states the difference from
tensor-only signatures explicitly; values remain evaluator data.
See [the runtime ABI and next dependencies](runtime_abi.md).
The same qualified library can now be replayed with independent argument/reference
archives and no compiler, preserving the original qualification and recording
new evaluator-only receipts. This proves scalar values vary without recompilation.

Workstream C now additionally has guarded signed integer domain parameters
(ABI v4), exact symbolic runtime prefixes, untouched readwrite output preservation,
and native/Merlin checks over empty, tail and full domains using the same library.
Negative and proved-empty domains retain their actual semantics and ownership
evidence. General source loop/while domains, strides and alias/mutation remain
open. ABI v5 now binds named dense dimensions to typed runtime scalars and
actual input extents, emits dynamic allocations/results and runtime logical
launch bounds, and reuses one actual Merlin library over independent shapes
including zero. These are actual extents rather than relabeled capacities.
Compiler-only signature v2 retains the scalar bindings without evaluator shapes.
Two-axis tiled addition also preserves runtime dense strides using exact
symbolic cancellation, with changing and zero-element matrices through both
Merlin memory profiles. Non-dense strides, shape expressions and alias contracts remain
open. See [the dimension contract](dynamic_shapes.md).
Runtime M/N/K GEMM additionally retains its source K loop and tiled contractions
through actual Merlin libraries in both profiles, with direct/transposed RHS,
changing strides, zero extents and ragged tails. No kernel-specific production
rewrite or whole-tensor replacement is needed.
Read-only indexed gathers now also retain runtime allocation dimensions and
independent index/result counts through the same compiler/library path, with
active tensor bounds checks, exact inactive fills and empty storage cases.
General scatter, writable-buffer reads and aliasing still need separate proofs.
Workstream E now has actual f16 tensor storage through native and Merlin host
ABIs in both profiles, retaining typed f32 promotions/narrowing and nonzero
f32 contraction accumulators. Rounding/subnormal/overflow and inactive-state
cases have independent references. BF16/fp8 encoding, broader numeric policies
and by-value half scalars remain open; see [half precision](half_precision.md).

Workstream D additionally preserves runtime value-only source `scf.for`
recurrences. Both memory profiles qualify zero/nonzero counts through the same
Merlin library. A checked post-fusion destination-reuse feature exposes original
loop-carried tensor storage to native bufferization while retaining its alias
and lifetime checks. Runtime source buffer-access loops still require separate
domain, initialization and effect proofs.

Guarded read-only buffer-access loops now derive a conservative IV interval
from actual entry-checked bounds while retaining the executable runtime trip
count. Both profiles qualify i32/i64 counts, negative/zero trips and row tails
through one library. Mask coordinates remain distinct from physical addresses;
access bounds do not grant complete initialization. Runtime source stores now
carry conditional tensor state and preserve incoming readwrite values on zero
trips or an unexecuted store branch. Possibly empty loops retain only entry
initialization facts; guarded positive loops retain proved post-body facts.
Both profiles qualify read/modify/write recurrences and preinitialized/positive
fresh outputs through Merlin. Unknown branch initialization still refuses.
The full memory profile currently uses the same proof witness; independent
memory admission, broader headers/while/pointer state and non-dense runtime layouts
remain required.

Workstream B now additionally has independent memory admission through the
explicit `guarded-buffer` profile. Loaded-index writes retain the full upstream
payload while executable checks enforce active bounds, per-program ownership,
per-store lane uniqueness and fresh-output coverage. i32/i64 indices, repeated
source stores, partial updates and zero/extreme inactive cases execute natively;
one actual Merlin library also runs independent runtime shapes. The package
replaces the tensor witness with a byte-bound admission report and declares its
grid-derived per-output-element guard cost. This closes that witness dependency for
the qualified private-output profile. Aliasing, general
pointer state and source effects remain open; see [guarded memory](guarded_memory.md).
Static guarded kernels also bind into actual model2MLIR captures, with ordinary
and ragged tiled GEMM, block pointers and descriptors compiled/executed through
Merlin. Per-call scratch accounting binds typed kernel results to model resource
declarations instead of estimating metadata from the final model result size.
Scalar indirect loads/stores additionally normalize to unit tensors before
the same upstream memory conversion, retaining exact source masks/fills and
index/value word types. Scalar and tiled cases both qualify through native
execution and a reused Merlin library over independent runtime dimensions.
V2 admission additionally preserves writable indirect observations before and
after source stores in runtime recurrences. Source-base gather views are
adapted to actual memory loads before CSE; structured loads retain their copies.
Sticky reader/writer tags allow shared readers, reject cross-program conflicts
in either order, and cover every i64 program-ID fragment. Completion cells
enforce read initialization and fresh-output coverage. Typical grids retain
16 bytes of metadata per element; wider domains require 24/32. Native checks
at fragment boundaries avoid huge launch experiments, and scalar/tiled
recurrences reuse one actual Merlin library over changing runtime inputs.

The explicit `source-words` address policy now reuses upstream's supported
`offset-bit-width=64` option. Its accumulator sign-extends each pointer operand
after original integer computation, without a local pointer converter. The
default structured range gate remains unchanged. Separate source-word stage
bytes, hashes, pass sequence and function/grid access intent are retained in
the `shared-source-word-buffer` profile; see [the contract](source_words.md).

Qualification covers loaded-index i32 wraparound, narrow chains with later
i64 cancellation, scalar/tiled masks, runtime `for` pointer state and final
values, nonlinear grid permutations and independent words/shapes through one
Merlin library. Real static model bindings retain contractions and loops.
The consumer's closed readonly-capture destination reuse keeps fused reads and
ordinary alias/lifetime checks, including zero-trip loops. Same-base pointer
branches and while state now have a narrow control-edge adapter before the
unchanged upstream stage. All branch/backedge roots participate in a fixed
point; source word math, masks, regions and final pointer results remain.
Actual Merlin libraries qualify independent shapes/counts and zero trips;
complete while destinations retain reuse when the condition forwards state
unchanged. Mixed-base pointer tensors, broader while ownership, aliases and
broader effects remain open. Preserve useful tiling and reuse as those profiles expand.

Nested pointer branches inside runtime `for`/`while` now also qualify through
Merlin, retaining helper inlining, selected row addresses, recurrence and final
pointer observations. The earlier repeating-control lifetime refusal remains
in `.scratch/nested-pointer-control/` as a diagnostic. Its resolution uses
generic conditional destination planning: each branch must qualify before
changes; complete computations retarget unused destinations, while captured
values use an explicit pure value copy into carried state. The original
branches, predicates and loop remain; native alias/lifetime checks stay enabled.
Deeper nested branches, independent choices/counts, zero trips, empty/tail
dimensions and actual selected-branch out-of-bounds failures are maintained tests.

Conditional reductions with nonzero initial state also qualify. Their original
reduction/init reads remain intact; a complete pure value copy supplies the
loop destination after closing the producer graph against unknown effects and
unproved memory captures. One unchanged Merlin library retains all results and
inputs across independent invocations. Computed/permuted while before-state,
broader shape/ownership, mixed-base joins, alias groups and broader effects
remain open. Do not infer universal coverage from this composition.

Tiled online attention now retains two contractions and three online softmax
states through actual Merlin execution. Late state copies preserve old maximum
reads before updates; independently synthesized state may read actual private
snapshots without claiming those buffers are immutable. Loop-local capture
producers and pure padding/conditional values remain checked. This also restores
the earlier tensor/buffer runtime-read-loop qualification; the failed broader
closure run is retained in `.scratch/branch-state-complete.log` and its eight
cases pass in `.scratch/branch-state-regressions.log`. See the maintained
[attention profile and explicit numerical scope](attention.md).

The final conditional-state regression run passed 506 tests with all existing
toolchains enabled (four block-pointer deprecation warnings), and Merlin's
destination/deallocation suites passed 76 tests. After that run, attention's
profile matrix was expanded: all nine profile tests pass for tensor, standard
buffer and guarded source-word emission, including eight independent query/key
shapes per compiled library. Extra tensor output state is checked as invariant.
The maintained tensor demo is `out/online-attention-tensor/`, with separately
verified compiler/native execution under `out/online-attention-tensor-merlin/`.
The guarded demo and qualification remain under `out/online-attention/` and
`out/online-attention-merlin/`. Accelerator execution remains unperformed.

Finite mixed-base pointer joins/backedges and lane selection now execute through
the source-word path. The fixed-point source-entry base sets become logical
buffer selectors paired with element offsets. Predicated accesses stay at each
original memory event, preserving intervening mutations, masks/fills, integer
words and source IDs. Actual Merlin libraries qualify scalar/tiled selections,
changing for/while bases, selected updates, shape transforms and independent
empty/tail inputs. An unselected empty buffer is valid; selecting an invalid
active address aborts. A narrow Merlin extension recognizes immutable tensor
read captures and private snapshots when proving fresh loop-state ownership;
the destination/deallocation suites pass 83 cases. See the maintained
[selected-pointer contract and source reuse boundary](selected_pointers.md).

Additional native/Merlin cases preserve raw FP16/32/64 transport bits and i32
values, nonuniform fills, wrapped i32 address producers and cancellation across
separately widened increments. These 13 cases pass; provenance and compatibility
metadata explicitly record the upstream zero refinement of absent masked fills.
General alias groups, non-dense views, pointer observations, broader loop
ownership, numeric/effect profiles and target materialization remain open.

The installed binary's cat/gather version gap now has a source-verified bridge.
Cat chooses lhs-then-rhs through native join/transpose/reshape, keeping original
and normalized inventories distinct. Tile gather adapts Meta's parallel
generic/extract expansion and checks indices before extraction. Native and
actual Merlin cases now include selected-pointer cat/split/repeated gather and
two-axis value gathers across all three memory profiles, with bit-exact FP32
transport, i32 values and actual invalid-index aborts. This closes the measured
composition gap without a target build or physical lane mapping.

The selected-memory regression passed all 541 producer tests with every existing
native/model2MLIR/Merlin environment enabled (four legacy block-pointer warnings).
After that run, the version bridge passes 12 pointer-shape/block-pointer cases,
and direct tile gather passes 42 cases across all three memory profiles. Wide
i64 indices are checked before index casts, including high-bit truncation traps.
The runnable `out/selected-memory/` package and its separate qualification at
`out/selected-memory-merlin/` pass compiler staging, native execution and all 45
artifact checks. Accelerator execution remains unperformed.

A real model2MLIR model now binds two selected-pointer/gather calls with distinct
scalar specializations and learned weights. Parent data/goldens remain byte
identical; actual Merlin compilation/execution passes. The visible derived
bundle `out/selected-memory-model/` verifies all 101 artifacts. Both package
writers now use the same representation-difference declarations; bound models
list each kernel's profile explicitly in additive compatibility metadata.

The next direct source probe found scalar maps and histogram unhandled, while
numeric unsplat already lowers. Scalar maps now have a qualified `pack=1`
expansion retaining their scalar branch/loop region and tuple results inside
parallel Linalg. Fourteen cases pass native and actual Merlin checks across
all three memory profiles. Packed grouping and mapped source-memory events
still need separate contracts; histogram remains unhandled. See
[the expansion, provenance and remaining scope](elementwise_map.md).

The combined memory/model regression then passed all 590 collected tests with
all existing toolchains enabled (four legacy block-pointer warnings). Scalar
maps were added after collection and pass their separate 14-case qualification.
Pointer unsplat now also retains a one-element logical selector/offset through
the unchanged upstream extraction, with 20 pointer-unsplat/control cases passing
native and actual Merlin execution. These counts describe measured cases, not
universal opcode/type/effect coverage. The goal remains open for aliases,
layouts, packed/memory maps, histogram, numeric/effect profiles and target work.

The same real selected-memory model also passes with half weights and inputs.
The manifest loader already preserved f16, but the bound-model size-budget gate
rejected it. That gate now admits two-byte elements; both actual model dtype
cases and the existing half profile pass all nine tests. This extends real
model-package qualification without changing tensor storage or the ABI.

Scalar-map qualification now covers original scalar while recurrences as well
as for loops: all 26 map cases pass with native and actual Merlin tools. The
source-backed Ascend comparison is recorded in the map contract; promoting a
scalar branch to a tensor select or duplicating packed lane arguments is not
used to justify these semantics.

Tile-local histogram now also reaches actual Merlin execution through a linear
indexed tensor-state expansion, retaining histogram composite intent. All 34
cases qualify runtime tiles, both integer input widths, exact masks/padding and
extreme values. Matching masked row views pass all three profiles; flat output
pointers into dynamic matrix results pass the guarded profile and retain the
existing tensor-witness rank refusal. The initial compiled LLVM probe reuses
counter storage without per-element allocation or full-bin copies. Ascend's
high-level histogram and target builtin are recorded as source boundaries,
without introducing an opaque portable fallback. See [the contract](histogram.md).

[Actual logical-grid bounds and exact runtime partitions](logical_grid_bounds.md)
now admit unchanged unmasked dynamic rows in tensor and buffer profiles, with
separate access/coverage proofs. Mixed unmasked rows and clipped columns also
qualify after exact dense uniform-origin recovery. Independent ownership,
source-word arithmetic and fresh-output checks remain intact.

Complete regression after the grid/event extension on 2026-11-09:
**718 passed**, four legacy block-pointer
warnings, with all existing native/model2MLIR/Merlin environments enabled.
The tensor histogram demo at `out/histogram-tensor/` passes separate compiler
staging/native execution under `out/histogram-tensor-merlin/`, with all 38
qualification artifacts verified. The f32 and f16 selected-memory model demos
each verify all 101 bound artifacts. That measurement used Merlin anchor
`5f989aaf19f1a2326e10012be6d1436456893f4b`; its 83 destination/deallocation tests
and documentation/structure gates passed for the loop-read extension.

The initial production grid-proof suite passed 87 cases, including native and
actual Merlin execution over independent domains and the 3,484-scenario affine
oracle, with two strict expected failures. Both source-word conditional-store
cases now pass through actual Merlin after [isolated native event conversion](source_words.md#memory-inside-structured-regions).
Every event is converted by the official pointer pass in a typed straight-line
template and substituted at its original region position. The parent receives
arithmetic conversion only. V2 word evidence retains both actual native reports
and intermediate bytes; bundle checks reconstruct the complete typed event
inventory/placement. Merlin's 77 standalone/derived intake tests and structure/
documentation gates pass at anchor `0aae21d4aee40b1aadf76caf745edb501fc326d5`.
Universal opcode/type/effect and accelerator coverage remain open.

The region-event extension also passes 97 existing source-word/pointer/control/
selected-memory/model/runtime-memory regressions. Nine new cases check scalar
and tiled memory in `for`/`while`, raw fill/result bits through native and actual
Merlin libraries, and rehashed evidence/type tampering. A real model2MLIR GEMM
package also qualifies branch-local tile contractions and K loops with original
weights and independent PyTorch goldens retained.

The subsequent [unsigned high-product/Philox extension](integer_products.md)
passes 35 new focused cases and the 21 existing shared-tensor/buffer regressions.
It adapts Ascend's standard extended unsigned multiply, retaining i32/i64 word
semantics and parallel tensor dataflow. Original Triton integer Philox rounds
execute through actual Merlin libraries with runtime seed/counter bindings and
all four results, against an independent reference checked by three published
known-answer vectors. A checked neutral-default adapter resolves native
`arith.trunci overflow<none>` versus the older xDSL schema; poison-producing
flags still refuse. The complete 718-case run predates this arithmetic addition;
the 35+21 results qualify the changed production path separately.

Validation on 2026-11-09: the complete producer suite passed 492 tests with all
existing native/model2MLIR/Merlin environments enabled (four legacy block-pointer
deprecation warnings). Merlin's destination/deallocation suites passed 57 tests;
its documentation/structure gates passed. `out/pointer-while/` contains the
visible source-word package, and `out/pointer-while-merlin/` passed separate
compiler staging, native library execution and all 45 artifact checks.
Accelerator execution remains unperformed. These are measured cases and custody
checks, not universal TTIR or target coverage.

A closed full-pipeline integration probe now compiles triton-shared's later
mixed linalg/tensor/memory body through Merlin and checks variable prefixes and
input preservation. The old categorical consumer incompatibility is therefore
not a reason to avoid the full shared pipeline. Follow [the full memory bridge
requirements](full_memory_bridge.md) to generalize its typed functional boundary,
memory/numeric contract, source correspondence and true dynamic dimensions.
The explicit `--memory-profile buffer` now implements the initial typed wrapper,
complete package/model binding and Merlin compiler/library qualification within
the existing tensor memory proof scope. Its additional artifacts bind the actual
continuation bytes. Independent memory admission, non-dense runtime layouts and
broader effects remain open; this is not universal source coverage.

### Typed atomic and logical-view extension

The typed atomic profile now admits bitwise f32/f64 exchange while preserving
integer RMW/CAS, scopes, original old values and the public tensor/scalar ABI.
V4 memory admission makes the new numeric contract explicit; Merlin staging
independently checks it. Native/Merlin raw-word comparisons cover exceptional
float representations, masks and collisions. A real model2MLIR capture keeps
weights and both tuple results. See [atomic memory](atomic_memory.md).

Ordinary equal-width pointer views now use original-storage memory events with
register bitcasts and `prov.logical_word_views` annotations. Source integer
fills remain bit patterns, including those representing NaNs; masks, selected
roots, source loops and zero trips are retained. f16/i16, f32/i32 and f64/i64
execute through native and Merlin interfaces without whole-buffer conversion.
See [logical views](pointer_views.md).

Next, expand atomic word updates on these storage tensors into exact standard
atomic regions and legalize them through the native ownership/lifetime boundary.
This is required for Triton's floating min/max source algorithm. Floating add's
rounding/subnormal/NaN policy is a distinct obligation. Broader packed numeric
formats, descriptor indexed effects, external implementations/diagnostics,
aliases/layouts, maintained typed coverage and target materialization remain
open. Neither a numerical package check nor provenance closes those gaps.

The atomic word-region extension now implements all nine i32/i64 RMWs on
f32/f64 storage and preserves Triton's sign-conditioned floating min/max
algorithm. Standard generic atomic regions retain source bits and old values;
a checked native feature emits one integer LLVM atomic per primitive while
retaining ownership/lifetime gates. Native and actual Merlin raw-word tests
cover exceptional float representations, and real min/max model2MLIR captures
retain weights and tuple results. Source region/selection compositions now
qualify, including repeated inlined helpers with distinct memory contexts.
IEEE f64 addition also qualifies through standard upstream atomic conversion,
native/Merlin region and collision tests, and a genuine f64 model2MLIR model.
It records round-to-nearest-even, subnormal preservation and unspecified
arithmetic NaN payload/sign without changing the package ABI. Next, implement
other floating policies and the other universal
gaps. This remains bounded coverage rather than arbitrary-kernel completion.

F32 addition now has explicit IEEE/FTZ policy selection. Both use the same
package and Merlin interface; FTZ retains standard executable bit arithmetic,
including NaN quieting after canonicalization. IEEE preserves subnormals and
also retains explicit quieting in its standard atomic region.
The native FTZ adapter reconstructs complete typed atomic bodies and exact
internal atomic-load/CAS retry helpers through ownership/lifetime verification.
No policy is inferred from kernel names, provenance or target capabilities.

The constant audit also exposed the f64 negative-zero quieting gap. New f64
captures now use `ieee-f64-rne-add-quiet-nans`, with executable quieting in the
standard atomic region and the shared typed native retry boundary. The old
standard-addf profile remains readable with its original weaker NaN contract.
Artifact/receipt names and the tensor/scalar ABI are unchanged. Tests construct
source constant words explicitly, so a Python constexpr zero-sign conversion
cannot make the test exercise a different TTIR operation.

Integer-word CAS now additionally retains f16/i16, f32/i32 and f64/i64 storage
views through the working Merlin interface. Closed standard atomic regions
preserve exact comparisons and old values; integer native helper results avoid
half floating ABI conversions. Source region/selection/rank-two compositions,
constants and actual model2MLIR tuple captures are qualified. The stage-v2
typed-template variant is an explicit package difference, with complete producer
replay and consumer custody checks. This closes word CAS within the private
typed profile; other numeric formats, descriptors,
external effects/synchronization, maintained coverage and staged accelerator
materialization remain completion requirements.

Ordinary selected memory now retains different equal-width storage dtypes per
root, including exact fills, selected writes/read-after-write observations and
final pointer state. Closed selector permutations keep immutable tiles and
carry scalar table indices through original loops. The 52 focused cases,
75 existing pointer/event regressions and three true model2MLIR captures reach
the actual Merlin interface. Native ownership/lifetime rules are unchanged.
See [mixed storage](mixed_storage.md); general computed selector recurrences,
aliases/layouts and the remaining workstreams above are still completion gates.

Selected integer atomic events now additionally retain each root's integer or
equal-width floating storage. The consumer uses complete source context plus
actual storage captures, with separate logical CAS template types. All nine
i32/i64 RMWs and i16/i32/i64 CAS qualify through native/Merlin execution,
including collisions, source regions and repeated helpers. Five real model2MLIR
captures preserve learned weights and all three tuple references. The 76
execution cases, six profile refusals, 18 existing execution regressions and
290 consumer declaration/staging cases pass. Public ABI and artifact schemas,
primitive checks, producer reconstruction and lifetime gates remain unchanged.
This closes that typed composition; broader numeric/effect, descriptor,
runtime/alias and accelerator-materialization requirements remain open.

BF16 standalone tensors now additionally qualify through actual Merlin libraries
in all three memory profiles. Twenty focused cases retain exact uncomputed
words, IEEE RNE casts, runtime shapes/selection loops and BF16/F32 contractions
with original nonzero accumulators. The combined low-precision/floating-atomic
regression passes 97 cases; native seed and transport gates remain checked.
The visible `out/bfloat-matmul` package retains `linalg.matmul`, and its separate
Merlin qualification records host execution. See [BF16](bfloat_precision.md).
Exact BF16 source inputs/results and safetensors weights now also bind into
real model2MLIR captures while preserving parent archives and receipts. Eight
model cases cover all three memory profiles and unique non-NaN legacy embeddings;
15 protocol cases plus a rehashed NaN corruption refusal give 24 focused cases.
The 62 existing model/atomic/half regressions, 30 model2MLIR capture checks and
50 Merlin staging/constant-reader checks pass. The visible observation/matmul
models pass 81/80 artifact checks including actual Merlin host execution.
Legacy widened NaNs still require original words. Dynamic/alias model
contracts, direct BF16 arithmetic
policies, FP8/scaled-dot, external effects, descriptors and accelerator
materialization remain completion gates; no numeric or ownership check is waived.

The subsequent [lifted-constant gate](lifted_constants.md) now reaches actual
Merlin execution. New model2MLIR captures bind exact source constants in
safetensors with a complete typed argument profile. Compiler staging checks
the original prepared graph and complete parent ABI, excluding evaluator
archives and their value digests. Real copy models cover nine tensor carriers;
lifted BF16 observation/matmul models retain all results, tiling, original
contractions and nonzero accumulators in all three memory profiles. Checked
[Boolean byte-slot normalization](boolean_memory.md) retains logical masks,
fills and source IDs. Zero-tolerance integer validation now detects individual
i64 bits. Rehashed source/type/shape/payload failures remain refusals.
Merlin's combined constant/staging/runtime regression passes 153 checks, with
two existing optional cross-toolchain cases skipped; docs/structure gates pass.
Visible lifted observation and matmul packages execute through Merlin.
The final 33-case model run and 13-case Boolean suite pass, alongside 36
integer/reference regressions and 48 model2MLIR source capture checks. Saved
final packages pass 82/81 artifact checks; accelerator execution is unperformed.

The [package observation registry](package_coverage.md) now joins all 50 pinned
definitions to receipt-bound package/native/Merlin outcomes and available
typed source forms. Its explicit 13-package selection observes 22 definitions;
18 occur in captures of nine host-qualified packages. The ten registry/corpus
tests pass. Native original-source attribute dictionaries remain explicitly
unavailable rather than empty; their exact source bytes stay archived.
A maintained full corpus/refusal matrix and independent source dispositions
remain open, alongside dynamic/alias model contracts and target materialization.
Noncanonical Boolean storage, broader low-precision
arithmetic, FP8/scaled dot and external effects still require executable
contracts; these transport results do not close them.

Source `tt.assert` now additionally reaches the actual Merlin interface using
the existing upstream scalar/lane-wise CF conversion. All 24 focused cases
retain condition/message provenance and source loop/branch positions, exercise
successful and failing invocations, and qualify a real three-result model2MLIR
capture in all memory profiles. The 41 existing diagnostic/guard/consumer cases
also pass. The saved standalone qualification/model pass 45/80 artifact checks.
The explicit 15-package registry now observes 24 definitions, with 20 present
in captures of host-qualified packages. See [the assertion contract](source_assertions.md).

The next effect/package gates are concrete extensions of the original scope:

1. Admit effect-only kernels with no tensor result through an explicit package
   and Merlin signature profile. Keep actual void control/effects; never invent
   a tensor result to fit the current package writer.
2. Represent initial expected-failure outcomes and diagnostic references,
   including always-failing source assertions. Bind actual execution/termination
   evidence without fabricating a successful numeric golden or success receipt.
3. Add a length-aware diagnostic ABI before accepting embedded NUL messages or
   broader print/formatting effects. Retain exact bytes, source order and failure
   scope through the selected consumer implementation.

External functions/assembly, synchronization, broader low-precision arithmetic,
runtime/alias models, the complete maintained corpus/refusal matrix and target
materialization remain required work. This does not redefine the goal around
the successful tensor-returning or host-executed cases.

The first two effect/package gates above are now implemented in
[ABI v6 and observable-outcome packets](effect_abi.md). Real zero-argument,
scalar and readonly dynamic tensor kernels keep their actual void signatures;
initially failing tensor kernels retain declared results without publishing
values. Independent JSON references bind exact return/assertion-failure outcomes
and byte streams. Invocation-phase records distinguish source failures from
setup errors, timeouts and unrelated signals. Native observations never become
compiler inputs: Merlin independently stages and compiles the portable payload.

The 40 focused producer/consumer/CLI cases pass, including all memory profiles,
empty/tail/full domains, loop zero trips, repeated use of one library and
rehashed outcome refusals. Another 73 existing numeric/diagnostic/model/coverage
regressions and 230 Merlin staging checks pass. A further rehashed-signature
case verifies that failure cannot hide a real tensor result. Saved constant-fail
and fail-before-store demos reach actual Merlin execution with no numeric golden.
The literal portion of the third effect gate is now implemented through
[length-aware assertion lowering](literal_diagnostics.md). Complete NUL/UTF-8
messages survive real native and Merlin execution without changing portable
CF operations or public signatures. The nine end-to-end cases include all
memory profiles, rank-two/dynamic conditions, reused libraries, original CF
messages, BF16 and CAS compositions, and truncated-reference refusal. The
native compiler's progress log is separately bound so it cannot pollute
source stdout. The combined frontend regression passes 75 cases. Merlin's
70 native/serialization/BF16 cases and docs/structure
gates pass. General print/formatting and broader external/synchronization
contracts remain open, along with the original numeric/runtime/alias/corpus/
target work.

The next print portion now has a working
[logical diagnostic profile](logical_prints.md). Registered `tt.print` stays
through actual shared tensor/memory conversion, then expands into standard
scalar/string printing and executable formatting arithmetic at the logical ABI
boundary. Physical thread replicas/interleaving are abstracted explicitly,
while values, source control positions, exact words and logical coordinates
remain executable. All 53 focused cases, seven CLI cases, 68 existing frontend
regressions and 84 Merlin native/binding/staging checks pass.

The package gate for **returning tensor values and diagnostic streams
together** now has an explicit [mixed reference profile](mixed_effects.md):
complete typed result archives, per-slot word/value policies, compiler
exclusion, one-call evaluation and reused-library verification. Numeric i64
recurrence retains its width/control/data reuse through the supported upstream
prepass selection; native functional inputs receive the upstream readonly
argument contract. Both checks preserve the original memory/ownership gates.
Pointer diagnostics need logical-address semantics; broader numeric, external/
assembly/synchronization, corpus/runtime/alias and target work stays in scope.

The next numeric-closure increment reuses the actual shared clamp/precise-math
converters and independently qualifies f16/BF16/f32/f64 typed TTIR through
Merlin. Public f32 builtins also bind into a real model2MLIR capture with learned
bounds and eight preserved results. Exact-rational references and the explicit
IEEE policy check rounding, subnormals, infinities, NaN modes and signed zeros.
The selected BF16 scalar-math native feature retains source BF16 payload and
storage, uses upstream f32 emulation/rounding casts, and leaves word transport
and ownership/lifetime checks separate. See [the full contract](scalar_math.md).
Packed low precision, scaled dot, external implementations, runtime/alias
models, the maintained corpus and applicable target materialization remain open.

The FP8 reader prerequisite now preserves all four pinned TTIR FP8 types and
all 256 constant words in each format through frontend/Merlin serialization
and the native MLIR reader. Decimal constants use an optional standard codec;
raw NaN/zero encodings retain their source bytes. Type-only parsing requires
no codec. This is reader evidence, not additional successful kernel/numeric
coverage. Continue with the explicit FP8 storage ABI, numeric rounding and
overflow, then contractions and targets in [the cast plan](float_casts.md).

The next vertical slice now executes FP8 transport for all four formats through
the public tensor interface and reused Merlin libraries. Original FP8 types,
masks, runtime loops and every encoding remain intact in tensor, buffer and
guarded-buffer profiles. Exact word receipts and separate byte checks retain
NaNs/zeros; numeric comparisons decode the declared format. Standard native
byte lowering suffices. Merlin's value-type classification and C storage
accounting recognize FP8 without relaxing alias, release or lifetime rules.
Numeric casts still need explicit executable lowering: direct extension probes
leave unresolved conversion casts. Continue numeric closure, not a transport-only
definition of success.

FP8 widening now additionally retains portable `arith.extf` and executes its
scalar native expansion through the Merlin interface. All 48
format/destination/memory-profile combinations preserve exact finite values,
nonfinite classes, source control and adjacent original word results over
every input encoding and reused libraries. Independent rational references
and codec cross-checks support the explicit IEEE/word policies. The selected
native feature preserves every ownership callback and typed storage. Continue
with rounding-aware narrowing, equal-width formats, source saturation policy,
broader models/scaled contractions, external/effect closure and targets;
the full workstreams and denominator above remain unchanged.

Standard-format casts now additionally preserve source RTNE/RTZ through
executable standard MLIR expansions. All seven narrowing/format pairs qualify
both modes in tensor/buffer/guarded profiles (42 combinations), including
exhaustive f16/BF16 source encodings, adjacent wider midpoints, exact raw source
results and reused libraries over empty/ragged shapes. An f32 round-to-odd
bridge avoids f64→16-bit double rounding without a missing native helper.
The real public RTZ source demo uses independent rational references. Intake
caller provenance fields survive upstream rewriting and emission, with complete
source-ID intent mappings for merged operations; tags grant no semantics.
FP8 narrowing and distinct-format changes now also have executable RTNE/RTZ
and three explicit overflow/infinity contracts. The 127-case slice covers all
f16/BF16 words, wider source boundaries, every distinct FP8 pair and actual
Merlin libraries across three memory profiles. Twelve source-loop cases retain
per-step rounding under changing counts/dimensions/scalars; computed NaN class
and exact adjacent words remain separate policies. The native implementation
uses standard integer significand rounding before widening/BF16 legalization,
without changed storage or ownership. A global selected profile is explicit;
heterogeneous per-cast backend contracts are still an open requirement.
Mixed FP8→f32 contractions now additionally preserve named tile structure,
original C, runtime M/N/K and K padding through 24 actual Merlin packages.
All sixteen input pairs qualify in tensor form; every same-format pair also
qualifies in the buffer profiles. Twelve signed-zero/downstream-add packages
retain those separate observations. The complete 93-case slice includes
precision applicability, typed-zero and direct repair/refusal checks. Native
generalization uses an official pass without changing portable named ops.
Source-width signed quotient intervals also support actual `cdiv` loop bounds
without supplying affine alignment facts. See [the source-backed contract and
scaled-dot implementation order](fp8_contractions.md). All A–G requirements and
the full definition/type/attribute/composition denominator remain intact.
Continue FP8 arithmetic, heterogeneous policies, broader and scaled dot,
runtime/alias and external/effect closure, maintained corpus and applicable
target materialization. Standard cast closure is not full TTIR closure.

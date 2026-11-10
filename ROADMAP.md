# Roadmap: Triton kernel to a Merlin capture package

**Updated 2026-11-10.** The architecture in [plan.md](plan.md) is accepted as the starting point:
tile-first conversion, a single semantic emission path, and original intent carried through
annotations/provenance while computations lower into standard MLIR. This roadmap tracks delivery;
the design document explains the transformations. Milestones below are capability gates, not dates.
The active [completion plan](docs/completion_plan.md) orders the remaining frontend,
compiler-staging, payload-materialization and coverage work with explicit evidence gates.
[Scaled contractions](docs/scaled_dot.md) now preserve FP4 K/M/N packing,
optional E8M0/E4M3FN scales, mixed operand types and original C through standard
tiled MLIR and real Merlin packages. The internal-precision profile is explicit;
runtime/ragged scaled compositions and applicable target materialization remain.
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

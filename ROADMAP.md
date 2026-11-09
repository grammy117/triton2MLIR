# Roadmap: Triton kernel to a Merlin capture package

**Updated 2026-11-08.** The architecture in [plan.md](plan.md) is accepted as the starting point:
tile-first conversion, a single semantic emission path, and original intent carried through
annotations/provenance while computations lower into standard MLIR. This roadmap tracks delivery;
the design document explains the transformations. Milestones below are capability gates, not dates.

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
accept both model interfaces; target/staged execution remains unqualified.

Permuted loads/stores, data-dependent `scf.if` memory merges and affine pointer-to-index
recurrences and clipped affine tail masks now have native checks. Ragged M/N/K GEMM retains
its K loop and tile sizes, including transposed weights. The next boundaries are runtime
scalars/shapes, arbitrary masks, grouped or
nonlinear indexing, block/irregular descriptors, general while/pointer state and numeric profiles.

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
checks. General combiners/scans, numeric edge policies and optional composite recognition remain
capability work; no dense attention replacement or universal reduction coverage is claimed.

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
MLP pair passes with identical source weights and eager goldens. Derived receipts are explicit;
Torch-only receipt admission, general mutation schemas, multiple-model-output references and
staged/device execution remain separate gates.

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

Expand the existing shared tensor path from its tested static-grid/bounded-loop profile to
nonlinear/live pointer state, block and irregular descriptors, guarded runtime shapes/masks,
general `while`, scans and explicitly qualified mixed precision. Preserve earlier mask,
ownership, source accounting and native numerical regressions. Inventory all remaining failures
against the [measured corpus](docs/coverage.md), not a claimed opcode percentage.

The binder now preserves model2MLIR's actual model ABI and learned parameters, and Merlin's
reader/runtime-generation interface is checked. Derived-receipt admission, general mutation and
model-result schemas, plus staged/device materialization, remain explicit consumer gates.
Backend-observing operations still need the effectful/backend-bound profiles defined above;
they cannot silently become pure tensor math.

# Completion plan: TTIR kernels through the Merlin compiler interface

Goal: all valid kernels in each pinned supported TTIR version reach a working
Merlin compiler interface with their observable computation/effects represented,
the useful algorithm retained, and applicable execution evidence. The existing
199-case regression suite and interface/runtime-generator demos establish a
baseline, not completion. Previous implementation turns made concrete progress;
this goal begins with remaining requirements still open.

## Completion evidence

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
local contribution/identity rules; commits remain local. Update this plan with
verified evidence and unresolved requirements, not just intentions.

## Execution evidence and next handoff

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
Workstream A remains open for runtime/effectful signature admission and
accelerator staged materialization.

Workstream B now has loaded signed i32/i64 indexed reads from immutable logical
tensors, including both upstream gather representations in the qualified slice.
Native and actual Merlin host checks cover exact inactive-lane behavior, explicit
fills, repeated indices, multidimensional allocation coordinates and active bounds.
See [the executable contract and remaining limits](indexed_memory.md). General
scatter/writable gathers, undefined-lane propagation, true dynamic tensor ABI
and broader numeric/effect profiles remain open. Continue with dynamic tensor ABI
and independently owned/effectful indexed writes; do not substitute an opaque
wrapper or treat native host cases as all-target closure.

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
open. The remaining dynamic-shape work must introduce dimensions bound to
actual tensor extents, corresponding runtime launch bounds and result allocation;
it cannot relabel static capacities as dynamic tensors. See [the runtime contract](runtime_abi.md).

Workstream D additionally preserves runtime value-only source `scf.for`
recurrences. Both memory profiles qualify zero/nonzero counts through the same
Merlin library. A checked post-fusion destination-reuse feature exposes original
loop-carried tensor storage to native bufferization while retaining its alias
and lifetime checks. Runtime source buffer-access loops still require separate
domain, initialization and effect proofs.

The next control slice is guarded read-only buffer-access loops: derive a
conservative IV interval from the actual entry-checked bounds, retain the runtime
trip count and zero-trip behavior, and grant no write/initialization facts from
possibly unexecuted iterations. That proof should feed the independent full
memory boundary as well as the tensor representation.

A closed full-pipeline integration probe now compiles triton-shared's later
mixed linalg/tensor/memory body through Merlin and checks variable prefixes and
input preservation. The old categorical consumer incompatibility is therefore
not a reason to avoid the full shared pipeline. Follow [the full memory bridge
requirements](full_memory_bridge.md) to generalize its typed functional boundary,
memory/numeric contract, source correspondence and true dynamic dimensions.
The explicit `--memory-profile buffer` now implements the initial typed wrapper,
complete package/model binding and Merlin compiler/library qualification within
the existing tensor memory proof scope. Its additional artifacts bind the actual
continuation bytes. Independent memory admission, true dynamic dimensions and
broader effects remain open; this is not universal source coverage.

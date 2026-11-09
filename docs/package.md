# Package contract and explicit model2MLIR differences

Baseline inspected: model2MLIR `feat/triton-integration` at `035bb5a`, particularly
`m2m/capture/bundle.py`, `m2m/capture/provenance.py` and `m2m/transforms/externalize.py`.
This document specifies the implemented R0/R1 package, not every future roadmap profile.

## Shared tensor profile (`t2m.capture-receipt.v2`)

The [upstream-based frontend](shared_tensor.md) uses the same core filenames with an explicitly
versioned ABI and receipt. `shared-demo` writes complete examples; `shared-tensor` accepts saved
TTIR plus `--abi`, optionally `--inputs` (`in0`, `in1`, ...) and `--reference` (`out0`, `out1`, ...)
NPZ archives. Without both arrays and references it writes IR/provenance only, with no receipt.

In the original v1 tensor ABI every buffer, including writable incoming values, is an argument.
Writable results follow declaration order. The function retains its kernel name. Specialized
integer scalars are recorded in `spec.json` rather than becoming runtime tensor arguments.
`golden.npy` stores result zero; `extra.npz` stores any additional results as `out1`, `out2`, ... .
That use of `extra.npz` is an explicit convention difference, not implied m2m compatibility.

Before writing any bundle, the writer executes the serialized MLIR through the native host LLVM
path and compares every result with the supplied independent references at explicit tolerances.
`execution.json` binds those checks to the exact IR hash. It makes no NaN-payload/signed-zero-bit,
universal source-equivalence, original Triton-execution or consumer/device-execution claim.
`verify-bundle` dispatches both receipt versions and verifies all artifact/source/ABI bindings.
The full list of differences is also embedded in each package's `compatibility.json`.

`t2m.shared-tensor-abi.v2` adds a `write` buffer effect for fresh model results.
These buffers become results without incoming function arguments. Conversion checks every
observed load against preceding stores and proves complete coverage from bounded, disjoint
tiles with equal total cardinality, or an exact clipped-prefix partition for ragged tails.
Unwritten holes or initial reads are refused. `readwrite`
continues to preserve incoming storage values. Existing v1 read/readwrite specs retain their
canonical serialization; package receipts continue to describe the actual materialized ABI.

The [bound-model profile](model_integration.md) keeps model2MLIR's actual core
data files, function ABI and learned weights. It uses a distinct
`t2m.bound-model-receipt.v1`, with original capture/FX evidence retained under
`parent/` and explicit TTIR binding/native/consumer evidence for the derived IR.
Optional compiler staging and native Merlin materialization add explicit
`consumer/compiler/` and `consumer/host/` products; the qualification input receipt
is retained separately before the final receipt is augmented. These products
do not change the portable model ABI or claim accelerator execution. See
[the consumer contract](merlin_consumer.md).

`t2m.shared-tensor-provenance.v2` is used when original TTIR needs an official
intake rewrite before the shared optimizer can parse it. It records the original
native-verified typed operations, block/region structure and exact source bytes,
plus a separate normalized operation inventory. The native binding does not
expose complete original attribute dictionaries; those remain in `source.ttir`,
and this limitation is explicit. `prov.normalized_ids` / `prov.normalized_ops`
and emission/memory facts reference the normalized graph. They are not presented
as original-operation correspondence. The package receipt/functional ABI remain
unchanged; the difference is source provenance and normalization evidence.

## Shared buffer profile (`t2m.capture-receipt.v2`)

ABI v5 adds [named runtime dimensions and logical launch expressions](dynamic_shapes.md).
Per-lane masked writes additionally retain `prov.source_scatters` and
`memory.source_scatters` observations with exact source IDs and the proof kind;
their functional tensor/scalar ABI and receipt schemas are unchanged.
The existing `f16` tensor element type now has native and Merlin host execution
support using actual NumPy float16 storage. No hidden widening or by-value half
scalar convention is introduced; see [the half profile](half_precision.md).
The actual tensor/result allocation extents are dynamic; they are distinct from
ABI v4's static capacities and changing prefixes. Native execution receives an
explicit shape-allocation contract derived from `spec.json`. Static packages
keep their existing ABI and artifact conventions.
Dense multidimensional runtime strides are recovered by exact symbolic
offset/stride cancellation. Public shapes retain every named axis; raw memory
flattening is internal to the buffer profile. No new schema or hidden stride
argument is introduced. Non-dense layouts still require another contract.

`--memory-profile buffer` selects `shared-functional-buffer`, alongside the
default `shared-functional-tensor`. The public tensor/scalar ABI, references and
model binding stay the same. The payload additionally contains standard
`memref` and `bufferization` operations around retained linalg/tensor computation.
Writable state is private; incoming readwrite values are copied before updates.
Fresh outputs still require complete initialization and grid ownership proofs.

The receipt additionally binds `memory-witness.mlir`, `memory-input.mlir`,
`memory-stage.mlir` and `memory-stage.json`. Provenance binds the witness and
continuation hashes; the report binds the continuation input/output bytes.
Bound models retain these files under each kernel's directory. Merlin copies
them into compiler staging without evaluator inputs or references. This initial
profile requires the tensor source-memory proof and does not imply broader
opcode admission. See [the memory boundary](full_memory_bridge.md).

## Shared core artifacts

The additional explicit `shared-guarded-buffer` profile uses the same public
tensor/scalar ABI with runtime memory admission instead of an affine tensor
witness. `memory-admission.json` replaces `memory-witness.mlir`; the other three
memory continuation artifacts remain. The report and function attribute declare
the grid-derived guard cost (16, 24 or 32 bytes per writable element), included
in standalone evaluator budgets.
Checks retain source masks, loops and payload while rejecting active bounds
violations, conflicting writers and fresh-output holes. This initial profile
supports standalone staging/execution and static derived-model binding. Bound
models retain each kernel's evidence and declare guard scratch summed over
inlined calls, independently checked by Merlin against typed kernel results.
See [the exact memory scope](guarded_memory.md).
V2 admission retains writable observations and complete-write initialization,
with sticky reader/writer tags and actual source-base memory reads before
optimization. Its `memory-effects` continuation boundary is explicit. Existing
v1 evidence retains its original write-only contract and `memory-ir` stage.

| Artifact | MVP convention |
|---|---|
| `model.mlir` | One `forward` with tensor inputs/result, standard dialect operations, `prov.level="linalg-on-tensors"`. Generic assembly preserves discardable annotations. |
| `weights.safetensors` | Valid empty safetensors, because this kernel has no learned/captured constants to bind. |
| `weights.safetensors.manifest.json` | Function-argument ordinal string → `{kind:"input",name:...}`. No output-pointer argument remains. |
| `inputs.npz` | Positional arrays keyed `in0`, `in1`, etc.; f32 shapes agree with the function ABI. |
| `input_order.json` | Manifest input name → positional input index. |
| `golden.npy` | Single output from an independently supplied reference; checked against emitted IR before writing. |
| `extra.npz` | Empty, valid NumPy archive; lifted constants/quantization metadata are not fabricated. |

`prov.weights_file` uses the relative name `weights.safetensors`. An existing output directory is
never overwritten. The receipt is written last. `verify-bundle` checks the recorded artifact bytes
and source/spec/output bindings; hashes establish byte identity, not external authorship or a
complete mathematical proof.

## Differences and additions

| Area | Explicit difference |
|---|---|
| Receipt identity | `t2m.capture-receipt.v1`, not m2m's Torch-specific receipt. Consumers requiring the latter need a declared adapter/schema change. |
| Source graph | `source.ttir` and `provenance.json` describe TTIR SSA, not an FX `frontend-trace.json`. No fake `prov.aten` or Torch framework identity. |
| Source intent | Caller descriptions have `origin="caller-description"` and `grants_semantics=false`. Derived operation family/kind comes from the actual translated opcode. |
| ABI | `spec.json` supplies shapes/effects, positive strides/storage offsets, grid, constexprs and allocation preconditions. The output pointer becomes a result; readwrite outputs also have incoming tensor arguments. |
| Payload | Retained tile grid in `scf.forall`; a consumer must support it or explicitly legalize it. Current Merlin staged support is not certified. |
| xDSL | 0.68.0, versus the surveyed m2m environment's 0.65.0. Local bindings add missing **upstream standard** SCF/tensor ops, not a t2m dialect. |
| Numeric evidence | Serialized-IR reference interpreter versus a supplied NumPy golden. Optional native verification/LLVM legalization reports are separate, hash-bound checks; no native/accelerator execution claim. |
| Parameters/results | MVP has no parameters and exactly one result. General weights, dtype/profile negotiation and multiple results are roadmap work. |
| Source locations | Stable operation/value IDs and TTIR are retained. Original Python file/line reconstruction is deferred; debug paths are excluded from TTIR hashes. |

`compatibility.json` repeats these differences in every complete bundle. `meta.json` keeps
conversion/execution/consumer statuses distinct. None of the additions is necessary to interpret
the arithmetic itself; required ABI preconditions still have to hold at any real call boundary.

Shared `t2m.capture-receipt.v2` packages can be qualified directly through Merlin
with `qualify-merlin`, preserving the source package. The separate
`t2m.merlin-kernel-qualification.v1` directory contains compiler-only staging and
native-host execution products. Entry-symbol adaptation, functional readwrite
results, multi-result reference ordering and provenance projection are explicit;
see [the consumer contract](merlin_consumer.md). This does not rename the source
receipt or assert accelerator qualification in the original capture metadata.

ABI v3 adds typed runtime scalar declarations with explicit source positions.
Exported MLIR keeps plain scalar argument types after logical tensor inputs;
the usual `inputs.npz` keys store zero-dimensional typed values for invocation.
Merlin passes these by value through its existing scalar ABI. This differs from
a tensor-only captured signature; it is not scalar specialization or a rank-zero
tensor boxing transform. See [the runtime ABI](runtime_abi.md).

Reusing an already qualified library produces a separate
`t2m.merlin-kernel-invocation.v1` evidence directory. Its compiler snapshots and
artifact relations identify the exact original qualified library. New inputs,
references and observed results are evaluator artifacts; the source capture and
original compiler/qualification directory remain immutable.

ABI v4 extends runtime scalar declarations with an optional inclusive signed
integer `bounds` interval. The frontend emits entry assertions and scalar/tile
analysis retains symbolic parameters. Merlin invocation signatures retain those
intervals and validate values. This permits runtime memory prefixes over static
logical allocation capacities; readwrite outputs preserve incoming values
outside the active prefix. Tensor shapes/launch sizes are not relabeled dynamic.

## Provenance schema (`t2m.provenance.v1`)

The trace contains the canonical TTIR and spec hashes, tool versions, intent and its origin,
source-operation IDs/opcodes/typed SSA edges, each operation's semantic disposition and recovered
index/access facts, emitted-op IDs with source correspondence, established conditions/preconditions,
numeric profile and the exact output MLIR hash. Bundle writing adds final-byte count and measured
reference-interpreter agreement. User mutation of the converted IR invalidates this trace; a
recorded transform must regenerate provenance before bundle writing can proceed.

The MLIR uses the shared `prov.family` / `prov.op` / `prov.region_id` convention, plus
`prov.source_ids`, `prov.emission_id`, `prov.frontend`, `prov.intent`, `prov.intent_origin`, source/spec
hashes and logical grid/tile facts. Source IDs refer to the trace, not device instruction IDs.
Several emitted ops may implement one source op, and address-only source ops can become analyzed
access facts. The trace distinguishes that from eliminated unused pure values.

## R1 tensor-memory additions

A declared rank-one view maps logical index i to element address
`storage_offset + strides[0] * i`, relative to the source pointer argument. The converter proves
that each access fits that view, then exports its **logical tensor values**, without physical
storage holes or pointer arithmetic. `inputs.npz` materializes those logical values. Binding back
to an original strided buffer is a caller/consumer ABI step, not a hidden hardware operation.
Default contiguous fields are omitted from canonical specs to preserve original R0 spec hashes.

`readwrite` adds the incoming logical tensor to the input manifest and returns its updated value.
One source store is currently supported; reads must precede it and stay in the same program's
owned output tile. Values outside a partial store's predicate are preserved. This standalone
functional ABI does not add mutation support to model2MLIR's custom-kernel registry.

`tensor.extract_slice` + `tensor.pad` represent rectangular masked loads, with an explicit
`tensor.cast` to the proved tile size for xDSL 0.68's pad verifier. Stores insert cropped active
contributions only. Missing `other` is tracked as undefined until a store proves it unobservable;
an implementation fill for those lanes never becomes an observed source value. The trace records
mask/access facts, stored-value definedness, source cache/eviction hints and identical-read reuse.

The optional `verify-mlir` report binds the exact input MLIR hash and distinguishes registered-op
verification, LLVM-dialect legalization and execution. It never modifies an existing capture bundle
or pretends those checks were performed by a bundle's original receipt.

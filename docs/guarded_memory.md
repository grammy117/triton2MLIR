# Guarded standard memory through Merlin

`shared-tensor`, `shared-demo` and `bind-model` accept `--memory-profile guarded-buffer`.
This explicitly selects `shared-guarded-buffer`: the public functional ranked
tensor/scalar ABI is unchanged, while the body retains upstream standard
`linalg`/`tensor`/`scf`/`memref`/`bufferization` operations. The original logical
grid remains an unmapped `scf.forall`. No hardware thread mapping or GPU launch
argument is introduced.

```bash
python -m t2m shared-demo --kernel indirect-store --memory-profile guarded-buffer \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python \
  --out out/indirect-store
python -m t2m qualify-merlin out/indirect-store --help
```

The existing `tensor` and `buffer` profiles require static/symbolic memory
proofs. This profile instead inserts executable memory admission checks before
the full triton-shared memory continuation. Loaded, non-affine store indices
therefore do not need a fabricated affine ownership proof or tensor witness.
Original payload operations, masks, values, source loops and numeric policies
remain in the program. The checks validate the invocation; they do not resolve
conflicts by serializing the kernel or replacing it with a reference algorithm.

Scalar gather/scatter events are deterministically lifted to one-element
tensors because the upstream tensor converters skip scalar offsets. Exact
source index/value types, predicates and fills are retained; a gather result
is extracted back to its original scalar type. The same upstream conversion
and executable memory checks then apply, with the original logical program
count unchanged. The admission report records each scalar normalization and
its source IDs.

## Memory contract

All source buffers have independent dense logical storage. Read-only buffers
are borrowed; writable buffers use private state. Readwrite state starts as a
copy of its incoming tensor. Reads from writable buffers must observe incoming
state or completed source writes. Pointer escapes, wrapping descriptors,
pointer-carried state and aliases still require separate contracts.

For each memory event, generated loops consume the exact source index and mask
SSA values. Only active lanes check `0 <= flat_index < actual_allocation_size`.
Inactive indices are not used to address guard storage. Structured prefixes are
checked against their tile; structured addresses sum the element offsets and
local-IV/stride products used by upstream pointer analysis.

Each writable element has program-owner tag words and an i64 completion cell.
An owner word pairs 31 bits of the linear logical program ID with their bitwise
complement. Standard `memref.atomic_rmw ori` accumulates these tags and a writer
bit. Distinct programs have distinct equal-population tags: their union cannot
fit inside either program's tag. A writer rejects any earlier foreign tag; a
reader rejects a foreign tag when a writer bit is present. Shared reads without
a writer remain valid. All fragments participate, so larger logical grids have
the same conflict behavior. A read and write in either order across programs
fail; no ordering is invented between logical programs.

Completion cells begin at zero for copied readwrite state and minus one for
fresh outputs. A store exchanges its active cells with one, rejecting duplicate
active lanes in that event, and clears them to zero only after the original
payload store. A writable read requires zero. This permits ordered repeated
reads/writes by one program while rejecting reads before initialization.
These atomics update compiler-owned verification metadata; they grant no source
atomic permission. The payload retains its parallel iteration semantics. A
failed check aborts before a functional result is returned.

Fresh write-only results are checked for complete write coverage after the
logical grid. Unwritten holes fail instead of receiving invented values.
Readwrite results preserve untouched input values. Empty results have vacuous
coverage. Source address-width validation and contraction/cast policy checks
still run before upstream pointer promotion and memory conversion.

The profile needs **16 extra bytes per writable element** for grids with at
most 2^31 programs. Grids with at most 2^62 programs need 24 bytes; the remaining
i64 grid domain needs 32. One owner word covers each 31-bit ID fragment, plus
one completion cell. The bound is derived from declared maximum launch sizes;
actual tensor sizes determine allocations. Guard work is proportional to
executed memory lanes and fragment count. Allocation products are checked before
allocation. Native and standalone Merlin evaluators include the declared guard
storage in their bounded allocation budgets. Standard ownership-based buffer
deallocation releases private allocations. These costs are explicit; the
proved tensor/buffer profiles remain available when guards can be eliminated
by an established proof. No performance equivalence is claimed.

## Package and consumer boundary

`memory-admission.json` replaces `memory-witness.mlir`; the package still
includes `memory-input.mlir`, `memory-stage.mlir` and `memory-stage.json`.
The admission report records the ABI/internal argument layout, every guarded
access, source IDs, effect restrictions, storage cost and continuation hashes.
`provenance.json` and the receipt bind the actual evidence bytes.
New reports use `t2m.memory-admission.v2`, recording the owner-word count and
sticky-tag observation protocol. V1 packages remain readable with their
original write-only admission contract and 16-byte declaration.

V2 uses the explicit upstream `memory-effects` stage, stopping before CSE and
canonicalization. Upstream unstructured gathers create `to_tensor restrict`
views over source memory. The adapter replaces their `tensor.extract` uses
with typed `memref.load` operations and removes those views before optimization.
This preserves mutable observations and avoids competing restrict promises on
source bases. Read-only source declarations remain in the ABI/evidence. Native
linalg operations expose recursive memory effects; downstream CSE/LICM therefore
see the actual reads. Structured loads already copy into distinct allocations
and retain that snapshot. No payload store is rewritten or serialized.

Merlin independently checks these declared relations in compiler-only staging,
copies their evidence without evaluator arrays/goldens, and retains the scratch
contract in its compile signature. Staging establishes custody and consistency;
it does not prove source equivalence or guard correctness. Native execution,
independent reference comparison and actual Merlin host-library execution are
separate checks. Static model binding also preserves this profile and its
kernel evidence. It sums the declared guard storage at each inlined call,
records that cost on the model function and in binding correspondence, and
Merlin independently checks it against the typed kernel results and call
inventory. The host budget includes this scratch even when model results are
smaller than intermediate kernel outputs. Dynamic kernel scratch inside a
bound model requires a further resource contract.

Tests cover loaded i32/i64 permutations, repeated source stores, partial
readwrite results, inactive extreme indices, empty allocations, fresh coverage,
within-store/cross-program conflicts, active out-of-bounds accesses, rank-two
structured accesses and repeated independent shapes through one Merlin library.
Both scalar and tiled memory events exercise native execution and a reused
Merlin library, including empty/tail allocations and negative masked indices.
Runtime indirect recurrences retain reads before and after stores, including
fresh initialization. Native tests cover shared readers and both cross-program
read/write orders. Generated guards are also executed directly at 31/62-bit ID
boundaries without launching huge grids; this checks the fragment protocol,
not whole-kernel execution at those launch sizes.
They also check evidence corruption and refusal of unqualified effects.
Actual model2MLIR captures qualify ordinary tiled/ragged GEMM, block pointers
and tensor descriptors through Merlin, retaining original weights, inputs and
goldens. A two-call binding test checks per-call scratch accounting.

## Reused implementation

Source pin: Meta triton-shared
`17f0c234ce474bee334af36e351c62eca7852836`.
`lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp`
(`ScatterConverter`) supplies the unchanged parallel masked-store payload;
`test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir` establishes
its mask convention. `lib/Analysis/PtrAnalysis.cpp` defines element offsets and
stride accumulation. Structured operation schemas come from
`include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td`.
The executable optimizer has its separate recorded pin, as described in
[the reuse map](triton_shared.md). No upstream converter was vendored.

This extends irregular store admission, not all TTIR effects or all descriptor,
numeric and control-flow combinations. Source atomics, volatile observations,
backend assembly, alias groups and cross-program synchronization remain open.

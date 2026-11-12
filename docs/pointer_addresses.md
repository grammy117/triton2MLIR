# Closed pointer-integer displacement expressions

TTIR `ptr_to_int` and `int_to_ptr` no longer imply an unavoidable physical
address observation. A closed expression that only computes a displaced pointer
can be re-raised without choosing or materializing its base address. The
original source and typed casts remain in the inventory; executable output is
the same standard tensor/memory computation used by other logical pointers.

## Semantic boundary

The pinned Triton casts use scalar/tensor i64. Analysis carries a source pointer
`P` and an executable byte displacement `D`, representing `P + D` modulo 2^64.
Supported address edges are modular addition/subtraction of independent integer
displacements, identity integer bitcasts, matched scalar/tensor shape operations
and selection between two closed pointer/displacement pairs. Source integer
computation, signed extension and wrap remain executable. A selection retains
both its pointer base and matching displacement.

Conversion back to a pointer of byte width `W` requires structural evidence
that every displacement lane is divisible by `W`. A bounded low-zero-bit analysis
handles constants, arithmetic, bitwise masks, fixed valid shifts and shape/select
edges. It never interprets example inputs or caller alignment annotations as
proof. The resulting pointer is `addptr(P, signed_i64(D) / W)`, with a real
source pointer view where necessary. Because `W` divides 2^64, the reconstructed
byte address is identical modulo 2^64, including negative and wrapped
displacements. No root address, base alignment or absence of unsigned address
wrap is assumed by this identity. Actual access bounds, initialization, view
compatibility and ownership still require the existing memory gates.

The whole address-use closure is validated before mutation. Escaping integers,
address comparisons/diagnostics, mixed integer/address selection, arbitrary
integer-created pointers, unsupported address algebra and poison-producing
overflow flags refuse with source locations. Reorderable address reshapes need
a paired lane-order contract; they cannot independently reorder bases and
displacements. Integer-valued region-carried addresses remain further work;
pointer-valued region state uses the existing pointer-control adapter.
Divisibility proof is limited to 8,192 SSA nodes and depth 128; an expression
retains at most 4,096 source IDs. Unproved cases are explicit refusals.

## Reuse and scalar memory

Reviewed upstream code at Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`:

- [Base/offset traversal](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp)
  retains integer-created pointers as roots and materializes pointer offsets
  before pointer-to-integer observations. That representation does not by itself
  bind an arbitrary absolute address to our owning tensor roots.
- [Pointer/integer fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonToPtr/cast_with_int_ptr.mlir)
  deliberately retains physical pointer integers in `tptr`/`ptr`; passing it is
  different from re-raising a closed expression into portable tensor dataflow.
- [Scalar and lane-wise memory converters](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp)
  and [predicated gather fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/UnstructuredToMemref/gather_mask_with_other.mlir)
  provide the existing scalar/parallel memory boundary and inactive fill behavior.

Archived Microsoft `050b93ae`'s
[conversion registration](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToLinalg/TritonToLinalg.cpp)
supplies arithmetic/shape and legacy pointer conversion, rather than this closed
address-use proof. No upstream source was copied for the new analysis. Actual
shared passes still convert the recovered pointers and arithmetic; no toolchain
build, Triton-CPU dependency or target-specific opcode was added.

The scalar probe also exposed missing tensor-witness handling for upstream
scalar gather/scatter. Tensor and buffer profiles now reuse
`guarded_memory.normalize_scalars`, the already checked one-element carrier.
It preserves scalar masks/fills and offset arithmetic, supports scalar stores
through the existing ownership/initialization proofs, and avoids unresolved
upstream scalar affine memory operations in the buffer continuation. The tile
analysis recognizes a genuine single-element constructor without changing its
address or inventing an ownership fact.

## Package and evidence

`closed-pointer-integer-displacement` transformations and
`prov.closed_pointer_addresses` retain original source IDs, address/element
widths and the executable expansion boundary. `compatibility.json` explicitly
records this representation and any scalar unit carrier. Public tensor/scalar
ABI, source capsule and receipt schemas are unchanged. Provenance supplies
traceability; the use closure, divisibility analysis and memory guards supply
their respective executable obligations.

`tests/test_pointer_addresses.py` exercises real Triton sources, independent
word references, runtime selected bases and tiled loops, signed strides,
empty/zero-step roots, masks and original output words. Separate scalar tests
exercise all three memory profiles. Source-use/refusal tests retain the original
module and SSA use graph after a failed analysis. This is a bounded capability;
escaping physical observations, arbitrary address algebra, general region/alias
compositions and accelerator performance remain open completion requirements.

Measured selection: four i8/i16/i32/i64 address kernels execute through native,
Merlin host and actual RISC-V/Spike; every allocation word matches independently
computed references. Their libraries replay sixteen additional runtime domains.
Three masked scalar-memory libraries additionally pass native/Merlin host and
nine runtime replays across tensor, buffer and guarded profiles. Three scalar
subtraction cases pass native execution in all profiles. All twenty focused
checks pass, including physical escape, unproved alignment, overflow flags,
proof budgets and paired lane-order refusals.

`out/closed-pointer-addresses/summary.json` independently reopens all seven
libraries, four target receipts, 25 host replays and a public i64 CLI capture.
`out/closed-pointer-address-inventory/coverage.json` combines those seven
packages with the previous inventory: 113 packages, 81 standalone qualifications
and fourteen maintained product cases. All 50 pinned definition names remain
observed in selected native/Merlin-host-qualified captures. This is typed
composition evidence, not per-operation execution or arbitrary-kernel coverage.

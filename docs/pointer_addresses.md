# Closed pointer-integer displacements and differences

[Integer-address region state](integer_address_regions.md) now extends these
closed expressions through original for/while/if interfaces. Paired pointer
bases and modular byte state preserve source control and memory effects;
physical observations and unproved compositions still refuse.

The latest [same-base difference](#same-base-byte-differences) selection has
eleven actual native/Merlin-host/Spike libraries with fifteen exact-word results,
48 further host replays and four real bounds aborts. Two complete models also
retain all twenty original results through those execution boundaries. The
latest combined inventory reopens 310 packages and 251 standalone qualifications.
Absolute observations,
broader region/alias/effect forms and applicable accelerator scope remain open.

TTIR `ptr_to_int` and `int_to_ptr` no longer imply an unavoidable physical
address observation. A closed expression that computes a displaced pointer or
cancels the same established base can be re-raised without materializing that
base address. The
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

The whole address-use closure is validated before mutation. Escaping absolute integers,
address comparisons/diagnostics, mixed integer/address selection, arbitrary
integer-created pointers, unsupported address algebra and poison-producing
overflow flags refuse with source locations. Reorderable address reshapes need
a paired lane-order contract; they cannot independently reorder bases and
displacements. Closed integer-valued region state uses the
[paired address adapter](integer_address_regions.md); pointer-valued region
state uses the existing pointer-control adapter.
Divisibility proof is limited to 8,192 SSA nodes and depth 128; an expression
retains at most 4,096 source IDs. Unproved cases are explicit refusals.

## Same-base byte differences

A real source pattern derives a logical coordinate from two pointer integers:

```python
base = x + origin.to(tl.int64)
pointer = tl.where(lane % 2 == 0, base + offset, base - offset)
distance = (pointer.to(tl.uint64) - base.to(tl.uint64)).to(tl.int64)
index = origin.to(tl.int64) + distance // word_bytes
value = tl.load(x + index, lane < n, other=-17)
```

This previously refused at `arith.subi`, even though the base cancels. For any
64-bit P, including wrapped source addresses, `(P + D1) - (P + D2)` equals
`D1 - D2` modulo 2^64. Only the byte difference becomes an ordinary integer;
the absolute pointer integers remain closed and disappear from executable IR.

`pointer_differences.py` establishes the common anchor from actual pointer SSA.
It retains each source addptr increment, signed-widens it only after its
original integer computation and scales it by that addptr's pointee byte width.
Pointer bitcasts retain the accumulated bytes. Matched source shape operations
transform the byte offsets; scalar anchors are uniform across those lanes.
Nonuniform anchors require the same actual value and shape-operation path.
Same-anchor pointer selection retains its real condition and both displacements.
An otherwise opaque pointer value may cancel only against that same SSA value
and matching lane path. Names, provenance, sample addresses and independently
declared allocations establish no root equality.

The new terminal is a standard modular i64 subtraction. Its consumers retain
their original signed/unsigned arithmetic, divides, masks and memory operations.
The full source-use check still rejects absolute observations, unproved
cross-root subtraction, poison-producing flags and unsupported source fields.
Failure after an earlier prepared difference leaves the entire original module
and SSA use graph intact. Analysis has the existing depth/node/source budgets
and a 32,768-operation per-difference recipe budget. Provenance and field order
do not alter a valid matched-lane proof.

This reuses the Meta/shared base/offset boundary. Its current PtrToInt handling
re-materializes scalar pointers and refuses tensor pointer integers, so merely
running that pass does not produce the needed portable cancellation. Ascend
`8f0ade66580245e30ad97093526a4018e4e18fc7`,
`TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.cpp` (`parseAddPtr`,
`parseBitcast`) and `unittest/Analysis/PtrAnalysisBitcastTest.cpp` supply the
per-increment byte-unit rule and independent byte-address interpretation.
The narrow cancellation adapter leaves actual memory conversion with shared;
it introduces no target address space, launcher ABI or physical pointer value.

`closed-pointer-integer-difference` records join the existing
`prov.closed_pointer_addresses` correspondence. Generated compatibility metadata
names byte cancellation separately from pointer reconstruction. Public ABI,
source capsule and receipt schemas are unchanged. General absolute integer
region state, unproved region/lane correspondences, physical observations,
external synchronization and accelerator qualification remain further work.

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

## Same-base difference qualification

`tests/test_pointer_differences.py` captures i8/i16/i32/i64 pointers selected
within one root, converts their byte differences back into runtime load
coordinates and returns both loaded words and original signed byte differences.
Its original runtime loop, signed stride, masks and logical grid stay
executable. Their runtime replays cover empty domains, zero trips with an
unobserved invalid origin, both stride signs and multiple logical programs.
Three i64 byte-displacement cases additionally retain arbitrary full-width
modular words in tensor, buffer and guarded-source-word profiles. Their
references use independent Python integer arithmetic and allocation indices.

The seven libraries and actual Spike images retain **eleven complete results
as exact words**, with **32 additional unchanged-library host invocations**.
Four actual Merlin host guard invocations reject active addresses beyond the allocation;
no successful receipt or output archive is produced. Bounds, initialization,
write ownership and lifetime gates remain enabled.
`out/pointer-differences/summary.json` reopens each receipt/replay and the observed guard outcomes.
The combined `out/pointer-difference-inventory/coverage.json` reopens **304
packages, 247 standalone qualifications and fourteen product cases**, with all
fifty names observed under selected native/Merlin-host profiles.

Twelve focused difference contracts pass, including actual-root equality,
source-width wrap before byte scaling, unchanged source/use graphs on refusal,
poison flags, unmatched lane permutations, provenance/field-order independence
and bounded analysis. The 27-case address selection also passes; it includes
the seven new libraries and the twenty earlier source/native/address tests.
Those checks overlap and do not establish universal address or target coverage.

Four additional i8/i16/i32/i64 source-pointer cases now qualify narrow i32
increment arithmetic before byte scaling. Actual runtime words include signed
extremes and overflowing multiply/adds. The original i32 operations execute
before signed extension; their four complete results match exact i64 byte
words through native/Merlin host/Spike and sixteen additional host replays.
`out/pointer-difference-widths/summary.json` reopens these receipts and a
separate native-qualified public saved-source CLI capture. CLI JSON references
use `--effect-reference`; `--reference` selects numeric NPZ archives.

## Complete model qualification and next region boundary

`scripts/capture_pointer_difference_model.py` captures real i32/i64 model2MLIR
models with saved weights/bias, a strided read, a dependent read from its result,
an empty call, a zero-trip call with an unobserved invalid origin and ten original
results each. Those include actual byte distances, value outputs and the original
input-derived snapshot. Independent CPU references use Python logical indices;
no compiler root/address helper supplies the goldens.

Both guarded-source-word models pass native/Merlin host/Spike with **all twenty
original results as exact words**. Parent weights, manifests, inputs, constants
and complete reference archives remain byte-identical. All five source runtime
scalars bind through the existing checked typed-call specialization; public
model shapes stay static. `tests/test_pointer_difference_model.py` reproduces
the models and `out/pointer-difference-models/summary.json` reopens their target
receipts. The latest `out/pointer-difference-width-inventory/coverage.json`
reopens **310 packages, 251 standalone qualifications and fourteen product
cases**, retaining all fifty names under selected native/Merlin-host profiles.

The formerly refused `carried_integer_address` case in
`t2m/examples/pointer_address_kernels.py` now converts with its actual `scf.for`
and the complete closed-pointer contract. The
[region-state qualification](integer_address_regions.md) additionally executes
for/while/branch recurrences and alternating roots through the same native,
Merlin-host and Spike interfaces. The earlier expected refusal has become a
positive source/control check. Broader root equivalence, aliases, external state,
numeric/source combinations and applicable targets remain completion work.

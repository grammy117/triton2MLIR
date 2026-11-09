# Functional tensors from the shared middle layer

`t2m.shared_tensor.convert_shared(ttir_path, TensorABI(...), new_work_directory,
optimizer=existing_executable)` now runs real triton-shared passes and adapts
their output into standard tensor MLIR. This is a broader, experimental path
alongside the small reference frontend; it is not a replacement hand-written
TTIR arithmetic converter.

The pipeline verifies/canonical-prints the original source, inventories every
operation including private helpers and typed SSA edges, stamps source name
locations, inlines helpers, then runs the safe structured-memory and upstream
arithmetic-to-linalg passes. Helper inlining runs in a separate optimizer process:
the surveyed prebuilt tool's combined inliner/arithmetic pipeline triggers a
missing `func` dialect-interface assertion. The isolated native passes succeed.

The adapter turns structured loads into `tensor.extract_slice` plus exact prefix
padding, and stores into ordered `tensor.insert_slice` versions. It forwards
registered standard tensor/linalg/arithmetic/math/structured-control-flow
computation, including regions, rather than maintaining an arithmetic-op allowlist.
Copy-free static `bufferization.alloc_tensor` becomes `tensor.empty`.

Source name/callsite locations survive upstream rewriting and become inline
`prov.source_ids`, `prov.original_ops`, region and emission IDs. The exact original
source and complete inventory remain in the sidecar. A missing descendant is
reported honestly: locations establish origin, not an independent proof of every
upstream elimination or rewrite.

The implementation also adapts an xDSL 0.68 parsing bug: dense hexadecimal float
literals are IEEE bits, not numeric integers. Without this typed-token fix,
`dense<0xFF800000>` silently becomes a large positive fill instead of negative
infinity. Masked softmax native execution tests catch this corruption. NaN
constant payload preservation remains refused.

## Qualified slice and boundaries

Real Triton source examples now cover 2-D add, masked softmax with private
reduction helpers, and IEEE matmul. Tests run their **serialized output** through
native verification, LLVM legalization and host execution, comparing against
independent NumPy references with declared tolerances. These are example checks,
not exhaustive numeric equivalence or Merlin/device certification.

This profile currently requires:

- One to three logical grid axes, independent contiguous positive static tensor
  shapes, and structured descriptors matching logical strides up to an axis permutation.
  Multi-program rectangular writes must have disjoint affine owned tiles;
  each nontrivial grid axis must advance a distinct tensor axis by at least
  its tile size. Readwrite loads stay within that same owned tile.
- Affine descriptor offsets over logical programs and statically bounded loop
  IVs, rectangular masks and specialized integer scalar arguments. Clipped
  prefixes may vary with logical program IDs and bounded source loop IVs.
  Bounded `scf.for` regions preserve their tensor accumulators and read-only tile
  loads; writable buffers become additional loop-carried tensor versions, including
  nested read-modify-write loops. Live pointer-valued loop arguments still need
  descriptor-state normalization; supported affine pointer increments are described below.
- `scf.if` carries source tensor results and writable buffer versions through
  both regions. A missing source else returns incoming memory values. Definite
  initialization facts are intersected across branches, so a store in one arm
  cannot justify a load or a fresh output in the other arm.
- `readwrite` buffers retain incoming values. Fresh `write` outputs omit incoming
  arguments only after store-dominance and complete-overwrite checks. Writable
  results follow declaration order; local stores observe previous versions.
- Explicit IEEE dot precision. For nonzero/source-defined accumulators, the
  adapter checks upstream's unique-use, identity-map pointwise-add pattern,
  restores C as the contraction destination and removes the post-contraction
  addition. Ambiguous/changed patterns fail. An adversarial native test distinguishes
  this recurrence from zero-matmul-then-add; no GPU bitwise-equivalence claim is made.
- Registered standard output only. Volatile observations, atomics, external
  math/assembly and hardware/layout operations require separate profiles.

f16/bf16/f32/f64 and integer tensor types are represented; actual dtype/op
combinations remain qualified by native checks and the test corpus. A supported
type name is not a claim that all operations on that type work.

The grid becomes an unmapped `scf.forall`; division/remainder decode logical
coordinates from its iteration variable. Per-program tensor versions preserve
local ordering, then publish only the owned output slices. Bounds and source
integer widths are checked symbolically before upstream index promotion, without
enumerating elements/programs. Native tests execute 2-D tiled add and matmul and
refuse conflicting writes, cross-tile readwrite loads and out-of-bounds domains.
Source numeric and consumer qualifications remain separate. K-loop matmul tests
retain two source K iterations and a source accumulator inside the logical grid.
Nested-update tests check all six memory updates rather than only a last store.

Permuted structured loads/stores now extract logical slices in buffer-axis order
and use standard `linalg.transpose` to preserve the source tile order. Masks are
cropped/padded in the appropriate source order. Bounds resolve degenerate stride
ambiguities using a bounded search; non-bijective/broadcast strides stay refused.
Tests execute transposed-weight K-loop GEMM and transposed fresh-output stores.
The original model2MLIR MLP kernel additionally exercises transposed weights,
bias broadcasts, activations and a masked four-row output.

Conditional tests execute both data-dependent arms, preserve incoming values on
an unwritten arm, refuse partially initialized fresh results, and combine source
branch results with nested loop memory updates. Branch-returned pointers and
branch-dependent address bounds still need separate descriptor/range rules.

Runtime shapes/scalars, irregular memory, aliasing, general pointer
recurrences, scans and mixed-precision policy need further work. The upstream
stage driver remains available for these investigations; successful stage
conversion is not counted as complete functional-tensor/package coverage.

Upstream pointer-to-index recurrences now retain invariant affine increments,
including nonzero lower bounds, nonunit loop steps and final values used after
the loop. Flat affine base offsets become tensor coordinates using exact
quotient terms and bounded remainders. Tests execute pointer-increment GEMM and
a post-loop load, and reject its out-of-bounds variant. Nonlinear or unproved
recurrences remain refused.

When upstream leaves an unused pointer carrier for a post-loop result, the
adapter carries recovered scalar coordinates through the loop instead. An
upstream poison initializer is discarded only when all uses initialize pointer
arguments proved unused inside positive-trip-count loops. Observable poison
never becomes a zero or another invented value.

## Legacy block pointers and official intake normalization

`tt.make_tensor_ptr` / `tt.advance` now use Triton 3.7.1's official
[`triton-rewrite-tensor-pointer` pass](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp).
It expands descriptor shape/stride/offsets, boundary-check axes and padding into
ordinary pre-layout pointer arithmetic and masks. No GPU layout or target backend
is selected. Existing tensor memory, tail/ownership and recurrence analyses then
emit the same portable tiled `tensor` / `linalg` / `scf` computation.

The installed shared optimizer does not recognize these legacy operations.
A bounded, isolated pinned-Triton worker therefore verifies and inventories the
original source before running the official pass. The original graph and exact
source, including block shape/order and control flow, remain separate from the
normalized graph consumed by shared. [Provenance v2](package.md) makes that
boundary and the native attribute-binding limitation explicit. Diagnostic files
retain the shared tool's original rejection and the normalization report.

Native tests cover masked block-pointer add with inactive programs and ragged
M/N/K GEMM with loop-carried `advance`. Grids, fixed source tiles and the K loop
remain executable structure. Incorrect declared dimensions are refused; source
loop-carried affine offset bounds also check integer overflow before promotion.

The same adapter now also runs the official
[`triton-rewrite-tensor-descriptor-to-pointer` pass](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp)
for kernel-local `tt.make_tensor_descriptor`, `tt.descriptor_load` and
`tt.descriptor_store`. Descriptor shapes, tile offsets and zero padding become
executable pointer/mask operations. Canonicalization folds the pass's unused
NaN-padding alternative before the checked tensor adapter runs. Observable NaN
payloads remain outside the qualified numeric profile. The pipeline/report lists
the exact official passes; no TMA instructions, physical descriptor layout or
hardware launcher enters the tensor ABI.

Native checks cover descriptor add and ragged 35×36 by 36×20 GEMM; the latter
retains its three K iterations and fixed 16×16 tiles. Declared-buffer mismatches
remain refusals. Externally supplied descriptor arguments, descriptor gather/
scatter/reduce and other numeric/alias policies still require separate adapters;
running the rewrite is not counted as their complete tensor coverage.

## Dynamic rectangular tails

The adapter now recognizes the signed affine clamp emitted by upstream
`MaskAnalysis`: `max(min(start + tile, limit), start) - start`, including its
extra `min(tile, ...)`. This is matched structurally against an exact clipped
prefix, tied to a recovered address axis and checked against the declared
tensor dimension. Arbitrary predicates remain outside this rule.

Loads use dynamically sized `tensor.extract_slice`, exact source padding and
a checked cast back to the static source tile type. Stores crop those tiles
before insertion. All-inactive extra programs use bounded zero-length slices.
Axis permutations retain dynamic dimensions through `linalg.transpose`.
No padding value is invented for an omitted source `other`.

When surplus columns cross a row stride, quotient/remainder of the flattened
base offset loses the original axis boundary. The adapter instead recovers
upstream's separate element-offset contributions when each divides exactly by
its matched logical stride. Their weighted sum preserves the original address;
each access still needs a bounded mask and a bijective axis mapping. Native
GEMM checks cover surplus programs in both grid dimensions, including transposed
weights, and refuse masks extending past the actual declared buffer.

Fresh output coverage uses a closed-form clipped affine partition check,
including ragged final tiles and surplus inactive programs. Equal nominal tile
cardinality would be incorrect here. Publication coordinates/extents are
materialized outside source branches so nested stores cannot leak scoped SSA
values. Provenance records the maximum tile sizes, limits, logical offsets and
coverage rule; the executable operations implement the masks and fill values.

Native checks now include ragged M/N/K GEMM, transposed weights, nonzero-fill
reductions with stores in both data-dependent branches, partial readwrite
outputs, surplus programs, out-of-bounds refusal and incomplete fresh-output
refusal. A >2²⁵-element conversion checks symbolic coverage without allocating
or executing that tensor. These checks preserve source grids, K loops and tile
computation; they do not qualify arbitrary dynamic masks or runtime dimensions.

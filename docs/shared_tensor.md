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
  shapes, and structured descriptor rank/strides matching the logical buffer.
  Multi-program rectangular writes must have disjoint affine owned tiles;
  each nontrivial grid axis must advance a distinct tensor axis by at least
  its tile size. Readwrite loads stay within that same owned tile.
- Affine descriptor offsets over logical programs and statically bounded loop
  IVs, static rectangular masks and specialized integer scalar arguments.
  Bounded `scf.for` regions preserve their tensor accumulators and read-only tile
  loads; writable buffers become additional loop-carried tensor versions, including
  nested read-modify-write loops. Pointer-valued loop arguments remain refused.
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

Dynamic/irregular memory, aliasing, pointer
recurrences, scans and mixed-precision policy need further work. The upstream
stage driver remains available for these investigations; successful stage
conversion is not counted as complete functional-tensor/package coverage.

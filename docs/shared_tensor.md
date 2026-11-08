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

- One logical program, independent contiguous positive static tensor shapes,
  and structured descriptor rank/strides matching the logical buffer.
- Statically resolved descriptor offsets, masks and specialized integer scalar
  arguments. Source compute loops may survive; memory/pointer loop recurrences
  still require an adapter and fail explicitly.
- All writable buffers are incoming `readwrite` tensors, returned in declaration
  order. Multiple local stores observe previous versions. No output initialization
  or complete-overwrite proof is invented.
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

Larger grids must remain undecided iteration spaces with established ownership,
not become serialized programs. Dynamic/irregular memory, aliasing, pointer
recurrences, scans and mixed-precision policy need further work. The upstream
stage driver remains available for these investigations; successful stage
conversion is not counted as complete functional-tensor/package coverage.

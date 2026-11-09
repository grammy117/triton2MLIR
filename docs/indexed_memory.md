# Indexed reads as portable tensor dataflow

The shared adapter now consumes both upstream `tts.gather` and the rank-one,
unit-stride, zero-base `tts.make_gather_scatter_tptr` + `tts.load` form. It keeps
triton-shared's per-lane parallel generic and predicated extraction pattern,
adapting its memory boundary to an immutable logical tensor. Source: Meta
`17f0c234ce474bee334af36e351c62eca7852836`,
`lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp`
(`GatherConverter`), its `gather_scatter_all_mask.mlir` test, and
`lib/Conversion/StructuredToMemref/StructuredToMemref.cpp` (`rewriteGather`).
The adapted pattern retains its MIT copyright/license notice in
[shared_gather.py](../t2m/shared_gather.py). The archived Microsoft converter is
recorded separately in [the reuse map](triton_shared.md); it is not an independent
correctness oracle.

The executable expansion retains the logical launch grid and result tiles. A
`linalg.generic` consumes the index tensor and any per-lane boolean mask. Its
active branch checks the signed i32/i64 offset in i64 arithmetic before casting
to `index` and extracting from the logical tensor. A standard
`tensor.collapse_shape` view preserves the original flat offset for dense
multidimensional allocations, including trailing unit axes. It introduces no
per-lane row/column division. A native identity-layout bufferization check
confirms the source allocation remains a memory view without an input copy.
The inactive branch yields the exact explicit scalar `other`, without executing
a bounds check or extraction. Structured gather descriptors retain their exact
prefix predicate through `linalg.index`; they do not depend on the general mask
path happening to be selected upstream.

Repeated indices and reads across logical programs reuse the same immutable
tensor. No read ownership or dense copy of that allocation is introduced. The
selected host consumer may schedule the logical grid serially, while emitted
frontend MLIR retains its parallel dataflow. Provenance records source IDs,
memory family/kind, representation and active bounds/mask facts; arithmetic and
effect semantics stay in the executable operations.

The current profile requires a read-only contiguous allocation with fewer than
2^31 elements and a static positive tile shape. ABI v5 also supports named
runtime allocation dimensions and a separately named index/result count. The
actual guarded dimensions determine the active bounds check and row-major
coordinates; declared maxima only bound the arithmetic domain. Every suffix
stride product must fit that domain, including when a leading dimension is
always zero. Runtime indices are tensor data,
not compile-time specializations. Their full source-width signed interval feeds
the existing address arithmetic check; an unproved overflowing source expression
still refuses before upstream index promotion. Active out-of-allocation indices
fail an assertion instead of performing an invalid extraction. This does not
define previously undefined source accesses as valid reads.

Native checks cover i32/i64 indices, repeated indices, multiple programs, ragged
tails, rank-one/multidimensional allocations, unit dimensions, all-inactive
extreme-negative lanes, explicit nonzero fill and active invalid indices. An
end-to-end package also executes through Merlin's actual compiler and host ABI
against the original independent reference. Source indices with unproved
overflow, irregular writable-buffer reads, collision-prone scatter and masked
loads with undefined `other` remain refused. These need explicit definedness,
initialization/ownership/effect and numeric work; they are not counted as covered.

Both memory profiles now qualify runtime-sized matrix allocations through
Merlin, reusing one library per profile/index type across independent shapes.
The corpus includes empty storage with all lanes inactive, repeated indices,
ragged counts and i32/i64 extreme-negative inactive indices. The tensor profile
additionally checks that an active index below the maximum capacity but outside
the actual allocation fails before extraction, including zero-sized storage.
The collapsed tensor's actual dimension supplies the active limit; no runtime
coordinate division is introduced. Empty storage permits no active read.
The buffer profile retains upstream memory operations
and the checked tensor witness; undefined source accesses are not qualified.
General indexed writes and writable-buffer observations remain separate gates.
The [masked-write adapter](masked_writes.md) now separately admits regular
affine addresses with nonrectangular lane predicates, represented upstream as
`tts.scatter`. General indirect indices still need their own conflict semantics.

```bash
python -m t2m shared-demo --kernel indexed-gather --out out/indexed_gather \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/indexed_gather --out out/indexed_gather_merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
python -m t2m verify-merlin out/indexed_gather_merlin
```

`shared-demo --kernel dynamic-gather` emits the ABI v5 variant. The package keeps
allocation shape `("m", "n")` and index/result shape `("count",)` symbolic;
example sizes do not enter the compiler-only view.

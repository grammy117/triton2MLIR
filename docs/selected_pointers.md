# Selected logical buffers with source-word offsets

The explicit `guarded-buffer` / `source-words` path now supports finite mixed-base
pointer state. Public arguments/results remain the existing functional
tensor/scalar ABI. Internal pointers become a logical buffer selector plus an
i64 element offset, with no GPU address space, physical pointer or launcher ABI.

Source-entry buffer identities propagate through pointer arithmetic, selections,
shape operations and every branch/backedge to a fixed point. Every path must
retain the same element type and refer to a declared buffer. Pointer observations,
unknown producers and pointee-changing casts remain located refusals. General
aliases between the declared buffers still need a separate ABI contract.

## Transformation and granularity

`pointer_control.py` shares the same source-word offset conversion for fixed and
selected bases. `pointer_select.py` adds selectors only where roots can differ:

- Uniform program/branch selection stays scalar i32, including tensor pointers.
  Lane-varying choices use an i32 tensor with the pointer's logical shape.
- For/while/if state gains only the required selector fields. While before and
  after state are handled separately, including different arities.
- Pointer shape transformations apply the same logical transformation to both
  fields. Original arithmetic producing offsets retains its word width; pointer
  increments are sign-extended separately into i64.
- At each original load/store, predicated accesses resolve the selected buffer.
  The load is not hoisted into the earlier pointer-selection branch. Intervening
  stores remain visible, including reads after writes through a selected pointer.

Only the selected root is read/written for an active lane. Unselected buffers may
be empty or have invalid dormant offsets. Source masks and fills remain value
dataflow. Per-root loads use a disposable zero fill and the final selection
retains the original fill, including nonuniform tensors. When source `other` is
absent, inactive undefined values use the same zero refinement as the pinned
upstream gather converter; this is recorded explicitly and grants no source
memory initialization fact. Active bounds, read initialization, ownership and
duplicate-store checks remain executable through the existing guarded profile.

The resulting standard MLIR keeps source tiles, predicates, loops and reuse.
Multiple predicated accesses are an explicit representation difference: a later
consumer may combine their selector/base calculation using its own legal memory
representation. The frontend chooses no physical memory mapping. Performance
has not been benchmarked.

## Source/provenance and qualification

The existing receipt binds exact TTIR, typed source operations, normalized
`word-input.mlir`, native stage output/pipeline and admission evidence. The
transformation report records base argument ordinals, selector granularity and
original memory-event source IDs. `compatibility.json` records the selected-base
representation and any inactive-undefined refinement. Source intent grants no
rewrite or effect permission; Merlin independently checks the artifact contract.

Maintained tests cover scalar/tiled branches, lane-varying selection, different
base extents and offsets, intervening writes, selected read/modify/write updates,
for/while changing bases and final pointers, join/transpose/reshape, zero trips,
empty domains and actual selected out-of-bounds errors. Actual Merlin libraries
reuse independent inputs, choices, counts and shapes without recompilation;
original input bytes and compiled library bytes remain unchanged.

Pointer `tt.unsplat` now also extracts the unique logical selector/offset from a
one-element pointer tensor. It uses the same source-word conversion and the
upstream numeric `UnsplatConverter` (`tensor.extract`), without observing a
physical address. Same-base, uniform-branch and lane-selected forms pass native
and actual Merlin execution over independent empty/scalar/ragged domains. The
pointer-unsplat/control regression passes 20 cases; one library per form handles
all shapes and choices. The source element type and original memory event remain
unchanged.

Cat, split and repeated intra-tile gather now also retain selectors and offsets
through native and actual Merlin libraries, over empty/scalar/ragged inputs.
The installed optimizer predates the surveyed Meta cat/gather converters.
Native Triton first verifies the original cat source and inventories its graph;
a bounded intake adapter expands equal rank-one inputs to join/transpose/reshape
in lhs-then-rhs order. This chooses an ordering allowed by
[Triton 3.7.1 cat semantics](https://github.com/triton-lang/triton/blob/v3.7.1/include/triton/Dialect/Triton/IR/TritonOps.td#L472).
Original source and normalized graphs remain distinct, and package compatibility
records the choice. It does not select a physical layout or require a new build.

Intra-tile `tt.gather` expands directly to parallel `linalg.generic` with
`tensor.extract`, adapting Meta's `TensorGatherToLinalgPass.cpp` and its two-axis
fixtures at the recorded revision. The adapter matches the actual typed source
operation; a caller marker attribute cannot authorize the rewrite. Executable
axis-index checks precede extraction. Pointer normalization applies the same
gather to both selector and offset, including repeated indices. Static numeric
tensors are admitted; pointer tensors first need the source-word path. The
12-case pointer-shape/block-pointer regression passes, including the existing
official descriptor and block-pointer intake paths.

Value-only tile gathers separately pass 42 native/Merlin cases across tensor,
buffer and guarded source-word profiles. Both axes of a two-dimensional tile
retain repeated indices and reduced axis extents. FP32 raw bits (including quiet
NaN payloads and signed zeros) and i32 values are checked independently. Negative
and upper-bound indices abort before extraction. One library per axis/profile
replays three index/value inputs with the capture and original inputs unchanged.
Both i32 and i64 indices retain their actual width. Bounds are checked in i64
before any cast to MLIR index; positive and negative values whose low 32 bits
look in-bounds abort. The captured library and original inputs remain unchanged.

The runnable package example exercises an intervening write followed by a
lane-selected read/modify/write. Results are the new X state, new Y state and
selected values, in ABI declaration order:

```bash
python -m t2m shared-demo --kernel selected-memory \
  --memory-profile guarded-buffer --address-policy source-words \
  --optimizer .scratch/toolchain/triton-shared-opt \
  --python /path/to/native-env/bin/python --out out/selected-memory
```

Additional native and Merlin cases compare raw FP16/FP32/FP64 value bits,
including signed zeros, infinities and quiet NaN payloads, and i32 values.
Nonuniform tensor fills remain observable on inactive lanes. Two address cases
separately check original i32 multiply/add wrap and cancellation of successive
sign-extended pointer increments. Each compiled library runs empty, scalar and
ragged inputs with independent runtime choices and encoded offsets. These
checks qualify memory transport; they do not grant floating arithmetic or
pointer-observation permissions. The 13 value/word cases pass with all native
and Merlin environments enabled.

Reuse boundary: Meta triton-shared `17f0c234ce474bee334af36e351c62eca7852836`,
`lib/AnalysisStructured/PtrAnalysis.cpp` reconciles branch pointer states and
selects bases. Its `if_returning_mixed_structured_ptrs.mlir` fixture demonstrates
mixed structured/unstructured results. That path promotes offsets to index;
this adapter keeps source integer words before the unchanged official
`offset-bit-width=64` stage. `UnstructuredToMemrefPass.cpp::GatherConverter`
documents zero when masked `other` is absent. Existing upstream gather/scatter
conversion, memory admission and Merlin lowering remain reused.

# Per-lane masked writes within owned tiles

The shared adapter now admits nonrectangular store masks on proved affine dense
tiles. Meta triton-shared represents these as `tts.scatter`, even when the
addresses are ordinary `program_id * tile + arange` expressions. Rejecting all
scatter operations therefore rejected useful regular-memory kernels too.

Source references at Meta revision
`17f0c234ce474bee334af36e351c62eca7852836`:

- `lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp`,
  `ScatterConverter`, and `test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir`.
- `lib/Analysis/PtrAnalysis.cpp`, `mulState`, for independent address-axis facts.
- `lib/Analysis/MaskAnalysis.cpp`, `parseAnd` and `parseCmp`, for separating
  necessary rectangular bounds from arbitrary predicate payloads.

The local analysis inspects the already-converted linalg indexing maps, fills,
local indices and scalar SSA. It derives an exact address polynomial, checks
each original integer width, matches dense strides and cancels them exactly.
It does not reimplement upstream computation conversion or enumerate program
instances or lanes. Unresolved loaded indices and nonlinear lane maps refuse.
Term, SSA and axis-search budgets keep the proof bounded.

Mask conjunctions contribute necessary inequalities; disjunctions retain only
facts common to both branches. Other predicates remain executable tensor data
and grant no bounds facts. A tail is admitted only when these facts prove every
active source lane stays within each logical tensor axis. Whole-tile write
ownership is checked independently, without serializing unproved conflicts.

The tensor expansion selects each new value or its initialized previous value
inside the owned tile. It retains the original mask, source loops and logical
forall. Safe rectangular slices clip tails, and masked lanes keep their previous
values. A possibly partial store grants no fresh-output initialization credit:
the incoming state or a covering earlier source store must establish the values
that it preserves. Zero-element slices clamp their actual offsets even when the
source coordinate is constant.

The buffer profile retains the original upstream predicated `memref.store` and
its dataflow. It uses the tensor proof as a witness, without substituting the
tensor select expansion into that memory payload. This preserves the sparse
store opportunity for downstream lowering; the tensor profile expresses the
same functional state update. No performance benchmark is implied.

Provenance records `prov.source_scatters`, original source IDs, the proof kind,
and a `memory` / `masked_store` family on the generated select region. The
package sidecar records these observations under `memory.source_scatters`.
No receipt or public tensor/scalar ABI schema changes. Labels do not replace
the executable mask, bounds, initialization or ownership checks.

Native checks cover periodic and data-dependent predicates, AND/OR masks,
repeated updates inside a source loop, incoming readwrite state, preinitialized
fresh outputs and constant-offset empty slices. Both profiles qualify actual
Merlin libraries over sizes 0, 1, 7, 8, 9 and 13 without recompilation. Inactive
readwrite values include bit checks for signed zero and a NaN payload. The
reference retains the source FP32 addition order. Conflicting tiles, unproved
tails and uninitialized fresh state refuse before memory continuation.

Multidimensional address and mask proofs also follow upstream projected
broadcasts, exact static `tensor.expand_shape` reassociations and typed
`linalg.transpose` permutations. Real two-axis kernels qualify all combinations
of direct/transposed destination storage and direct/transposed source traversal
in both profiles. One library per variant handles `(0,0)`, `(0,7)`, `(5,0)`,
`(1,1)`, `(5,9)` and `(9,13)`. Missing per-axis bounds refuse even when some
flattened addresses could fall in another logical row. Refusals identify the
original store ID; all-unit views retain their exact zero-index check.

```bash
python -m t2m shared-demo --kernel masked-matrix --memory-profile buffer \
  --out out/masked_matrix --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/masked_matrix --out out/masked_matrix_merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
```

`shared-demo --kernel masked-store` exercises periodic/data-dependent predicates
inside a source loop. The default tensor profile emits the functional select
expansion; the explicit buffer profile retains predicated memory operations.

General indirect scatter, mask-dependent ownership, aliasing and writable-buffer
indexed observations remain open. The full memory profile still depends on its
checked tensor witness; independent memory admission remains a separate gate.

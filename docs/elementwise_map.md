# Scalar regions mapped over logical tiles

`tt.map_elementwise` / `tt.map_elementwise.return` now have a standard expansion
for `pack=1` with numeric tensors and an inlined scalar value region. Inputs may
have different element types, but all input/result shapes must match. The public
tensor/scalar ABI and package schemas do not change.

The adapter creates one parallel `linalg.generic` with identity indexing maps
and fresh tensor destinations, cloning the actual scalar region. It preserves
multiple results, arithmetic attributes, structured branches and scalar
for/while recurrences. A scalar branch remains control flow: the adapter does
not replace it with eager evaluation of both sides. Runtime loop bounds remain
runtime values. No element is assigned to a hardware lane or warp.

`prov.composite_family`, `prov.composite_id` and `prov.map_pack` retain map intent
on the generic. The normal source-name locations still connect cloned scalar
operations to the original helper/callsite inventory. A `ttir-elementwise-map`
report records shape, arity, expansion and the undecided physical mapping.
These annotations describe computation; they grant no arithmetic/effect rewrite.

The pinned Meta and archived Microsoft converters have no region-map adapter.
This implementation follows the scalar-region generalization boundary used by
upstream MLIR Linalg (`Generalization.cpp` and its named-op fixtures at LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`). Source signature/packing semantics come
from Triton 3.7.1 `TritonOps.td`, `Ops.cpp::MapElementwiseOp::verifyRegions` and
`language/core.py::map_elementwise`. Existing upstream helper inlining,
arithmetic lowering, memory adaptation and Merlin lowering remain reused.

Twenty-six maintained cases pass with native and actual Merlin tools enabled.
Tensor, buffer and guarded source-word profiles execute f32/i32 maps with tuple
results, per-element zero-divisor branches and runtime for/while recurrences. One
library per dtype/profile runs empty, scalar and ragged inputs with different
iteration counts; inputs and library bytes remain unchanged. The finite numeric
references do not establish every floating or integer exceptional policy.

Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7` additionally has a
`MapElementwiseDecomposeConverter` in `TritonOpConverter.cpp` and executable
`test_map_elementwise_ops.py` fixtures. Its boundary promotes scalar operations
to tensor operations, converts if regions to tensor selects, maps packed
arguments to whole input tensors and does not support scalar while. That is
different from retaining the scalar region. This adapter keeps branch execution
and while conditions scalar inside the generic; its grouping gate does not
assume that packed lanes are identical. No Ascend backend or custom op is needed.

Packed maps remain a separate grouping contract: flattening physical packed
groups into independent elements could change cross-element dataflow. Pointer
inputs or source-memory operations inside a map also need an effect adapter
that preserves each event, mask and source order. Both forms currently produce
located refusals, with real source regression cases. They are retained in the
completion scope; no opaque successful fallback is emitted. Histogram remains
another independently observed gap from the same source probe. Numerical
unsplat already lowers through the shared path.

# Guarded reductions with fresh tensor allocations

The maintained product corpus exposed a real source-to-interface gap: upstream
softmax lowering creates rank-zero `bufferization.alloc_tensor` results for
its reductions, but guarded-memory admission rejected that standard operation.

The adapter now admits static, dense, numeric, copy-free allocations in the
default memory space. The original allocation stays in the executable MLIR.
It is not replaced with `tensor.empty`: `alloc_tensor` explicitly establishes a
fresh buffer during One-Shot Bufferize, which can matter for aliases and reuse.
Source locations/provenance, reduction order, masks and ordinary memory guards
remain intact. Native ownership/deallocation still determines buffer lifetimes.

The contract comes from LLVM/MLIR at
`a47bddccec30255619bb8c37fa59700e661d4e66`:

- `mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td` defines
  allocation, optional copy, sparse hints and physical memory-space semantics.
- `mlir/lib/Dialect/Bufferization/IR/BufferizationOps.cpp` materializes the
  new allocation and optional copy during bufferization.
- `mlir/test/Dialect/Bufferization/Transforms/one-shot-bufferize.mlir` checks
  independent allocations, copied contents and nondefault memory spaces.

The guarded extension checks the complete zero-operand allocation form and
exact operand segments before admitting it. Copies, dynamic sizes, sparse
hints/encodings, physical spaces and unknown semantic properties keep their
own admission requirements; provenance cannot grant them allocation permission.
This is tensor temporary storage, with no GPU storage class or lane mapping.

The 39-case allocation/guarded-memory suite passes. Four actual Merlin
libraries cover 13- and 37-element softmax with both structured and source-word
addresses. They retain the original rank-zero allocations and reductions and
compare with independent NumPy references at explicit `rtol=1e-5, atol=2e-6`.
Positive/adversarial allocation checks retain default-space freshness and
reject copy/hint/space changes. Existing scatter/bounds/ownership regressions
also pass. Broader numeric, effect and target coverage remains in the
[completion plan](completion_plan.md).

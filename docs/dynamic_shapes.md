# Runtime dimensions and logical launches (ABI v5)

`t2m.shared-tensor-abi.v5` declares named tensor dimensions tied to bounded,
nonnegative runtime integer scalars. A rank-one buffer with `shape=("n",)` emits
`tensor<?xf32>`, with an executable `tensor.dim`/`cf.assert` check against scalar
`n` for every incoming buffer. Fresh results allocate their actual runtime
extent. Static signatures retain their existing ABI versions.

`GridExtent("n", 8)` declares a logical launch extent `ceildiv(n,8)`. The emitted
unmapped forall and source program-count values use that runtime expression.
The maximum count is used only for range proofs. Zero dimensions yield zero
programs and zero-element results. No hardware launch or target mapping is chosen.

Complete-output proofs require the mask limit to equal the actual named result
dimension. Variable launches must cover that same dimension: their ceil-divisor
may be at most the source tile stride. Surplus masked programs are valid;
potential holes refuse. Ordinary per-program ownership, initialization, integer
width and source memory checks still apply. Bounds over a maximum capacity alone
cannot prove this contract.

Both tensor and mixed-memory conversion preserve native runtime allocation and
launch behavior. Native tests invoke the same serialized module with sizes
0, 1, 7, 8, 9 and 13; the payload observes the real program count. The JIT caller
uses the explicit `t2m.runtime-shapes.v1` allocation contract derived from the ABI,
checks tensor/scalar consistency and budgets output bytes before allocating.
It does not guess an output size from example data.

Merlin stages this contract without evaluator data using
`merlin.compile-signature.v2`. Symbolic input/result dimensions and named scalar
bindings drive invocation checks and result allocation. Both profiles qualify
through the actual compiler/library path, then reuse exactly one library over
independent shapes including zero. The payload observes the actual program
count, so a captured or worst-case launch count would fail those checks.

```bash
python -m t2m shared-demo --kernel dynamic-count --memory-profile buffer \
  --out out/dynamic_count --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
```

Dense multidimensional buffers now use exact symbolic stride analysis over
triton-shared's per-axis offsets and strides. For example, a row contribution
`pid_m * BM * n` with stride `n` recovers logical row `pid_m * BM`. This is an
algebraic cancellation, with no emitted division by a runtime extent that may
be zero. Source integer-width checks still precede pointer re-raising. Axes
must match the declared dense layout exactly; unaligned or nonlinear residual
coordinates, ambiguous axes and unmatched layouts refuse. Axis recovery has
an eight-axis and bounded-search analysis budget.

This adapts Meta revision `17f0c234ce474bee334af36e351c62eca7852836`,
`lib/Analysis/PtrAnalysis.cpp` (`mulState`, `accumulateTargetOffset`), rather
than replacing upstream computation conversion. Its runtime-stride GEMM fixture
is `test/Conversion/StructuredToMemref/kernel-03-matrix-multiplication.mlir`.

Real two-axis tiled matrix addition qualifies both memory profiles through
Merlin. One library per profile handles `(0,0)`, `(0,7)`, `(5,0)`, `(1,1)`,
`(5,9)` and `(9,13)` without recompilation. The tensor form retains tiled
slices, pointwise computation and parallel publication; the buffer form retains
upstream memory operations with dynamic dense flattening. A disagreeing
declared row stride refuses before the buffer continuation is emitted.

Runtime M/N/K GEMM also qualifies both profiles, including transposed RHS
storage. The emitted computation retains `linalg.matmul` on 16×16 tiles
inside the original runtime K `scf.for`; M/N govern the unmapped logical grid.
One library per profile/storage layout handles empty M/N, K=0 with nonempty
zero-filled results, exact tiles and ragged `(35,21,37)` dimensions. Numerical
checks use independent NumPy matmul at explicit `rtol=1e-5`, `atol=6e-6`;
they do not establish universal floating-point equivalence. Tile loop structure
is checked independently of the numerical reference.

```bash
python -m t2m shared-demo --kernel dynamic-matmul --memory-profile buffer \
  --out out/dynamic_matmul --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/dynamic_matmul \
  --out out/dynamic_matmul_merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang \
  --rtol 1e-5 --atol 6e-6
```

Runtime read-only indexed gathers also qualify multidimensional allocations,
independent index/result counts, zero-element inactive reads and repeated-index
reuse in both profiles through one Merlin library per profile/index type.
See [the active bounds and effect contract](indexed_memory.md).

Unmasked unit rows now use the actual runtime logical launch bound, with
independent exact-partition coverage for fresh results. Mixed unmasked rows and
clipped column tiles retain their dense coordinates via exact uniform-origin
recovery. See [the launch-domain proof and negative cases](logical_grid_bounds.md).

Non-dense multidimensional runtime strides, general indexed writes, arbitrary shape
expressions, model-call signature adaptation and aliasing still need their own
proofs. Compiler staging establishes declared type/binding consistency; actual
guard semantics, memory coverage, numeric and execution evidence remain separate.

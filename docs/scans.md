# Portable associative scans

`tt.scan` / `tt.scan.return` now expand after native helper inlining and before
the shared tensor passes. The [TTIR scan definition](https://github.com/triton-lang/triton/blob/v3.7.1/include/triton/Dialect/Triton/IR/TritonOps.td)
declares an associative combiner. The expansion preserves operand order within
that contract, including reverse traversal and multiple tuple components.
Triton's [scan lowering](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp)
confirms that reverse means flip/scan/flip, which matters for noncommutative
combiners. No hardware lowering from that implementation is imported.

The surveyed shared `CumSumConverter` and `cumsum.mlir` fixtures at Meta
`17f0c234` handle a single addition combiner and emit nonstandard `ttx.cumsum`.
They do not supply the general functional tensor expansion required here.

The adapter emits an inclusive doubling network. At distances 1, 2, 4, …,
`tensor.extract_slice` selects prior/current partial prefixes, `linalg.generic`
combines them in parallel, and `tensor.insert_slice` publishes the next stage.
Inputs for both sides come from the preceding stage. Prefixes remain inclusive;
the first element is copied and no identity is invented. Independent dimensions
remain parallel. Rank, scan axis and all tuple element types are preserved.

This is a standard baseline with logarithmic depth and O(n log n) work along
the scan axis. Original combiner source, composite IDs, axis, reverse flag,
stage distances and algorithm provenance remain available for downstream scan
recognition and a future work-efficient implementation. No target performance
or bitwise floating-point equivalence is claimed. Reassociation is confined to
the source associative-scan operation; ordinary surrounding arithmetic is not
given a reassociation permission.

The current profile accepts equal positive static tensor shapes and inlined
scalar `arith` / `math` combiners. Unsupported regions/effects receive located
refusals. Tests execute f32/i32 sums, products and maxima, forward/reverse tails,
two-axis tuple scans with an associative noncommutative affine-composition
combiner, and a single-element negative-zero case. Unit source dimensions may
have structured stride zero; broadcast nonunit axes remain outside that rule.
Upstream's all-unit constant-zero gather/scatter spelling is adapted to the same
single-element tensor versions. Nonzero indices or nonunit buffers do not gain
general irregular-memory support from this rule.

Actual model2MLIR sum/product and tuple-result scan models now pass complete
native comparison against their unchanged PyTorch captures and optional Merlin
interface/runtime generation. See [the model recipe](model_integration.md).

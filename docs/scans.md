# Portable associative scans

[Captured tensor lookups](scalar_tile_reads.md) now retain immutable table
snapshots in the original scalar body, including guarded element reads and
runtime recurrence after source table mutation. Source assertions and direct
pointer-memory effects keep their separate audit and effect requirements.

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
pure scalar `arith` / `math` computation, including nested `scf.if`, `scf.for`
and `scf.while` value regions. Every nested operation, block argument and captured
value is checked structurally; unknown operations, unqualified tensor values, pointers, memory,
calls and observable effects receive located refusals, including unused effects
inside an unselected branch or a one-element scan. All source scan candidates
validate before any expansion is attached. Original loops and yields are cloned
into each parallel stage, preserving per-invocation state and source locations.
Direct uniform scalar return captures also remain executable.

Structured combiners add derived `prov.scan_combiner_control` counts on each
stage and `combiner_control_flow`/`combiner_source_ids` in the transformation
record. These describe the actual source body and grant no associativity,
numeric, memory or termination proof. The scan's associative contract remains
the source premise; arbitrary surrounding arithmetic gets no reassociation.
No loop is unrolled or replaced with a recognized mathematical function.

Tests execute f32/i32 sums, products and maxima, forward/reverse tails,
two-axis tuple scans with an associative noncommutative affine-composition
combiner, and a single-element negative-zero case. Unit source dimensions may
have structured stride zero; broadcast nonunit axes remain outside that rule.
Upstream's all-unit constant-zero gather/scatter spelling is adapted to the same
single-element tensor versions. Nonzero indices or nonunit buffers do not gain
general irregular-memory support from this rule.

Actual model2MLIR sum/product and tuple-result scan models now pass complete
native comparison against their unchanged PyTorch captures and optional Merlin
interface/runtime generation. See [the model recipe](model_integration.md).

## Structured combiner qualification

The original real Triton segmented scan refused at its scalar `scf.if`.
It now retains a noncommutative `(value, head)` combiner, mixed tuple types and
source padding through all three memory profiles. Separate real kernels retain
modular multiplication's actual shift/add `for` loop and Euclid's remainder/
`while` recurrence. Both traversal directions execute correctly. No source
operation is replaced by an opaque segment/product/GCD call.

Fourteen native cases cover i32/i64 and selected exact f32 values, forward/reverse
traversal, empty/singleton/ragged runtime domains, modular overflow and values
above 2^53. Ten actual Merlin libraries and Spike images retain all sixteen
standalone results as exact words; forty further host invocations reuse the same
libraries. Thirteen direct contract cases check nested unused effects,
all-candidate validation before edits and direct captured scalar returns. The
existing scan/model/control and structured-state regression selection passes
52 checks. Effectful combiners, arbitrary arithmetic/source-version compositions
and accelerator correctness/performance remain open.

The surveyed Meta and archived Microsoft converters register the same restricted
`CumSumConverter`; their addition-only `cumsum.mlir` fixture remains the upstream
reference. The structural-region extension uses existing standard SCF and Linalg
cloning; it introduces no source dialect, target mapping or new runtime helper.
Public ABI, receipt schemas and compiler artifact rosters are unchanged.

Two additional complete model2MLIR captures now retain learned integer weights,
dependent segmented/loop scans, forward/reverse tuple results, an empty-domain
call and the original input-derived snapshot. A value above 2^62 exercises
modular product overflow and negative intermediate results. Both the `for`
product and `while` GCD models retain all seven original results, unchanged
weights/inputs/goldens and exact words through native/Merlin host/Spike.

`out/scan-control/summary.json` independently reopens twelve target receipts,
thirty complete exact-word results and forty additional host runtime replays,
alongside a public original-source CLI capture. The combined inventory at
`out/scan-control-inventory/coverage.json` reopens 141 packages, 104 standalone
qualifications and fourteen product cases. All fifty pinned operation names
remain observed under selected native/Merlin-host profiles; this does not prove
every type, attribute, kernel composition or applicable target. Reproduce the
whole-model checks with `tests/test_scan_control_model.py`, or capture a parent
model using `scripts/capture_scan_control_model.py` in model2MLIR's environment
(`--gcd` selects the while model).

# Fresh tensor outputs from rectangular source loop nests

Complete rectangular scalar-store nests now initialize fresh functional tensor
results without an incoming output argument or private write-tracking arrays.
The actual source traversal, outer-loop bias reuse, scalar word arithmetic and
post-loop observations remain executable as standard tensor/SCF computation.

`t2m/examples/nested_overwrite_kernels.py` supplies real 2D and 3D Triton kernels.
They include row/column traversal, a permuted 3D traversal, runtime unit/empty
axes and branches using different complete traversals. A representative body is:

```python
for col in range(cols):
    value = tl.load(bias + col)       # Reused through the inner loop.
    for row in range(rows):
        index = row * cols + col
        tl.store(out + index, tl.load(x + index) ^ value)
```

## Source-derived coverage

`loop_write_coverage` reuses the existing bounded polynomial reader over actual
native pointer/index SSA. Every admitted loop has signed zero-to-bound,
unit-step semantics and a nonnegative upper bound independent of induction
variables. The exact product of the original bounds must equal the result's
actual element count. An unconditional scalar store must implement a
mixed-radix bijection from those induction values to flat result ordinals.
Recovering coefficient-one digits and successive bound products admits a
different traversal order without element enumeration or factorial search.

Runtime unit-axis substitutions apply only at a reachable store, where each
enclosing nonnegative bound is positive. A zero bound makes the exact result
product zero instead. Original entry domains and loops are unchanged. Source
arithmetic retains its independent width/overflow checks, and every active
access retains bounds checking. Coverage becomes available after the complete
outer loop, including the zero-trip case; it cannot initialize an earlier read.

An ordinary branch join may retain whole-write initialization when both arms
independently establish it, including flat and nested proofs. Otherwise only
common partial facts survive. Wrong bounds, duplicate indices, triangular
domains, nonunit steps, masked/one-arm writes and reads before coverage cannot
gain complete credit. This proves initialization, not parallel write ownership.

The existing Meta/shared dense offset/stride algebra remains the source reader.
The upstream review pins Meta `17f0c234ce474bee334af36e351c62eca7852836` and
historical Microsoft `050b93ae9c737111f1ca425d793419b32427a971`, including
`lib/AnalysisStructured/PtrAnalysis.cpp` and
`test/Conversion/StructuredToMemref/scalar_store_nested_loop.mlir` in both trees.
Meta's `tensor_indices_loop_iterargs_nested.mlir` additionally exercises nested
carried index state. Their analyses and supported passes remain in use; their
memref ABI does not itself establish this functional-output initialization
proof. No upstream source blocks were copied for the new bijection analysis.

## Complete model and consumer changes

`scripts/capture_rectangular_model.py` captures two real Triton calls:

```python
rows, cols = x.shape
first, seen0 = transform(x, self.row_bias[:rows].clone(), rows, cols, 0)
second, seen1 = transform(first, self.col_bias[:cols].clone(), rows, cols, 1)
return second, first, seen0, seen1, x, self.row_bias, self.col_bias
```

Model2MLIR resolves these runtime slice bounds from actual FX scalar SSA, even
on static parameter tensors. Checked signed clipping, source shape guards,
empty branches and both parameter slices remain. No registered opaque ATen
call is treated as a kernel implementation. The exported model domain remains
rows 2..7 and columns 2..9; standalone kernels independently admit zero axes.

Static zero tensor extents now pass the producer and independent Merlin shape
reader with exact rank/type. This is an explicit accepted-value-domain extension
under unchanged wire schemas; older positive-only readers reject it. Logical
grids remain positive. Merlin's initialization gate recognizes actual static
zeros, dynamic literal zeros and enclosing Boolean edges that prove one of an
empty value's own dimensions zero. Unknown/wrong-dimension controls and tagged
quant-inner placeholders retain their original requirements. Active accesses
to zero-sized storage remain actual aborts. Empty values grant no access,
ownership, mutation/copyback or effect permission.

## Qualification and remaining work

The focused source/model selection passes 51 checks. Seven nested-loop host
libraries cover both traversals, 3D/unit axes and both complete branch arms;
two more libraries cover static empty write/readwrite ports. The complete model
retains all seven results through changed-shape/parameter replays and two actual
serial images. Comparisons retain exact i64 words, including values above 2**53. Independent
native slice tests cover signed-i64 extremes, clipped/empty results and large
positive steps on static and dynamic tensors. Earlier failed logs are retained.

Producer `7c9f0a9b00a675d0b928449d243c22b2689724ea`, model2MLIR
`a3ce8ed153e64b1948bfa3c1c005f184f10ee1c7` and Merlin
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba` supply the implementation.
The additional original-model-domain check passes and actually aborts two
safe-kernel shapes outside the exported model domain. The overlapping producer
regression passes 188; independent Merlin checks pass 150 with two skips,
model2MLIR checks 68 and targeted new-file Ruff passes. Merlin structure/docs
gates pass. These finite counts are separate from arbitrary-kernel coverage.

`out/nested-output-coverage/summary.json` independently reopens all nine kernel
packages, the complete model and original parent archives, saved output/reference
words, replay receipts and eighteen actual serial qualifications. It retains
25 initial, 91 replay and 46 target results, six actual active-access aborts,
two original-model-domain aborts and 36 native runtime-bound slice invocations.
Fourteen selected implementation/test files match actual committed bytes across
all three repositories. Reopening passes in `.scratch/nested-output-report1.log`;
the report records actual implementation revisions and selected source hashes,
without claiming a full dependency closure or accelerator performance.

Partial row-local initialization, unions of conditional/masked source writes,
general while initialization, resizing/permuted ownership, irregular parallel
dependencies, shared/async effects and applicable accelerator execution and
performance remain full-goal work. Provenance records actual source loop IDs,
digit order, unit axes and coverage evidence; it supplies no semantic permission.
The historical operation-name inventory is unchanged.

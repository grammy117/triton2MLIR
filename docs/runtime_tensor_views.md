# Runtime dense views through the Merlin interface

Explicit `tensor_views="dense-row-major-runtime-v1"` connects checked runtime
dimensions to execution-owned binding. It selects
`execution_view_binding_profile: t2m.execution-checked-call-views.v2` under the
existing binding v4. Older readers reject the new profile/policy. Static v1
adapters retain their meaning; runtime policy requires explicit execution-owned
mode. Parent archives, public tensor ABI, receipt v2 and signature v6 are unchanged.
Compatibility metadata names the actual selected view profile.

## Source-driven geometry

Kernel input axes come from actual declared runtime dimension scalars. The
adapter reifies tensor dimensions and checks element counts before standard
collapse/expand operations. Multiple dynamic input axes and same-rank changes
of geometry are supported: two `tensor<?x?>` types do not prove that their
runtime geometries agree. Logical values preserve row-major element order;
these adapters grant no physical alias or mutation permission.

Returned kernel dimensions come from actual computation. A changed result rank
can infer one unknown axis when its other static axes have nonzero product.
Count equality enforces divisibility. Multiple unknown result axes or a zero
static factor cannot be recovered from a flat count alone; they need additional
original coordinates or a more precise kernel tensor ABI. Sample shapes and
provenance cannot fill that gap. Compatible result types retain the kernel's
actual geometry, including multiple already-declared dynamic axes.

Dimensions must be nonnegative. Standard `arith.mului_extended` on `index`
checks high-word overflow and the signed index range before reshaping. An empty
domain selects unit factors before products, preserving zero-element values
even when unused extents would overflow. Actual source/destination counts must
match. Numeric data operations, rounding, source tiling, scans, loops and logical
grids retain their original computation.

Merlin independently reconstructs every dimension operand, overflow/count guard
and adapter, then compares the complete original model/kernel SSA and locations.
Unknown profiles, changed operands and removed checks refuse after outer hashes
are repaired. Source fingerprints bind the new independent implementation.
Compiler staging still excludes invocation arrays and references.

## Real program and retained evidence

`scripts/capture_compact_model.py --matrix-views` captures a real PyTorch model
with independent exported row/column dimensions and three actual Triton kernels:

```python
original = x + self.weight                 # dynamic matrix
count = count_positive(original, x.numel())
n = count.item()
torch._check_is_size(n)
torch._check(n <= 31)
first = compact(original, n, x.numel())     # tensor<1x?xi32>
second = prefix(first, n, n)
return second, first, original, x, count, self.weight, first[:, ::2]
```

Count and copy select tensor computation; packing retains five parallel scan
stages and guarded source-word scatter. All seven original results and source
guards remain. One host library replays four independent matrix/parameter
domains, checking all twenty-eight results. Empty, full, tail and signed-i32
overflow cases retain exact words. Three actual Spike images check twenty-one
complete results, including empty and full selections.

The selected frontend suite passes 32 checks. Merlin's 47-check selection
includes actual aborts for negative dimensions, mismatched counts, unsigned and
signed product overflow, plus zero-domain and ambiguous-axis checks. Model2MLIR's
50-check provenance selection separately closes a reproduced `sym_numel` trace
gap: exact source dimension products retain static factors and shared exported
symbol multiplicity after CSE. Original range/equality guards remain in MLIR.
The trace adds a `numel_dimension_queries` relation without changing its schema;
it supplies no arithmetic rewrite, layout or source-equivalence permission.

`out/runtime-tensor-views/summary.json` reopens source packages, exact outputs,
unchanged-library replays, raw target receipts and observed native guard failures.
These selections overlap and do not increase historical opcode inventory counts.

Captured source-layout options now separately compose through
[executed source grouping](executed_source_layouts.md). Runtime non-dense source-view
coordinates, physical/mutating aliases, general ownership/effect/numeric compositions
and applicable accelerators remain open. Functional host/serial execution establishes no GPU correspondence
or preserved performance.

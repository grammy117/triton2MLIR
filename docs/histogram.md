# Logical tile histograms through Merlin

`tt.histogram` now has an executable standard expansion with original source
correspondence. The adapter accepts positive static rank-one source and counter
tensors using the six pinned Triton integer widths: 1, 4, 8, 16, 32 and 64,
with an optional matching i1 mask. Public tensor/scalar ABI schemas remain
unchanged, including named runtime allocation and logical-grid dimensions.

The expansion initializes a bins tensor once, then carries it through one
`scf.for` over the source tile. A selected, in-range element reads and increments
its indexed bin using `tensor.extract` / `tensor.insert`. Mask selection happens
before inspecting an element; masked lanes contribute nothing. Negative and
above-range values contribute nothing. Without a histogram mask, source padding
values are ordinary elements and remain observable in the counts. No atomic
memory effect is invented for this tile-local value operation.

Boolean samples denote zero/one. Wider samples keep the existing signed
interpretation and are extended before checking the bin interval. Counter
increments retain the declared integer width and modular word arithmetic.
The public Triton language creates i32 counters; other legal counter widths
are supported in verified TTIR without inventing a by-value subbyte ABI.

This is O(elements + bins) logical work and does not unroll a bins-by-elements
comparison matrix. The source tile and unmapped logical grid remain intact.
`prov.composite_family`, `prov.composite_id`, `prov.histogram_bins` and
`prov.histogram_algorithm` retain histogram intent on the loop, while the trace
records mask, widths, work shape and original IDs. A consumer may recover a
parallel/private-bin implementation by proving the rewrite; provenance alone
does not permit an effect or arithmetic change. Physical histogram implementation
and memory placement remain consumer decisions. Performance is not benchmarked.

The initial Merlin probe's LLVM also reuses one counter allocation: the input
loop contains indexed load/add/store, with no allocation or full-bin copy on
its backedge. This is a structural observation of the compiled probe, separate
from universal storage-reuse or performance claims.

## Reuse and version boundary

Meta `17f0c234` and the archived Microsoft arithmetic converters contain no
histogram expansion. Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7`
`TritonHistogramToHFusionConversion` preserves a high-level histogram; its
`HistogramConverter` uses an architecture-specific `__builtin_histogram` and
HIVM settings. Those operations cannot become successful portable opaque calls.
The source-backed comparison includes its histogram MLIR fixtures and
`test_histogram.py`. Here standard tensor state supplies the executable form
while annotations retain the high-level operation.

The triton-cpu source at `ef84a29ac523dcef1dadaf5677ffc881f1009fce`
`ConvertHistogramOp.cpp` separately shows mask exclusion and bin comparisons.
Its unrolled vector comparison strategy is not the expansion used here. No
Triton CPU, GPU or Ascend backend is imported or executed by this path.

## Measured scope and remaining limits

The visible example emits the tensor form and validates its independent NumPy
reference. Buffer and guarded profiles are selected with the existing flags:

```bash
python -m t2m shared-demo --kernel histogram --out out/histogram-tensor \
  --optimizer .scratch/toolchain/triton-shared-opt \
  --python /path/to/native-env/bin/python
```

All 34 original histogram cases pass native and actual Merlin checks. Tensor, buffer and
guarded source-word profiles qualify matching row views with runtime row masks.
The guarded profile additionally retains flat output pointers into the same
dynamic rank-two result ABI. The tensor witness currently refuses that rank
mismatch; maintained negative cases retain this existing proof limit. Three
additional profile cases now qualify unmasked dynamic rows through actual
Merlin libraries using [logical-grid bounds and exact partitions](logical_grid_bounds.md).

One compiled library per source width/mask/profile/layout handles changing input
and program counts, empty inputs/results, surplus tiles, tails, repeated bins,
negative and extreme values, and exact masked versus unmasked padding counts.
Inputs and library bytes remain unchanged. Global atomic accumulation of tile
histograms and every device implementation are
separate capabilities; tile-local histogram support does not certify them.

## Narrow integer source review

The generic-operation audit found that a real captured i8 histogram and local
i8 gather were refused by unnecessary i32/i64-only admission checks. The
existing indexed expansions now accept all six `TT_Int` widths from pinned
[TritonTypes.td](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/Triton/IR/TritonTypes.td).
They keep the same tensor state, source masks, original IDs and guards.

The gather route retains Meta's indexed `linalg.generic`/`tensor.extract`
algorithm and signed `arith.index_cast` interpretation, including i1 true
extending to -1 and failing an active index guard. No physical lane mapping is
introduced. Reviewed upstream implementation and tests are
[the converter](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TensorGatherToLinalg/TensorGatherToLinalgPass.cpp)
and [the end-to-end fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TensorGatherToLinalg/gather-e2e.mlir).
Although the TTIR prose permits smaller non-axis dimensions, the pinned native
verifier rejects them; that shape constraint remains unchanged.

`tests/test_narrow_indexed_values.py` exercises internal i1/i4/i8/i16 samples,
counters and indices through the actual public `StagedProgram` interface.
Independent integer references check mask exclusion, empty inputs, counter
wraparound and changed inputs on an unchanged library. Gather failures check
actual source-domain aborts before a result is published. Separate cases
capture real i8 Triton histogram/gather kernels. Subbyte values remain internal;
ordinary declared storage types transport their exact inputs/results.

All eighteen new public-interface cases pass, as do ten selected existing
i32/i64 histogram/gather and active-index-refusal regressions. Retained logs
are `.scratch/narrow-index-first.log` and
`.scratch/narrow-index-regressions.log` in the qualification checkout. The
separate 107-case baseline passed before this width extension; it is not a
whole-selection qualification of the new compiler candidate.

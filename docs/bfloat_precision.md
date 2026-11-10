# BF16 tensor execution and archive storage

The existing logical `bf16` tensor ABI now executes through native LLVM and
Merlin's actual host library in the tensor, buffer and guarded-buffer profiles.
Portable MLIR retains BF16 operands, F32 computations/contraction accumulators,
source narrowing operations and original provenance. No GPU layout, physical
address space or BF16 hardware instruction enters the frontend.

## Storage and package compatibility

NumPy archives carry BF16 tensors as **raw `uint16` words**. MLIR signatures
still say `bf16`; these words are never interpreted as source integer math.
The descriptor ABI, two-byte element size, artifact names and v2 receipt schema
are unchanged. Package compatibility metadata explicitly declares this encoding
and the selected native compiler requirements. BF16 inputs and references with
other archive dtypes are refused.

Numeric checks widen the words by shifting them into binary32 before applying
the declared tolerance and NaN classification. Comparing raw integers would
misinterpret signs, NaNs and numeric tolerances. Separate bitwise checks establish
copy/selection preservation, including signed zeros and signaling NaN payloads;
computed NaN payload/sign are not generally specified.

This encoding differs from model2MLIR's legacy BF16 NumPy archives, which widen
values to F32. Those archives cannot establish original signaling-NaN payloads.
Real BF16 model-call binding still needs receipt-bound exact input/result words
and BF16 safetensors weight materialization. No existing capture is silently
reinterpreted or relabeled as a qualified BF16 model.

## Reused arithmetic and native transport

Meta triton-shared revision `17f0c234ce474bee334af36e351c62eca7852836`
provides `FpToFpConverter` in
`include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp`.
Its `convert_2d_elemwise_arith_unary.mlir`, `reducesum_scalar.mlir` and
`reduce_extend_fp32_precision.mlir` tests establish BF16 casts/storage and the
upstream reduction boundary. These conversions still supply the computation;
this work adds archive transport and consumer qualification.

The host pipelines select upstream **`arith-expand{include-bf16=true}`**.
LLVM revision `a47bddccec30255619bb8c37fa59700e661d4e66`,
`mlir/lib/Dialect/Arith/Transforms/ExpandOps.cpp` and
`mlir/test/Dialect/Arith/expand-ops.mlir`, provide exact BF16/F32 bit expansion
and round-to-nearest-even narrowing with separate NaN handling. The installed
torch-mlir provider/version and actual compiler recipe remain recorded separately
from these source references. Direct host BF16 conversion flushed a minimum
subnormal in the qualifying test; the official expansion preserves it.

BF16 control-flow copies exposed a second native issue: LLVM's host lowering
can introduce a floating conversion around a BF16 phi, flushing subnormal words.
Merlin's explicit **`preserve_bfloat_transport`** feature therefore represents
remaining scalar LLVM loads/stores, constants, bitcasts, selects and CFG values
as i16 after complete LLVM conversion. Opaque pointers, two-byte GEP sizes,
effects, attributes and public descriptors stay unchanged. Constants obtain
their exact APFloat bits through upstream folding. Arithmetic, scalar function
ABIs, vectors, nonneutral floating flags and incomplete conversion refuse before
edits; provenance grants no additional numeric permission.

Read-computed loop seeds additionally require a complete unused-init parallel
generic with an independently proved output seed. Only numeric inputs and
qualified readonly/private captures supply its values. Unknown owners and init
reads supply no freshness fact. Native bufferization, alias and lifetime checks
remain mandatory; no free or ownership violation is suppressed.

## Execution evidence and remaining scope

Twenty focused cases cover all 65,536 BF16 words through tiled copies and
widening, 3,584 binary32 narrowing patterns across signs/exponents/ties,
empty/ragged runtime shapes, runtime selection loops and zero trips, and BF16
contractions with nonzero F32 accumulators. Uncomputed results match exact words;
casts check exact non-NaN words and NaN classification. Small integer-valued
contraction inputs avoid assuming a backend reduction tree. Repeated invocations
retain the same compiled Merlin library. The combined BF16/FP16/floating-atomic
regression passes 97 cases. Merlin's seed and transport tests pass separately.

Scalar indirect BF16 memory uses the existing one-element normalization in the
guarded source-word profile. Scalar dynamic accesses in the other two profiles
retain their existing ranked-tensor refusal. By-value BF16 scalars, direct BF16
arithmetic policies beyond the measured F32 computation/cast route, other
rounding modes, BF16 atomics, FP8/scaled-dot, complete model archives and
accelerator materialization remain completion work.

The visible package keeps a real source contraction and original accumulator:

```bash
T2M_MERLIN_ROOT=/path/to/merlin python -m t2m shared-demo \
  --kernel bfloat-matmul --out out/bfloat-matmul \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-python
python -m t2m qualify-merlin out/bfloat-matmul \
  --out out/bfloat-matmul-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/merlin-python --python /path/to/native-python \
  --merlin-clang /path/to/clang
```

The saved package and separate Merlin qualification pass native/reference,
compiler-input exclusion, whole-body lowering and host execution checks.
This is measured host coverage, not universal BF16 or accelerator coverage.

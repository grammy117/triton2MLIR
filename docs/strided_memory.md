# Runtime non-dense source allocations

Padded, column-major, signed/reversed and zero-stride readonly views now have
qualified tiled GEMM packages through Merlin for f32, f16 and BF16 inputs with
original f32 C/results. The existing guarded source-word path already
represents these addresses. The new vertical slice exposes and qualifies that
path without adding a layout ABI or replacing the kernel with whole-matrix math.

## Data and abstraction boundary

Inputs carry the **owning one-dimensional allocations**, including prefix,
padding and suffix storage. Typed runtime parameters carry each origin and
both element strides; capacity parameters bind the actual owning tensor sizes.
Logical M/N/K remain separate runtime dimensions. A dense incoming C and dense
result keep their logical M×N shapes. Negative and repeated addresses retain
their original meaning; independent source allocations remain an ABI premise.

This differs from supplying a dense logical M×K or K×N matrix and treating its
bytes as non-dense storage. Neither the input archives nor the compiler silently
repack a view. The reference constructs actual NumPy views with byte strides
into each owning allocation, decodes their logical values independently, and
executes ordered f32 products/additions through the original padded BK loop.
Uncomputed sentinel padding helps detect incorrect active accesses.

Actual triton-shared unstructured conversion remains responsible for pointer,
mask, gather and arithmetic lowering. Its source-word offset policy preserves
source integer widths before signed i64 pointer-offset accumulation. Read
against the pinned Meta
[`gather_no_loop`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonToUnstructured/gather_no_loop.mlir)
and
[`gather_simple_loop`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonToUnstructured/gather_simple_loop.mlir)
tests; the installed optimizer remains the separately content-pinned `0c27923`
distribution. No source/backend identity is inferred from those reference pins.

The portable body retains one named 16×16 contraction inside its original
runtime K loop with a 16×16 f32 carried tensor and 16×16×32 source tiling.
Gathered operand tiles and original C remain executable dataflow. There is no
GPU address space, warp mapping or physical matrix instruction. Runtime
admission retains bounds, ownership, initialization and active-lane checks.
Those checks are required execution evidence; this slice supplies no permission
to erase guards or assert accelerator performance.

## Provenance and validation

Source SSA, signed runtime stride/origin inputs, masks/fills and source-emission
correspondence remain in the existing trace and memory evidence. The common
emission boundary now also preserves `prov.intent` when a function is rebuilt
for standard-memory profiles. The caller description remains labeled as such
in the trace and grants no arithmetic, memory or numeric semantics.

Three real typed packages reuse their compiled Merlin libraries across ten
additional layouts/domains each: row/column padding, column-major views,
reversed axes, mixed signs, repeated reads, multiple tiles, tails, empty M/N,
zero K and a C=`2^24` accumulator-order case. Zero-K invocations tolerate
unobserved origins outside empty input storage. Two active out-of-bounds cases
per type abort through the actual compiled guard and remain refused invocations,
with no output archive or successful invocation receipt. They cannot inherit
the successful source-assertion outcome contract.

The final three typed integration cases and 14 existing source/emission cases
pass. The visible BF16 allocation package reopens 33 producer and 52 Merlin
qualification artifacts. The explicit 37-package observation report remains
31/50 definitions observed and 27 in host-qualified captures. These counts
describe selected compositions and do not close all type/attribute coverage.

Public tensor/scalar ABI v5 and package schemas are unchanged. This profile
qualifies source allocations and dense results; a pure logical-tensor witness
for arbitrary non-dense views, aliases between declared allocations,
learned model compositions and applicable target
materialization remain work in the [completion plan](completion_plan.md).
[Strided writable outputs](strided_outputs.md) additionally retain the complete
incoming output allocation and reject conflicting stores through the same path.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel strided-bfloat-dot \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/strided-bfloat --python /path/to/native-python
python -m t2m qualify-merlin out/strided-bfloat --out out/strided-bfloat-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`strided-dot` and `strided-half-dot` select the other input formats. BF16 native
materialization requires the explicit Merlin compiler-support root; examples,
references and source storage values never become compiler inputs.

# Measured coverage and remaining boundaries

Use `shared-coverage` to rerun a fixed, bounded native pipeline against actual
upstream fixtures. It inventories operations structurally, retains every input,
output and diagnostic, and records content hashes and the corpus Git revision.
It never executes fixture `RUN` comments. Negative tests and inputs intended for
other pass boundaries stay visible in the denominator.

```bash
python -m t2m shared-coverage /path/to/triton-shared/test/Conversion \
  --out out/coverage --optimizer /path/to/triton-shared-opt \
  --stage tensor-ir --budget 120
```

## Native corpus measurement, 2026-11-07

Corpus: Meta `17f0c234ce474bee334af36e351c62eca7852836`, all 255 files in
`test/Conversion`, split into 339 native MLIR test cases. Optimizer binary SHA-256:
`2af44b4536b2dffbdb90b877b04e97cd2b10fe333020606fa4a1814b3c34926d`.
The [distribution/source pins](triton_shared.md) differ from the corpus revision.

| Observation | Tensor stage |
|---|---:|
| Cases measured / discovered | 339 / 339 |
| Native-verified source cases | 335 |
| Successful native stage conversions | 310 |
| Structurally inventoried cases containing `tt.func` | 323 |
| Successful stages in that observed TTIR subset | 309 |
| Source inventories unavailable | 1 (fp8 type unsupported by xDSL 0.68) |

These numbers measure a reusable middle layer. They are **not** functional tensor
ABI coverage, numeric equivalence, or Merlin materialization coverage. A whole
opcode is not certified merely because it appeared in a passing composition.
The complete per-case/opcode report is generated as `coverage.json`; intermediate
failures are retained alongside it.

Failures include source/tool version differences (`tt.cat`, structured-state
schemas), pass-specific fixtures containing pointer-tensor forms that this full
stage does not accept, conditional block-pointer conversions, and missing native
dialect interfaces when the inliner is given some already-standard functions.
Changing the expected input boundary is different from adding TTIR support.

## Actual product evidence

The [shared functional tensor path](shared_tensor.md) now executes source-captured
2-D add, masked softmax/reduction helpers, IEEE matmul and ordered multiple-output
stores through standard MLIR and native LLVM. Its op decomposition comes from
triton-shared; the local adapter checks the tensor ABI and memory effects.
The existing reference frontend additionally qualifies rank-one masked tiled
pointwise kernels with an unmapped logical grid and symbolic ownership checks.

The principal work still needed for broad functional tensor coverage is:

1. General logical grids and memory/pointer recurrences with established output
   ownership; preserve useful loops and reuse rather than serializing launches.
2. Strided/permuted/block descriptors, dynamic bounds and irregular access with
   defined gather/scatter conflict behavior and supported alias/effect contracts.
3. Mixed-precision, cast, scan/combiner and contraction numeric policies. A source
   attribute that upstream drops cannot be repaired by provenance alone.
4. Consumer model-call/parameter binding and staged support for all retained
   payloads. Standard MLIR verification and host execution do not discharge these.
5. Explicit effectful/backend profiles for atomics, synchronization, volatile
   observations and hardware intrinsics. Arbitrary hardware-observing assembly
   has no universal equivalent in a pure mathematical tensor ABI.

The archived Microsoft implementation is useful evidence for historical patterns
and regressions. Its monolithic converter imports many of the same patterns as
the staged pipeline, including reductions, matmul, reshape and a cumsum-specific
scan converter. It is not another complete universal converter that removes these
ABI/numeric obligations. See the [pinned source comparison](triton_shared.md).

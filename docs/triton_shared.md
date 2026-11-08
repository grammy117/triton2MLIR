# Reusing triton-shared

triton-shared already provides much of the transformation machinery needed by this project.
Use its analyses, converters and corpus as the primary reuse path for broader kernel coverage.
triton2MLIR supplies the functional tensor ABI, logical launch normalization, intent/provenance,
semantic qualification and model2MLIR/Merlin package contract.

The surveyed source is pinned to **17f0c234ce474bee334af36e351c62eca7852836**, with Triton pin
**142586f90aac9b917af1081e4697b085027e13b2**. The stock-wheel path currently uses Triton 3.7.1;
these are separate version contracts, not assumed to be interchangeable.

## Source-backed reuse map

| Roadmap capability | Existing upstream mechanism / fixtures | Work owned here |
|---|---|---|
| R1 multidimensional/strided access | `TritonToStructured`, `TritonStructured` descriptors; `addptr_2d_example.mlir`, `masked_ldst_2d.mlir` | Bind shapes/strides/effects to logical tensors; prove bounds and ownership; preserve origin. |
| R1 masks and mixed address/value uses | `MaskAnalysis`, `UseAnalysis`; scaled masks and mixed-use regression cases | Retain source fill/undefined semantics and all observations; qualify supported compositions. |
| R2 loops and pointer recurrences | Structured-state prepass, loop converters; `addptr_for_used_after_update.mlir`, nested-loop/branch fixtures | Normalize complete-launch grid/effects and functional outputs; retain recurrence and source correspondence. |
| R2 dot/batched dot | `MatmulConverter`, `dot.mlir`, `batched_matmul.mlir` | Check numeric profile, real accumulator semantics and mixed precision; preserve source tiling/reuse. |
| R3 reductions/softmax | Region/reduction conversion and fused-softmax/layer-norm fixtures | Review supported combiners, NaN/tie/precision semantics; optional m2m composites and package agreement. |
| R6 modulo/gather/scatter | Structured/unstructured access passes, gather/scatter descriptor and conversion fixtures | Bound indices, alias/conflict semantics and declare the accepted tensor/effectful profile. |

These are observations from upstream code/tests, not measured local coverage percentages.
The pinned dot converter initializes matmul to zero and adds a nonzero C afterward; it does not
by itself qualify every source accumulation order or `inputPrecision` contract. Its structured
pass exposes `use-unsafe-mask`; the adapter explicitly disables it. Reuse needs numerical/effect
qualification at our boundary, with existing upstream source as the starting implementation.

## Runnable stage adapter

```bash
export TRITON_SHARED_OPT=/path/to/existing/triton-shared-opt
python -m t2m shared-lower kernel.ttir --out out/shared_structured --stage structured
python -m t2m shared-lower kernel.ttir --out out/shared_compute --stage tensor-compute
python -m t2m shared-lower kernel.ttir --out out/shared_full --stage full
```

An explicit `--optimizer` overrides discovery. `--optimizer-sha256` can require exact binary
contents. The adapter records observed binary/input/output hashes and the reference source pins;
it does not pretend a binary's source revision was verified from a hash or nearby checkout.

| Stage | Boundary |
|---|---|
| `structured` | Run safe structured-pointer analysis; retain source tensor/control-flow computation. |
| `tensor-compute` | Reuse structured/unstructured and arithmetic-to-linalg passes. Project memory operations and the upstream per-program ABI may remain. |
| `full` | Run upstream's complete experimental conversion, including ptr/memref memory boundaries. |

Each writes `source.ttir`, `lowered.mlir`, `stderr.log` and `stage.json`. The report marks
`portable_conversion_complete=false`, `tensor_abi_compatible=false` and source correspondence/
numerics unqualified. It deliberately emits no `model.mlir` or capture receipt. A successful
upstream stage is a useful reusable result; it is not yet the promised Merlin tensor package.

The optimizer verifies each pass and prints generic assembly, which preserves attributes and
provides a structural intake for a later adapter. Invocation disables MLIR threading, defaults
to a 30-second timeout, caps output files at 16MiB and never overwrites an output directory.
It never builds a native toolchain or uses triton-cpu/the reference CPU driver.

## Qualification and next integration

1. Run the pinned upstream fixture corpus with an existing, revision-qualified optimizer. The
   current VM has reference sources but no discovered `triton-shared-opt`; opt-in compiler tests
   skip until `TRITON_SHARED_OPT` is set. Process-transport tests are labeled separately.
2. Prefer the structured/tensor-compute boundary for functionalization. The complete memref pipeline
   remains valuable as a differential reference, with its different ABI recorded explicitly.
3. Add a descriptor-to-logical-tensor adapter and whole-launch wrapper using the declared kernel
   ABI. Reuse upstream arithmetic/region conversion, preserving source grid and recurrence.
4. Establish actual source-to-emitted correspondence across rewrites. Counts, textual similarity
   and whole-function annotations do not prove per-operation attribution or work retention.
5. Qualify numerics/effects, then emit the existing versioned capture package with its explicit
   compatibility differences. Until then, the small stock-wheel frontend remains a working
   contract/reference path; broad upstream coverage is not claimed as local frontend coverage.

No native build was started for this integration. Building a version-matched Triton/LLVM stack on
the shared VM requires a concrete resource plan and approval under the machine rules.

Primary source references:

- [Structured conversion and options](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonToStructured/Passes.td)
- [Arithmetic, matmul and loop patterns](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp)
- [Full pipeline boundary](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp)

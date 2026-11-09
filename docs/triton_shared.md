# Reusing triton-shared

triton-shared already provides much of the transformation machinery needed by this project.
Use its analyses, converters and corpus as the primary reuse path for broader kernel coverage.
triton2MLIR supplies the functional tensor ABI, logical launch normalization, intent/provenance,
semantic qualification and model2MLIR/Merlin package contract.

The [full memory bridge probe and plan](full_memory_bridge.md) now demonstrates
the full shared pipeline through Merlin's actual whole-body compiler. Its later
memory form is usable at that consumer boundary; general functional ABI,
memory/numeric/source admission and dynamic profiles still require the stated work.

The surveyed source is pinned to **17f0c234ce474bee334af36e351c62eca7852836**, with Triton pin
**142586f90aac9b917af1081e4697b085027e13b2**. The stock-wheel path currently uses Triton 3.7.1;
these are separate version contracts, not assumed to be interchangeable.

The installed shared binary also predates the surveyed cat/gather converters.
The native-verified intake adapter can expand rank-one cat to ordered
join/transpose/reshape, while the tile-gather adapter reuses Meta's parallel
generic/extract form with executable index checks. This is an explicit version
bridge, with separate original/normalized source graphs and package differences;
see [the qualified selector compositions](selected_pointers.md).

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

### A bounded prebuilt toolchain

`scripts/fetch_shared_opt.py --out .scratch/shared-opt` downloads a content-pinned
AMD Triton-XDNA wheel, verifies its published SHA-256 and the standalone optimizer's
SHA-256, and extracts only `triton-shared-opt`. It installs no backend and compiles
nothing. Peak disk use is approximately 900 MB; the temporary wheel is removed.
The moving release tag is protected by exact content pins and fails on replacement.
Set `TRITON_SHARED_OPT=$PWD/.scratch/shared-opt/triton-shared-opt` afterward.

The surveyed executable reports LLVM 24.0.0git. Its distribution source is AMD
`7180fccfa0283456eb083a5718104da8156ae50b`, with shared submodule
`0c27923cbcb51f1565ae39c2de0e2245babb6067`; this differs from the Meta survey pin.
The script records this distinction and does not claim reproducible-build attestation.
All four real stage tests pass on this host with this executable, including the
`tensor-ir` stage that retains `tt.func` and logical program IDs for our ABI adapter.

The distribution builds its Triton IR library from the `triton-lang/triton-cpu`
fork, as recorded by its pinned `.gitmodules`; this is a build-provenance difference.
Only its standalone **shared IR optimizer** is extracted. We do not install or
invoke that fork's target backend, CPU driver, launcher or runtime. Source capture
still uses the pinned stock Triton wheel, and host qualification uses standard
MLIR/LLVM through existing torch-mlir. A compiler built directly against the Meta
survey pin remains a separate, unperformed source-build qualification.

The real [corpus coverage report](coverage.md) and [functional tensor adapter](shared_tensor.md)
supersede the initial stage-only status below. Reused arithmetic conversion now
feeds complete, natively checked example packages, within the adapter's explicit scope.

The historical Microsoft repository is archived, at surveyed revision
`050b93ae9c737111f1ca425d793419b32427a971`. It contains both the earlier monolithic
`TritonToLinalg` implementation and the staged experimental pipeline inherited by
the Meta repository. These are related implementations, not independent coverage
oracles. Prefer the staged implementation; consult old patterns/tests when a new
pipeline fails rather than silently selecting an unqualified legacy pass.

Sources: [AMD packaging](https://github.com/amd/Triton-XDNA/blob/7180fccfa0283456eb083a5718104da8156ae50b/setup.py),
[historical Microsoft conversion](https://github.com/microsoft/triton-shared/tree/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToLinalg).

1. The pinned corpus has now been measured using the content-pinned executable described above.
   Source revision/build provenance differences remain explicit; tests still skip on machines
   without `TRITON_SHARED_OPT`. Process-transport tests are labeled separately.
2. Prefer the structured/tensor-compute boundary for functionalization. The complete memref pipeline
   remains valuable as a differential reference, with its different ABI recorded explicitly.
3. The descriptor-to-tensor adapter and whole-launch wrapper now preserve qualified static grids,
   source computation and bounded loop recurrences. Extend that boundary to pointer-valued state,
   dynamic/permuted/irregular descriptors and effectful branches.
4. Source name/callsite locations now establish emission origin across rewrites. Counts, textual
   similarity and whole-function annotations still do not prove every elimination or work retention.
5. Complete example packages now have native numerical/effect checks and a versioned tensor ABI.
   General type/numeric/effect coverage and consumer binding remain work; broad upstream pass
   success is not counted as product support.

No native build was started for this integration. Building a version-matched Triton/LLVM stack on
the shared VM requires a concrete resource plan and approval under the machine rules.

Primary source references:

- [Structured conversion and options](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonToStructured/Passes.td)
- [Arithmetic, matmul and loop patterns](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp)
- [Full pipeline boundary](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp)

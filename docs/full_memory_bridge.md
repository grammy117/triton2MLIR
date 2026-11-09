# Reuse the full shared pipeline through a functional memory boundary

The closed integration test [test_full_memory_bridge.py](../tests/test_full_memory_bridge.py)
now captures real runtime-prefix TTIR, runs triton-shared's full pipeline, clones
its complete standard body into a functional tensor wrapper, and compiles that
body through Merlin's actual whole-body LLVM/library path. Separate evaluation
checks `n=0,1,7,13` and input preservation. This is a closed prototype, not generic
portable conversion, source admission, or universal coverage.

This changes an earlier architectural assumption: the full pipeline is usable
with the payload-preserving Merlin compiler. Its result still contains linalg
and tensor computation, standard memory views/transfers, regions and allocations.
It does not require replacing that dataflow with LLVM instructions or a GPU launch
ABI at the public boundary. The tensor-stage adapter remains valuable for directly
re-raised tensor slices and established ownership/initialization proofs.

Source boundary: Meta `17f0c234ce474bee334af36e351c62eca7852836`,
`TritonToLinalgExperimentalPass.cpp` and `TritonArithToLinalg/ConversionPatterns.hpp`.
The source pins and executable pin remain distinct as documented in the
[reuse map](triton_shared.md). Upstream appends **three program counts, then three
program IDs**. The prototype binds these inside an unmapped logical `scf.forall`;
they do not become public hardware launcher arguments.

The prototype makes read-only tensor-to-buffer views, allocates private writable
state, copies the incoming output state, retains all upstream payload operations,
and returns the resulting logical tensor. Those copies establish functional
semantics; dropping them requires a proven fresh/overwritten or writable-alias
contract. Scoped native bufferization/deallocation remains Merlin's responsibility.
The current xDSL materialization binding had incorrectly inferred an identity
layout from a tensor type. Merlin now checks the native compatible-shape/element
relation and result/access rules, accepting the actual strided destinations.

## Supported initial profile

`shared-demo`, `shared-tensor` and `bind-model` now accept an explicit
`--memory-profile buffer`. The default remains `tensor`; there is no automatic
fallback. Both profiles retain the logical tensor/scalar function ABI. The
buffer profile preserves the upstream standard `memref`/`bufferization` payload
inside an unmapped logical grid, with private writable state and tensor results.
Dense multidimensional inputs have checked linear views for upstream raw pointer
arguments. Program counts and IDs remain internal logical values.

This initial profile **requires the existing tensor functionalizer's source
bounds, initialization and grid-ownership analysis as a semantic witness**.
It extends the supported representation and consumer boundary; it does not
increase source opcode coverage by itself. Numeric repairs, including nonzero
dot accumulators, run before the memory continuation. Unknown effects and
unproved write conflicts still refuse before that continuation runs.

Four additional artifacts bind the transition: `memory-witness.mlir`,
`memory-input.mlir`, `memory-stage.mlir` and `memory-stage.json`. Provenance and
the stage report bind their actual bytes. Bundle verification and Merlin's
compiler-only stager check those relations independently. Bound models retain
the evidence under `kernels/0/`, alongside original parent data and FX evidence.

Native and actual Merlin compiler/library tests cover runtime prefixes,
multidimensional addition, ragged GEMM, descriptors and tuple scans. Actual
model tests cover ordinary, ragged, block-pointer and descriptor GEMM bindings
without replacing model weights, inputs or goldens. These are the qualified
examples within the existing semantic proof scope, not universal conversion.

## Profile invariants and remaining gates

1. Add an explicit mixed tensor/memref memory profile alongside the existing
   tensor profile. Retain the same logical tensor/scalar public boundary and
   state the payload/package differences. A backend binary or an opaque call is
   not the portable result. Select profiles by declared capability, not kernel
   names or an automatic unproved fallback.
2. Preserve original/native-verified TTIR inventory and source names before
   running the full pipeline. Keep original and normalized graphs separate.
   Account for generated ABI memory operations separately from source operations.
3. Bind typed source arguments and the full launcher ABI exactly. Flatten dense
   logical buffer views where the upstream raw-pointer ABI expects linear memory;
   a rank-N tensor is not automatically a rank-one pointer view. Preserve
   specialized/runtime values and layout/alias constraints explicitly. Preserve
   source region-carried types through ranked-to-unranked buffer casts where
   needed; correct remaining native/xDSL binding differences rather than changing
   nested SSA types to placate a verifier.
4. Preserve numeric policies before memory lowering, including real dot
   accumulators, precision/cast modes and original address-width arithmetic.
   Interpose the existing scan/accumulator adaptations where required; do not
   assume full-stage success establishes those semantics.
5. Establish memory observation bounds, initialization and effects from the
   source or executable guards. The prototype's one known footprint is not a
   generic alias or grid-independence proof. Retain atomics/order/scope, volatile
   observations and dependencies when admitted; unresolved cases stay refused.
6. Retain useful program/tile structure and data reuse. Generic memory
   functionalization must not serialize unproved conflicts or hide a changed
   algorithm. Fresh output input omission still requires complete initialization;
   a more general state-transition ABI may retain incoming writable state with
   that difference explicit.
7. Extend to actual dynamic tensor dimensions, launch extents and result
   allocation tied to runtime arguments. Compare multiple shapes through one
   compiled library, including empty/tail domains. Static capacity with a varying
   prefix remains a different capability.
8. Add complete package/compiler-only staging and real model binding evidence,
   retaining parent FX/weights/goldens. Qualify every result and all observed
   effects through the actual consumer, including supported target materialization.

Next generalize the memory proof independently of constructing a pure tensor
witness, reusing upstream analyses and retaining the full pass composition.
Keep both public profiles explicit. The supported wrapper removes a consumer
incompatibility; it does not discharge the remaining dynamic, numeric or effect
gates above.

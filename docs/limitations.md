# Limitations, expected encounters and closure work

This inventory accompanies the [closure plan](completion_plan.md). It explains
what users will encounter and why; it is not a declaration that closure has
already been achieved. Generic computation remains required. A failed proof or
specialized pattern is a reason to try broader standard MLIR with explicit
storage/effects, not to erase the computation or label it GPU-specific.

## Hardware or unavailable-source semantics

| Limitation | Why a kernel might use it / when it appears | Why portable lowering cannot infer it | Current remedy |
| --- | --- | --- | --- |
| Arbitrary inline assembly | Hand-tuned attention/GEMM, packed arithmetic, special instructions, approximate math and device diagnostics. | Assembly text can observe physical state or have undocumented value/effect semantics; a purity label is insufficient. | Default intake refuses with source location. Existing optional qualified integer providers remain available. Supply a typed semantic implementation or use ordinary Triton operations. Assembly coverage is not required for generic-op closure. |
| Physical addresses, clocks and hardware identities | Pointer tagging, device allocators, alignment tricks, clock-based profiling and lane/core-specific algorithms. | Logical tensor roots and iteration coordinates do not define a numeric GPU address, timer or physical participant identity. | Preserve logical pointer offsets/round-trips where proved. Escaping physical observations require an explicit backend contract or remain unsupported. Do not invent addresses or clocks. |
| Post-layout GPU operations and physical descriptor bytes | Saved backend IR, warp-specialized kernels, architecture-selected layouts and DMA descriptors. | These encode a particular device's storage/participation rules rather than pre-layout tensor computation. | Capture earlier TTIR and use logical descriptors. Physical descriptor/runtime implementations belong in target support. This frontend does not decompile GPU binaries. |
| General cross-program/external synchronization and async completion | Producer/consumer pipelines, inter-CTA queues, device/host publication and asynchronous shared-memory staging. | Serializing a launch or dropping a wait can change permitted observations, ordering and lifetime. The existing invocation-local profile does not define arbitrary external participants. | Qualified local phases/effects remain supported. A synchronous portable expansion needs a real equivalence/dependency argument. Otherwise require an explicit execution provider and report backend-required. |
| Unknown external function implementation | Vendor device libraries, custom approximate math and private helper libraries. | A function name/prototype does not define its arithmetic, rounding or side effects. | Existing qualified external math profiles work. Supply a typed implementation/effect contract; unspecified external semantics cannot count as portable support. The generic call operation itself is not being discarded. |
| Exact agreement with every GPU arithmetic mode | Tensor-core precision modes, FTZ/subnormal behavior, packed low-precision formats and approximate device functions. | Different selected source/backend numeric policies can produce different words. Tolerances or provenance cannot silently grant a changed policy. | Select an implemented named numeric profile and expose differences. A missing generic arithmetic implementation remains work; unspecified hardware behavior needs an explicit contract. |

Cache hints, physical coalescing and placement choices may become provenance
when they do not affect values/effects. Logical tiles, loaded-value reuse,
accumulation order, masks and dependencies remain executable. A generic load
fallback must keep its actual root, indices, guard, fill and source ordering.

## Current product boundaries

| Boundary | When it comes up | Current state / next action |
| --- | --- | --- |
| Automatic launch discovery | A PyTorch `triton_op` wrapper launches several kernels, performs surrounding arithmetic, uses views/mutation, symbolic grids, autotuning or hidden scratch. | The new automatic adapter currently covers static pure single-launch wrappers with fresh returned outputs. It rejects unaccounted wrapper work. Existing explicit model/storage bindings cover additional cases; broader automatic wrapper composition remains open. |
| Source/version compatibility | A model uses a different Torch/Triton release or saved IR from another compiler pipeline. | The TTIR frontend is pinned to Triton 3.7.1. Automatic discovery probes Torch 2.10 and records its separate source-capture version boundary. New versions need compatibility qualification rather than silent parsing. |
| Public physical alias/copyback | Callers expect Python-visible mutation, overlapping storage identity or shared external memory. | Functional state/results and selected explicit storage adapters exist. Unimplemented physical invocation behavior must be requested explicitly; returning a changed tensor does not imply caller copyback. |
| Generic address/shape/ownership compositions | Loaded/nonlinear indices, selected roots, changing loop-carried storage, value-sized intermediates and unusual valid control-flow combinations. | Many qualified paths exist, but universal composition closure has not been established. Use checked roots/extents/shape/scalar bindings and generic logical memory/control expansions; remaining valid generic failures are release blockers, not hardware exclusions. |
| Packet-sharing experiment | Optional narrow integer assembly inside a dynamic recurrence. | An uncommitted packet-preserving path exposes a Merlin lifetime-proof failure. The committed scalar word recipe has qualification; no lifetime check may be bypassed. This optional assembly optimization is distinct from generic-op support. |
| Demo breadth | Training, dropout, multi-head/causal attention, large models or arbitrary dynamic sequence lengths. | The delivered demo source is two complete small single-head pre-norm inference blocks with ReLU FFNs and no dropout/causal mask. It demonstrates automatic integration and correct computation, not those unmeasured model configurations or performance. |
| Target execution/performance | A user expects an accelerator executable or the GPU kernel's throughput. | Native and public Merlin host execution plus selected serial images establish their own scope. Accelerator scheduling, resources, runtime ABI, correctness and timing need separate target qualification. Portable MLIR success is not a speedup claim. |

The final closure report must retain this distinction: hardware exclusions,
missing user/source contracts, automatic-adapter limits, and unfinished generic
implementation are different outcomes. None becomes a successful portable
lowering merely because a diagnostic or provenance record exists.

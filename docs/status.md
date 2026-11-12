# Progress toward arbitrary TTIR kernels reaching Merlin

Updated 2026-11-11. **The end-to-end pipeline works for a broad qualified
selection; arbitrary-kernel coverage remains incomplete.** The goal is a
model2MLIR-like package whose executable MLIR preserves the kernel's algorithm
and observable semantics, enters Merlin's ordinary compiler interface, and
executes correctly on each applicable qualified target.

## Progress against the end goal

The project has passed the minimum demo and basic interface stages. Real
kernels and enclosing model captures can already travel through the complete
pipeline. The remaining work is to make that working path cover the full
declared source and consumer scope, rather than establish another frontend.

| Part of the goal | Working capability | Practical limit |
|---|---|---|
| Portable computation | Standard Linalg/Tensor/SCF/arith/math/cf; tiled contractions, runtime K reuse, reductions/scans, branches, loops and selected packed computation. | Every type, attribute, region and composition is not yet qualified. |
| Memory and state | Masks/fills, signed runtime strides, selected pointers, closed pointer-integer displacements, logical descriptors, owning-root updates, typed/byte aliases and selected atomic profiles. | Physical descriptors/escaping addresses, general shared/external state, synchronization and arbitrary alias/dependency compositions remain open. |
| Numeric intent | Typed accumulators, low-precision/scaled contractions, composite dot profiles and independent per-cast FP8 contracts. | Explicit software profiles do not establish equivalence to every GPU backend's arithmetic or all normalizer/model combinations. |
| Provenance | Original source, complete operation accounting, separate normalized graphs and emission correspondence. | Custody and intent annotations grant no omitted arithmetic, memory or effect semantics. |
| Package and enclosing model | Standalone and model2MLIR-bound captures, weights/lifted constants, runtime kernel scalars, complete tuple results, dependent state and selected source tensor views. | General physical/mutating views, dynamic public model ABIs, value-returning TTIR entries/CFGs and broader binding compositions remain open. |
| Consumer execution | Real native execution, reusable Merlin host libraries and selected complete model/kernel images executed on RISC-V/Spike. | Functional CPU execution does not qualify accelerators, hardware placement or preserved performance. |

In roadmap terms, R0's minimum demo is delivered. R1–R5 have working
end-to-end paths for tensor memory, tiled/control-flow kernels, reductions,
online attention and model integration. Their qualified forms are documented;
this does not close every variation. R6–R8 remain partial: irregular/dynamic
compositions, general effects and arbitrary-kernel semantic coverage are the
main completion work. Target correctness and performance remain separate gates.

Useful tile structure, reuse and recurrence remain executable. The logical
grid stays independent of warps, threads, cores and physical memory placement.
Source-language assembly implementations are explicit optional adapters with
standard expansions. Unknown semantics produce located refusals; an opaque
fallback is not counted as portable success.

## What the measurements establish

The maintained inventory at
`out/closed-pointer-address-inventory/coverage.json` reopens **113 packages,
81 standalone qualifications and fourteen product cases**. All **50 pinned TTIR
definition names** occur in source, native- and Merlin-host-qualified captures
under selected profiles. This is breadth across a pinned dialect inventory,
not a percentage of arbitrary kernels or a per-operation execution proof.

Later selections add independently checked complete models, runtime target
ABIs, effects, source views and precision compositions. They remain separate
from that inventory; counts must not be added together as unique coverage.
Selected whole models retain learned weights, dependent calls, original
snapshots and every result through Merlin host and actual Spike execution.
The [coverage record](coverage.md) links the concrete receipt selections and
their limits.

Recent work closes a source-preservation gap across official normalization:
casts, composite precision and selected assembly are audited and made executable
before native rewrites or helper folding. Original and transformed graphs remain
separate. Pointer/descriptor allocation state and complete learned-state models
now compose with those contracts through native, Merlin host and Spike.
The newest descriptor/assembly selection reopens six standalone kernels and two
complete models: eight actual target images, fourteen exact-word results and
26 additional host runtime replays. The [source assembly contract](source_assembly.md)
records this finite selection; it does not establish arbitrary assembly/effects.

[Closed pointer-integer arithmetic](pointer_addresses.md) now additionally
recovers actual logical bases and proved byte displacements without materializing
physical addresses. Four i8/i16/i32/i64 kernels pass native/Merlin host/Spike.
Three scalar-memory libraries add all-profile mask/state handling; the seven
libraries replay 25 additional runtime invocations. Escaping address values stay
explicit refusals. This adds a portable implementation for another concrete
semantic family, while general addresses/effects and target scope remain open.

## What still separates this from the end goal

1. **Close composition gaps.** Extend protected source contracts through
   the remaining type/attribute, view/mutation, runtime/alias, region/CFG and
   numeric/effect combinations. Allocation wrappers and enclosing models work
   for qualified forms; individual operations do not establish their closure.
2. **Give hardware-dependent observations concrete semantics.** General
   synchronization, physical addresses/descriptors, external implementations,
   broader assembly and shared effects need portable definitions or consumer
   implementations. Preserve useful dependency/reuse structure; annotations
   alone cannot replace behavior.
3. **Qualify applicable targets.** Extend Merlin scheduling, payload
   materialization and execution across the retained computations/effects.
   Accelerator correctness and performance preservation require their own
   evidence; host/Spike success supplies neither automatically.
4. **Make the denominator match the claim.** Cover supported source versions
   and type/attribute/kernel combinations with original-source accounting,
   independent references, negative cases and complete consumer results.

The [completion plan](completion_plan.md) gives the execution order and gates.
The [package contract](package.md) records explicit model2MLIR differences.
The next milestone is to turn the opcode inventory into a maintained semantic
matrix with concrete unsupported forms, then close those forms through the
same package and consumer. Priority remains remaining effect/external/packed
families, runtime/shared-state bindings and applicable targets. More passing
variants of an already-qualified fixture cannot substitute for those gaps.
There is no defensible arbitrary-kernel completion percentage yet: the
50-name inventory measures a different denominator.

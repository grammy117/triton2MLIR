# Progress toward arbitrary TTIR kernels reaching Merlin

Updated 2026-11-12. **The end-to-end pipeline works for a broad qualified
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

[Captured tensor lookups](scalar_tile_reads.md) now retain loaded immutable
table values across intervening source writes, scalar lookup bounds and runtime
reduction/scan recurrence. Twenty-four i32/i64 libraries pass native/Merlin host/
Spike with all 48 results exact and 96 additional host invocations. Actual
existing-library bounds aborts and provenance-independent outcomes are checked.
See the [review handoff](handoff.md) for reproduction and remaining limits.

[Scalar TTIR region composition](scalar_regions.md) additionally closes the
conversion-order gap for checked precise math, clamps, casts, numeric bitcasts
and existing pure external mappings. Twenty-four floating/integer libraries
pass native/Merlin host/Spike and 96 further host invocations. BF16-to-f64 uses
an exact standard f32 bridge; region/control/effect checks remain mandatory.

[General reductions](reductions.md) now retain complete pure scalar bodies,
coupled tuple order/types, captures and structured control with real source
initialization. Twenty-four libraries and four complete models pass native,
Merlin host and Spike, retaining all 80 complete target results as exact words
and 96 further unchanged-library host invocations. Remaining scalar TTIR
builtin/region compositions and memory/effect/backend forms remain open.

[Exact IEEE source literals](ieee_literals.md) now retain original NaN sign,
payload and signaling bits in standard executable constants/bitcasts, including
allocation wrappers and complete enclosing models. Sixteen libraries and four
models compare 56 exact-word results through native/Merlin host/Spike, with
48 additional unchanged-library runtime invocations. Broader numeric/effect,
alias/shared-state and accelerator closure remains open.

[Structured scan combiners](scans.md#structured-combiner-qualification) now
retain scalar branches and actual for/while recurrences in parallel scan stages.
Ten libraries and two complete learned-state models pass native/Merlin host/
Spike, retaining thirty complete target results as exact words. Effectful
combiners, broader numeric/alias/source-version compositions and accelerator
coverage remain open.

[IEEE f16/BF16 atomic addition](half_atomics.md) additionally closes the missing
half arithmetic profile through native/Merlin host and actual Spike. Four
libraries and two complete models retain twenty target results, including twelve
exact-word model results. Remaining numeric/effect/shared-state and accelerator
forms stay open.

[Original multi-block CFGs](cfg_intake.md) now also reach the interface through
LLVM's official structuralization, with original successor/block arguments
retained and normalized IDs kept separate. Qualified early returns and numeric
backedges/phi values execute through all memory profiles; a complete learned-state
model preserves dependent updates and every original result. General CFG forms
and value-returning public entries remain further work.

## What the measurements establish

The maintained inventory at
`out/tile-lookup-inventory/coverage.json` reopens **237 packages,
192 standalone qualifications and fourteen product cases**. All **50 pinned TTIR
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
The descriptor/assembly selection reopens six standalone kernels and two
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
The next milestone closes remaining region forms, tensor intermediates/captures, valid
memory/external effects, runtime/shared-state bindings and applicable targets
through the same package and consumer. Checked scalar arithmetic now expands
before region admission; arbitrary builtin/effect bodies retain their separate
requirements. The pinned source verifier forbids map writes, atomics and volatile loads, so those are
invalid-source refusals rather than missing portable map-store implementations.
More passing variants of an already-qualified fixture cannot substitute for those gaps.
There is no defensible arbitrary-kernel completion percentage yet: the
50-name inventory measures a different denominator.

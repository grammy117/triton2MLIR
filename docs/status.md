# Progress toward arbitrary TTIR kernels reaching Merlin

Updated 2026-11-11. **The end-to-end pipeline works for a broad qualified
selection; arbitrary-kernel coverage remains incomplete.** The goal is a
model2MLIR-like package whose executable MLIR preserves the kernel's algorithm
and observable semantics, enters Merlin's ordinary compiler interface, and
executes correctly on each applicable qualified target.

## Established paths

| Part of the goal | Working capability | Practical limit |
|---|---|---|
| Portable computation | Standard Linalg/Tensor/SCF/arith/math/cf; tiled contractions, runtime K reuse, reductions/scans, branches, loops and selected packed computation. | Every type, attribute, region and composition is not yet qualified. |
| Memory and state | Masks/fills, signed runtime strides, selected pointers, logical descriptors, owning-root updates, typed/byte aliases and selected atomic profiles. | Physical descriptors, general shared/external state, synchronization and arbitrary alias/dependency compositions remain open. |
| Numeric intent | Typed accumulators, low-precision/scaled contractions, composite dot profiles and independent per-cast FP8 contracts. | Explicit software profiles do not establish equivalence to every GPU backend's arithmetic or all normalizer/model combinations. |
| Provenance | Original source, complete operation accounting, separate normalized graphs and emission correspondence. | Custody and intent annotations grant no omitted arithmetic, memory or effect semantics. |
| Package and enclosing model | Standalone and model2MLIR-bound captures, weights/lifted constants, runtime kernel scalars, complete tuple results, dependent state and selected source tensor views. | General physical/mutating views, dynamic public model ABIs, value-returning TTIR entries/CFGs and broader binding compositions remain open. |
| Consumer execution | Real native execution, reusable Merlin host libraries and selected complete model/kernel images executed on RISC-V/Spike. | Functional CPU execution does not qualify accelerators, hardware placement or preserved performance. |

Useful tile structure, reuse and recurrence remain executable. The logical
grid stays independent of warps, threads, cores and physical memory placement.
Source-language assembly implementations are explicit optional adapters with
standard expansions. Unknown semantics produce located refusals; an opaque
fallback is not counted as portable success.

## What the measurements establish

The maintained inventory at
`out/grouped-memory-model-inventory/coverage.json` reopens **106 packages,
74 standalone qualifications and fourteen product cases**. All **50 pinned TTIR
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

Most recently, precision-sensitive casts are protected before official pointer,
descriptor and composite-dot normalization. Original external descriptor
arguments now pass selected rank-one/two mixed-policy kernels through native,
Merlin host and Spike, with changed fields and empty roots replaying unchanged
host libraries. Source auditing and actual expanded-ABI validation remain
separate mandatory gates. Allocation-root/enclosing-model per-cast compositions
still need protection before their own initial normalization.

## What still separates this from the end goal

1. **Close composition gaps.** Extend protected source contracts through
   allocation wrappers and enclosing models, with an explicit custody chain.
   Continue general view/mutation, runtime/alias, region/CFG and numeric/effect
   combinations. Working individual operations do not establish their closure.
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
The interface and abstraction decisions are established; the largest remaining
work is semantic composition and consumer/target closure.

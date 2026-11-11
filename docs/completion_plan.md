# Completion plan: all TTIR kernels through the Merlin interface

Updated 2026-11-11. **The full goal is active and incomplete.** Every valid
kernel in each pinned supported TTIR version must reach a working Merlin
compiler interface with its observable computation/effects represented, useful
algorithm retained, and applicable execution evidence. Host-only support,
accepting opaque calls, or counting parsed operations does not meet that goal.

This document records current requirements and the next execution order.
Detailed historical evidence remains in [coverage.md](coverage.md), the linked
capability contracts and Git history; earlier open items are superseded only
by explicitly qualified implementations.

## Accepted architecture

- Parse pre-layout TTIR and reuse actual supported Triton/triton-shared passes.
  Keep original source inventory separate from normalized graphs.
- Emit tile-first standard `linalg`/`tensor`/`scf`/`arith`/`math`/`cf`, with an
  explicit executable memory/effect profile where functional tensors alone
  cannot represent the source. Preserve useful tiling, K reuse, accumulators,
  masks, structured control and online recurrence.
- Carry source intent, recovered structure and established facts separately.
  Provenance never authorizes omitted arithmetic, effects, aliases or order.
- Keep the logical grid independent of warps, cores and physical placement.
  Abstract implementation-only mappings after checking dependencies. Physical
  observations require concrete semantics and consumer support.
- Supply a model2MLIR-like package with executable MLIR, weights/constants,
  semantic ABI, inputs, independent references, original source/correspondence
  and byte-bound receipts. Keep [package differences](package.md) explicit.
- Retain complete payload accounting and native ownership/lifetime gates.
  A successful reader or runtime generator is insufficient execution evidence.

The source-backed [reuse map](triton_shared.md) records Meta/shared, archived
Microsoft and Triton boundaries. Use the existing pinned tools; no implicit
Triton/LLVM rebuild or Triton-CPU dependency.

## Current delivery state

| Boundary | Established capability | Remaining completion requirement |
|---|---|---|
| Frontend and portable dataflow | Real shared conversion, logical grids, structured loops/branches, tiled contractions, reductions/scans and qualified shape/index operations. | Complete types/attributes/compositions and source versions; no generic arbitrary-kernel claim. |
| Merlin compiler boundary | Standalone and bound-model compiler-only staging, actual whole-body host lowering and reusable native libraries. | Applicable target scheduling/materialization/execution and every retained operation/effect accounted for there. |
| Package and model interface | Typed bindings, learned/lifted constants, all tuple results, outcome/effect references, runtime kernel scalars and explicit dense model value views. | Physical/non-dense/mutating model views, general value-returning TTIR entries/CFGs and broader runtime/alias compositions. |
| Memory and state | Gather/scatter profiles, active masks/bounds, runtime signed strides, selected pointers, descriptors, writable allocation roots, typed/byte aliases and integer atomics. | Remaining conflicting/shared/external effects, physical descriptors, independently strided accumulators and arbitrary dependency compositions. |
| Numeric and effects | Qualified standard/FP8 casts, low-precision/scaled contractions, selected math/RNG, assertions/prints, integer and selected typed floating atomic profiles. | Remaining arithmetic/rounding policies, heterogeneous cast contracts, effectful combiners/maps, arbitrary external implementations, address observations and synchronization. |
| Coverage evidence | Complete pinned 50-definition inventory, upstream-stage measurements, receipt-bound package registry and maintained product/refusal driver with actual Merlin libraries. | Complete type/attribute/kernel-composition matrix and per-form applicable target evidence. |

The current `out/packed-map-inventory-coverage/coverage.json` selection reopens
**69 packages, 46 standalone qualifications and thirteen maintained product
cases**. It observes **50/50 source definitions**, with **49 represented in
native- and Merlin-host-qualified captures**. The current corpus provides eleven
actual libraries and retains two located refusals. Source assembly has no
qualified host capture; coupled/asymmetric packed maps still refuse, while mapped reads qualify through the
explicit guarded source-word profile. These are composition observations, not a completion
percentage or per-operation execution measurement. The [product corpus](product_corpus.md)
and [coverage v2](package_coverage.md) keep source/refusal/execution limits
machine-readable. The older byte-atomic selection retains its 34/50 source,
30 host-definition measurements.

The corpus also exposed guarded softmax's rejected rank-zero tensor allocations.
The [allocation admission fix](guarded_reductions.md) preserves the actual fresh
allocation and native lifetime semantics; four real Merlin libraries and the
39-case allocation/guarded-memory suite pass. Nineteen corpus/registry checks
exercise actual success/refusal/failure, effects, reuse and changed evidence.

Latest implementation: [byte-backed atomic words](byte_atomic_words.md) retain
all nine i32/i64 RMW kinds and i16/i32/i64 CAS, both logical byte orders,
whole-word bounds/alignment, per-byte guards and old values through Merlin.
Selected roots and for/while pointer state qualify in 40 cases. Old-value tensor
recurrences add 20 cases; different-width typed roots add eight. Each recurrence
library replays five domains. Four real model captures retain learned state and
all seven results; typed CAS also passes the public model CLI. Merlin's nested
snapshot/destination/integer-body suite passes 119 checks with lifetime gates
retained. This closes the previously failing nested CAS recurrence slice, not
the remaining numeric/effect/target work.

[Mapped reads](memory_maps.md) now retain source pointer/block edges, scalar
branches and for/while reads through Merlin. Seven libraries preserve the basic
map and five-result snapshots around ordinary kernel writes across i32/i64/f32.
Thirty-four map/memory regression checks and the refreshed maintained corpus
pass. The pinned verifier forbids writes inside maps; source reads and ordinary
writes outside those maps retain their actual effects and guard gates.
Twelve further libraries now carry selected logical roots across map arguments,
pointer results and mixed pointer/numeric results; 34 existing selector
regressions pass. Two real i64/f32 model captures retain learned state and all
fourteen results through Merlin; the i64 model also qualifies via public CLI.
The [complete map contract](memory_maps.md) records the exact composition scope.

[Layout-independent packed DAGs](packed_maps.md) now additionally qualify when
typed SSA establishes identical slot-local functions. Eight actual libraries
retain pack two/four, i32/i64/f32, rank-one/two tiles and runtime recurrences,
with four additional domains per library. Actual dependencies and arithmetic
attributes are checked independently of provenance. Coupled/asymmetric slots,
packed regions/reads, source assembly and applicable targets remain open.

## Required evidence for every supported form

Keep these gates separate for each opcode/type/attribute/composition:

1. Original native-verified source inventory and disposition, including helper
   regions and exact source bytes. Normalized IDs remain a separate graph.
2. Executable tensor or explicitly declared effectful MLIR expansion. Unknown
   behavior must remain a located refusal until implemented, never a successful
   opaque fallback. Refusal is incomplete coverage, not goal completion.
3. ABI/package compatibility: tensors, shapes, runtime scalars, parameters,
   alias/mutation/result order, dependencies and explicit schema differences.
4. Compiler-only Merlin staging with byte-bound source correspondence and no
   evaluator inputs/goldens supplied to compilation.
5. Payload-preserving consumer lowering with every retained operation accounted
   for. Reading a signature or generating a C harness does not prove this gate.
6. Native and applicable target materialization/execution against independent
   numeric/effect references, with original algorithm structure and provenance
   checked separately from numerical agreement.

Byte custody, recipe validation, runtime guards, numeric tests and independent
source equivalence are different evidence. Existing custody records do not
establish universal source equivalence or `source_closure_verified`.

## Remaining work and execution order

Finish coherent source-to-consumer slices and commit each. Keep the full
workstreams below active; do not indefinitely extend one already-qualified
memory fixture while leaving the missing families unexamined.

1. **Expand the maintained coverage/refusal matrix.** All 50 definitions are
   now observed across explicit retained packages and current product cases;
   49 occur in host-qualified captures. Continue with type/attribute and kernel
   compositions, source dispositions and actual failures by stage. Preserve
   implemented-but-unqualified forms and precise refusals in the denominator.
   A scalar map or reversible pointer cast does not cover arbitrary packing,
   memory effects or physical address observations. Keep upstream-only pass
   successes separate from actual package/consumer execution.
2. **Close numeric and effect families.** Use the matrix to select actual
   implementation gaps, starting with packed/effectful maps, remaining numeric
   contracts, external/assembly implementations and physical observations.
   Reuse official expansions where their semantics agree; preserve source
   width, rounding boundaries, accumulator order and recurrence. Implement
   unsupported behavior rather than hiding it behind provenance.
3. **Complete runtime and shared-state bindings.** Extend existing logical
   allocation roots to physical/non-dense/mutating model views, independently
   strided C and remaining alias/layout/shape compositions. Keep original
   pointer units, untouched words, zero/tail domains and actual guard failures.
4. **Qualify applicable Merlin targets alongside frontend work.** Carry the
   same standard payload into scheduling/materialization, preserving tile/reuse
   intent and effect dependencies. Implement missing generic consumer lowering
   in Merlin; target-specific legality, kernels, mapping and ABI belong in the
   target OOT support package. Record actual target compile/execution evidence;
   a host result or backend binary hidden behind a call is insufficient.

## Workstream acceptance requirements

| Workstream | Required behavior and evidence |
|---|---|
| A. Compiler boundary | Preserve Torch capture and derived Triton receipt/source paths; reject corrupted/stale correspondence; exclude evaluator inputs; lower the entire body; compare every result/effect. See [consumer](merlin_consumer.md), [model binding](model_integration.md). |
| B. Irregular memory | Loaded/repeated/negative indices, arbitrary inactive out-of-bounds lanes, exact fills, multidimensional gathers, initialization, partial output, conflicting scatter and logical-grid dependencies. Retain real state or explicit effects. See [indexed memory](indexed_memory.md), [allocation state](allocation_state.md), [byte aliases](byte_aliases.md), [descriptors](external_descriptors.md). |
| C. Runtime ABI | Reuse compiled code across valid shapes/scalars/strides/grids, including empty/tail domains; reject overflow and mismatched bindings; preserve aliases/mutation and all results. Distinguish dynamic kernel ABI from static public model capture. See [runtime ABI](runtime_abi.md), [model scalars](runtime_model_arguments.md), [value views](model_tensor_views.md). |
| D. Control/index structure | Preserve general logical div/rem/grouped traversal, pointer state, branches/while, bijective/conflicting grids, zero/nonzero C and ragged attention's online state without materializing a dense score replacement. |
| E. Numeric closure | Types/signedness, rounding/ties/NaN/zero/overflow, reductions/arg rules, mixed-precision dot/scaled dot, external math and RNG dependencies. Qualify per-step rounding and reuse. See [casts](float_casts.md), [precision](dot_precision.md), [scaled dot](scaled_dot.md), [external math](external_math.md). |
| F. Effects/hardware boundary | Returned-old atomic values, collisions, order/scope/publication, external participants, volatile observations, async wait/barrier lifetimes and exact diagnostic outcomes. Physical address/assembly observations require actual semantics/dependencies. Preserve every effect and qualify its consumer implementation. See [atomics](atomic_memory.md), [effect ABI](effect_abi.md), [mixed effects](mixed_effects.md). |
| G. Maintained coverage | Pin source/tool identities and the complete operation definitions; record all source dispositions, exact typed forms, successful packages and located refusals. Reproduce kernel corpus, integrity and applicable target checks. Never equate observed opcode names with universal support. See [inventory](ttir_operations.md), [measurements](coverage.md). |

## Unresolved architectural boundaries

- **Pure dataflow versus effects:** private invocation atomics can retain
  executable state and old values, but that does not model external participants
  or publication. Synchronization may disappear only after dependency proofs;
  difficult effects cannot simply be erased.
- **Logical versus physical addresses/descriptors:** ordinary logical views
  and external descriptor fields work today. Raw descriptor bytes, pointer
  integers or hardware observations need an explicit compatible address/state
  contract and real consumer implementation.
- **Numeric backend contracts:** selected IEEE/FTZ/low-precision profiles are
  explicit. One global cast profile does not establish heterogeneous per-cast
  contracts or target equivalence. Reserved source enumerants with no defined
  format must remain identified rather than assigned invented semantics.
- **Abstraction and performance:** preserve algorithmic tiling/reuse/recurrence
  even when physical placement is postponed. Do not serialize useful parallel
  computation or replace a kernel with a coarse opaque name just to pass.
- **Tool provenance:** installed Triton, shared source checkout and the actual
  optimizer binary have separately recorded identities. Unverified source/binary
  correspondence must remain explicit; do not claim an unperformed native build.

## Completion and working rules

The goal is complete only when the full pinned valid operation/type/attribute
and kernel-composition scope passes every applicable gate above, with explicit
package differences and target dependencies. Unsupported/refused cases stay in
the denominator. A green selected suite or an acceptance-only package does not
satisfy arbitrary-kernel coverage.

Use bounded serial native work on the shared VM and the existing toolchains.
Follow repository contribution/identity rules, commit tested capabilities
frequently and push the authorized branches: Triton `main`, Merlin and
model2MLIR `ttir-experiment-grammy`. Do not rename branches, publish PRs, rewrite
published history, or alter unrelated user work. Keep current evidence and
remaining requirements here; keep detailed measurements in capability docs.

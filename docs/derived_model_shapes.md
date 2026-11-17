# Derived model dimensions through Merlin

The real PyTorch model in `scripts/capture_derived_shape_model.py` now retains
shared input axes, dynamic broadcasting and a stepped slice feeding two
dependent Triton calls:

```python
original = x + y + self.weight
trimmed = original[1::2]
first = copy(trimmed, trimmed.shape[0])
second = copy(first, first.shape[0])
return second, first, original, trimmed, x, y, self.weight
```

Both input lengths have the actual exported shared domain `[4, 31]`. PyTorch's
exporter requires that domain for this example; an earlier `[2, 31]` request
failed its source constraints. The scalar parameter is argument zero. Neither
the saved length 13 nor the parameter value supplies a compiler shape fact.

## Explicit contract extension

`t2m.symbolic-model-shapes.v2` selects `t2m.runtime-shapes.v3` and a
`logical-size-program.v1` typed DAG. Existing v1 model profiles/runtime v2
contracts retain their input-axis-only behavior. The independent consumer
publishes `merlin.compile-signature.v4`, with
`shape_scope: source-proved-derived-dimensions`, the public dimension bindings
and the checked `shape_program`. Older profiles refuse these additional fields.

The program uses public input-axis leaves and actual scalar constants, casts,
integer arithmetic, comparisons and selections. Fixed-width words retain their
original widths, modular operations and signed/unsigned interpretation. Index
comparisons and selection have no chosen native index width. Node identifiers
bind complete definitions and dependencies; readers reject malformed types,
unknown operations, cycles, extra nodes and altered definitions. The DAG avoids
duplicating shared source computations during serialization and evaluation.
The rules follow standard [Arith semantics](https://mlir.llvm.org/docs/Dialects/ArithOps/).

Model2MLIR [`4b0e7bd`](https://github.com/grammy117/model2MLIR/commit/4b0e7bdfa9af63a2a7fc277297ca35490b28adbb)
retains actual exported size-query and scalar arithmetic SSA,
including the `floordiv` that feeds this kernel. Source guards and index round
trips remain executable. The shape analyses recover bounds across actual
dominating equality assertions only when both word transports are injective.
An equality after a call, inside a sibling branch or between truncated words
cannot transfer another axis's range.

Producer checks and Merlin's separately owned implementation reopen original
calls and actual standard kernel/model bodies. Syntactically different source
expressions, such as the clipped slice count and exported floor division,
require a proof over actual dominating assertions and checked axis enclosures.
Bounded Z3 bit-vector queries supply that proof; only UNSAT admits equality.
Missing solvers or UNKNOWN/timeouts refuse. Install the optional `shape-proofs`
extra in frontend/consumer environments. The isolated native runner evaluates
the already source-checked program without importing Z3 or the compiler frontend.
The independent Merlin implementation is
[`49bd12b1a`](https://github.com/ucb-bar/merlin/commit/49bd12b1a8c206bf0e90c802f993c49243ca2fa2).

The profile caps programs at 4,096 nodes, 64 query levels, 16,384 visits and
4,096-bit source words. Each equivalence query has a two-second, one-million
resource-unit and 128 MiB solver budget. These are explicit resource refusals,
not target index widths or universal shape coverage.

## Storage and preserved computation

Output allocation uses the checked program and actual invocation input axes.
Merlin independently checks the complete executable body's result relationships
before allocating. A rehashed signature or even a well-typed replacement size
program cannot replace that source proof. Logical grids, tiles, source masks,
guards, arithmetic and every original result remain executable standard MLIR.
Source shape upper bounds still determine guarded scratch independently.

The existing positional invocation materializer and runtime input projection
retain their schemas. Public MLIR arguments/results, original parent archives,
weights, complete tuple ordering, receipt schemas and artifact rosters are
unchanged. The extension does not grant physical slice aliasing, public mutation,
numeric policy, effects or target placement. Provenance remains correspondence.

## Qualification and remaining scope

`tests/test_derived_shape_model.py` executes all three tensor/buffer/guarded-buffer
configurations through native, Merlin host and actual RISC-V/Spike. All seven
results are checked in each configuration. Twelve further host invocations
reuse unchanged libraries at lengths 4, 5, 7 and 31 with changed parameter words,
checking all 84 results and preservation of functional inputs. Nine actual
calls with length 3, length 32 or mismatched input axes abort through source
guards without publishing outputs. Rehashed output/program/profile declarations
and canonical but false programs refuse before invocation.

The pure program tests compare small source words against independent arithmetic
and conservative range enclosures, including casts, signed comparisons,
conditional bounds, sharing, invalid programs and unavailable/inconclusive
proofs. Independent Merlin tests check source guard dominance/injectivity,
conditional result selection, storage reconstruction and false declarations.
The final producer/model selection passes 141 checks in
`.scratch/derived-shapes-final1.log`, including legacy symbolic model bindings
and complete packages. Independent Merlin runtime/staging checks pass 174;
six further source dominance/injectivity checks pass separately. Model2MLIR
symbolic/domain checks pass 22. Selections overlap.
`out/derived-model-shapes/summary.json` reopens all three package/target receipts
and the twelve replay archives. The selection is separate from the historical
352-package inventory.

General runtime-supplied slice bounds, data-dependent dimensions, resizing loop
state, physical/mutating views, arbitrary model/effect compositions and applicable
accelerator implementations remain open. Retained algorithm structure does not
establish preserved target performance. Full arbitrary-TTIR coverage is incomplete.

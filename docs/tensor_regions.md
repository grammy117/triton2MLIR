# Local tensor computation in scalar regions

Map, reduction and scan bodies may retain positive static unencoded numeric
tensor intermediates, nested region computation and tensor state in structured
control. A real source example computes bitset intersection in small chunks:

```python
bit = tl.arange(0, 8)
state = tl.full((8,), 0, left.dtype)
for start in range(0, left.dtype.primitive_bitwidth, 8):
    shift = start + bit
    flags = ((left >> shift) & 1) & ((right >> shift) & 1)
    state = state | (flags << shift)
return tl.sum(state, 0)
```

This can serve as a reduction/scan combiner. The example's map also builds a
bit vector, computes a nested associative scan and reduces its last prefix back
to the original word. These are original Python/Triton captures, not fixtures
selected by kernel name or substituted mathematical answers.

## Conversion and abstraction

`value_regions.py` prepares one detached module and expands mixed nested
containers in postorder. Every candidate and complete descendant body must
pass before any prepared operation replaces the original module. Failures leave
the entire input unchanged, including already prepared valid siblings.

The existing first-element reduction and parallel scan expansions retain tuple
order, initialization, source locations and scalar bodies. Checked numeric
tensor values may remain local to branches/for/while state. Source shape/range/
splat operations stay in place for Meta/shared's converters. The archived
Microsoft converters use the same arithmetic foundation. The relevant pins,
native tools and historical source boundary remain in [shared reuse](triton_shared.md).

The pinned xDSL integer-width-cast bindings accept scalars only. For shaped
`arith.extsi`, `arith.extui` and `arith.trunci` inside these regions,
`region_integer_casts.py` applies LLVM's existing elementwise-to-Linalg pattern
before recursive region verification. It checks shape/width relations and keeps
the exact scalar operation, attributes and location. This does not weaken SCF
verification or reinterpret source rounding/poison policies.

Generated lookup-guard admission follows actual compiler clone edges. A source
annotation, copied location or serialized operation cannot recreate that
authority. Original source effect auditing still precedes normalization.
Unknown operations, pointer/memory state and unqualified effects remain located
refusals. Native verification and shared conversion remain mandatory.

Source/emission correspondence and `source-region-tensor-dataflow` records retain
the original container, local tensor operation/source counts and undecided
physical mapping. Tensor operations, loop state and dependencies are executable;
provenance supplies no semantics, ownership or storage placement.

## Native consumer boundary

The pinned native Linalg bufferization model does not describe scalar bodies as
repetitive regions. Repeated nested tensor computation exposed numerical errors:
captured zero seeds and shared functional destinations could be overwritten
between invocations. A passing first tile did not establish correct reuse.

Merlin's destination feature now protects captured tensor loop seeds and the
qualified functional write destinations at the real pre-bufferization seam.
An actual `bufferization.alloc_tensor` supplies independent invocation storage.
Used initial values receive a copy; unused initial values require allocation
without copying their data. Dynamic dimensions remain explicit. Local producers
and ordinary readonly captures keep their original values and paths.

The native evaluator uses the same checked preparation after its existing
canonicalization/CSE, recording the actual support hash. Portable package MLIR
keeps its standard tensor/Linalg/SCF computation; these allocations are native
materialization. Full Merlin bufferization, ownership, deallocation and lifetime
checks remain enabled. The source grid, tile sizes and useful parallel scan
stages remain undecided with respect to GPU lanes/warps or physical storage.

This requires the updated experimental Merlin consumer. Earlier consumer
revisions are not qualified for these nested tensor-state payloads. Public
tensor/scalar ABI, artifact rosters and receipt schemas are unchanged; generated
compatibility metadata records this requirement and the representation.

## Qualification and remaining scope

The current selection reopens **18 real source captures/libraries and 18 actual
Spike images**, comparing **18 complete exact-word results and 72 further host
runtime invocations**. The receipt-bound selection is saved locally in
`out/tensor-regions/summary.json`. These are separate from the November 12
237-package combined inventory; source-name breadth is not a completion score.

`tests/test_tensor_regions.py` checks real i32/i64 source captures for reduction
with tensor `for` state, reverse scan with tensor `while` state, and maps with
nested scan/reduction computation. Each runs in tensor, buffer and guarded-buffer
profiles, comparing complete exact-word results through native execution,
reusable Merlin host libraries and actual Spike images. Runtime replays include
empty/singleton/exact/ragged domains and preserve the same compiled libraries.
The independent reference uses ordinary integer bitwise intersection/identity;
it is never imported by lowering.

Contract cases retain transactional refusal, source properties, exact cast
operations and nested unused effects. Merlin checks captured/local initial
state, dynamic/static destinations and used/unused initialization separately.
Its 147-test destination selection and structure/documentation gates pass.
The focused frontend source-effect/map/scan/guard regression selection passes
85 checks. Changed-file Ruff and whitespace checks pass. Repository-wide Ruff
still reports four pre-existing unused imports in unrelated files.
The current 31-case region/guard selection also passes, including actual
unchanged-library bounds aborts and lookup-guard clone authority. The former
blanket refusal of an unused pure `tensor.empty` is now a positive numeric-value
case; every original call/read/write/assertion refusal stays enforced.

General tensor operations and dynamic intermediate shapes, direct memory or
external effects in combiners, arbitrary packing/type/source-version forms,
broader model bindings and applicable accelerator correctness/performance remain
full-goal requirements. This finite selection supplies no GPU-binary equivalence
or benchmark result. The [completion plan](completion_plan.md) remains active.

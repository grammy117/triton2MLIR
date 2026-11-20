# Branch-local source index ranges

Actual scalar branch conditions can now prove a source divisor's safe range
without excluding zero at function entry. The frontend retains the original
condition, lookup, state update and complete prior-state observation:

```python
before = tl.load(state)
tl.store(old, before)
if (divisor > 0) & (allow != 0):
    value = tl.load(x + 9 // divisor)
    tl.store(state, before + value)
else:
    tl.store(state, before ^ 3)
```

The source scalar domain includes divisor zero. It selects the actual else
branch and never executes the divide or lookup. This is not a sample-specific
specialization or a request to assume positivity at entry. Both returned state
and its earlier snapshot remain exact i64 values.

## Source proof and reuse

`source_predicates` reuses `source_index_ranges`' fixed-width comparison sets
and literal reader. It intersects the actual source signed interval with each
selected signed/unsigned comparison, retaining the sign-bit partitions before
taking a conservative signed hull. A nonzero set spanning both signs still
contains zero in that hull and cannot by itself authorize division. Empty
intersections grant no optional fact or unreachable-region rewrite.

The actual true region of `scf.if` supplies its condition; the false region
supplies its complement. A true conjunction and false disjunction each imply
both constituent predicates. A false conjunction or true disjunction does not.
XOR with an actual i1 literal handles boolean negation. Unknown conditions give
no extra facts; the condition DAG traversal is bounded at 4,096 visits.

`SourceScopes` records the resulting facts only in the selected original region
and its descendants. Original-definition checks still precede narrower use
facts. Earlier arithmetic, later accesses and sibling branches cannot borrow
the selected branch's facts. Checked value-preserving casts may propagate a
constraint to their source; an overflowing earlier product or unproved cast
does not gain permission from its condition. Provenance attributes grant no fact.

Primary source review at LLVM `a47bddccec30255619bb8c37fa59700e661d4e66` covers
[`IfOp` region successors](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Dialect/SCF/IR/SCF.cpp),
[`SCF` tests](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Dialect/SCF/ops.mlir),
[`SCF` canonicalization tests](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Dialect/SCF/canonicalize.mlir)
and [`Arith` range inference](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Dialect/Arith/IR/InferIntRangeInterfaceImpls.cpp).
The branch constraints are a local source proof, not a claim that an installed
native solver exports them to Python. Existing LLVM attribution/license and
the [loop-body proof](source_scope_ranges.md) retain their original scope.

## Native pointer boundary

The installed pointer pass can leave `tt.load`/`tt.store` unconverted when an
addptr is local to `scf.if`. The historical
[Microsoft converter](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp)
refuses every such addptr. The reviewed
[Meta converter](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp)
distinguishes a branch-local pointer from an escaping pointer result. Its
[`gather_no_loop`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonToUnstructured/gather_no_loop.mlir)
fixture supplies the existing base/offset conversion boundary.

The frontend first keeps the ordinary structured conversion. If actual source
loads/stores remain, it reuses `pointer_control` and `word_events` on the original
source-width-checked computation. Each native event template is substituted at
the original event; parent arithmetic converts separately. No event is hoisted,
no dependency is removed, and converted pointers are not run through the native
pointer pass twice. Successful structured tile conversions keep their original
path and its ownership/reuse proofs.

The retained initial stage and `native-structured-memory-residual` record explain
selection from actual IR. `native-isolated-checked-memory-events` distinguishes
this checked-address tensor continuation from the existing guarded source-word
profile. It grants no effect or ownership semantics. An earlier broad-routing
experiment lost structured proofs; its 18-failure regression log remains under
`.scratch/branch-memory1.log`. The corrected selector's regression passes.

## Package and qualification

The output remains standard tensor/SCF/arith computation with active bounds and
the existing source initialization/ownership checks. `source-branch-ranges`
records the source control IDs, narrowed-value count and original-definition
rule. Generated compatibility describes both analysis and native adaptation.
Public tensor/scalar ABI, receipts, artifact rosters and Merlin signatures are
unchanged. The unguarded buffer continuation retains its independent static
witness requirements; tensor-only runtime checks cannot authorize it.

Six real source forms cover positive, complementary, conjunctive, disjunctive,
unsigned and nonzero conditions. Each has four native invocations comparing
both complete results as exact words. Two additional configurations admit the
entire signed i32 divisor domain and unrestricted i32 allow words, including
signed minimum, negatives, zero and signed maximum. This confirms that the
proof does not depend on the small example's optional entry bounds.
Four unchanged Merlin libraries run six independent divisor/allow choices
each. Six actual serial images exercise both branch outcomes and full-width
controls, retaining both results and original inputs.
The public host rejects two out-of-domain scalar invocations before execution;
two separate native runs abort at the serialized executable scalar guards.
Those are distinct failure boundaries, not interchangeable execution claims.

The focused selection passes 66 checks, with one additional advisory-tag refusal
and two additional full-signed-domain native/host/serial qualifications.
The memory/control regression passes 109 and the existing pointer/event selection
23, with overlap. Exhaustive two-/three-/four-bit comparisons check every signed
interval, bound word and supported predicate against an independent value oracle.
Actual source negatives cover unsafe earlier/later/sibling accesses, unrelated
conditions, insufficient boolean implications and overflow before the branch.
Evidence reopens under `out/branch-source-ranges/summary.json`; final test logs
are `.scratch/branch-ranges4.log`, `.scratch/branch-memory2.log` and
`.scratch/branch-event-regression2.log`. Earlier incorrect test expectations
remain in their original logs and are not relabeled as compiler failures.
Reopening passes in `.scratch/branch-range-report1.log`, pinning implementation
`60f3bad`, all four source packages/libraries, 24 host replays, six target
receipts, 24 direct native invocations and the distinct refusal/abort outcomes.
The full-domain selection passes separately in `.scratch/branch-full-signed1.log`.

General comparisons between two varying values, disjunctive nonzero domains,
while-carried constraints, partial/nested initialization, irregular parallel
ownership, broader numeric/storage/effect compositions and applicable
accelerators remain incomplete. This selected kernel qualification does not
establish a new complete PyTorch model, arbitrary-kernel coverage or performance.

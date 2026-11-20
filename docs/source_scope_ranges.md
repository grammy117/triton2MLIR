# Source integer ranges inside executed loop bodies

The structured-address frontend now proves a nonzero divisor inside selected
actual source loops without rejecting valid zero-sized inputs at entry. This
closes the rank-three shared matrix-view example's tensor-admission refusal.
The original quotient/remainder indexing, loop order and reused bias remain;
there is no replacement kernel, whole-operation shortcut or GPU mapping.

## The source obligation

```python
for i in range(rows * cols):
    row, col = i // cols, i % cols
    value = tl.load(x + row * cols + col)
    tl.store(out + i, value ^ (i.to(tl.int64) * 3 + 1))
```

When `rows` and `cols` have checked nonnegative source-width ranges, execution
of this signed loop body proves that its upper bound is positive. A positive,
nonnegative product has positive factors. Thus `cols` is nonzero at these
division/remainder operations. If either dimension is zero, the body never
executes; zero remains permitted at entry and the result is empty. Changing
the ABI to exclude zero would change the source program's valid domain.

The actual bound product and loop induction update must first fit their source
integer widths. An overflowing product cannot gain permission from its later
use as a loop bound. A division computed before or after the loop also cannot
borrow a body-only fact. A sum bound does not prove its individual operands
positive. Nested body facts must not escape to the enclosing body.

## Analysis and upstream boundary

`shared_indexing.check_source_addresses` still audits source arithmetic before
triton-shared promotes recovered pointer indices. Computed facts are now cached
by the pair of SSA value and actual scope block. Before evaluating a value in
a narrower use context, the analysis validates its original definition context.
This preserves earlier poison/overflow refusals even when later reachability
would make the same expression safe.

`source_scope_ranges.SourceScopes` inherits facts through actual region parents.
For a signed `scf.for`, it reuses the existing constant-lower/positive-step
bounded-loop proof. A nonnegative lower bound permits the body-only positive
upper-bound fact. Nonnegative multiplication factors and already checked
value-preserving integer casts propagate positivity. An unproved control grants
no optional fact; actual induction/address uses still need their ordinary proof.
This analysis does not move, rewrite or eliminate source operations.

The program-point and loop-interface boundary follows LLVM/MLIR
[`IntegerRangeAnalysis.cpp`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp).
The local lexical product proof is not a claim that the native solver already
supplies these path constraints. Relevant upstream tests include
[`int-range-interface.mlir`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Dialect/Arith/int-range-interface.mlir)
and [`int-range-loop-iter-args.mlir`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Dialect/Arith/int-range-loop-iter-args.mlir).
These source files and tests were inspected at the recorded revision; no
native stack or solver was rebuilt.

Signed remainder inference adapts the sign/modulus and constant-divisor interval
strategy in LLVM/MLIR
[`inferRemS`](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Interfaces/Utils/InferIntRangeCommon.cpp),
under Apache-2.0 WITH LLVM-exception. The [upstream notice and full
license](licenses/llvm-NOTICE.md) are retained and included in distributions.
Source signed-width and zero-divisor checks remain mandatory. The current
native-safe profile also refuses signed
minimum divided/remaindered by minus one: native LLVM `srem` can be poison.
This is an explicit admission limit, not a claim that MLIR `arith.remsi`'s
documentation defines every such input as overflow. Tests exhaust all four-bit
intervals and values, with additional width/zero/overflow checks through i64.

## Granularity, provenance and package compatibility

The scalar-addressed matrix views use the existing serial tensor-state path:
complete flat storage travels through the source loops, with public logical
shapes restored at the boundary. No memref payload is required for the newly
admitted rank-three form. Actual bounds, initialization, source-order snapshots
and native ownership/lifetime checks remain.

`source-loop-body-ranges` transformation records retain source loop IDs,
narrowed-value counts, the reachability premise and definition-context rule.
Generated compatibility metadata explains the analysis. These annotations grant
no rewrite, initialization, ownership, effect, numeric policy or target support.
Public tensor/scalar types, receipts, artifact rosters and Merlin signatures are
unchanged. The guarded source-word path remains independently selectable; the
separate unguarded buffer path cannot inherit tensor-only runtime checks.

## Qualification and remaining scope

The source-level selection checks actual stock Triton capture, unsafe operations
before/after a loop, unrelated sum bounds, nested scope leakage and overflowing
earlier bounds. A nontrivial i64 XOR matrix kernel executes native code and
reuses one Merlin host library over six shapes, including `(0, 0)`, `(0, 7)`
and `(3, 0)`. Exact words above floating-point integer precision distinguish
the computation from a copy or omitted body.

The focused source/domain selection passes 39 checks, the memory regression
91 and the complete shared-view model selection 30, with overlap. All six
models select pure tensor computation. Together they retain 52 initial, 156
changed-invocation and 52 actual serial target results. Thirty coherently
rehashed claims refuse in both readers; six original source domains abort
without outputs. Earlier failure logs, including the corrected replay-test
API error, retain their original outcomes. Native and host matrix selections
each retain six exact results with valid empty domains.

Final logs are `.scratch/source-scopes9.log`,
`.scratch/scoped-memory-pinned1.log` and `.scratch/scoped-models-pinned1.log`.
The source-pinned reopening report is `out/source-scope-ranges/summary.json`.
It independently reopens complete parent archives, every model result, all
target receipts and the six native/host matrix invocations. The rank-three
comparison retains identical typed source-operation records after excluding
locations: emitted memref operations change from 51 to zero, with four tensor
insertions and three local range reports. This measures abstraction rather
than execution speed or physical allocation reuse. Reopening passes in
`.scratch/source-scope-report2.log`; the first invocation's missing Python
module path remains in its original failed log. Distribution checking in
`.scratch/source-range-wheel3.log` verifies the LLVM notice/license and existing
xDSL license bytes in the built wheel using the installed build backend.
The historical opcode inventory remains unchanged; these checks establish
semantic compositions and abstraction, not new operation-name breadth.

This closes one concrete reachability family. General conditional/while
predicates, richer arithmetic relations, partial/nested fresh initialization,
parallel irregular-write ownership, general storage/alias/effect compositions
and applicable accelerator correctness/performance remain incomplete. Nested
source-range tests do not by themselves qualify nested output ownership.

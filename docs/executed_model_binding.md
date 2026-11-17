# Internal sizes through executable model call contracts

[Executed call views](executed_call_views.md) additionally connect checked
dense/source adapters under an explicit extension profile. Complete models can
retain original non-dense view coordinates and snapshots alongside internally
computed result extents; fully runtime and physical/mutating views remain work.

The [real compaction example](compaction.md) now extends this path to three
actual count/parallel-scan/scatter/copy kernels. Its explicit mixed memory
selection retains the existing call, result and package profiles.

`bind_model(source, bindings, result_mode="execution-owned")` now binds a real
Triton-produced count controlling two later result allocations. This is an
explicit transformation profile, `t2m.model-kernel-binding.v4` with
`t2m.execution-checked-calls.v1`. Existing v1/v2/v3 binding and their precomputed
result-size proofs retain their behavior; there is no implicit fallback.

The real PyTorch capture in `scripts/capture_kernel_size_model.py` computes:

```python
original = x + self.weight
count = count_positive(original, x.shape[0])  # real tl.sum over loaded values
n = count.item()
torch._check_is_size(n)
torch._check(n <= 31)
first = prefix(original, n, x.shape[0])
second = prefix(first, n, n)
return second, first, original, x, count, self.weight, first[::2]
```

The source operation is count-then-prefix, not filtering: the prefix retains
the first `n` original elements. The count and every original model operation
execute once per invocation. Neither an independent size predictor nor a
captured golden determines the count or an allocation.

## Executable contracts and preserved structure

The adapter reuses the actual shared tensor/buffer conversion. Original typed
call/declaration identity, complete schema operands, captured scalar attributes,
runtime mappings and result order must select exactly one checked implementation.
Every implementation retains its original source/ABI/memory evidence.

At each call the emitted standard computation:

1. Reads exact scalar SSA or static singleton tensors. Captured attributes use
   their original typed constants, including declared f64-to-f32 rounding.
2. Checks declared runtime scalar bounds. Integer/index adapters check actual
   signed round trips; dimension scalars additionally check nonnegativity.
3. Checks each symbolic input dimension against the actual scalar and checks
   dynamic dimensions before a cast to a statically narrower tensor type.
4. Inlines the complete translated body with source/callsite provenance.
5. Checks actual returned symbolic dimensions and caller narrowing before
   replacing original result SSA uses.

Sizes may depend on arbitrary earlier executable computation without being
expressible as public input-axis/word programs. Tensor rank/storage/static
compatibility remains mandatory. No unknown calls or opaque implementations
survive binding. Kernel loops, tile reuse, masks, arithmetic and effects remain
in their original dataflow; hardware mapping remains downstream. Source locations
and original provenance survive selected serialization.

Upstream semantics were checked at LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`: Tensor `CastOp::areCastCompatible`,
the mismatched-constant case in `mlir/test/Dialect/Tensor/invalid.mlir`, and
Arith signed/unsigned index-cast implementations and canonicalization tests.
Standard [tensor casts](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcast-tensorcastop)
change shape information without checking runtime dimensions; explicit guards
are necessary. Round-trip checks retain the actual target's index conversion
without choosing a frontend index width.

Resource scope is explicitly `execution-allocator-accounted`. This selected
consumer mechanism checks actual allocation products/capacities and ownership;
it does not derive a global scratch estimate from saved output sizes. Static
scratch accounting in older profiles is unchanged.

## Qualified package and Merlin interface

```python
write_model_bundle(
    capture, bindings, output, work=work,
    python=model2mlir_python,
    merlin_root=merlin_root, merlin_python=merlin_python,
    merlin_host=True, result_mode="execution-owned",
)
```

The CLI equivalent is `bind-model --result-mode execution-owned --merlin-host`
with the existing capture/bindings, toolchain and output options. The selected
mode requires an existing Merlin checkout and host compiler. Other modes and
their ordinary torch-mlir native qualification retain their behavior.

| Boundary | Explicit selected profile |
|---|---|
| Original parent | Unchanged `m2m.capture-receipt.v1`, archives, public ABI and original FX graph |
| Call derivation | `t2m.model-kernel-binding.v4` / `t2m.execution-checked-calls.v1` |
| Qualified package | `t2m.bound-model-receipt.v2` |
| Compatibility | `t2m.bound-model-compatibility.v2` |
| Compiler correspondence | `merlin.derived-source-correspondence.v2` |
| Compiler signature and result ABI | `merlin.compile-signature.v6` / `merlin.execution-owned-results.v1` |
| Qualification record | `t2m.execution-owned-model-qualification.v1` |
| Actual host qualification | `t2m.merlin-host-qualification.v2` |

The source-only candidate remains under the selected work directory, with
`qualification: source-only-unqualified`. Compiler staging checks original
source/memory evidence and independently reconstructs the entire SSA expansion.
No evaluator arrays, result references or qualification report enter its staged
view. Public axes come from actual input types; original source-domain guards
remain executable. Actual descriptors supply result sizes after execution.
Only then are all original references compared and the final package published
with `qualification: qualified`. A contradictory reference prevents publication.

The original source-only input receipt remains under
`consumer/input-receipt.json`. Producer verification replays original binding
against retained located kernels and rechecks complete result/reference words,
source/library/observation hashes, shapes and input preservation. Published
libraries reopen without compilation. Common parent/kernel retention and byte
custody are shared with the older package path.

Native qualification uses the actual Merlin host compiler with an independent
original PyTorch reference. It does **not** claim a second independent native
compiler implementation. The receipt, compatibility metadata and coverage
record that backend explicitly. This closes package/host integration for this
selection; it does not qualify the separate native evaluator or target harness.

## Evidence and remaining integration

`tests/test_kernel_size_binding.py` captures the actual PyTorch model and both
TTIR kernels. The three tensor/buffer/guarded-buffer variants compile through
Merlin's existing `compile_owned_results` API. All 21 original result words
match the complete parent references. Fifteen further invocations reuse the
three unchanged libraries, checking all 105 results with changed input lengths
and parameter words, including zero and full computed counts.

`tests/test_executed_call_guards.py` separately exercises actual compiled
dimension disagreement, statically narrowing results, scalar bounds and lossy
i8 index transport. Negative tests require a started invocation, SIGABRT,
the actual guard diagnostic and no published output archive. These standard
MLIR adapter harnesses are not TTIR source-equivalence evidence.

Merlin [2963c543e](https://github.com/ucb-bar/merlin/commit/2963c543e757a2afad3e3788b34970e7b974a049)
now independently reconstructs the complete source expansion through
`frontends.execution_model_correspondence.checked_execution_model`. Its own
reader imports no producer analysis or evaluator. It reopens original typed
calls and retained kernel bodies, constructs every expected scalar/dimension/
cast guard, and compares the entire SSA graph and source locations. Changing
ordinary model arithmetic, removing a guard, changing a mapping/source record,
reordering calls or introducing a size program refuses even after rehashing.

The original retained `implementation.mlir_text` omits source locations.
V4 therefore additionally requires `located_implementations`, mapping each
ordinary implementation digest to the SHA256/byte count of `serialize(module)`.
The consumer verifies these actual located bytes and requires that stripping
locations reproduces the original implementation digest. Receipt v2 retains
this additional compiler artifact; the old unlocated artifact and
its source/memory evidence are unchanged. No location annotation supplies a
numeric, shape or effect fact.

The actual count model passes 27 independent reconstruction/refusal checks in
`.scratch/kernel-size-correspondence-final5.log` across the three memory profiles.
Twelve separately handwritten Merlin fixtures pass without the producer installed,
including a type-changing cast that structural comparison alone would miss.
These are correspondence checks, not additional execution or TTIR equivalence.
The complete 46-case binding selection passes in `.scratch/kernel-size-final1.log`.
Three final direct-execution checks in `.scratch/kernel-size-inputs-final6.log`
additionally verify unchanged caller input/parameter bytes for every initial
and repeated invocation. These selections overlap.

The complete package selection now passes 49 new/legacy checks in
`.scratch/executed-package-final5.log`, with overlap. The new selection includes
three qualified packages, all 21 initial results, fifteen published-library
replays checking 105 results, six actual source-domain aborts, three source-only
stages after removing evaluator members and a real failed-reference execution
that leaves no output package. Rehashed model/guard/order/located/profile and
result-observation changes refuse. Earlier binding/guard tests and the
independent-reader selections retain their original scopes.

Merlin's staged implementation is
[`e943745a3`](https://github.com/ucb-bar/merlin/commit/e943745a3dacac22f7d9071b1772527551c9635a).
Its 56 new/legacy staging/correspondence checks pass in
`out/build/execution-model-staging-final.log`. The producer's
`out/executed-model-packages/summary.json` reopens all selected package/result,
replay, abort, source-only stage and failed-publication evidence and checks
current consumer source fingerprints. These are separate from the historical
352-package inventory.

Further work includes the separate native evaluator, generated C and target
harnesses, physical/dense/source-layout view adapters, runtime resizing
ownership and broader source/effect/numeric compositions. The older staged v6
profiles keep their original admission checks. Full arbitrary-TTIR coverage,
accelerator correspondence and performance preservation remain unqualified.

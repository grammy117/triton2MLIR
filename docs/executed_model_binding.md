# Internal sizes through executable model call contracts

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

This milestone is **binding plus direct whole-body Merlin host execution**.
Normal `write_model_bundle`/receipt verification and independent Merlin package
staging do not yet admit v4; they must reconstruct original call/body/adapter
correspondence before selecting signature v6. The existing staged v6 selection
still covers already admitted older binding profiles. Further work includes
native qualification and target harness integration, physical/dense view
adapters, runtime resizing ownership and broader source/effect compositions.
No new target, performance, universal-op or historical-inventory claim follows.

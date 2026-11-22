# Defined helper calls and source intent

Real Triton helpers may use `noinline=true` to control physical call boundaries,
code size and register pressure. The pinned upstream inliner honors that Boolean
hint. Leaving it in place can strand a defined `tt.func` behind a `func.call`
during tensor conversion, even when the complete body is portable computation.

The frontend now validates the native Boolean attribute and actual defined body,
then moves a true hint into descriptive provenance before the existing upstream
inliner runs. It does not implement a second inliner. False/absent hints leave
the program unchanged; malformed hints and missing implementations refuse before
any planned hint changes. The original source and operation inventory retain
the Boolean attribute. The conversion trace and emitted module's
`prov.source_call_intent` retain function/body source IDs, the original grouping
intent and `grants_semantics=false`.

Values, types, complete returns, source control and effects remain executable.
This transformation does not remove SCF `no_inline` boundaries used by the
effect expansion. It cannot supply an external implementation or authorize
arithmetic, aliasing, effect or scheduling changes. Physical call boundaries,
code size and register pressure may change; a consumer may outline only after
its own SSA, liveness and effect checks. Public ABI and receipt schemas are
unchanged, and generated compatibility metadata discloses this tradeoff.

The implementation reuses the Boolean decision in Triton's pinned
[inliner interface](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/IR/Dialect.cpp)
and its [existing inlining test](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/test/TritonGPU/inline.mlir).
The latter demonstrates upstream body exposure; its post-layout GPU operations
are not admitted into the portable frontend. The shared conversion's
[call test](https://github.com/facebookincubator/triton-shared/blob/17f0c234/test/Conversion/TritonPtrToMemref/call.mlir)
remains a reference for its distinct function/type boundary.

Five native-verified public Merlin programs qualify the original counterexample
through tensor, buffer and guarded-buffer memory routes, repeated stateful calls
with every returned value, and FP8 storage-backed arguments/results with owned
replay. Eighteen library invocations include changed values, negative/zero
updates, tails and empty inputs. Seven focused contract checks cover hint types,
transactional refusal, missing bodies and exact false/absent preservation.
This is focused qualification, not the final 128-case release verdict or a claim
that recursive/external calls acquire an implementation.

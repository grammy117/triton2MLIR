# Source assertions through Merlin

`tt.assert` now reaches the actual Merlin interface as executable standard
control flow. Scalar i1 checks become `cf.assert`; tensor i1 checks use a
lane-wise `linalg.generic` containing `cf.assert`. The converter is the actual
upstream implementation, not a local replacement or a provenance-only marker.

## Reused implementation and semantics

The source boundary is Triton v3.7.1
[`TT_AssertOp`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/Triton/IR/TritonOps.td):
false conditions print a diagnostic and abort the program. Meta
[`AssertConverter`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp)
implements scalar/tensor conversion; its
`test/Conversion/TritonArithToLinalg/triton_assert.mlir` fixtures cover tensor
ranks one through three at the pinned stage. The archived Microsoft source
also registers this converter in `TritonToLinalg.cpp`; its old monolithic pipeline is not used.
The selected standalone optimizer enables `assert-to-cf` by default.

The adapter accepts the source i1 condition and literal message, retaining
their types, source IDs and original control positions. It records
`source-fail-stop-assertions` and `prov.source_assertions`. Every retained
assertion also carries the ordinary source/emission provenance. Conditions,
messages and failures remain executable; metadata grants no effect-removal
permission. Ordinary registered canonicalization may fold proved true checks.

`capture_ttir(..., debug=True)` explicitly enables Triton's source diagnostics.
The default `debug=None` preserves existing backend defaults. Debug capture
also inserts Triton's own integer overflow checks; those remain part of the
captured graph and receive the same treatment as user assertions. No kernel
name or annotation selects production lowering behavior.

## Software boundary and package differences

Portable `model.mlir` contains standard tensor/linalg/SCF/CF operations and an
unmapped logical grid. No device instruction, GPU printf ABI or opaque helper
replaces an assertion. A downstream target must implement the observable
diagnostic/failure effect. This profile qualifies native Merlin execution,
not accelerator diagnostic handling.

Public tensor/scalar arguments, result order, core filenames and receipt
schemas are unchanged. Compatibility metadata declares the diagnostic effect
and native runtime dependency. Source messages use the upstream
``Assertion `message` failed`` form. Native evaluators use the host libc
`stdout`/`setbuf` ABI so the LLVM lowering's `puts` output survives the following
`abort`. Both successful checks and actual failures are tested; failure text
reaches the caller, and failed invocations receive no success receipt.

The current text transport requires NUL-free messages. Embedded NULs are a
located refusal until a length-aware diagnostic ABI can preserve them.
[Observable-outcome packages](effect_abi.md) now separately support void kernels
and initial expected failures, retaining actual result types and independent
termination/stream references without dummy results. Other print, external
function, assembly and synchronization effects retain their existing gates.

## Executed cases and visible demos

The focused source suite has 24 cases: scalar and rank-one/rank-two conditions,
all three memory profiles, empty/tail/full domains, original loop/branch
placement, zero trips, independently skipped branches inside running loops,
inactive loads, preserved incoming destination state,
located schema refusals and actual failing native/Merlin invocations. One
qualified Merlin library handles new inputs and both scalar/tensor failures.
The 41 diagnostic transport, memory-guard, consumer and invocation regressions
also pass, including literal percent characters, quotes, newlines and UTF-8.

A real model2MLIR capture returns the checked result, original input and
captured state. All three references and the original parameter bytes survive
native and Merlin execution in all three profiles. Scalar source specializations
are checked against the captured custom-call arguments. The model uses a
functional update of a cloned initial value; it does not mutate a Torch argument.

```bash
python -m t2m shared-demo --kernel checked-copy --out out/source-assertions \
  --optimizer .scratch/toolchain/triton-shared-opt \
  --memory-profile guarded-buffer --address-policy source-words \
  --python /path/to/native-python
python -m t2m qualify-merlin out/source-assertions \
  --out out/source-assertions-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
/path/to/model2MLIR-python scripts/capture_assertion_model.py \
  --out out/source-assertions-model/capture
python -m t2m bind-model out/source-assertions-model/capture \
  --bindings examples/model_assertion_binding.json \
  --out out/source-assertions-model/model \
  --optimizer .scratch/toolchain/triton-shared-opt \
  --memory-profile guarded-buffer --address-policy source-words \
  --python /path/to/native-python --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --merlin-host \
  --merlin-clang /path/to/compatible-clang
```

Use fresh output directories. The saved standalone Merlin qualification and
model package pass 45/80 artifact checks respectively. The subsequent explicit
15-package report at `out/source-assertion-coverage/coverage.json` observes 24
of 50 definitions, with 20 present in captures of Merlin-qualified packages.
Both new source-assertion packages have actual Merlin evidence. These are
selection-specific composition observations, not universal opcode coverage.

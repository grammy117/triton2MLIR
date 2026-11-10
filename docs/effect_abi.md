# Effect-only kernel ABI

`t2m.shared-tensor-abi.v6` explicitly selects `effect_only: true`. It permits
readonly tensor inputs, runtime scalars, static specializations and a logical
grid, including an empty argument list. The function has no tensor results.
Writable buffers cannot disappear into this profile: they require the usual
functional result ABI. Earlier schemas keep their existing serialization and
continue requiring writable results.

The converter retains the standard source computation and observable effects.
Scalar and dynamically shaped tensor assertions already execute through the
native JIT with a real void entry in all memory profiles. No result placeholder
or artificial allocation is introduced. Named dimensions and range declarations
retain their existing executable guards.

## Observable-outcome package

`t2m.effect-kernel-receipt.v1` now packages void kernels and initially failing
tensor-result kernels. It keeps source, portable `model.mlir`, spec, provenance,
inputs, argument manifest and valid empty standalone weights. `reference.json`
replaces the numeric golden:

```json
{
  "schema": "t2m.effect-reference.v1",
  "termination": "assertion_failure",
  "stdout": "Assertion `constant source assertion` failed\n",
  "stderr": ""
}
```

The other termination is `return`. Streams are compared as exact UTF-8 bytes.
Expected assertion diagnostics must correspond to actual source assertions,
including plain source `cf.assert`. Expected termination is an independent
caller reference. Successful tensor results still require numeric references.
When actual source prints precede a failure, the expected stdout must end in
an actual source assertion diagnostic. The complete independent stream is
still compared byte-for-byte; source print metadata never supplies its values.
Initially failing tensor kernels retain their actual declared result ABI;
they publish no fabricated values.

Returning tensor kernels now also use an explicit
[mixed reference v2](mixed_effects.md), adding `reference.npz` and complete
per-result word/value policies while comparing all tensors, exact streams and
termination from one call. Their ordinary tensor-result ABI remains intact;
effect-only ABI v6 is reserved for real void entries. Original reference v1
packets retain their existing meaning.

The package omits `golden.npy`, retains an empty valid `extra.npz`, and declares
these differences in `compatibility.json`. A void function has no tensor
result. An empty result archive records no values; an aborting function writes
no result archive or completed evaluator record. This is an explicit package
difference from model2MLIR and the tensor receipt, not an FX capture.

## Observed execution and compiler custody

The producer actually executes serialized standard MLIR through the existing
native JIT. `native/` retains program/arguments, legalized LLVM, exact byte
streams, termination, command/resource limits and an invocation-start marker.
The marker binds program, implementation and arguments after materialization
and before the entry call. Missing markers, setup errors, timeouts, other
signals, wrong diagnostics and unexpected published values cannot qualify as
a source assertion failure.

Merlin `281a8294b004f7a0756af7bf4dd249a41a21ab84` independently stages and
compiles the portable payload. Its `standalone-effect-kernel` profile preserves
actual types and empty rosters in existing compile-signature v1/v2. It never
reads outcome references, native observations, input values or their individual
digests. Source messages remain source constants. Native legalized IR is
evidence, not the compiler input or a replacement portable body.

`t2m.merlin-effect-qualification.v1` binds whole-body lowering, code generation,
the actual library and a separate observed evaluator call. Its status is
`matched-expected-outcome`, which can include assertion failure. It does not
claim normal return or a numeric result. `t2m.merlin-effect-invocation.v1`
reuses that exact library with new arguments/references and a pinned source,
compiler and library snapshot. New invocation data causes no recompilation.

These are producer observations and byte/ABI custody, not historical execution
attestations or universal source-equivalence proofs. Compiler input exclusion,
memory/alias checks and target gates remain separate.

## Public commands

```bash
python -m t2m effect-demo --case constant-fail --out out/effect-constant-fail \
  --optimizer .scratch/toolchain/triton-shared-opt --python /path/to/native-python
python -m t2m qualify-merlin out/effect-constant-fail \
  --out out/effect-constant-fail-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

Other demos are `empty-return`, `scalar-check`, `tensor-check` and
`fail-before-store`. `literal-check` additionally exercises embedded NUL and
Unicode through the explicit [length-aware diagnostic path](literal_diagnostics.md).
Memory profiles keep their existing selection/evidence
rules. Saved TTIR uses `shared-tensor --effect-reference reference.json`;
`--inputs` is optional only for an empty input roster. Effect references exclude
numeric references/tolerances. `invoke-merlin` accepts outcome JSON for a
qualified effect library. Use fresh output directories.

The focused suite passes 40 cases across ABI, actual producer/native outcomes,
Merlin qualification/reuse, refusals and CLI routes; 73 existing regressions
also pass. Merlin's corresponding staging suite passes 230 cases. The saved
`out/effect-constant-fail` and `out/effect-real-result` packages pass 28 artifact
checks each; their separate Merlin qualifications pass 49 each. Both outcomes
are explicit source assertion failures. The latter still declares one real
tensor result, with no value published.

The explicit 17-package observation report at `out/effect-coverage/coverage.json`
retains all 50 definitions, with 24 observed and 20 present in captures of
Merlin-qualified packages. Both effect packages record exact outcome policies
separately from numeric tolerances. These are selected composition observations,
not universal opcode execution coverage.

## Remaining effects

[Logical print events](logical_prints.md) now also execute through actual Merlin
for readonly/void entries and initial tensor failures. `print-words` and
`print-hex` are public demos. Physical print replicas and the selected numeric
formatting/runtime are explicit differences. Mixed tensor results plus exact
diagnostic references remain a required package extension.

Assertions now retain embedded NUL and Unicode through the explicit
[length-aware literal transport](literal_diagnostics.md). New packets bind
`native/compiler.log` separately from source streams; original v1 packets
remain readable. Broader `tt.print` forms, arbitrary binary diagnostic formats,
external implementations, assembly and synchronization still need
executable contracts. Accelerator materialization/diagnostic handling remains
unqualified. The full goal also retains numeric, runtime/alias and maintained
corpus/refusal requirements.

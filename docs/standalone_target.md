# Standalone kernels with a runtime target ABI

`scripts/qualify_spike_model.py` also accepts a verified standalone numeric
`t2m.capture-receipt.v2` package. Dynamic tensors and native scalar arguments
reach Merlin's ordinary RISC-V build and Spike execution without enclosing the
kernel in a static model capture or specializing its MLIR to an invocation.
This uses the same portable payload, source provenance, compiler-only staging
and consumer ownership checks as the host path.

## Compiler and runtime boundary

Merlin's optional `runtime_signature` selects a checked
`merlin.compile-signature.v1/v2` from compiler staging. The runtime generator
reopens the actual `forward` types and exact positional `inN` archive roster.
Static dimensions must agree with those types; dynamic dimensions bind to
named integer scalars, with their declared guarded bounds checked. Dtypes,
scalar rank, canonical Boolean words and complete value byte budgets must
agree before generation. The present C transport accepts unencoded dense
tensors of rank at most eight and ranked tensor results. Other layouts and
effect-only target entries remain explicit completion requirements.

`image/cgen/runtime-abi.json` records `merlin.runtime-invocation-abi.v1`: source,
signature and invocation archive hashes, concrete tensor shapes, native scalar
types/words and `mlir_argument_specialization: not-performed`. This is runtime
transport metadata, separately bound into build preparation and image identity;
it supplies no numeric, alias, ownership or source-equivalence permission.

Native scalars remain actual i1/i8/i16/i32/i64/f32/f64 arguments. Private
descriptor storage holds their bytes for the C harness; a typed C bridge passes
each scalar by value to the unchanged MLIR C interface. Neither that storage
nor the archive's zero-dimensional array converts an MLIR scalar into a tensor.
Native and Merlin host runners also copy exact scalar storage rather than
rounding F32 through a Python float. This requires Merlin's
`ScalarArg.from_storage` API. BF16 inputs retain original uint16 source words;
the C generator uses an exact bit-container projection for its existing storage
emitter, with no archive or source-ABI change.

Each bare-metal image embeds its invocation arguments. Changing values builds
a new harness/image, while the kernel MLIR and compiled object can stay
identical. The current command rebuilds the object and checks its hash; it does
not implement an object cache or a persistent target runtime.

## Qualification and replay

The source package's inputs and complete independent references are the default.
Pass both `--inputs` and `--reference` to qualify a different standalone
invocation. Input archives contain exact `in0..inN` fields; references contain
every `out0..outN`, including empty results. Changed invocation archives remain
outside compiler staging. Only the input path reaches the build worker;
references remain evaluator data.

```bash
python -m t2m shared-demo --kernel dynamic-count --out out/runtime-target-source \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python
python scripts/qualify_spike_model.py \
  --capture out/runtime-target-source --out out/runtime-target \
  --merlin-root /path/to/merlin --python /path/to/merlin-env/bin/python \
  --comparison words
python scripts/qualify_spike_model.py \
  --capture out/runtime-target-source --out out/runtime-target-changed \
  --inputs changed-inputs.npz --reference changed-reference.npz \
  --merlin-root /path/to/merlin --python /path/to/merlin-env/bin/python \
  --comparison words
python scripts/qualify_spike_model.py --verify --out out/runtime-target-changed \
  --merlin-root /path/to/merlin --python /path/to/merlin-env/bin/python
```

The external receipt is `t2m.spike-kernel-qualification.v1`, with scope
`functional-riscv-standalone-kernel`. Source capture/ABI schemas and parent data
stay unchanged. Bound models retain their separate existing qualification schema
and static public model contract. See [package differences](package.md).

Verification re-derives the runtime ABI from actual source types, staged
signature and invocation bytes, then decodes complete retained target wire
frames. Updated outer hashes cannot substitute different output shapes.
Every original input has a readonly byte observation. Empty tensors require an
explicit zero-byte frame; a program with no inputs records that distinct case.
Changed archives, missing frames, altered output/input bytes and undeclared
compiler members refuse. Numerical agreement remains separate from proof of
source equivalence and algorithm preservation.

## Measured scope

`tests/test_spike_kernel.py` runs five source cases through ten actual target
images and compares eighteen complete result tensors as exact words:

- One dynamic-count kernel at n=7, 0, 1 and 13, with unchanged program/object
  hashes. n=14 violates its declared bound and refuses before an image succeeds.
- All seven native scalar types, including wide integers, signaling NaNs,
  signed zero and changed values. Native, Merlin host and target preserve the
  original scalar-word observations; both target objects have the same hash.
- A no-input numeric program and a BF16 copy with exceptional source encodings.
- Source-grouped mapped memory with five results, runtime scalars and nonempty/
  empty domains. Both invocations preserve the same compiled object.

The test also changes runtime shapes behind recomputed outer hashes and mutates
external input archives; verification rejects both. Existing whole-model target
and scalar-runtime regressions check the shared paths. The final combined run
passes 39 tests, including a fifteen-result bound-model target regression;
Merlin's runtime/C-generator suite passes 116 checks with two optional bitcode
checks skipped because the selected toolchain has no `llvm-as`.
`out/spike-standalone-runtime/summary.json` pins the ten target receipts under
`.scratch/pytest-standalone-target-qualified/` and records the separate legacy
regression. The public changed-invocation CLI demo is
`out/standalone-runtime-target-demo/qualification.json`; both execution and
subsequent CLI verification pass.

These are finite functional CPU qualifications. Broader types/attributes,
physical/mutating views, shared/external effects, general value-returning TTIR
entries, accelerator execution and performance remain open. Existing tool and
memory/time budgets from [target execution](target_execution.md) still apply;
no toolchain is built or installed by this command.

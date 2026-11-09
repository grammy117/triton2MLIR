# Compiler-only staging and actual Merlin host materialization

The optional Merlin path now has two explicit levels:

- `--merlin-root` checks real interface/runtime generation and stages the derived
  model through `merlin.frontends.compile_inputs.stage_compile_inputs`.
- `--merlin-host` additionally compiles that staged program with Merlin's whole-body
  scalar pipeline, links an actual native shared library, and invokes it through
  `merlin.llvmlower.abi.HostModel`. It compares against the unchanged original
  PyTorch golden outside compilation.

```bash
python -m t2m bind-model out/descriptor_model_capture \
  --bindings examples/model_descriptor_binding.json --out out/descriptor_host \
  --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --merlin-host --merlin-clang /path/to/compatible/clang \
  --rtol 1e-5 --atol 6e-6
```

The compile worker receives only the staged compiler directory and a build
directory. Its inputs contain the full program, signature, weights and separate
parent/TTIR source correspondence. Evaluator arrays, goldens, execution reports
and their digests are excluded. The later invocation worker receives the
materialized argument archive and the compiled library; reference comparison
remains outside both compilation and invocation. No source-kernel name, golden
or provenance ID selects lowering behavior.

The scalar pipeline is derived from Merlin's existing pass recipe, including
allocation ownership/deallocation. This adapter adds logical-forall lowering
for the selected native host and permits bufferized loop results. It does not
rebuild a model from a partial list of recognized contractions. The exported
frontend IR retains its logical grid, source K loops and scan network. Host
serialization is a consumer scheduling choice, not a frontend hardware mapping.

Functional tensor inputs are borrowed immutable values. The staged entry marks
them `bufferization.writable = false`; otherwise upstream function-boundary
bufferization can reuse and mutate parameters. The tuple-scan qualification
exposed that bug. Execution now checks input bytes and two calls with separately
initialized result buffers, as well as the complete-model golden. These are
measured cases, not universal semantic proofs.

`consumer/interface.json` uses `t2m.merlin-interface-check.v2` and records compiler
staging separately. The stager's exact input receipt is archived as
`consumer/input-receipt.json`; the final package receipt subsequently includes
all consumer products. This avoids a self-referential receipt. A consumer failure
removes the outer success receipt while retaining diagnostics.

The optional `consumer/host/execution.json` uses
`t2m.merlin-host-qualification.v1`. It binds staged/prepared IR, actual LLVM,
shared library, observed outputs, reference and the selected toolchain. Native
objects are explicitly named consumer products, not a replacement for portable
`model.mlir`. `verify-bundle` checks their internal source/library/reference
relations as well as artifact hashes. Missing reports cannot retain a host-PASS
metadata claim. Old interface-only receipts remain readable.

This qualifies actual native-host materialization and execution. Accelerator
materialization, dynamic/scalar/effectful ABI profiles and direct standalone
kernel staging still need their own gates; these checks do not close them.

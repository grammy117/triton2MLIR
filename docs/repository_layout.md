# Merlin repository-layout integration

The `ttir-experiment-grammy` Merlin branch merges upstream
`refactor/repository-layout` at `2451b8134a7995d4e73efa3a465f975a1d532727`.
The triton2MLIR adapters now require that layout and the retained TTIR extensions.
The initial integrated Merlin commit is `dd7b62801d8fba37ba5e2dc88fc0c04fd2cebd1e`.
The public-stage extension below additionally requires
`compiler.frontend.intake.staged_projection` and the revised public preparation
services, delivered by Merlin `e1c2c7f88ac454b42b68052ccfa87ea605fff39c` on
`ttir-experiment-grammy`.
An unmodified upstream checkout does not supply those extensions. Historical
receipts and the previous adapter remain available at their recorded Git revisions;
there is no second namespace or automatic fallback to the retired layout.

The package boundary remains source-bound standard MLIR plus explicit ABI,
provenance and receipt members. Tensor computation, logical-grid iteration,
reuse, effects and allocation requirements retain their existing meanings.
Evaluator arrays and golden outputs remain outside compiler staging. Select
execution-owned results explicitly; their actual descriptors determine output
shapes, with finite allocation and ownership checks.

| Previous Merlin owner | Current owner |
| --- | --- |
| `frontends.compile_inputs.stage_compile_inputs` | `compiler.frontend.inputs.stage_compile_inputs` |
| Receipt/shape helpers in `frontends` | `compiler.frontend.intake` |
| `frontends.linalg_mlir` | `compiler.frontend.linalg_mlir` |
| `xdsl_dialects` | `compiler.dialects` |
| `llvmlower.pipeline` | `compiler.lower.native.pipeline` |
| Optional numeric/effect/ownership transformations in `llvmlower` | `compiler.lower.options` |
| `llvmlower.abi` | `compiler.lower.model.oracle` |
| `llvmlower.c_runtime` | `compiler.lower.c.runtime` |
| `runtime.backends.spike_model` | `runtime.backends.baremetal.model_build` and `image_link` |
| `merlin/runtime/` C sources | `src/merlin/runtime/native/` |

Names in the table are relative to `merlin.`. Imports, emitted API names and
source hash rosters use the current implementation owners, including embedded
worker scripts. Existing report schemas remain unchanged; new reports name and
hash the refactored sources. A source-roster hash alone does not establish a
complete dependency closure or qualify an accelerator.

Fresh standalone kernel/effect stages now also support generic
`load_compile_inputs` / `StagedProgram` preparation. The current Merlin builder
adds a closed `source/intake.json` projection and a non-transforming
`merlin.imported-source-policy.v1` document. Loading replays the same source,
typed ABI and effect checks, reconstructing every staged member. Resealing a
changed body, signature, source binding or numerical policy does not supply those
relationships. Public preparation retains the complete program byte-for-byte,
including scalar inputs, dynamic shapes, void effects and execution-owned results.

The selected backend must explicitly admit that policy. `MerlinHostBackend`
uses the same existing whole-body Merlin native pipeline as standalone
qualification; it introduces no source decomposition or arithmetic policy:

```python
from merlin.api import StagedProgram, compile
from merlin.compiler.frontend.inputs import stage_compile_inputs
from t2m.merlin_backend import MerlinHostBackend

stage_compile_inputs(capture, compiler, result_mode="execution-owned")
compiled = compile(
    StagedProgram(compiler),
    backend=MerlinHostBackend(
        merlin_root=merlin_checkout,
        python=merlin_python,
        compiler_python=native_python,
        clang=selected_clang,
    ),
    output=compiled_directory,
)
```

Omit `result_mode` for the ordinary declared-shape/void ABI. Compilation
publishes the source projection under `frontend/`, the actual signature under
`frontend/signature.json`, and the library under `host/model.so`. Runtime
arguments and references remain outside compilation. The backend reports
compilation only; actual invocation and reference comparison are separate.
It accepts no unimplemented deployment/options and supplies no accelerator
execution or performance verdict.

This is a compiler-stage extension: original triton2MLIR capture/receipt schemas
and standard MLIR payloads remain unchanged. Restage older standalone captures
to obtain the new projection. Bound-model generic public preparation still
requires its own source/parameter/state projection; its existing dedicated
native/host/bare-metal path remains available. Neither path establishes every
backend's support.

The 35-check public staging selection passes in
`.scratch/public-kernel-staging-final.log`. Its five compiled libraries execute
thirteen successful changed-input invocations, retain all twenty result arrays,
and preserve one actual source assertion abort. Thirty existing consumer/effect
checks and the source-roster check also pass. The shared stage path additionally
materializes actual atomic and execution-owned serial images; all three results
compare as exact words. These separate image checks do not make the host-only
public adapter a deployment backend.

`out/public-staging/summary.json` reopens the public compilation artifacts and
selected committed source identities. The fresh, valid bound-model failure is
retained in `.scratch/public-bound-model-frontier/boundary.json`: receipt staging
passes, but its generic loader lacks the model replay projection. Continue that
source/parameter/state path before claiming complete public-model integration.
Selected source hashes establish no full dependency closure or accelerator/
performance qualification.

Set `T2M_MERLIN_ROOT` to the merged checkout and `T2M_MERLIN_PYTHON` to an
environment installed from it. Native MLIR and cross-tool selections remain
explicit. Reinstall Merlin in the selected environment after the layout change;
old editable-install metadata and console wrappers can retain retired owners.

The focused migration checks use real tuple scan, descriptor matmul, symbolic
shapes and PyTorch scan/scatter compaction, including empty outputs and negative
custody/initialization witnesses. Merlin's opt-in frozen TTIR replay is available
as `build_tools/checks/replay_ttir_frontend.py`; it records its exact source and
adapter hashes under ignored `out/build/ttir-replay/`.

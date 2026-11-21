# Merlin repository-layout integration

The `ttir-experiment-grammy` Merlin branch merges upstream
`refactor/repository-layout` at `2451b8134a7995d4e73efa3a465f975a1d532727`.
The triton2MLIR adapters now require that layout and the retained TTIR extensions.
The integrated Merlin commit is `dd7b62801d8fba37ba5e2dc88fc0c04fd2cebd1e`.
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

The newer generic `load_compile_inputs` / `StagedProgram` service retains its
own m2m profile and precision-policy requirements. The retained triton2MLIR path
uses `stage_compile_inputs` and the native/host/bare-metal adapters. Acceptance
through that path does not establish admission through every new public service.

Set `T2M_MERLIN_ROOT` to the merged checkout and `T2M_MERLIN_PYTHON` to an
environment installed from it. Native MLIR and cross-tool selections remain
explicit. Reinstall Merlin in the selected environment after the layout change;
old editable-install metadata and console wrappers can retain retired owners.

The focused migration checks use real tuple scan, descriptor matmul, symbolic
shapes and PyTorch scan/scatter compaction, including empty outputs and negative
custody/initialization witnesses. Merlin's opt-in frozen TTIR replay is available
as `build_tools/checks/replay_ttir_frontend.py`; it records its exact source and
adapter hashes under ignored `out/build/ttir-replay/`.

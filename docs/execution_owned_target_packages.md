# Execution-sized TTIR results through target packages

The real count/parallel-scan/scatter/consumer model now reaches the ordinary
Merlin model builder and actual RISC-V/Spike execution with signature v6.
All seven results retain original shapes and exact words, including the internal
count, parameter, snapshots and stepped slice. Output extents come from the
executed descriptors; no independent size predictor or evaluator value enters
compiler staging or lowering. This is a functional serial CPU target selection.

```bash
python scripts/qualify_spike_model.py \
  --capture /path/to/qualified-compaction-package \
  --result-mode execution-owned --comparison words \
  --merlin-root /path/to/merlin --python /path/to/merlin-env/bin/python \
  --out out/compaction-target
```

The explicit mode selects `t2m.spike-owned-result-qualification.v1`, separate
from legacy preallocated-result target receipts. Parent capture/package receipts,
public MLIR and original archives remain unchanged. Runtime input projection v3
materializes the complete positional model roster as exact source words, including
parameters. Changed model invocations supply positional `--inputs` and complete
`--reference` archives together; these do not rewrite the source package.
The image's historical weights-blob section carries this separately generated
invocation blob; it is not the original safetensors container layout. Original
container/archive bytes remain receipt-bound and unchanged in the package.
Old/default qualification retains its existing receipt and signature requirements.

The ordinary Merlin builder dispatches explicit signature v6 to its shared
source-only lowerer and generated C driver. It observes the selected compiler's
actual layout/index contract, uses the existing bounded allocator, serial context,
startup/console/linker and exact selected math archive, and records actual compile/
link commands. Loaded image, stack and arena must fit disjointly inside the
declared span. The current executor is one nonconcurrent hart with HTIF readback;
unimplemented nondefault build options refuse rather than being ignored.

The explicit observed-shape profile requires every input/result shape and complete
payload. Success also needs DONE and a successful observed process. An encoded
guest allocator failure can truncate to host exit code zero, so process status
alone never qualifies an image. Original source guards and runtime allocation/
descriptor checks remain active. The serial selection grants no public mutation,
alias, numeric-policy, synchronization or accelerator permission.

Retained verification reopens all artifacts, replays independent Merlin source
staging, reconstructs the driver from actual source/invocation bytes, re-decodes
the raw wire and compares every original reference. Rehashing a changed payload
or source-correspondence declaration cannot substitute for those checks. Extra
unaccounted artifacts also refuse. Verification runs neither the simulator nor
tools selected by serialized paths.

`tests/test_compact_model_target.py` qualifies four actual whole-model images:
the original PyTorch invocation plus changed empty/full/overflow cases, checking
all 28 results. A source-domain failure emits no results/DONE; a complete wrong
reference cannot publish qualification. Three further checks reject coherently
rehashed payload/correspondence records and unaccounted artifacts. Evidence is
retained in `out/compact-model-target/summary.json`.
The combined frontend target/host/legacy selection passes 57 checks; one optional
legacy capture selection is skipped. The separate Merlin runtime selection
passes all 142 checks, including actual images and arena exhaustion.

Remaining work includes runtime/zero-trip resizing ownership, v4 physical/view
adapters, broader numeric/effect/source compositions, additional applicable
executors and accelerator correctness/performance. The historical TTIR opcode
inventory is unchanged. [Generated hosted transport](owned_result_driver.md)
and the older [runtime seams](execution_result_targets.md) retain distinct scopes.

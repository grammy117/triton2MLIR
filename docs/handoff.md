# Review handoff — 2026-11-12

The source-to-package-to-Merlin pipeline works for the qualified selections.
**Full arbitrary-Triton-kernel coverage remains incomplete.** This is a review
boundary requested by the user, not a completion claim.

## Repositories and publication

| Repository | Branch | Relevant implementation |
|---|---|---|
| `/scratch/grammy/blast/triton2MLIR` | `main` | `e7f4733ab364c733436cbdb9bba1371e4d9fecbd` — captured tensor reads; the following qualification commit contains this handoff and refreshed status. |
| `/scratch/grammy/blast/merlin-graph-proof-intake` | `ttir-experiment-grammy` | `4770c046bed7b64c4f81862e68f2f0cfae7da2c4` — independent initial-state copies and retained guards; `1a408cd4b577380bfec4d3e79c139b6ac5e0af32` — checked padding values. |
| `/scratch/grammy/blast/model2MLIR` | `ttir-experiment-grammy` | Existing integration remains separate; no new source edits in this slice. |

The unrelated dirty `model2MLIR/m2m/ir/import_fx.py` is preserved. Its SHA-256 is
`8d4000b64f18802086bcf55325f21e268f795df4ccaefea40266e7d23a7a1379`.
Do not reset or include it in these changes. Continue the verified Graham Anderson
Git identity and ordinary incremental commits; no co-author trailers or history rewrites.

## What changed

- `t2m/scalar_tile_reads.py` reuses shared's singleton projection and the existing
  guarded gather lowering. Captured tensor values become standard element reads.
  Shared/conditional projections retain one lookup at the original gather position.
- Reduction/scan checks admit exact typed reads and literals. Only guard operation
  identities created by that transform receive generated-guard admission. Tags,
  locations and serialized copies cannot grant it. Original assertions and hidden
  pointer-memory effects retain their source audit.
- Lookup family, original source IDs, logical shape/axis and source layout hints
  remain annotations. Bounds, values and control are executable without them.
- Merlin constructs an independent initial-state copy only after closing its
  producer/capture proof. Closed padding supplies a readable value, not ownership.
  Guards, zero-trip behavior and native alias/lifetime gates remain enabled.
- Public package ABI, artifact rosters and receipt schemas are unchanged.
  [Package differences](package.md) and generated compatibility metadata state
  the representation. [The design/evidence contract](scalar_tile_reads.md)
  describes the hardware/software boundary.

## Evidence to inspect

The combined inventory reopens **237 packages, 192 standalone qualifications
and fourteen product cases**. All fifty pinned TTIR definition names occur in
selected native/Merlin-host-qualified captures. This measures opcode breadth;
it establishes neither universal operation support nor a completion percentage.

The new state selection has **24 libraries and 24 actual Spike images**, with
**48 complete exact-word results and 96 additional host runtime invocations**.
Each kernel loads a table, mutates its source storage, then uses the old snapshot
in runtime reduction/scan recurrence. Both returned results are checked: the
changed table and output. Cases cover i32/i64, both table axes, three memory
profiles, empty/tail inputs, zero steps, overflow and different runtime tables.

VM-local generated evidence, intentionally ignored by Git:

- `out/tile-lookup-state/summary.json` — target receipts and runtime counts.
- `out/tile-lookup-inventory/coverage.json` — explicit combined package selection.
- `.scratch/pytest-tile-lookup-state2/` — the 24 complete captures/libraries/images.
- `.scratch/tile-read-regressions3.log` — **177 passed**, one upstream deprecation warning.
- `.scratch/tile-read-final.log` — **21 passed**, covering guard identity, annotation
  independence, a current-package target case and actual bounds aborts inside both
  combiners in an existing Merlin library.

Merlin's destination selection passed **135 tests** after the padding fix;
structure and documentation gates passed. Earlier ownership/deallocation checks
also passed, separately. Test selections overlap; do not add them as unique coverage.
There are no known unresolved failures in the listed final selections.

## Reproduction on this VM

Use the existing environments and a fresh project-local pytest basetemp.
No toolchain rebuild, install, GPU or Triton-CPU dependency is required.

```bash
cd /scratch/grammy/blast/triton2MLIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_SHARED_OPT="$PWD/.scratch/toolchain/triton-shared-opt"
export T2M_NATIVE_PYTHON=/scratch/grammy/blast/model2MLIR/.venv/bin/python
export T2M_MERLIN_ROOT=/scratch/grammy/blast/merlin-graph-proof-intake
export T2M_MERLIN_PYTHON=/scratch/grammy/blast/merlin/.venv/bin/python
export T2M_MERLIN_CLANG=/scratch/grammy/blast/merlin/third_party/llvm-build/bin/clang-23
export MERLIN_COMPILER_PYTHON="$T2M_NATIVE_PYTHON"
export MERLIN_CLANG="$T2M_MERLIN_CLANG"
export MERLIN_MLIR_INSTALL=/scratch/grammy/blast/merlin/third_party/llvm-build
export MERLIN_RISCV_GCC=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc
export MERLIN_NM=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm
export MERLIN_SPIKE=/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/spike

"$T2M_MERLIN_PYTHON" -m pytest -q tests/test_scalar_tile_read_contract.py \
  tests/test_scalar_tile_reads.py --basetemp=.scratch/pytest-review-contracts
"$T2M_MERLIN_PYTHON" -m pytest -q tests/test_tile_lookup_state.py \
  --basetemp=.scratch/pytest-review-table-state
```

Keep execution serial on this shared VM. Missing environment variables can skip
integration tests, so inspect the final pass/skip summary. The 24-image selection
took about four minutes with the existing tools.

## Limits and next work after review

These are native-verified **typed TTIR fixtures**, not new Python AST captures,
GPU execution/correspondence evidence or a performance benchmark. No new whole-model
capture was added for this lookup family; existing model integration evidence
remains separately scoped. Floating/other-width lookup compositions and general
tensor intermediates are not qualified by this selection.

Keep the full [completion plan](completion_plan.md) active after resumption:
close remaining builtin/region and typed capture forms; give direct memory,
shared/external effects and synchronization concrete semantics; extend physical/
mutating views, runtime/alias/CFG/value-returning interfaces; cover supported source
versions and applicable accelerator materialization, correctness and performance.
Do not erase difficult source operations or count opaque fallbacks as completion.

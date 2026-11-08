# Repository conventions

Read this file, [ROADMAP.md](ROADMAP.md), and the relevant design/package documentation before
changing the compiler. These rules apply to every contributor and coding agent in this repository.
The commit conventions follow Merlin's `CLAUDE.md`; the ownership/layout rules below are specific
to triton2MLIR.

## Purpose and abstraction boundary

Translate a Triton kernel into the same kind of tensor computation that model2MLIR supplies to
Merlin. Preserve useful tiling, fusion, reuse, recurrence and numeric behavior. Emit standard
`func`/`linalg`/`tensor`/`scf`/`arith`/`math`/`cf`, or an explicitly declared high-level profile with a
deterministic standard expansion. Hardware mapping and physical memory placement belong downstream.

- Keep the logical grid independent of warp/thread/core mapping. Do not introduce per-target
  lowering cases, device opcodes, physical address spaces or runtime launch ABIs in the core path.
- Keep numeric/effect semantics in executable operations. Provenance cannot make an omitted
  mask, accumulator, cast, synchronization requirement or memory effect correct.
- Derive behavior from source IR, types, shapes, effects and established facts. Never select
  production behavior by kernel name, provenance ID, example data or golden output.
- Unknown or unproved behavior is a located refusal, not an opaque call counted as successful
  portable conversion. Preserve the complete source-operation inventory and its disposition.
- Caller intent describes the source; it does not grant arithmetic, effect or rewrite semantics.
  Preserve `prov.family`, `prov.op`, `prov.region_id` and source/emission correspondence through
  serialization and any supported transformation.

## Reuse triton-shared and upstream MLIR

[triton-shared](https://github.com/facebookincubator/triton-shared) already implements structured
pointer/mask/use analyses, region handling and many arithmetic conversions. Treat that work as
the first source of implementation patterns, fixtures and reusable passes, not merely background.

Before adding a decomposition or analysis:

1. Read the relevant upstream implementation **and tests** at a recorded revision.
2. Prefer a supported upstream pass/adapter or a narrowly adapted analysis over another independent
   implementation. Explain the remaining tensor-ABI/provenance work and any actual incompatibility.
3. Preserve source attribution, copyright and license notices when adapting upstream code/fixtures.
   Do not confuse source-license attribution with commit co-authorship.
4. Record the version/pipeline boundary. A full upstream pipeline containing memrefs does not make
   all its earlier tensor/structured stages unusable; neither does a tensor stage by itself satisfy
   our functional ABI, proof or provenance contract.
5. Do not build a large Triton/LLVM stack implicitly. Use existing tools or saved fixtures for
   bounded experiments; request approval for significant builds under the machine rules.

Keep all Triton compiler-internal imports in `t2m/frontend.py` (or a deliberate successor adapter).
Pin/probe its API. Parse operations, regions, blocks and types structurally; do not add regex-based
IR lowering. The existing checked constant-printer bridge is an isolated binding limitation, not
permission to recover arbitrary IR semantics from printed lines.

## Ownership and artifacts

- `t2m/`: installed frontend, analyses, builders, transforms and artifact contracts.
- `t2m/examples/`: small real kernel sources shipped with the demo.
- `tests/`: focused conversion, semantic, refusal and integration tests. Small source/IR fixtures
  may be tracked when they establish a contract and carry any upstream attribution.
- `docs/`, `README.md`, `ROADMAP.md`, `plan.md`: durable contracts, design and current delivery status.
- `out/`: generated bundles/reports/demos. `.scratch/`: disposable probes, pytest temporary files,
  packaging output and local experiments. Both are ignored. Never commit caches, wheels, venvs,
  captured weights or large generated artifacts.

Keep model2MLIR package differences explicit in `docs/package.md` and generated compatibility
metadata. Do not label a Triton receipt/trace as an m2m receipt/FX trace without meeting that schema.
Keep frontend conversion, native verification/legalization, reference execution and staged/device
execution as separate measured statuses.

## Validation and working cadence

Implement one reviewable capability at a time. Include semantic positive and negative cases for
compiler changes: tails, masks/fills, dtypes, initial values, effects/aliasing and unsupported forms
as relevant. Do not weaken refusal checks or substitute provenance for numerical/effect evidence.

```bash
python -m pytest -q --basetemp=.scratch/pytest
# Optional: use an existing environment with native torch-mlir bindings.
T2M_NATIVE_PYTHON=/path/to/native-env/bin/python python -m pytest -q --basetemp=.scratch/pytest
git diff --check
```

Use the supported Python environment explicitly. Report skipped/unavailable checks honestly.
Native verification/LLVM legalization is not native or accelerator execution. Once appropriate
checks pass, commit the coherent change and continue; do not repeat broad testing without a new
change, failure or unresolved concern.

This is a shared VM. Stay under `/scratch/grammy`, `/scratch/gram117anderson` or
`/home/eecs/gram117anderson`; no sudo, system package installs or user namespaces. Check CPU/load
before substantial work, cap workers, use project scratch instead of `/tmp`, and stop jobs you
start when finished. Ask before destructive actions, outward publication or sustained significant
CPU/RAM/disk use. Ordinary reversible implementation and targeted tests remain authorized.

## Commits and attribution

- **Commit frequently:** after each coherent, tested capability/fix or documentation change.
  Do not accumulate multiple milestones until the end of a long session. Existing uncommitted
  work may be captured as an honest baseline; do not fabricate a historical sequence of states.
- **Format:** `type(scope): imperative summary`; type is one of
  `feat|fix|docs|chore|refactor|test|perf|build`. Keep the **entire subject at most 72 characters**.
  Use the optional wrapped body for the problem, resulting behavior and relevant validation.
- Describe the file/code change, not the agent's process. Avoid `WIP`, `checkpoint`, `snapshot`,
  `phase 2`, `step 3`, `finalize`, task IDs and results narratives in subjects.
- No emoji, `Co-Authored-By` trailers, generated-by messages, AI/tool attribution or other
  unsolicited authorship decoration. This applies to the subject, body and trailers.
- Use **Graham Anderson <gram117anderson@berkeley.edu>** as both author and committer for new user
  commits. Never use the VM hostname email. Preserve other contributors' authorship when replaying
  their commits.
- **Before every commit**, run `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` in this
  repository. If either differs, inspect environment/config origins and correct it before committing.
- Review the staged diff and stage only the intended change. Do not revert, hide or include another
  contributor's work. Stay on the current branch; use a separate worktree when isolation is needed.
- **After every commit**, inspect the actual commit object's full author, committer and message.
  Verify identity and absence of forbidden trailers; do not rely on intended flags/config alone.
- Commits are local by default. Pushes, new PRs and other publication require user authorization.
  Never rewrite published history without explicit authorization; an authorized force push must
  protect the expected remote tip with an explicit `--force-with-lease`.
- Before declaring an authorized push complete, verify the remote tip and, when available,
  attribution to GitHub login `grammy117`. Report attribution failures rather than claiming success.

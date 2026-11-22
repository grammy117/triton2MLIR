# Pinned TTIR-to-Merlin qualification and review handoff

The finite v1 qualification is complete for the declared Triton 3.7.1 source,
ABI and semantic profiles. The existing release checker passes all five gates:
pinned policy/fixtures, no demonstrated open generic gaps, complete acceptance,
actual semantic/package evidence and independent Merlin baseline source custody.
This establishes a usable pinned frontend and interface, **not support for every
arbitrary Triton kernel or every type/attribute/control composition**. A valid
generic counterexample remains a bug and reopens closure. Explicit product limits
below are not successful portable translations.

## Evidence

| Check | Actual result |
| --- | --- |
| Frozen acceptance | 128 passed, no failures or skips; all 35 fixture hashes unchanged. Includes K1–K12, M1–M7, contract/publication refusals and one serial-target smoke. Expected refusals are not portable-program counts. |
| Original generic TTIR definitions | All 49 have reviewed profiles and original-source witnesses in qualified packages. Optional assembly is the fiftieth definition. This is not universal semantic coverage. |
| Verified final artifact selection | 72 unique packages; 58 standalone qualifications. Model host receipts remain separate. |
| Merlin regression replay | 2,600 passed; two pre-existing llvm-as prerequisite skips, outside the zero-skip acceptance selection. Current consumer source hashes match the replay. |
| Focused closure extensions | I4 transport, logical low-precision print, FP8 arithmetic/accumulators and per-event atomic addition, numeric entry/result ports, and defined noinline helpers have separate exact-source qualification. Those retained focused runs are not silently relabeled final whole-suite executions. |
| Effort | 9.20 hours charged from the original forty-hour start; the helper correction used 0.67h of the existing five-hour reserve. No budget reset. |

The executed compiler revision is `460cac6aa996eb0f7385f220e72f28a06f85357c`;
Merlin is `cd2fe16bb61c62b929255b3a5fc4f67691b789df` on `ttir-experiment-grammy`, and
model2MLIR is `f3f0de1ae130faebbd27bd36374d11e1e211daa2` on the same branch name. Producer
and consumer started clean. The pre-existing model2MLIR docstring whitespace
edit remains unmodified and is recorded in the implementation manifest.
Subsequent handoff changes only update documentation/evidence.

Actual records are under `out/v1-closure/acceptance-128/`: `command.json`, copied
`manifest.json`, `pytest.xml`, `process.json`, `implementation.json`,
`tool-identities.json`, `package-observations.json`, `package-selection.json`,
`semantic-audit.json`, `closure.json` and `summary.json`. `join-packages.py` retains
the exact mechanical evidence join. The frozen policy retains its freeze-time
status wording; `closure.json` is the current measured verdict. Review the profile/limitations and actual
source/ABI/numeric/effect receipts; an observation count is not a proof.
Intentionally damaged or unqualified artifacts are recorded as selection exclusions,
not successful portable evidence. Installed SDK/executable identities are recorded. Pinned upstream source hashes
are reference identities, not an independently established SDK build receipt.

## Required transformer example

The actual model in `scripts/capture_transformer_model.py` is:

```python
class TwoBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular = TorchBlock()
        self.triton = TritonBlock()

    def forward(self, x):
        first = self.regular(x)
        return first, self.triton(first)
```

Both blocks are complete pre-norm layer-normalization, Q/K/V self-attention,
output projection, residual and ReLU-FFN blocks, with independent parameters.
The second block is the real JIT kernel in
`t2m/examples/transformer_kernels.py`, containing eight `tt.dot` operations.
Fake launch tracing automatically discovers its source, grid and ports. The
PyTorch CPU implementation supplies evaluator references only; its values are
excluded from compiler staging. Original, changed and zero inputs compare both
outputs from one artifact-only published library.

See [the demo and reproduction command](transformer_demo.md). The final selected
13-token demonstration has a verified byte-identical review copy at
`out/two-transformer-demo-v1/`; its `summary.json` records exact error/byte evidence.
`out/v1-closure/acceptance-128/demo-handoff.json` binds all 235 copied files to the final selected test output.
Both output matrices pass on all three inputs; the largest observed absolute
error is approximately `1.32e-6` under `rtol=3e-4`, `atol=3e-5`.
Inspect `capture/parent/model.mlir`, `launch-translation/`, `bound/model.mlir`,
`compiled/` and `published/`. The example is single-head inference, D16/H32,
without dropout/causal masking, training, GPU execution or a speedup claim.

## Reopen or reproduce on a26

From the repository root, with the existing user-space environments:

```bash
source .scratch/goal-tools.env
/scratch/grammy/blast/merlin/.venv/bin/python scripts/check_v1_closure.py \
  --run out/v1-closure/acceptance-128 \
  --semantic-audit out/v1-closure/acceptance-128/semantic-audit.json \
  --merlin-replay /scratch/grammy/blast/merlin-graph-proof-intake/out/build/ttir-replay/layout-y3727ubm/replay.json
/scratch/grammy/blast/merlin/.venv/bin/python scripts/run_transformer_demo.py \
  --out out/review-two-transformer --tokens 13
```

The second command builds a fresh demo; choose another fresh output directory
if that name exists. Tool paths/versions are retained in `tool-identities.json`;
[the demo guide](transformer_demo.md) explains the environment variables when
reproducing elsewhere. The complete acceptance argv is in `command.json` and
its copied manifest; use a fresh basetemp/JUnit path for another run.

## Abstraction and fallbacks

Standard tensor/Linalg/SCF/Arith/Math computation retains logical tiles, reuse,
recurrences, masks/fills, source-width arithmetic, original accumulators and
actual effects. Logical grid coordinates remain independent of warp/core
placement. Source intent and operation correspondence travel as provenance;
annotations grant no missing computation or effect permission.

Explicit buffer/event paths are qualified generic fallbacks with checked logical
roots, indices, bounds, initialization and source-order state. They can add guard
storage, per-event checks or materialization. Exact FP8 host math and atomic
addition can expand to software f64 casts and byte-CAS loops; this changes cost.
Defined noinline helper bodies now use the existing upstream inliner, retaining
original call intent while disclosing code-size/register-pressure/call-boundary
changes. The emitted package's compatibility metadata records these differences.

The public consumer is Merlin `StagedProgram`, preparation and whole-body host
compilation. [Package differences](package.md) stay explicit: TTIR traces and
receipts are not FX/m2m traces or receipts. Numeric value/result ports use v3
source allocation configuration; public tensor/result receipt schemas remain
unchanged. I4 is one canonical unpacked uint8 cell per logical value.

## Limits and stopping decision

[The limitations inventory](limitations.md) states each limit's expected use,
reason and remedy. The material boundaries are arbitrary assembly or unknown
extern implementations; physical addresses/clocks/lane identities; post-layout
GPU/DMA IR; external/cross-program publication and async completion; unspecified
GPU numeric modes; reserved FP6 packing; finite resource domains; source/version
compatibility; physical alias/copyback; additional unqualified word views and
control/ownership compositions. Generic computation is not relabeled hardware
because it is difficult.

Automatic model integration currently requires one pure static launch with fresh
returned outputs. Wrapper math, multiple launches, mutation/views, hidden scratch,
symbolic/autotuned launch metadata and physical descriptors need explicit bindings
or a separately qualified adapter. By-value numeric inputs/returns are supported
through checked storage ports; raw pointer/descriptor objects need their actual
logical fields. Recursive/uninlineable APIs and multiple public library entries
are not universally supported by the selected normalization/entry interface.
No arbitrary-kernel, universal-composition, GPU-performance or new-target claim
is made. These limits are visible in the deliverable rather than omitted from a
full-closure claim.

The finite plan stops here: review the candidate and decide any next scope
explicitly. Do not restart the historical roadmap, extend the corpus, build a
new backend/solver or spend the unused effort budget automatically. A new real
regression in the admitted profile requires an exact source witness and bounded
correction; a new source version, model wrapper class or target is separate work.

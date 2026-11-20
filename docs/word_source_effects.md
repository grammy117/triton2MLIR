# Actual word-source effect admission

Word-stage byte custody and source-effect admission are separate obligations.
Both the producer and Merlin now parse actual typed `word-input.mlir` before
legacy stage early returns. A consistently rehashed envelope cannot authorize
an unselected volatile observation, atomic primitive or raw physical effect.

## Confirmed gap and resulting behavior

A retained probe placed real captured TTIR containing three volatile load
sites and an atomic RMW inside ordinary legacy and isolated word envelopes.
Before this change, byte-custody checks and independent memory admission
accepted those envelopes. That demonstrated missing primitive admission, not
source equivalence or successful execution of the altered source. The byte
helpers retain their custody-only scope; the memory gates now perform actual
source-effect admission first.

`t2m/word_source_effects.py` reuses the existing producer atomic/volatile
primitive contracts. Merlin's `kernel_word_source_effects.py` independently
parses source and reconstructs primitive facts without importing producer or
evaluator code. Each actual effect requires:

- An explicitly selected compatible effect profile.
- Matching retained source identity and original primitive fields.
- Actual atomic kind, value type, order and scope; selected f32 add policy.
- Actual typed volatile flags and matching observation declarations.
- Nondestructive word stages v3/v4 for volatile observations.

Lost volatile flags and deleted or changed primitive declarations refuse.
Raw GPU/physical dialect operations and inline assembly must already have
their supported standard expansion before this boundary. Original inventories
may retain unreachable declarations; admission does not invent their execution.

The checks compare the historical f64 IEEE source label under legacy stages
without changing the original evidence. This grants no new NaN arithmetic,
payload, address, storage-view, dependency or ownership permission. Existing
address, effect, replay and native lifetime checks remain separate obligations.

## Upstream and package boundary

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and archived Microsoft
`050b93ae9c737111f1ca425d793419b32427a971` supply the existing pointer conversion
boundary. Their load conversion does not preserve the volatile request; atomic
conversion carries kind/order/scope. The existing nondestructive
[volatile path](volatile_reads.md) and [mixed-effect path](atomic_volatile_effects.md)
retain executable semantics around that supported upstream machinery. This
change adds admission checks, not another decomposition or compiler build.

Public ABI, capture receipts and required artifact rosters are unchanged.
Previously accepted envelopes with unsupported actual effects now refuse.
Host and target qualification fingerprints additionally include the independent
reader, so old images are not relabeled as having passed the new gate. See
[package differences](package.md).

## Verification and remaining work

```bash
python -m pytest -q tests/test_word_source_effects.py \
  --basetemp=.scratch/pytest-word-source-effects
```

With the existing source/native/Merlin environments enabled, the 26 checks
include actual whole-function and isolated-event conversions. Six consistently
rehashed source changes refuse in both complete memory gates. The independent
Merlin frontend selection passes 861 checks, including six full staging cases
that leave no published compiler directory. A retained compatibility audit
reopens 293 actual word-source contexts from 352 historical packages, with no
primitive-admission failures. These scopes overlap and add no opcode count.

Complete i64/f32 mixed-effect models additionally pass 11 checks, with one
expected i64 skip for the f32-only public policy test. All 26 initial, 78 replay
and 26 serial-target results retain exact words; public CLI global/override
binding retains 26 further host results. Twelve changed model claims refuse
and two source domains actually abort. Original parent archives remain intact.

Retained-record reopening exposed a separate source-list omission in the
execution-owned target builder. `qualify_spike_owned.py` now pins the independent
reader too, and the actual model target test requires its exact current bytes.
Two new target images were built from the same unchanged model packages; the
updated integration test body passed for both. The earlier functional images
remain retained but lack this pin and are not counted as current-reader images.
`out/word-source-effects/summary.json` reopens original before/after probes,
real-source refusals, six no-publication staging cases, historical source audit,
complete model/replay/CLI results and both fresh source-pinned target records.
This report keeps the historical audit and new execution scopes separate.

This gate proves selection and primitive fields only. General shared/external/
asynchronous effects, byte-backed volatile views, source equivalence, broader
type/control/storage/numeric compositions and accelerator correctness/performance
remain completion work. The arbitrary-kernel goal remains open.

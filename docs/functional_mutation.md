# Mutating Triton calls as explicit functional state

The actual Torch `auto_functionalized_v2` operator now reaches the execution-owned
Merlin interface for distinct whole mutable Tensor bases. Ordinary observations
and updated state remain separate SSA results. Source review uses Torch 2.10's
`auto_functionalized_v2_dense` and `_generate_new_op_kwargs_from_bases` in
`torch/_higher_order_ops/auto_functionalize.py`; only the actual operator object
is recognized. Printed names and caller provenance cannot introduce effects.

## Complete state and package contract

Model2MLIR emits `m2m.functional-kernel-bases.v1` with the original source schema,
complete mutated argument names, original base indices, schema operand indices,
updated-result indices and raw tuple slots. Ordinary source returns precede
updated bases. A void source op reserves a None tuple slot; the unused slot
is accounted for without inventing a Tensor. Every actual Tensor result survives.

The explicit `t2m.functional-kernel-binding.v1` extension to binding v4 additionally
requires each updated base to map to the same retained readwrite source buffer
on both sides of the call. Equal shapes/types cannot authorize substituting an
ordinary fresh result. Producer replay and Merlin's independent
`execution_functional_state.py` reader check this separately. Original complete
SSA, scalar/shape guards, kernel memory evidence, locations and lifetime gates
remain required. Legacy binding refuses this profile.

Public input or registered-buffer mutation additionally selects the source-derived
`m2m.functional-state-results.v1` contract on the entry. It records every state
result's original input/kind/target and every user-result ordinal. These are
functional state-result ports, not writable borrowed input storage. The portable
invocation returns updated state explicitly; a caller can carry it into a later
invocation. Physical public aliasing and automatic in-place caller copyback need
their own invocation support. No annotation grants either behavior.

Bundle eager observation preserves initial input and buffer archives and records
every endpoint result in the actual exported order. Input copies preserve shared
input storage; buffers are restored after the reference forward. Shared input/
buffer storage and metadata-changing buffers currently refuse. `golden.npy`
remains the first complete exported result, including a leading state port when
present; `goldens.npz`, metadata ABI and the receipt cover the full output roster.
The parent capture remains unchanged when bound into the derived package.

## Actual program

`scripts/capture_mutating_state_model.py` captures two dependent calls to a real
Triton kernel with `mutates_args={"state"}`:

```python
before = state.clone()
first = update(state, state.numel(), 3)
second = update(state, state.numel(), -2)
return before, first, second, state, x, self.weight
```

The kernel loads old state, stores its observation, then writes old state plus
bias using its original 8-lane tiles and tail masks. Three model forms use
internal tensor state, a public input and a registered buffer respectively.
The public input form returns its updated value both as an explicit state port
and wherever the ordinary model result refers to that mutated input. Earlier
snapshots remain original values. Source result order is `(observed, state)`,
explicitly reordered from the translated buffer ABI `(state, observed)`.

Internal/public captures keep runtime length types and their original `[2, 31]`
domain; registered-buffer state retains its actual static length 17. Exact i64
parameter/input words exercise values beyond floating-point integer precision.
The registered CPU implementation supplies independent PyTorch references;
compiler output never supplies goldens or allocation facts.

## Qualification

The final `tests/test_mutating_state_models.py` and existing model-binding
selection passes 23 checks. Three complete packages retain all twenty initial
host results, sixty results across nine changed-input/parameter invocations and
twenty results from three actual Spike images using independently changed
invocations. All comparisons retain exact words and unchanged libraries; original
parent archives remain byte-identical. Twelve rehashed profile/result/base claims
refuse in both producer and independent Merlin readers. Two invalid public length
domains actually abort without output. Merlin's independent functional-state/
correspondence/view selection passes 40 checks. Model2MLIR's focused capture/
trace/result/constant/word/quality selection passes 102; its overlapping kernel/
mutation/session selection passes 38. `out/functional-mutation/summary.json`
reopens the receipts, all arrays, refusal records and actual abort outcomes.

Reproduction uses the existing environments documented in the README:

```bash
python -m pytest -q tests/test_mutating_state_models.py tests/test_model_binding.py \
  --basetemp=.scratch/pytest-mutating-state
```

## Remaining scope

General mutated views, shared mutable bases, optional/Tensor-list mutation,
clone overrides, metadata-changing state, arbitrary returned/public aliases,
automatic caller copyback and broader effect/numeric/source/target compositions
remain completion work. Functional state results do not establish a public
imperative invocation API, accelerator correctness or performance. The historical
352-package/50-name inventory remains unchanged.

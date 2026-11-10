# BF16 model2MLIR calls through Merlin

BF16 kernels now bind into real model2MLIR captures and execute their complete
models through native LLVM and Merlin host libraries. Logical BF16 MLIR types,
original contractions/accumulators, source calls and provenance remain intact.
No Triton CPU backend or GPU execution supplies the reference.

## Exact source data and compatibility

model2MLIR revision `a60c3d87ba6bc4f97c0ec96d48c08b0ffdd03d69`
adds optional `tensor_words.npz` and `meta.json.tensor_words`, schema
`m2m.tensor-word-archives.v1`. Each entry identifies an original archive/key,
logical BF16 dtype/shape and a raw uint16 word array. Source words come from an
equal-width tensor view before any floating conversion. The capture receipt
binds this additional evaluator artifact. Legacy F32 value archives stay
unchanged, including the first/tuple golden files.

The model adapter checks the exact parent receipt, inventory, unique bindings,
logical dtype, shape and agreement with the legacy values. First-result aliases
must agree in raw words even for NaNs. Numeric comparisons decode BF16 before
applying tolerances. Separate execution checks compare uncomputed outputs and
integer bit observations exactly, including NaN payloads and signed zeros.
Computed NaN payload/sign are not universally specified.

Legacy archives without this optional artifact also work when each BF16 value
has its unique exact binary32 embedding: low 16 bits zero, no NaNs, with signed
zero and infinities retained. This derivation is explicit in argument evidence.
NaNs and nonembedding values require original words; no payload is guessed and
no reference answer becomes a compiler input or replacement source input.
A partially declared word archive never falls back to legacy recovery.

BF16 safetensors parameters/buffers use Merlin's existing
`runtime.captured_constants.verify_capture_constant` reader. Its implementation
and tests validate manifest argument/type/shape, the complete bounded file
partition and exact receipt-pinned payload bytes. No Python float or weight
repacking intervenes. Original weights, manifests, value archives and optional
word archives remain byte-identical in the derived package.

This differs explicitly from the standalone BF16 package: its primary NumPy
arrays already carry raw uint16 words. Bound models preserve the parent's F32
value files and consume the additional word witness or proved legacy embedding.
Both retain BF16 in portable MLIR and use the same native cast/transport support.
The derived receipt remains `t2m.bound-model-receipt.v1`, with the original m2m
receipt and metadata under `parent/`; no FX trace is invented for TTIR.

Compiler staging selects only actual compiler members. Tests prohibit reading
`tensor_words.npz` and verify that neither its values nor its digest appear in
the staged compiler artifacts. Actual source weights remain compiler data.
Native and Merlin reports bind every independent result-reference member;
the package verifier rechecks references against the typed final model outputs.

## Measured models and remaining requirements

The observation model makes two calls through one translated BF16 kernel using
an input, learned BF16 weight and BF16 buffer. All six ordered results survive:
two BF16 selections, two original integer-word views and two F32 computations.
Exceptional data includes subnormals, signed zeros, infinities and signaling
NaNs. The matmul model keeps BF16 operands, a learned nonzero F32 accumulator
and `linalg.matmul`; small integers make its independent reference exact without
assuming a reduction tree.

Both models pass all three memory profiles. Two explicit legacy-layout
projections also pass with unchanged source/weights/value files and finite
data. Fifteen protocol cases include scalar/buffer/lifted argument
materialization and malformed/unbound/contradictory archive refusals. A separate
rehashed NaN input corruption still fails the independent integer result
reference and cannot publish a successful package. These are 24 focused cases;
62 existing model/atomic/half regressions also pass. model2MLIR's capture checks
pass 30 cases, and Merlin's staging/constant-reader checks pass 50.

Scalar/lifted argument transport is protocol-tested; this does not remove
Merlin's existing general lifted-constant compiler-admission boundary. Dynamic
model ABIs, aliases/mutation, by-value BF16 calling conventions, broader numeric
policies and accelerator materialization remain completion requirements.

## Visible end-to-end packages

```bash
# Existing source-capture and frontend/native environments; set their paths.
/path/to/m2m-python scripts/capture_bfloat_model.py \
  --mode observe --out out/bfloat-observe-model/capture
T2M_MERLIN_ROOT=/path/to/merlin python -m t2m shared-demo \
  --kernel bfloat-observe --out out/bfloat-observe --python /path/to/native-python
python -m t2m bind-model out/bfloat-observe-model/capture \
  --bindings examples/model_bfloat_observe_binding.json \
  --out out/bfloat-observe-model/model --memory-profile guarded-buffer \
  --address-policy source-words --python /path/to/native-python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-python \
  --merlin-host --merlin-clang /path/to/clang
python -m t2m verify-bundle out/bfloat-observe-model/model
```

Set `TRITON_SHARED_OPT` to the existing shared optimizer. The matmul route uses
`--mode matmul`, `shared-demo --kernel bfloat-matmul`, and
`examples/model_bfloat_matmul_binding.json`. The saved observation/matmul model
bundles pass 81/80 artifact checks respectively, including actual Merlin host
execution. Accelerator execution remains unperformed.

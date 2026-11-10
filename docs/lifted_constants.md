# Lifted source constants through Merlin

Literal tensors captured as exported function arguments now reach the same
Merlin interface as parameters and buffers. Constants retain their source
placeholder, tensor type, shape, original bytes and prepared-graph provenance.
No Triton CPU backend or GPU reference is involved.

## Source package and explicit differences

model2MLIR `f98c39a6e0e096f66370737d5cb5960d2e322fdf` externalizes lifted
constants into `weights.safetensors`. Manifest entries use `kind: constant`,
the actual exported `name`, `source_target`, reserved weight key
`__m2m_lifted__/<target>`, original dtype/shape and exact payload SHA-256.
Actual source names matter: attribute literals may export as `c_literal`,
whereas generated literals may use `c_lifted_tensor_0`.

The additive receipt profile `m2m.compiler-constants.v1` binds every constant
to its function argument index, source placeholder/target, weight key,
dtype/shape and exact payload digest. Existing receipt schema and value
archive filenames remain unchanged. Legacy `extra.npz` values remain evaluator
data, including the optional exact BF16 word archive. Old captures without
source-bound constant storage remain explicitly unbound at compiler intake.
Recapture them rather than guessing source bits from widened evaluator data.

The derived package preserves every parent data file byte-for-byte, retains
the original receipt under `parent/`, and declares this source extension in
`compatibility.json`. Merlin `46c23a51c67dcf8de1dd5a3347dbd07fe40f3086`
checks the original prepared graph, actual argument types, unchanged public
parent ABI, manifest and complete safetensors layout. It emits the separate
compiler-only `merlin.compiler-constants.v1` projection in
`consumer/compiler/compiler-constants.json`. This grants byte custody; numeric,
effect, alias/lifetime, source equivalence and target gates remain separate.

Compiler staging never reads evaluator arrays or copies their value digests.
Rehashed receipts cannot excuse a mismatched source argument, role, shape,
type, payload or missing/duplicate binding. Constant stubs refuse. The C runtime
uses the ordinary exact weight blob/alignment path; it does not reconstruct
literal constants as floating-point C expressions.

## Executed profiles

The real literal-copy model covers f16/bf16/f32/f64, logical Boolean and signed
i8/i16/i32/i64 in tensor, buffer and guarded-buffer profiles. The independent
runtime input is also returned, checking the complete public model ABI.
Floating copy outputs preserve original signaling-NaN payloads, signed zeros
and subnormals byte-for-byte. Large i64 values retain individual low bits.
This qualifies transport, not arbitrary floating arithmetic. At zero tolerance,
integer reference comparisons also avoid lossy floating promotion.

Boolean byte-slot implementation patterns use the checked
[logical Boolean memory normalization](boolean_memory.md); raw byte
observations and noncanonical Boolean storage remain separate qualifications.

The lifted BF16 observation model makes two calls and returns six results,
retaining masks, selection, exact words and arithmetic. The lifted matmul model
retains BF16 inputs/weights, a nonzero F32 literal accumulator and
`linalg.matmul`. All three memory profiles execute through both native LLVM
and actual Merlin libraries. Tests live in `test_literal_models.py` and
`test_lifted_bfloat_models.py`; Merlin additionally checks malformed source
profiles, compiler/evaluator separation and exact runtime transport.
The final model run passes 33 cases. Boolean memory has 13 focused cases;
integer/reference regressions pass 36 checks. The source capture suite passes
48 cases, and Merlin's constant/staging/runtime suite passes 153 with two
optional cross-toolchain skips. Merlin documentation/structure gates pass.

## Visible observation demo

Use existing native/MLIR environments and the pinned standalone optimizer.
First create the TTIR with `shared-demo --kernel bfloat-observe` if it is absent;
the binding below refers to `out/bfloat-observe/source.ttir`.

```bash
/path/to/model2MLIR-python scripts/capture_bfloat_model.py \
  --out out/lifted-observe-model/capture --mode observe --lifted
python -m t2m bind-model out/lifted-observe-model/capture \
  --bindings examples/model_bfloat_observe_binding.json \
  --out out/lifted-observe-model/model \
  --optimizer .scratch/toolchain/triton-shared-opt \
  --memory-profile guarded-buffer --address-policy source-words \
  --python /path/to/native-python --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --merlin-host \
  --merlin-clang /path/to/compatible-clang
python -m t2m verify-bundle out/lifted-observe-model/model
```

The matmul route uses `--mode matmul`, the `bfloat-matmul` source and
`examples/model_bfloat_matmul_binding.json`. Use fresh output directories.
The locally saved final packages are `out/lifted-observe-model/qualified` and
`out/lifted-matmul-model/qualified`; they pass 82/81 artifact checks respectively,
including actual Merlin host execution and the explicit compatibility metadata.
Accelerator materialization/execution, dynamic/alias model contracts,
FP8/scaled-dot, broader numeric policies and external synchronization remain
completion requirements.

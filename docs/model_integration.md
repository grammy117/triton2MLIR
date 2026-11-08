# Binding translated kernels into model2MLIR

`t2m.model_binding.bind_model(model_mlir, [KernelBinding(...)])` consumes the
actual model2MLIR `prov.kernel` contract. It replaces typed custom-kernel calls
with translated standard tensor bodies and preserves the enclosing model ABI.
No backend kernel table, CPU Triton backend or GPU runtime is needed.

Binding checks the call/declaration identity, complete schema operand names,
tensor argument/result types and every scalar specialization. Source buffer names
can map explicitly to schema names; source results can map explicitly to return
order. Each call must select exactly one implementation. Missing/ambiguous
bindings, mismatched scalars and incomplete operand/result mappings fail.

Inlined operations retain their source provenance, with callsite-prefixed region
and emission IDs. A binding sidecar includes the original call attributes and
kernel provenance. Resolved private declarations are removed. Math outside the
kernel call remains ordinary model computation. Binding itself establishes typed
SSA validity; native numerical and consumer execution remain separate checks.

`scripts/capture_model_binding_demo.py` creates a genuine model2MLIR capture of
a `torch.library.triton_op` GEMM plus surrounding model math and learned weights.
Run it with an existing model2MLIR environment. Its registered PyTorch CPU
implementation supplies the independent golden; the actual Triton body is
captured and translated separately with triton2MLIR's pinned frontend. This
avoids invoking a Triton target backend and keeps both environments independent.

The model2MLIR registry currently rejects argument-mutating custom operations.
Fresh `write` results satisfy its functional interface after initialization
checks. General mutation/schema negotiation and downstream staged payload
support still need explicit capabilities; the binder does not silently change
model2MLIR's registry rules or certify a target's materializer.

## Checked model bundle and Merlin interface

The `bind-model` command reads `t2m.kernel-bindings.v1` JSON descriptions, translates
the supplied TTIR specializations and binds every registered call. It retains the
parent's real weights, manifest, runtime inputs, constants, input order and golden
byte-for-byte. A native check materializes arguments by **function argument index**,
using the manifest and `input_order.json` rather than treating weight positions as
runtime input positions, and compares the complete model result with the unchanged
PyTorch golden. The initial model-bundle profile requires one captured model output.

```bash
# Existing, independent environments; no GPU or Triton CPU backend required.
/path/to/m2m-env/bin/python scripts/capture_model_binding_demo.py --out out/model_capture
python -m t2m shared-demo --kernel loop-matmul --out out/shared_loop_matmul --python /path/to/native-env/bin/python
python -m t2m bind-model out/model_capture --bindings examples/model_binding.json \
  --out out/bound_model --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --rtol 1e-5 --atol 5e-6
python -m t2m verify-bundle out/bound_model
```

Set `TRITON_SHARED_OPT` to an existing optimizer for both conversion commands.
The optional Merlin check invokes its actual `forward_signature` reader and
`c_runtime.generate`: `model_gen.h`, `model_io.h`, `model_call.c` and `weights.bin`
are retained under `consumer/runtime/` and hash-bound by the derived receipt.
This measures real interface ingestion/runtime generation; target execution and
staged accelerator materialization are separate unperformed checks.

The new receipt is **`t2m.bound-model-receipt.v1`**. The original m2m receipt, model,
FX trace and metadata remain under `parent/`; unchanged data artifacts satisfy
the parent receipt without duplicating weight files. The transformed MLIR has its
own binding/native evidence and bundled TTIR implementations. Its parent FX trace
is never relabeled as correspondence for changed IR. Merlin paths requiring a
Torch-only capture receipt still need explicit derived-receipt admission; generic
model interface/runtime generation accepts the actual tensor ABI.

## Original MLP pair

`scripts/capture_mlp_pair.py` captures model2MLIR's existing `mlp_basic` and
`mlp_basic_triton` workloads, retaining their actual initialization and weights.
The test extracts the original standalone JIT function through Python AST,
captures it with the pinned Triton frontend, and binds it by schema names into
the captured model. The plain and Triton captures have byte-identical weights
and identical eager goldens. Both complete models pass native execution against
those goldens and the optional Merlin interface/runtime-generation check.

This tests the original two-layer fused MLP plus the surrounding third layer,
rather than only substituting a standalone matmul result for a model output.

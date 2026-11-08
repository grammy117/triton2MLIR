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

# Native execution evidence

`t2m.native.execute_native(mlir_text, inputs, work, python=existing_python)`
executes the **serialized standard MLIR** with an existing torch-mlir installation.
It verifies registered operations, bufferizes a temporary copy, lowers through
LLVM, then invokes the host JIT. The distributed `model.mlir` remains tensor IR.
No Triton target backend or device runtime is used.

The subprocess copies inputs, checks their declared shapes/dtypes, supports multiple
tensor results, and records its native provider/version and exact input-IR hash.
It retains the legalized LLVM module and actual outputs for inspection. Its generic
strided-copy callback supports the MLIR unranked-memref runtime ABI without building
runner-utils. Tests execute a genuinely strided copy and check multiple-result order.

Execution is limited to 30 seconds, one native thread, 64 MiB of input/results and
one million elements per generic runtime copy. The initial runner accepts f32/f64
and signed i8/i16/i32/i64 storage. Other types and missing native legalizations fail.
The subprocess releases JIT allocations on exit; this is a qualification tool,
not a persistent production runtime or a security sandbox.

Native results still need an independently supplied reference. A numerical match
is evidence for those inputs and the stated tolerance; it does not prove all
numeric policies, execute the original Triton source, or certify Merlin/device
materialization. Those statuses remain separate in the execution report.

# Native execution evidence

Native evaluator workers make C stdout unbuffered through the host libc
`stdout`/`setbuf` ABI before invoking compiled code. Upstream `cf.assert`
lowers to `puts` followed by `abort`; buffering a piped stream otherwise loses
the diagnostic. Failure exceptions retain bounded stdout/stderr text, including
literal format characters, quotes, newlines and UTF-8. This applies only to
owned evaluator subprocesses; portable IR and compiler inputs are unchanged.
The worker still aborts on failure and produces no successful invocation
receipt. Missing native stdio support is an explicit runtime refusal.

`t2m.native.execute_native(mlir_text, inputs, work, python=existing_python)`
executes the **serialized standard MLIR** with an existing torch-mlir installation.
It verifies registered operations, bufferizes a temporary copy, lowers through
LLVM, then invokes the host JIT. The distributed `model.mlir` remains tensor IR.
No Triton target backend or device runtime is used.

The subprocess copies inputs, checks their declared shapes/dtypes, supports multiple
tensor results, and records its native provider/version and exact input-IR hash.
Before bufferization, every tensor argument receives upstream's
`bufferization.writable=false` contract. This includes the incoming tensor
version of a source readwrite buffer: native code must return its new version
without changing caller-owned storage. MLIR inserts private copies where
required while retaining the original loops and tensor dataflow. The native
execution report records this adapter. Observed calls still check the actual
input bytes after one invocation, and bind the complete output archive, source
program and arguments; a mutation remains a failure. The contract follows LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`,
`FuncBufferizableOpInterfaceImpl.cpp::isWritable` and
`one-shot-module-bufferize.mlir` readonly-argument tests.
It retains the legalized LLVM module and actual outputs for inspection. Its generic
strided-copy callback supports the MLIR unranked-memref runtime ABI without building
runner-utils. Tests execute a genuinely strided copy and check multiple-result order.

Execution is limited to 30 seconds, one native thread, 64 MiB of input/results and
one million elements per generic runtime copy, with a 512 MiB data-allocation
limit in the native subprocess. The initial runner accepts f32/f64
and signed i8/i16/i32/i64 storage. Other types and missing native legalizations fail.
The subprocess releases JIT allocations on exit; this is a qualification tool,
not a persistent production runtime or a security sandbox. Temporary bufferization
permits source loops to return newly computed buffers; this does not change the
functional tensor ABI of the exported module.

Native results still need an independently supplied reference. A numerical match
is evidence for those inputs and the stated tolerance; it does not prove all
numeric policies, execute the original Triton source, or certify Merlin/device
materialization. Those statuses remain separate in the execution report.

# Execution-sized results: next target integration boundary

The shared lowerer and serial allocator seams are now implemented. Merlin
`bb347b1de` emits source/prepared MLIR, checked bridge and LLVM before runtime
linking; layout/index values remain explicit caller premises. Merlin
`f2552dec1` adds explicitly selected nonconcurrent context and external failure
reporting without changing native TLS/stdio defaults. Source/owner/product
guards and the default 64-bit descriptor contract remain active.

Actual cross model/bridge objects and a linked Spike runtime fixture now pass.
The successful standard-MLIR image returns all nine results across three
invocations; a second image rejects allocation exhaustion before results or
normal completion. The complete 152-check selection and two retained image
rechecks pass. These are runtime-seam results, not production receipt-v2/v6
target-driver qualification. The unchanged Spike package driver still has the
original receipt gate described below.

The next implementation must connect actual v6 source-only staging, invocation
input transport and generated typed calls to the existing target build/resource
owners. It must then report every actual result shape/payload under an explicit
readback profile and independently match complete original model references.
Reusing the binary frame transport is possible, but fixed-shape output metadata
cannot substitute for observed dimensions. Keep source validation, compiler
inputs and runtime/evaluator arrays separated throughout that integration.

The first reusable C seam is now implemented and exercised by
[checked C calls](checked_c_results.md): source-typed aggregate invocation,
complete descriptor/owner validation and independent logical copies. The real
three-kernel compaction model passes all seven results and source-domain guards
from a standalone C executable. This does not yet change Spike staging, linking
or production dynamic-result readback. The source-boundary inventory below
records the original probe; the implementation order identifies the remaining
integration requirements.

The real [compaction package](compaction.md) passes Merlin host qualification.
The unchanged Spike driver still rejects receipt v2 before staging. The actual
probe is retained in `.scratch/compact-v6-spike-probe.log`; it reports
`target execution requires a bound model or standalone numeric kernel package`.
This is a package/harness boundary, not a failed target lowering of the kernel.
No target image was built and no Spike execution is claimed by that probe.

## Current source boundaries

Source inspected at triton2MLIR `b77da69` and Merlin `e943745a3`:

- `scripts/qualify_spike_model.py::qualify` accepts only the older bound receipt
  and standalone numeric receipt. Its stage worker supplies no selected result
  mode, and its runtime signature dispatch lists v3/v4/v5.
- Merlin's `owned_result_abi.prepare` already reads actual ranked tensor/scalar
  inputs and tensor results. Its C bridge uses upstream's official returned
  descriptor aggregate convention; reuse it rather than guessing tuple offsets.
- `owned_result_abi.pipeline` already retains ownership/deallocation and checked
  allocator seams while omitting the preallocated out-parameter rewrite. It
  currently selects an explicit 64-bit host index contract.
- `owned_results.compile_owned_results` observes native layout and links a host
  shared image. Its Python executor checks returned descriptors against live
  allocator records before copying payloads. Those checks are required in a
  target harness too; loading a descriptor is not evidence that it is valid.
- `merlin/runtime/abi/owned_results.c` tracks invocation allocations and frees,
  charges alignment/tracking overhead and aborts for invalid ownership/budgets.
  Its current context is host thread-local. Bare-metal context selection needs
  an explicit execution contract; silently discarding isolation is invalid.

## Implementation order

1. Add a reusable generated-C executor for execution-owned results in Merlin.
   It must validate actual rank/static dimensions/storage, signed offset/strides,
   arithmetic overflow, alignment, accessible footprint and total output bytes
   against live invocation allocations before any result payload access. Empty
   results make no access; duplicate owners release once; unreturned live
   owners still refuse. Keep the existing Python checks as an independent
   boundary and test the C executor with deliberately corrupt descriptors.
2. Separate owned-result source preparation/lowering from host shared-image
   linking. Reuse the same upstream return ABI and complete ordinary lifetime
   gates. Index/pointer width, data layout and allocation context must come from
   the selected execution/toolchain contract. Initially qualify only actually
   observed supported layouts; another width requires its own implementation.
3. Extend generic runtime generation and the Spike driver with an explicit
   execution-owned result mode. Reopen receipt v2, located kernels and complete
   independent SSA correspondence; select source-only signature v6 staging.
   Old/default modes must refuse it rather than preallocate guessed outputs.
   Runtime argument arrays may initialize a target invocation, but must not
   enter compiler staging or decide output sizes or compiler transformations.
4. Extend target readback to carry actual result shapes and payloads. Bind rank,
   storage, shape products, payload sizes and complete result order to the
   checked source signature and actual execution observations. Do not reuse a
   static-shape wire declaration with guessed dimensions. Preserve existing
   fixed-shape formats and make any new wire/profile difference explicit.
5. Qualify the committed internal-count model and real TTIR compaction on the
   actual selected target image. Reopen complete original PyTorch references,
   learned parameter/input words and all seven results. Include empty, sparse,
   full and overflowing-i32 examples, plus actual source-domain/allocation and
   unwritten-result failures. Target success must come from real image execution,
   not a host report or generated harness alone.

## Independent acceptance requirements

The generated-C executor must first work on the host with fixed saved compiler
products and changed inputs. Tests must prove one source execution per
invocation, output size discovery after computation, valid empty/strided views,
duplicate returned owners and independent result copies. Invalid descriptors,
allocation overflow, unknown frees and exhausted budgets must fail before
payload access or successful publication. Source locations and provenance
remain intact through preparation; no annotation grants ownership or numeric
permission.

Target integration must retain input preservation, complete observed result
types/shapes/words, all original scalar/domain guards, checked allocation
budgets and byte-bound compiler/tool/runtime artifacts. Package compatibility
metadata must distinguish the selected target/result transport from the
existing host-only qualification. No v6 target status may be inherited from
older v3/v4/v5 packages or the historical opcode inventory.

Runtime/zero-trip resizing ownership, general physical/mutating views,
unrestricted numeric/effect/source compositions and accelerator performance
remain separate full-goal obligations. This plan does not weaken their gates.

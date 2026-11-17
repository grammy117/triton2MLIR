# Execution-sized results: target integration

The [execution-owned target package](execution_owned_target_packages.md) now
connects actual signature-v6 source-only staging, generated typed calls, serial
allocator context, ordinary Merlin model building and complete observed-shape
readback to real RISC-V/Spike execution. Four qualified whole-model images retain
all 28 original/changed-invocation results. Original source-domain and complete
reference failures cannot publish success. Retained verification replays independent
source staging, driver reconstruction and complete input/wire/reference checks.

The [generated hosted driver](owned_result_driver.md) separately retains all
84 results across twelve changed-content invocations and two original source
guard failures. Merlin's current 142-check runtime selection covers production
image execution, zero-input functions, actual arena exhaustion, strict readback,
legacy invocation, compiler/index/ISA selection and memory admission.

These are explicitly selected functional serial execution profiles. Public source
MLIR and parent archives remain unchanged. No target mapping is inserted into
TTIR conversion. Accelerator correctness/performance, broader effects, physical
views and runtime/zero-trip resizing ownership remain open.

## Remaining integration scope

- Extend resizing ownership without weakening ordinary lifetime checks.
- Admit general physical/view and alias/effect combinations with executable
  contracts; provenance supplies correspondence, not permission.
- Implement additional executor/console/device/numeric options instead of
  discarding selected options. The current v6 image builder refuses them.
- Qualify every applicable source/type/attribute composition and target against
  complete independent references. Selected images and opcode names do not
  establish arbitrary-kernel completion.

## Initial source boundaries, before this implementation

The following source inventory and sequence record the original failing probe.
The explicit execution-owned route above now implements the serial package
integration; the default legacy route retains its original receipt requirements.
`.scratch/compact-v6-spike-probe.log` is historical refusal evidence, not a current
claim that the selected v6 path cannot build an image.


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

## Original implementation sequence

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

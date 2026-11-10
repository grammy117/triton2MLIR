# Original integer words with portable pointer-offset dataflow

Use `--memory-profile guarded-buffer --address-policy source-words` on
`shared-tensor`, `shared-demo` or `bind-model`. The Python API exposes the same
`address_policy="source-words"` argument. A binding JSON entry can select its
own address policy. The default remains `structured`; no failed range proof
silently changes policy.

This emits the explicit `shared-source-word-buffer` package profile. The public
functional tensor/scalar ABI and standard MLIR vocabulary stay the same as
[guarded memory](guarded_memory.md), including runtime bounds, initialization,
reader/writer ownership and lane-conflict checks. Physical pointer observations
and unsupported effects remain located refusals.

[Strided source allocations](strided_memory.md) additionally qualify padded,
signed/reversed, column-major and broadcast readonly addressing with runtime
capacities/origins. Owning input spans retain their padding bytes; the tiled
contraction and original K reuse remain inside the standard-memory payload.
This does not supply a pure logical-view witness or cross-allocation alias proof.

```bash
python -m t2m shared-demo --kernel indirect-store \
  --memory-profile guarded-buffer --address-policy source-words \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python \
  --out out/source-word-indirect
```

`shared-demo --kernel pointer-while` with the same flags emits a runtime-shaped
pointer/accumulator loop package, including its final pointer load. Pass that
bundle to `qualify-merlin` for the separate real compiler/library qualification.

## Address semantics and reused passes

The structured policy checks source ranges before pointer analysis promotes
expressions to `index`. The source-word policy instead keeps the original
integer arithmetic and uses the official
`triton-to-unstructured="offset-bit-width=64"` pass. Its initial offset is i64;
each narrower `tt.addptr` operand is sign-extended into that accumulator after
the source computation. Original i32 multiplication, addition, comparisons and
casts retain their types. This permits defined word wraparound that cannot be
represented by simply widening all arithmetic.

For example, an i32 loaded key `(2^30 + q)` multiplied by four wraps to `4*q`
before conversion to a pointer offset. Separately, two i32 pointer increments
followed by an i64 cancellation must accumulate as pointer offsets, rather
than wrap together as an i32 sum. These are different operations and both are
retained. The existing structured width gate remains unchanged.

Source pin: Meta triton-shared `17f0c234ce474bee334af36e351c62eca7852836`:

- `include/triton-shared/Conversion/TritonToUnstructured/Passes.td` declares the
  offset-width option.
- `lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp` implements
  typed offset accumulation and source `scf.for` pointer-state conversion.
- `test/Conversion/TritonToUnstructured/test_offset_bitwidth.mlir` checks the
  64-bit option, retaining original narrow arithmetic and extending pointer
  operands at their boundaries.
- `scf_for_ptr_in_iterargs.mlir` and `gather_reuse_loop_results.mlir` in that
  test directory cover carried offsets and final pointer results.

The pinned [Triton pointer lowering](https://github.com/triton-lang/triton/blob/142586f90aac9b917af1081e4697b085027e13b2/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp)
uses a GEP for each source pointer addition. The policy's software offset domain
is i64; it introduces no GPU address space, lane layout or hardware launcher ABI.
The native converter remains upstream. A narrow control-state adapter below
supplies missing region edges. The optimizer executable has its separately
recorded pin, as described in [the reuse map](triton_shared.md).

## Pointer-valued structured control

The pinned unstructured converter handles pointer-valued `scf.for` state but
does not reconcile pointer-valued `scf.if` or traverse `scf.while`. Its structured
converter has useful branch analysis, including
`test/Conversion/TritonToStructured/if_simple_addptr.mlir` and
`if_returning_mixed_structured_ptrs.mlir`, but does not supply this source-word
path with a qualified logical-buffer boundary for those cases.

`t2m/pointer_control.py` therefore runs before the unchanged native source-word
pipeline when those control edges are present:

1. Propagate source-entry buffer identities over pointer construction, shape
   transformations, selections, branch yields, loop initial values and every
   backedge to a fixed point. Every pointer must have a finite set of declared
   bases on all paths and retain its element type. Fixed-base state uses just
   the offset; mixed-base state adds the [logical selector](selected_pointers.md).
2. Represent pointer-valued region arguments/results as scalar/tensor i64
   element offsets. Follow `scf.condition` and `scf.yield` separately; while
   input/output state can have different arities. Preserve source predicates,
   region structure, value-state order and original narrow integer computation.
   Only the operand at each original pointer addition is sign-extended.
3. Reconstruct a local base-plus-offset pointer at each original load/store.
   The upstream converter then owns gather/scatter and arithmetic conversion;
   the existing guarded memory continuation checks the original active accesses.
   Selected bases resolve at the original memory event, preserving intervening
   writes, per-lane masks/fills and changing loop bases. Uniform selectors stay scalar.

The transformation report retains the affected source IDs and base/width
contract. `word-input.mlir` binds the exact resulting input to the native
stage. Original TTIR and its typed inventory remain retained independently.
Unknown pointer producers/observations, pointee-changing casts and address
escapes refuse; intent annotations do not bypass these checks.

## Memory inside structured regions

The pinned binary can also leave branch-local pointer preparation unresolved
when no pointer-valued result crosses the branch. `word_events.py` supplies a
version bridge for memory under `scf.if`/`scf.while`, including nested `for`:

1. The existing pointer-state analysis exposes entry bases plus exact i64
   offsets, retaining the computations producing each source word.
2. Every source load/store becomes a separate typed, straight-line template.
   One batched native `source-word-events` stage runs the official unstructured
   converter. No pointer-valued helper calls are inserted in the parent.
3. Exactly one converted gather/scatter replaces each event at its original
   position. Captures and results are checked structurally across parser
   contexts. Regions, branches, loop state and memory ordering stay intact.
4. A masked load with an explicit fill uses local zero in the template, then
   selects the original scalar/tensor fill with the original mask. This avoids
   upstream scalar-fill inference without changing nonuniform fills or their
   bits. Missing source fills retain the existing upstream refinement policy.
5. The parent runs `source-word-compute`, an arithmetic-only stage. Running the
   pointer pass again on already-unstructured events is unsupported by the
   pinned converter and is deliberately excluded.

The pinned source is Meta's `TritonToUnstructuredPass.cpp` and
`test/Conversion/TritonToUnstructured/gather_no_loop.mlir` at the revision above.
The adapter adds region placement and typed correspondence; native pointer and
arithmetic lowering remain upstream. It supplies no hardware mapping or opaque
portable calls. Source-word arithmetic, guard protocol and public ABI are unchanged.

Real scalar/tiled branches and runtime pointer/accumulator while loops now
execute natively and through Merlin. One unchanged library handles zero trips,
different loop counts, empty/tail dimensions and final pointer observations.
Actual invalid loop accesses abort. Nested branches and pointer-returning helpers
also qualify inside `for`/`while`, preserving runtime choices, accumulators and
final observations. Consumer conditional destination planning validates every
path before edits: complete computations reuse unused destinations, and captured
or closed pure computed values use explicit value copies. Reductions and live
original init reads retain their original bodies/order. Native alias/lifetime
checks remain mandatory. The while condition must forward complete state
unchanged; computed/permuted condition state needs separate ownership support.
Source word wraparound and wide cancellation remain checked across branches.

## Granularity and source intent

This policy represents accesses as offset tensors and parallel linalg gather/
scatter bodies rather than recovered structured slices. It retains source tile
shapes, masks/fills, program-grid arithmetic, contractions, recurrence state and
loop results. The logical grid remains an unmapped `scf.forall`. Source-base
reads become actual memory loads before optimization, as in guarded memory.

`prov.address_policy` and `prov.source_accesses` on the function and logical
grid record the policy and original access intent. The grid annotations survive
model-call inlining. Existing source IDs, exact TTIR, typed operation inventory
and normalized-source correspondence remain separate from these descriptions.
The attributes grant no rewrite, memory or numerical semantics; removing them
does not remove the executable checks or computations.

The structured path remains useful for directly visible slices and views.
The source-word path preserves irregular and nonlinear address computation
without an affine-proof requirement. This representation difference is explicit;
it does not claim equal optimization quality or discard tiling and reuse.

## Package and consumer evidence

In addition to guarded memory's four artifacts, the receipt binds:

| Artifact | Meaning |
|---|---|
| `word-input.mlir` | Exact inlined/intake-normalized input before the source-word stage. |
| `word-tensor.mlir` | Exact native upstream output before guard insertion. |
| `word-stage.json` | V1 native stage report, or explicit `t2m.source-word-stages.v2` chain with both actual native reports, event template input/output, typed bindings and parent computation input. |

Admission/provenance bind the policy and stage hashes. Merlin independently
checks the inventory, byte relations, declared pipeline and access-intent
consistency, and copies the evidence into compiler-only staging. Static model
bindings retain it under each kernel. Evaluator arrays and goldens stay outside
the compiler view. These checks establish custody/consistency, not universal
source equivalence. Native execution, independent references and actual Merlin
library execution remain separate evidence.

For v2, bundle validation reconstructs the complete template inventory from
`word-input.mlir`, substitutes the converted events with checked types at the
original positions, and compares the resulting parent bytes to the actual
compute input. Hashes alone cannot replace this reconstruction. Merlin checks
the declared version, byte/pipeline chain, same optimizer identity and source
binding inventory, retaining the existing custody-only boundary. The existing
three artifact names are retained; package differences explicitly name v2.

New scalar/tiled `for`/`while` tests reuse actual Merlin libraries over false
branches, negative/zero trips, empty/ragged domains and repeated observations
of writable state. Native and Merlin outputs are separately compared as raw
f32 words for signed zero, NaN payloads, infinities and nonuniform fills. These
cases do not establish other numeric/effect profiles or device execution.

A real model2MLIR GEMM capture additionally binds branch-local tiled GEMM
through the v2 path. Actual Merlin execution retains contractions and K loops,
compares independent PyTorch goldens, and preserves original model weights and
evaluator artifacts. Standalone and derived packages share the same evidence
checks; no FX trace is invented for the Triton implementation.

Tests cover scalar/tiled i32 wraparound, narrow pointer chains with i64
cancellation, runtime pointer recurrences and final pointers, zero trips,
empty/tail dimensions, nonlinear grid permutations, actual conflicts and
corrupted evidence. One Merlin library runs independent encoded words and
runtime shapes; another retains changing pointer-loop counts. Real model2MLIR
GEMM captures retain learned parameters, surrounding model math, contractions
and loops through the same interface, including block/descriptor intake.

Broader while ownership profiles, alias groups,
physical pointer observations and broader numeric/effect profiles remain open.
This policy expands address coverage; it is not full TTIR coverage.

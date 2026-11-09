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

```bash
python -m t2m shared-demo --kernel indirect-store \
  --memory-profile guarded-buffer --address-policy source-words \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python \
  --out out/source-word-indirect
```

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
   backedge to a fixed point. Every pointer must have exactly one base on all
   paths and retain its element type. A mixed-base branch or backedge refuses
   before native lowering; it needs an explicit buffer-selector representation.
2. Represent pointer-valued region arguments/results as scalar/tensor i64
   element offsets. Follow `scf.condition` and `scf.yield` separately; while
   input/output state can have different arities. Preserve source predicates,
   region structure, value-state order and original narrow integer computation.
   Only the operand at each original pointer addition is sign-extended.
3. Reconstruct a local base-plus-offset pointer at each original load/store.
   The upstream converter then owns gather/scatter and arithmetic conversion;
   the existing guarded memory continuation checks the original active accesses.

The transformation report retains the affected source IDs and base/width
contract. `word-input.mlir` binds the exact resulting input to the native
stage. Original TTIR and its typed inventory remain retained independently.
Unknown pointer producers/observations, pointee-changing casts and address
escapes refuse; intent annotations do not bypass these checks.

Real scalar/tiled branches and runtime pointer/accumulator while loops now
execute natively and through Merlin. One unchanged library handles zero trips,
different loop counts, empty/tail dimensions and final pointer observations.
Actual invalid loop accesses abort. The consumer reuses complete after-region
destinations only when the while condition forwards the complete state unchanged;
computed/permuted condition state needs separate consumer ownership support.
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
| `word-stage.json` | Native stage report binding those input/output bytes and the explicit pass sequence. |

Admission/provenance bind the policy and stage hashes. Merlin independently
checks the inventory, byte relations, declared pipeline and access-intent
consistency, and copies the evidence into compiler-only staging. Static model
bindings retain it under each kernel. Evaluator arrays and goldens stay outside
the compiler view. These checks establish custody/consistency, not universal
source equivalence. Native execution, independent references and actual Merlin
library execution remain separate evidence.

Tests cover scalar/tiled i32 wraparound, narrow pointer chains with i64
cancellation, runtime pointer recurrences and final pointers, zero trips,
empty/tail dimensions, nonlinear grid permutations, actual conflicts and
corrupted evidence. One Merlin library runs independent encoded words and
runtime shapes; another retains changing pointer-loop counts. Real model2MLIR
GEMM captures retain learned parameters, surrounding model math, contractions
and loops through the same interface, including block/descriptor intake.

Mixed-base pointer tensors, broader while ownership profiles, alias groups,
physical pointer observations and broader numeric/effect profiles remain open.
This policy expands address coverage; it is not full TTIR coverage.

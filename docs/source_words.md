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
No pointer converter is reimplemented or vendored. The optimizer executable has
its separately recorded pin, as described in [the reuse map](triton_shared.md).

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

Mixed-base pointer tensors, general pointer-valued branches/while, alias groups,
physical pointer observations and broader numeric/effect profiles remain open.
This policy expands address coverage; it is not full TTIR coverage.

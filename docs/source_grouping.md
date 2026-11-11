# Source-specialized packed tensor dataflow

Coupled and asymmetric `tt.map_elementwise` computations now have an explicit
source-layout path to standard MLIR and actual Merlin host libraries. This is
separate from the default [slot-independent proof](packed_maps.md): the source
layout is part of the computation's specialization, not a placement hint.

## Meaning and upstream boundary

The pinned Triton LLVM map conversion groups consecutive per-thread register
values. They need not be consecutive tensor coordinates. For a real rank-one
256-element map, the initial four-warp/32-thread specialization pairs elements
128 positions apart; two warps produce different groups. Inferring adjacency
would change both cross-slot arithmetic and asymmetric output placement.

The frontend runs Triton's actual `add_convert_to_ttgpuir` pass, then uses the
native Gluon builder's tensor-layout and `to_linear_layout` APIs. Native
`LinearLayout` supplies the forward mapping and inverse. No encoding text,
layout picture or printed operation line is parsed for semantics. The installed
Triton version and exact native-library bytes are recorded; dependency source
closure remains unestablished.

This boundary is **the caller-selected initial TritonGPU specialization**. It
does not claim to recover a final optimized GPU layout or establish equivalence
to an unobserved GPU binary. Changing the selected source layout can change
coupled/asymmetric math. The frontend never silently selects one to make an
otherwise ambiguous map pass. No device code generation, launcher or GPU
execution is invoked.

Source references at Triton `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`:

- [Map lowering and operand/result packing](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L572-L640).
- [Native tensor-layout conversion bindings](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/src/gluon_ir.cc).
- [Native linear-layout APIs](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/src/linear_layout.cc) and [their tests](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/test/unit/tools/test_linear_layout.py).

## Portable computation and checks

The source pass retains owned map identities. All operands/results must have
the same logical shape and actual coordinate mapping. Pack size must divide
the observed register value count. The current profile requires a bijection;
both compositions of the forward/inverse maps are checked on their binary
bases, without enumerating tensor elements. Replicated layouts remain refused.

The emitter creates unmapped parallel logical groups. Each group extracts its
actual source slots and runs the complete original scalar region once, keeping
cross-slot reuse, branches, loops, result order and unused control execution.
It writes one complete row of each grouped result tensor. Parallel Linalg then
restores ordinary tensor coordinates with the checked inverse. No scalar body
is repeated once per output slot and no physical thread/warp is assigned.

The guarded-memory adapter admits these parallel updates only after inspecting
standard SSA: one IV owns a complete row, every output is updated once, bounds
match output extents, and the body is value-only with no external effects or
physical memory. Overlapping rows, duplicate updates, mismatched domains and
external calls refuse. Source buffer guards, ownership and native lifetime
checks remain in force. An annotation cannot authorize this admission.

## Interface and custody

Use the existing API's optional `source_layout` dictionary or the public CLI's
`--source-layout path.json`:

```json
{"target":"cuda:80","num_warps":4,"threads_per_warp":32,"num_ctas":1}
```

Every field is explicit. Integer geometry is bounded before the native worker;
the source pass has a 30-second timeout and bounded output. Model binding JSON
may supply the same `source_layout` object per kernel specialization.

`source_layout_specialization` is an additive provenance field containing the
exact annotated/inlined source, specialized TTGIR, source options, native
implementation identity and map-coordinate facts. Hashes bind both source
forms. Transformation records must match the complete witness roster. Package
and model-kernel verification checks that contract without requiring Triton or
executing tools named by saved JSON. This is byte custody and checked
coordinate structure, not universal source-equivalence proof.

The public tensor/scalar ABI and receipt schemas stay unchanged. Compatibility
metadata names the specialization. The emitted payload contains standard
tensor/SCF/Linalg/arithmetic; the TTGIR appears only in provenance. Consumer
placement, storage and target legality remain downstream decisions.

## Measured qualification and remaining work

Seven actual native/Merlin libraries qualify i64/f32, pack two/four, rank-one/two
tiles, differing source warp counts, all three memory profiles and a coupled
zero-divisor branch/while recurrence. Six libraries replay 24 additional empty,
scalar, full-tile and multiprogram tail domains. References retain wide integer
words and all results; functional input preservation is checked.

Twenty-three direct matrix/witness/parallel-row checks and the existing
59 packed/unpacked/memory-map regressions pass. A concrete public CLI bundle
and its Merlin qualification are retained at `out/source-grouped-map-demo/`.
Broader kernel/type/attribute coverage, packed source memory, replicated layouts,
final optimized-layout correspondence and applicable target execution remain
open. CPU host qualification supplies no accelerator or performance evidence.

# Source-specialized packed tensor dataflow

Coupled and asymmetric `tt.map_elementwise` computations now have an explicit
source-layout path to standard MLIR and actual Merlin host libraries. This is
separate from the default [slot-independent proof](packed_maps.md): the source
layout is part of the computation's specialization, not a placement hint.

[Packed source reads and pointers](grouped_memory.md) additionally use that
contract with the existing guarded/source-word memory path. One parallel scalar
region computes all slots before coordinate restoration; read snapshots and
per-slot root selectors remain executable.

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

## Captured model contract

A source specialization can change the mathematical function. A bound model
therefore requires `source_layout_arguments`, mapping all four source options
to distinct original `prov.kernel_args` scalar names. The selected values must
equal those original typed attributes: target is a string; geometry is i64.
Missing, changed or mistyped selections refuse. This is semantic specialization,
not a way to choose consumer hardware.

The additive `t2m.captured-source-layout.v1` profile records the complete mapping
in `kernel-bindings.json`. Bundle verification and Merlin compiler staging
independently reopen the original parent call. Kernel provenance must match its
binding copy. Public model ABI/data and existing receipt schemas stay unchanged;
the new profile and its custody requirements are explicit package differences.
Readers predating this profile do not provide this additional validation.

`scripts/capture_grouped_map_model.py` creates real model2MLIR captures with
learned weights, two dependent calls and four results. Its independent PyTorch
reference defines the declared initial-stage grouping directly; it reads no
frontend witness or emitted IR. These captures qualify that semantic profile,
not the final GPU launch suggested by the recorded Triton body.

For the i64/two-warp arithmetic example, capture the shipped kernel first:

```python
from pathlib import Path
from t2m.examples.grouped_map_kernels import grouped_map
from t2m.frontend import capture_ttir
out = Path("out/source-grouped-map-kernel")
out.mkdir(parents=True, exist_ok=False)
names = ("x", "y", "out_mix", "out_asymmetric")
(out / "source.ttir").write_text(capture_ttir(grouped_map,
    {**{name: "*i64" for name in names}, "n": "i32", "B": "constexpr",
     "PACK": "constexpr", "MATRIX": "constexpr"},
    {"B": 256, "PACK": 2, "MATRIX": False}))
```

Then use the explicit [binding configuration](../examples/source_grouped_map_binding.json)
and existing independent Python environments:

```bash
/path/to/m2m-python scripts/capture_grouped_map_model.py \
  --out out/source-grouped-parent --dtype i64 --warps 2
python -m t2m bind-model out/source-grouped-parent \
  --bindings examples/source_grouped_map_binding.json --out out/source-grouped-bound \
  --python /path/to/native-python --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --merlin-clang /path/to/clang --merlin-host
```

Use the ordinary [target qualifier](target_execution.md) for functional RISC-V
execution. It does not feed source geometry into consumer placement.

## Measured qualification and remaining work

Seven actual native/Merlin libraries qualify i64/f32, pack two/four, rank-one/two
tiles, differing source warp counts, all three memory profiles and a coupled
zero-divisor branch/while recurrence. Six libraries replay 24 additional empty,
scalar, full-tile and multiprogram tail domains. References retain wide integer
words and all results; functional input preservation is checked.

Twenty-three direct matrix/witness/parallel-row checks and the existing
59 packed/unpacked/memory-map regressions pass. A concrete public CLI bundle
and its Merlin qualification are retained at `out/source-grouped-map-demo/`.
Three complete i64/f32 captures additionally qualify two/four source warps,
cross-slot/asymmetric arithmetic and coupled branch/while state. Their 300-element
domains exercise complete packs, multiple programs and ragged tails. All twelve
results match exact independent reference words through native LLVM, Merlin host
and actual RISC-V/Spike execution. Original weights, inputs and references remain
unchanged. The target selection is `out/spike-source-grouped-models/summary.json`;
the public binding/target CLI example is `out/source-grouped-model-demo/`.
Missing/mismatched selections and changed binding witnesses refuse; Merlin's
81-case derived-model suite includes typed-call and rehashed-profile refusals.

Broader kernel/type/attribute coverage, wider packed-memory profiles, replicated layouts,
final optimized-layout correspondence and broader target execution remain open.
Functional CPU qualification supplies no accelerator or performance evidence.

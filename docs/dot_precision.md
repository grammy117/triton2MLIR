# F32 dot precision and composites

F32 `bf16x3` and `bf16x6` now reuse Triton's actual `F32DotTC` pass directly
on pre-layout TTIR. It requires no GPU layout, target compilation or backend
runtime. Shared continues to own arithmetic/memory conversion; the existing
tensor/scalar ABI, logical grid, source K loop and tile reuse remain intact.
TF32/TF32x3 additionally have an explicit logical E8M10 numeric profile with
standard arithmetic expansion; no device opcode remains in the portable body.

## Executable algorithm and reuse

Read and tested against Triton
[`F32DotTC.cpp`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp)
and [`bf16x3-matmul.mlir`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/test/TritonGPU/bf16x3-matmul.mlir)
at `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`. The installed API is
`passes.ttgpuir.add_f32_dot_tc(manager, False)` in pinned-version Triton 3.7.1.
The adapter records the actual native library's byte count/SHA256 and the
normalized output hash. The referenced source revision is explicitly separate
from a build receipt; no binary/source identity is invented.

The pass splits each f32 operand into successive BF16 parts using RTNE casts,
exact widening and f32 residual subtraction. BF16x3 evaluates `(mid,high)`,
then `(high,mid)`. BF16x6 first evaluates `(mid,mid)`, `(low,high)`, `(high,low)`,
then those two terms. It replaces correction NaNs with zero, evaluates the
`(high,high)` contraction, then adds original C. Every partial contraction
retains its own accumulator through shared's standard conversion. C remains
after the composite as specified by the official algorithm. Source outer-loop
rounding and zero-trip behavior stay executable.

Operand splits are shared across three/six named `linalg.matmul` operations.
No full-tensor replacement, opaque helper or physical matrix instruction is
introduced. Original effect/numeric admission runs before greedy normalization;
the native precision worker also refuses volatile reads. A negative original
imprecise-accumulation bound cannot disappear into generated default bounds.
Other original accumulation bounds stay in source intent and grant no new
arithmetic approximation.

## Provenance and numeric boundary

Original and normalized TTIR remain distinct typed graphs under the existing
v2 provenance schema. Original operation records retain the selected precision,
C value identity, input types and StringAttr caller intent. Function and named
contraction attributes carry `prov.kernel_original_dot_precisions`: a **kernel
source inventory**, not a claimed per-operation original-source bijection.
Normalized contraction IDs and memory facts refer to their own graph.

The selected host uses standard f32 products and ascending-K additions with
gradual underflow. Independent rational BF16 rounding, f32 residuals, products,
partial accumulations, NaN clearing and final C addition supply references.
Exact non-NaN words and NaN class are checked separately. This does not qualify
every GPU's Tensor Core accumulation order, subnormal/FTZ behavior, rounding
or performance. A capable target must establish its own matching numeric
profile before selecting fused physical instructions. Compatibility metadata
records this implementation boundary; public tensor storage/ABI is unchanged.

## TF32 numeric profile

`logical-tf32-e8m10-rtz-v1` explicitly selects ten fraction bits with f32 range,
gradual underflow, toward-zero dot-input quantization and computed NaN class.
The integer expansion preserves signs, including zero, and recognizes NaNs
before clearing fraction bits. A small NaN payload cannot turn into infinity.
Direct TF32 keeps original C **inside** its single named contraction.

TF32x3 reuses the same pinned native pass with `emuTF32=True`. Its
[`tf32x3-matmul.mlir` test](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/test/TritonGPU/tf32x3-matmul.mlir)
establishes partial-dot order and correction handling. The adapter expands only
the pass-generated pure, pack-one `cvt.rna.tf32.f32` with matching types and
constraints. High parts round to nearest with ties away from zero; f32
subtraction produces residuals, whose dot inputs then truncate to E8M10.
The two ordered cross terms, NaN clearing, high/high term and final original-C
addition remain executable. Original source assembly still needs its separate
qualification and refuses **before** the pass; identical text in source does
not acquire generated-conversion semantics.
Conversions of the same value are shared within their block; generated high
parts already have the selected format and are reused directly. This preserves
operand reuse without moving conversions across branch or loop scopes.

The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
defines RNA rounding, leaves the internal TF32 representation implementation
defined and leaves several matrix-accumulation details unspecified. Its WGMMA
contract explicitly truncates the lower thirteen input bits. These references
motivate this logical profile; they do not prove bit equivalence to every
Triton backend. Portable values remain f32 carriers with quantization explicit
in standard operations, rather than an invented native MLIR TF32 type.
Metadata records this choice, source precision and source C placement.
Physical matrix instructions need their own numeric/target qualification.

## Evidence and remaining work

Six real runtime packages cover both modes in tensor, buffer and guarded-buffer
profiles. Each qualified Merlin library replays changing/tail/empty shapes,
K=0, padded K=65 and nonzero C. A cancellation-sensitive C=`2^24` case checks
that original C was not moved into a partial contraction. Source NaNs,
infinities, BF16/f32 underflow/overflow boundaries and signed zeros have
independent references. Two original-bound refusals and a volatile-read refusal
complete the focused checks. The broader 83-case precision/block-pointer/FP8
run passes, with 24 separately qualified FP8 shape cases deselected.
The TF32 extension adds six runtime packages across the same three profiles,
including finite ties/residuals, identity operands, subnormal/overflow/NaN
boundaries, zero-trip and ragged domains, and changing shapes on reused Merlin
libraries. Independent exact rational E8M10 rounding supplies references.
Original assembly, volatile reads and invalid bounds remain refused. Separate
rank-two/rank-three normalization checks compose BF16/TF32 modes while retaining
ignored f16 precision flags; these checks alone do not qualify batched execution.
The combined evidence covers 36 focused cases, with twelve relevant cases
rerun after the reuse change and 75 separate block-pointer/FP8 regressions.
The explicit 34-package report remains 30/50 observed TTIR definitions and
26 in host-qualified captures; generated assembly is not original-op coverage.

```bash
python -m t2m shared-demo --kernel bf16x6-dot --out out/bf16x6-dot \
  --python /path/to/native-python
python -m t2m qualify-merlin out/bf16x6-dot --out out/bf16x6-dot-merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`bf16x3-dot`, `tf32-dot` and `tf32x3-dot` select the other modes.
The visible demo uses 16×16×32 tiles and
runtime M/N/K=17/19/37. Compiler staging never reads reference archives.
Broader accumulators, batched/runtime compositions, reserved
scaled FP6 storage, external/effect/alias profiles, corpus closure and applicable
target materialization remain in the full [completion plan](completion_plan.md).

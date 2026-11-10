# Portable scaled contractions

`tt.dot_scaled` expands into logical unpack/scale tiles, a standard contraction
with original C, and explicit result rounding. Public tensor/scalar ABI and
receipt schemas are unchanged. `prov.scaled_dot_contracts` and transformation
reports retain formats, packing, logical shapes, scale groups, C identity and
numeric profile. Annotations grant no arithmetic or physical mapping permission.

## Sources and reuse boundary

Pinned Triton `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`
[`TritonOps.td`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/Triton/IR/TritonOps.td),
`Ops.cpp`, `semantic.py` and `test_matmul.py` define operand segments, logical
dimensions and scale layouts. The enum contains seven formats: E4M3, E5M2,
E2M3, E3M2, E2M1, BF16 and FP16. The public language exposes five. Reserved
FP6 enums lack a pinned Triton storage/packing contract and still refuse.

Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7`
[`DotScaledConverter`](https://github.com/triton-lang/triton-ascend/blob/8f0ade66580245e30ad97093526a4018e4e18fc7/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp)
and `dot_scaled_optional_{lhs,rhs}.mlir` / optional-scale Python tests establish
the useful pattern: preserve packed operands, use unit scale when absent,
retain C and emit a contraction. Its FP4/FP8 path uses `hfusion.matmul_mx`,
which needs a standard expansion here. Triton's `DecomposeScaledBlocked`
requires GPU layouts and rejects M/N packing. We retain shared's source
verification, inlining, memory and region passes; the local adapter supplies
the missing computation boundary without a new backend or launch convention.

## Explicit numeric profile

The selected implementation is `logical-mx-f64-v1`. OCP MX v1.0 sections
6.1–6.2 leave internal precision and operation order implementation-defined.
See the [normative specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
The profile decodes/scales operands into f64, extends source C before the
contraction, computes f64 products/accumulation, then rounds once to the
declared result format. The selected host reduces in ascending logical K;
provenance grants no reassociation or alternate accelerator precision.
F16/BF16/f32/f64 accumulator/result formats are retained. BF16 extension uses
its exact f32 bridge. F64-to-16-bit results reuse the existing round-to-odd
bridge to avoid double rounding and unavailable runtime helpers.

This is an explicit implementation choice within MX's numeric boundary. It
does not reproduce Triton's target-dependent f16/BF16 software emulation or
claim bit identity with every GPU instruction. Compatibility metadata states
that difference. A target may fuse the composite after establishing its
numeric contract. F64 is the portable reference expansion; hardware support
and performance remain separate materialization requirements.

E8M0 code zero means `2^-127`, code 254 means `2^127`, and 255 means NaN.
Integer exponent construction implements these values exactly in f64.
Opposite extreme scales avoid premature operand overflow/underflow. Scale
NaNs propagate even when `fastMath` permits omission; computed NaN payloads
remain unspecified. E4M3FN scales retain their signed zeros and NaNs.
Absent scales mean exactly one.

## Logical dataflow and measured scope

FP4 stores adjacent even/odd logical elements in low/high nibbles. K packing
and independent M/N packing are implemented. Affine maps express packed
coordinates and scale-group reuse; unpack/scaling occurs once per operand
tile. RHS scales use output column followed by K group. E8M0 uses groups of
32; E4M3FN uses groups of 16, independently on either side. Rank-two tiles
use named `linalg.matmul`; rank-three tiles use a batched `linalg.generic`
contraction. No warp/core/shared-memory layout is chosen. Every selected
candidate validates before edits. Unknown properties, incorrect segments,
types/shapes, encoded/dynamic tiles and unresolved formats refuse.

The 65 cases in `tests/test_scaled_dot.py` cover 40 geometry/packing/optional
scale combinations, six located malformed-contract refusals, transactional
multi-op refusal, ten native mixed-format/batched compositions, three other
accumulator formats, all 256 E8M0 codes and opposite extreme exponents, and
four real TTIR-to-Merlin packages across tensor/buffer/guarded-buffer profiles.
Independent rational products, rounded f64 additions and final rounding
provide references. Exact non-NaN words and NaN class are checked separately.
Packages retain compiler-input exclusion, payload accounting, ownership,
lifetime and real one-call native/Merlin evaluation gates.

```bash
python -m t2m shared-demo --kernel scaled-dot --out out/scaled-dot \
  --python /path/to/native-python
python -m t2m qualify-merlin out/scaled-dot --out out/scaled-dot-merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

The static demo uses 16×16×64 logical tiles with M-packed A and N-packed B.

## Runtime FP4 loops and packed tails

`scaled-dot-dynamic` additionally retains runtime M/N/K, packed allocation
dimensions, scale-group extents and the source K loop over 16×16×32 tiles.
An odd logical K masks the unused high nibble of the last live byte on both
operands; arbitrary caller bits there cannot enter the contraction. Each
source dot rounds its result before the next source iteration. The frontend
does not replace the whole kernel with one higher-precision matmul.
Original source assertions enforce the packed-K/group relationships; actual
allocation dimensions remain runtime ABI data rather than inferred constants.

Three additional tests qualify tensor, buffer and guarded-buffer libraries.
Each library replays empty M/N, K=0, K=1/31/32/65, ragged output tiles and
changed scalar/shape data without recompilation. Independent rational
references retain per-tile rounding. Two inconsistent allocation relationships
per library fail through the original assertion and publish no tensor result.

The selected native consumer can copy a restricted import from actual private
storage into independent carried state before the loop, after validating the
complete producer/capture plan. This preserves zero-trip values and keeps
native alias/lifetime checks. The consumer's 160 destination/ownership cases
pass. The earlier 96 shared-tensor/float-cast regressions also pass.
The final combined scaled-dot/runtime/conditional-read/tail-proof run passes
87 cases without skips against the updated consumer.
The buffer demo is reproducible with `shared-demo --kernel scaled-dot-dynamic
--memory-profile buffer`, followed by `qualify-merlin` as above.
Its capture/qualification verify 30/49 artifacts. The independently reopened
30-package observation report retains 30/50 definitions observed and 26 in
host-qualified captures; an unobserved definition is not an unsupported-op verdict.

Three real model2MLIR bindings now retain packed parameters, scale buffers,
learned C/bias and complete tuple references through Merlin host execution;
see [model integration](model_integration.md). Runtime model scalar/shape
adaptation is now qualified under the explicit
[captured-argument profile](runtime_model_arguments.md), including guarded
per-call scratch derived independently from original model calls.
Remaining qualification includes broader runtime format/packing compositions,
broader model signatures/aliases, reserved FP6 storage, target numeric profiles and applicable
accelerator materialization. These remain in the full
[completion plan](completion_plan.md). This slice establishes no universal
opcode/type/attribute/kernel coverage.

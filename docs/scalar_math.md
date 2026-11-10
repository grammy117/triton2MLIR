# Clamp and correctly rounded scalar math through Merlin

`tt.clampf`, `tt.precise_divf` and `tt.precise_sqrt` now have independent
numeric qualification through the ordinary Merlin compiler interface. The
portable payload retains standard min/max, division and square-root operations,
tensor tiles, logical launch structure and original source correspondence.
No backend binary, approximate reciprocal or expanded integer oracle replaces
the kernel computation.

```bash
export TRITON_SHARED_OPT=/path/to/triton-shared-opt
python -m t2m shared-demo --kernel precise-math \
  --out out/precise-math --python /path/to/native-python
python -m t2m qualify-merlin out/precise-math \
  --out out/precise-math-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

The real public Triton kernel returns both clamp NaN modes, precise division
and precise square root as four ordered results. Masks and explicit inactive
fills remain executable. Its runtime extent reuses one compiled library over
empty, single-element and ragged domains. `scripts/capture_scalar_math_model.py`
also captures that kernel as a real model2MLIR custom call with learned lower
and upper bounds. Binding preserves all eight model outputs, weights and
original evaluator archives in every memory profile.

## Reused implementations and type boundary

Meta triton-shared `17f0c234ce474bee334af36e351c62eca7852836`,
`Conversion/TritonArithToLinalg/ConversionPatterns.hpp`, provides:

| Source operation | Standard computation |
|---|---|
| Clamp, `propagateNan = none` | `arith.maxnumf` followed by `arith.minnumf` |
| Clamp, `propagateNan = all` | `arith.maximumf` followed by `arith.minimumf` |
| Precise division | `arith.divf` |
| Precise square root | `math.sqrt` |

The archived Microsoft implementation registers the same precise-math
converters in `Conversion/TritonToLinalg/TritonToLinalg.cpp`; these are related
implementations rather than independent numerical oracles. The actual reused
optimizer remains the recorded `0c27923` distribution, not a claimed build of
the Meta survey revision. Upstream's elementwise arithmetic fixtures establish
the surrounding tensor-to-linalg route; its survey tree has no dedicated
precise-math exceptional-value tests.

Triton v3.7.1 (`f797708c0626e5f9840ca5b0a98790e2c7cb09ad`) declares the
three operations over `TT_FloatLike`. Native-verified typed TTIR tests cover
f16, BF16, f32 and f64. The public Python `tl.math.div_rn` and `sqrt_rn`
builtins restrict arguments to f32, so the real Python kernel uses f32. The
other dtype cases are explicit legal TTIR fixtures, not claims that the Python
builtins or every device backend accept those types. FP8/FP4/scaled dot and
other rounding/format profiles remain separate work.

## Numeric contract and independent reference

Precise math retains nearest-even rounding and gradual-underflow semantics
under the selected host's normal floating-point environment. Qualification
does not grant an alternate ambient rounding/flush mode. Source `clampf`
requires defined bounds: a NaN bound or lower bound greater than upper bound
is outside this qualification domain. NaN input behavior follows the actual
source `propagateNan` mode.

The independent reference in `examples/scalar_math_reference.py` decodes
binary floats as exact rational values. Division rounds an integer quotient
using remainder and parity. Square root compares integer squares with exact
rounding midpoints. Known 1/3, sqrt(2), midpoint, underflow, overflow and
signed-zero vectors check the oracle independently. Neither NumPy floating
division/sqrt nor the compiler's emitted arithmetic supplies those answers.

References use the explicit [mixed-result package](mixed_effects.md) with
empty expected streams and a complete `reference.npz` tuple. The additional
`{"kind":"ieee"}` policy requires exact finite/infinite/signed-zero words
and matching NaN positions, without fixing a computed NaN payload or sign.
It applies to clamp's propagating mode and precise div/sqrt. Clamp's
nonpropagating minNum/maxNum route allows either sign on a zero tie under
standard MLIR; its separate zero-tolerance value policy preserves that freedom.
No numeric tolerance hides a division or square-root rounding error.

The Triton source tests in `python/test/unit/language/test_core.py` use f64
intermediates narrowed to f32 for public precise-math references and test both
clamp NaN modes. Its clamp LLVM conversion in `ElementwiseOpToLLVM.cpp`
confirms the minNum versus propagating min/max distinction. Exact source
hashes are retained in `.scratch/strict-math-upstream/sources.json`.
The model's CPU reference likewise uses f64 then narrows: Torch's vector f32
sqrt differed by one ULP on several inputs and does not implement the precise
Triton builtin contract. The corrected CPU reference is captured before binding;
the model's resulting golden archives remain unchanged through Merlin.

## BF16 consumer implementation

The portable body keeps BF16 arithmetic. Only selected native materialization
uses Merlin's explicit `emulate_bfloat_elementwise` feature. It reuses
LLVM's `arith-emulate-unsupported-floats{source-types=bf16 target-type=f32}`
for scalar min/max/div, and the same extend/compute/truncate boundary for
scalar sqrt. Rounding occurs at each original operation, including loop
iterations. The original tensor and buffer element types remain BF16.

Upstream `arith-expand{include-bf16=true}` handles exact casts. The separate
existing `preserve_bfloat_transport` feature then handles scalar LLVM word
transport, retaining its arithmetic, vector and by-value-ABI refusals.
Unknown math, nonneutral flags and unsupported numeric forms fail closed;
the new feature never turns arithmetic into an integer-word reinterpretation.
Both native producer and Merlin report the actual implementation selection.
Compatibility metadata makes this dependency explicit. The implementation
requires Merlin's `ttir-experiment-grammy` branch with the BF16 math feature;
f16/f32/f64 math uses the established native conversion.

Every one of the 65,536 BF16 input words is checked through clamp/div/sqrt
with an adjacent exact raw-word copy. Computed NaNs use the IEEE policy;
uncomputed signaling NaNs, payloads and signed zeros must survive unchanged.
Separate runtime-loop/branch cases check per-iteration rounding, zero trips
and skipped updates while reusing the same actual library. Memory ownership,
input immutability, result completeness, alias/lifetime checks and compiler
reference exclusion remain active.

## Remaining closure

This extends measured numeric coverage in all three memory profiles and the
real model2MLIR interface. It does not establish all dtype/attribute inputs,
external math libraries, NaN payload-sensitive arithmetic, ambient FP-state
changes, packed low-precision arithmetic, scaled contractions, a maintained
full corpus or staged accelerator materialization. Those remain completion
requirements alongside runtime/alias and synchronization work.

The final focused run passes 95 scalar-math/reference/model/BF16 cases, and
Merlin passes 23 native/seam/transport cases plus its docs/structure gates.
The saved `out/precise-math` and `out/precise-math-merlin` pair passes 26/45
artifact checks. `out/precise-math-bfloat/{capture,qualified}` retains the
typed BF16 recurrence and its actual native implementation, also with 26/45
artifact checks. `out/precise-math-model` preserves the real model2MLIR
capture and passes all 66 bound artifact checks. The 23-package report at
`out/precise-math-coverage/coverage.json` observes 28 of the 50 pinned
definitions, with 24 present in captures of host-qualified packages.

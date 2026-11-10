# Pure external math through standard MLIR

`tt.extern_elementwise` now has a working Merlin interface for 52 typed default
libdevice declarations: 26 functions in f32 and f64. This is the explicit
`logical-libdevice-standard-math-v1` profile. Public tensor/scalar storage,
runtime shapes, source control and the package schemas stay intact.

## Reuse and boundary

Read against Meta shared
[`ConversionPatterns.hpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp)
and its
[`conversion tests`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonArithToLinalg/convert_extern_elementwise.mlir).
The configured content-pinned optimizer remains the separate `0c27923`
distribution. The adapter reuses its actual unary/binary conversion, completing
only f64 tanh/atanh: the surveyed table duplicates their f32 entries, and the
actual binary leaves the f64 declarations untouched. These two completions use
ordinary typed math operations and retain the original source locations.

| Function family | Standard computation |
|---|---|
| fabs, ceil, floor, trunc | abs and integral rounding |
| sin, cos, tan, asin, acos, atan, atan2 | trigonometric math |
| sinh, cosh, tanh, asinh, acosh, atanh | hyperbolic math |
| log, log10, log1p, exp, exp2, pow | logarithm, exponential and power |
| erf, sqrt, rsqrt | error function, square root and reciprocal square root |

All declarations require typed purity, exact arity, the library's exact f32/f64
prototype and matching scalar or static logical-tile shapes. An empty library
path and the default/`libdevice` name are checked before mapping symbols.
A custom library that exports the same symbol cannot inherit these semantics.
Impure calls, unknown symbols, mismatched prototypes and unqualified attributes
still refuse with source locations; they need actual implementations.

The native test runner now uses upstream `convert-math-to-libm` before the
remaining LLVM intrinsic conversion, matching Merlin's existing host pipeline.
Its actual pass list is retained in native execution evidence. Upstream
`math-expand-ops{ops=rsqrt}` expands reciprocal roots into sqrt and division in
the shared tensor computation stage. This avoids a nonstandard `rsqrt[f]`
host-library dependency. The [LLVM source and tests](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Dialect/Math/expand-math.mlir)
establish that expansion; inverse hyperbolics remain stable library calls.
The exact extra computation pass is accepted by both source-word custody
readers, alongside retained legacy pipelines. Unfiltered/different expansions,
changed pointer widths and duplicate passes remain refused. Receipt schemas
and event reconstruction obligations are unchanged.

## Numeric and provenance contract

This profile selects standard mathematical functions and the selected host's
libm/intrinsic implementation. It does not reproduce every NVIDIA library
approximation, NaN payload, floating environment effect or accelerator behavior.
The [libdevice specification](https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/functions.html)
defines a much broader API than this selected mapping. Backend-specific
approximations, explicit rounding variants, integer/bit operations, alternate
libraries and unknown external dependencies remain separate coverage work.

Each admitted source operation retains library identity, type, purity, target
math operation, profile and original source ID in the trace and
`prov.external_math_contracts`. Caller annotations and ordinary source/emission
correspondence remain independent of those arithmetic checks. An annotation
cannot grant a new implementation or erase an effect.

The example reference uses optional `triton2mlir[math-reference]` / mpmath at
110 decimal digits and exact rational rounding to f32/f64. Recurrences round
after each source iteration. Exact comparisons cover abs/integral rounding;
the other functions use explicitly recorded finite-value tolerances, NaN class,
infinity and separate signed-zero checks. These archives never enter compilation.

## Visible demo and qualification

```bash
python -m t2m shared-demo --kernel external-math --out out/external-math \
  --python /path/to/native-python
python -m t2m qualify-merlin out/external-math --out out/external-math-merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`external-math64` selects f64. The source kernel returns all 26 results, retains
runtime sin recurrences and a conditional, and replays changing/empty/ragged
shapes with the same compiled library in each memory profile. This is measured
composition coverage; it does not close arbitrary external or all-TTIR coverage.

The six real packages pass in tensor, buffer and guarded-buffer profiles.
There are 123 focused producer cases, 69 existing transport/native/math/
source-word regressions, two independent native reciprocal-root cases and
206 Merlin staging cases. The f64 guarded demo and qualification verify 33/52
artifacts. The explicit 36-package report observes 31/50 TTIR definitions and
27 in host-qualified captures. Unmeasured definitions and unsupported variants
remain visible rather than inheriting support from this mapping.

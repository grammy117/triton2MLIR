# FP8 contractions and original accumulators

The portable payload retains typed `linalg.matmul`, its original C, logical
output tiles and source K loop. FP8 operands stay one-byte FP8 tensors rather
than expanded full-size f32 matrices. Runtime M/N/K, ragged masks, typed-zero
K padding and zero-trip behavior remain executable. Physical threads, warps,
placement and device instructions remain downstream decisions.

## Reused implementation and numeric contract

The pinned Meta shared converter emits named matmul/batch-matmul, initially
computes into zero and subsequently adds source C. We reuse that conversion
and repair accumulator placement using source correspondence. Its zero test
also skips negative zero. Proved negative source constants therefore receive
an actual typed negative-zero destination when needed. An addition belonging
to a later source operation cannot become C. All selected repairs validate
before edits. Scalar splats and integer word bitcasts retain zero sign;
FNUZ's `0x80` NaN cannot masquerade as negative zero.

See Meta revision `17f0c234ce474bee334af36e351c62eca7852836`,
[`MatmulConverter`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp),
and `test/Conversion/TritonArithToLinalg/{dot,batched_matmul}.mlir`.
The configured optimizer is the separately recorded content-pinned
distribution; the survey revision does not identify its build source.

Pinned Triton
[`TritonOps.td`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/Triton/IR/TritonOps.td)
specifies that `inputPrecision` applies to f32 inputs and is ignored otherwise.
Non-f32 dot now admits all five declared settings. F32 modes other than IEEE
still require their actual arithmetic expansion and refuse. Source formats,
precision applicability, imprecise-accumulation bound, zero kind and intake C
value remain in `prov.contraction_contracts`, keyed by intake graph IDs.
Those annotations grant no new approximation, reassociation or hardware policy.

## Selected native implementation

Named linalg operations require canonical scalar bodies. Expanding an implicit
FP8 cast directly inside one violates native verification. Merlin's selected
`lower_fp8_widening` therefore validates every candidate, reuses
`linalg-generalize-named-ops`, recollects casts and expands their exact scalar
extensions. Generalized fills can carry scalar FP8 values; complete recursive
body validation still rejects unqualified FP8 arithmetic and calls. Every
stage/ownership callback remains. Portable `model.mlir` keeps named operations.

The selected host uses ordinary typed f32 products and accumulation. Every
finite FP8 product is exact and normal in f32. An independent rational product
and rounded-add oracle checks ascending source K, including padded lanes.
Exact non-NaN words and NaN class remain separate from unspecified computed
NaN payloads. This does not establish every GPU's reduction order or performance.

## Measured evidence

`tests/test_fp8_dot.py` covers 93 cases: 45 precision-applicability cases,
24 dynamic contraction packages, 12 signed-zero/downstream-add packages,
nine typed-zero proofs and three direct repair/refusal checks. The 24 packages
cover all sixteen FP8 input pairs in the tensor profile and all same-format
pairs in buffer and guarded-buffer profiles. Reused libraries check empty
domains, K=0, tails, multiple output tiles and original nonzero C. Cancellation
cases distinguish source C from an addition after the contraction. Negative
zero packages preserve its sign and keep a runtime downstream bonus separate.

Nine integer-range checks additionally qualify the source `cdiv(k,BK)` form
without inventing an affine layout fact. The 64 native narrowing/widening
cases cover named/batched contractions, fills, callbacks and body refusal.
These are measured compositions, not universal dot or accelerator coverage.
The saved demo and Merlin qualification verify 26/45 artifacts. The explicit
28-package observation report at `out/fp8-dot-coverage/coverage.json` records
29/50 definitions observed and 25 in host-qualified captures; this is not
whole-opcode closure.

Both cooperating readers now reject false dense splats by comparing complete
element bytes. Mixed `+0`/`-0` constants survive serialization and actual native
returns across wider and FP8 formats. The runtime format decoder also uses
exact binary scaling rather than an approximate `exp2` result, with all
65,536 BF16 and f16 encodings checked independently. These repairs grant no
new numeric approximation or target permission.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel fp8-dot --out out/fp8-dot \
  --python /path/to/native-python
python -m t2m qualify-merlin out/fp8-dot --out out/fp8-dot-merlin \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

The demo uses different FP8 formats, 16×16×32 tiles and runtime M/N/K of
17/19/37. The existing v2 result/effect package excludes references and their
digests from compilation. Public tensor/scalar ABI and byte storage are unchanged.

## Next: scaled contractions

`tt.dot_scaled` remains a required next implementation. The pinned definition
admits five operand formats, optional scales, rank-two/batched geometry and
K versus M/N packing. E8M0 integer scales use groups of 32; E4M3FN scales use
groups of 16. RHS scale storage follows output column and K group. Preserve
those relationships as tiled unpack/scale computation plus named contraction.

The official
[`DecomposeScaledBlocked`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.cpp)
and [`test_matmul.py`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/test/unit/language/test_matmul.py)
are reuse references. That fallback depends on GPU layouts, refuses M/N
packing and selects intermediate f16/BF16 arithmetic. It cannot silently
define our portable contract. Establish executable scaling, rounding,
scale-zero/NaN and `fastMath` behavior, then qualify every packing and
optional-scale form. Retain scaled-dot intent for a capable consumer's fused
implementation after its numeric proof. Broader accumulators, numeric/effect
profiles, model bindings, corpus and applicable target work remain open under
[the complete plan](completion_plan.md).

# Tiled online attention through the Merlin interface

The maintained f32 example emits standard MLIR for
`softmax(Q @ K.T / sqrt(D)) @ V` using online softmax state, not a dense score
matrix or an opaque attention operation. It keeps the source algorithm:

- One query tile is loaded before the key-tile loop and reused.
- Each iteration loads a key/value tile and performs two tile contractions.
- Running maximum, normalization and output accumulator remain carried tensors.
- The logical query grid remains an unmapped `scf.forall`; masks retain tails.

The example uses `D=BM=BN=16` and explicit IEEE dot input precision. Actual
runtime query/key counts use ABI v5, with tested bounds `M=0..35`, `N=0..37`.
Source code explicitly defines zero output when there are no keys. Fresh output
allocation, source access checks and provenance use the default tensor path or
the declared buffer/source-word profiles, with the same public functional
tensor/scalar ABI and [explicit model2MLIR package differences](package.md).
The tensor form carries three changing online states; an extra output-tensor
carrier is invariant. It introduces no whole score matrix or memory guard arrays.

```bash
python -m t2m shared-demo --kernel online-attention \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python \
  --out out/online-attention

python -m t2m qualify-merlin out/online-attention \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang \
  --rtol 2e-5 --atol 2e-6 --out out/online-attention-merlin
```

`--memory-profile buffer` retains the full standard memory payload behind that
interface. `--memory-profile guarded-buffer --address-policy source-words`
selects the separately declared [runtime admission profile](source_words.md).

## Consumer state and granularity

The old maximum is still needed to compute the rescaling factor after the new
maximum has been computed. Early physical reuse would require an extra owner
or a different schedule. The consumer therefore retains the original producer
and updates carried storage at the loop edge, after old-state reads finish.
Complete unused-init computations can still reuse their original destinations;
reductions/live init reads retain their bodies and use a pure complete-value
copy. Original branches, dependencies and arithmetic order remain. Native
bufferization and lifetime verification decide actual storage reuse.

Tests independently check three carried tensor states, two `linalg.matmul`
operations, tile-shaped intermediates and absence of a whole query-by-key
score tensor. Exact TTIR, typed source operations, original dot precision and
accumulator-restoration evidence remain retained. Caller intent describes the
algorithm; it supplies no rewrite or numerical permission. Consumer copies keep
source IDs/locations and introduce no arithmetic or physical memory mapping.

## Measured qualification

Native execution and one unchanged actual Merlin library run independent
query/key shapes, including zero queries, zero keys, one element, exact tiles,
multiple tiles and ragged tails. Every result is compared with an independent
dense float64 reference at `rtol=2e-5`, `atol=2e-6`; original inputs and compiled
library bytes remain unchanged. Dense reference computation is evaluator-only
and stays outside compiler staging.
Each emission profile has its own unchanged compiled library and independent checks.

This qualifies the finite f32 example and its interface. It does not establish
device-bit equivalence, all floating-point edge policies, causal/dropout/batched
attention, low-precision variants, general aliasing or accelerator execution.
Tiling and reuse are retained; performance has not been benchmarked.

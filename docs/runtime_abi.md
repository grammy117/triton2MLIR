# Runtime scalars and the remaining dynamic ABI

`t2m.shared-tensor-abi.v3` extends the standalone functional profile with plain
runtime scalar inputs. Each declaration has exactly `name`, `dtype` and
`source_arg` (its zero-based TTIR argument position). This supports interleaved
pointer/scalar source arguments. Pointer buffers retain declaration order in
the remaining source slots, followed by any legacy specialized integer scalars.
The exported function orders logical tensor inputs first, then runtime scalars
in their declaration order. Writable buffers remain tensor results.

```python
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

abi = TensorABI(
    "runtime_affine",
    (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")),
    grid=(2,),
    runtime_scalars=(RuntimeScalar("alpha", "f32", 1), RuntimeScalar("bias", "f32", 3)),
)
```

The source is `(x_pointer, alpha, out_pointer, bias)`; exported MLIR takes
`(tensor<13xf32>, f32, f32) -> tensor<13xf32>`. Scalars keep their actual
types and SSA dependencies through upstream splat/arithmetic/region conversion.
The adapter reuses triton-shared's `SplatConverter` and scalar arithmetic at
Meta `17f0c234`; it does not replace payload scalar arithmetic with another
decomposition table. `prov.runtime_scalars` and the spec retain source positions.
Runtime values are absent from that spec and compiler staging.

`inputs.npz` stores a typed zero-dimensional array for each scalar using the
usual positional `inN` key; this storage convention does not turn the MLIR
argument into a rank-zero tensor. The native runner packs scalar arguments by
value. Merlin staging marks them `kind: scalar` in its signature and excludes
tensor bufferization attributes. Actual invocation uses the pre-existing
`merlin.llvmlower.abi.ScalarArg` alongside tensor descriptors. Compiler source
and invocation data remain separate. An invocation value in a scalar declaration,
a tensor substituted for a scalar, wrong type/rank, invalid position or duplicate
name refuses rather than silently specializing or reinterpreting storage.

Existing ABI v1/v2 specs retain their serialization and specialized-constant
behavior. A complete package continues to use `t2m.capture-receipt.v2`; ABI v3 is
an explicit signature extension, not a relabeled Torch capture. Native and
Merlin host checks cover f32/f64/i32 payload values and runtime i1 source branches,
interleaved arguments, tails, integer wrapping and changed invocation values.
Supported scalar declarations also include i8/i16/i64; their broader operation/
composition matrix still needs qualification. This is not universal numeric or
target certification.

`invoke-merlin` now reuses a verified native library with new typed argument and
reference archives. It runs only the evaluator worker, needs no compiler, and
keeps the original qualification immutable. A separate
`t2m.merlin-kernel-invocation.v1` receipt binds the exact library/program,
argument bytes, observed outputs, independent reference and original compiler
snapshots. `verify-merlin-invocation` checks those bytes and internal relations.
Tests change scalar values with compiler paths deliberately unavailable, replay
multiple results, and refuse wrong rank/dtype/keys, references and changed libraries.

```bash
python -m t2m shared-demo --kernel runtime-affine --out out/runtime_affine \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python
# qualify-merlin builds this once; see docs/merlin_consumer.md.
python -m t2m invoke-merlin out/runtime_affine_merlin \
  --inputs new-inputs.npz --reference new-reference.npz --out out/new-invocation \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python
python -m t2m verify-merlin-invocation out/new-invocation
```

## Guarded memory domains (ABI v4)

Bounded signed integer parameters use `t2m.shared-tensor-abi.v4`:
`RuntimeScalar("n", "i32", 1, bounds=(0, 13))`. Bounds are inclusive, typed and
explicit. Entry `arith.cmpi` / `cf.assert` operations enforce them before source
memory operations. Merlin retains the interval in its invocation signature and
also checks it before calling the compiled library. This is an executable ABI
premise, not provenance used as a proof or capture of one runtime value.

Symbolic range analysis now keeps guarded parameters separate from source loop
IVs and program IDs. Shared `MaskAnalysis` at Meta `17f0c234` already supplies
the correct `max(min(start+tile,n),start)-start` expression, including inactive
tiles. The adapter recovers the exact clipped prefix with a symbolic limit and
rebuilds it for parallel publication. It preserves the logical grid, tile size,
explicit fill, source arithmetic and incoming readwrite values outside the mask.
Parameter offsets cannot masquerade as a complete fresh-output partition.

The `runtime-prefix` demo takes `(x, n, out)` in TTIR and exports
`(tensor<13xf32>, tensor<13xf32>, i32) -> tensor<13xf32>`. The shape here is the
allocation capacity. A single compiled library handles `n=0`, tails and full
capacity, returning untouched incoming output elements outside `[0,n)`.
Broader intervals can admit negative limits: the prefix is then empty. Proved
empty launches have disjoint empty writes even when every offset is identical;
their ownership annotation states that actual proof instead of inventing a tile
partition. Fresh outputs still require complete initialization of every result.

Native and actual Merlin host checks cover i32/i64 domains, surplus inactive
programs, negative/all-inactive limits, source guards and per-invocation bounds,
preserved inputs, and multiple domain sizes using exactly the same library.
The source receipt, compiler view and original qualification remain immutable.

```bash
python -m t2m shared-demo --kernel runtime-prefix --out out/runtime_prefix \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python
```

Dynamic tensor shapes/strides/grid sizes, general source while/loop bounds and
indexed writes remain open. Next: symbolic tensor dimensions and launch extents
linked to actual argument shapes, dynamic allocations/results, and one compiled
library invoked over independent shapes. Those need initialization/ownership
proofs and consumer allocation/binding changes. Allocation capacities above are
static; runtime prefix lengths do not close the dynamic-shape gate.

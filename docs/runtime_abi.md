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

Dynamic shapes/strides/grid sizes and runtime-controlled address/loop domains
remain open. Next: guarded scalar domain bounds and exact runtime prefix masks;
then symbolic tensor dimensions and launch extents linked to actual argument
shapes, dynamic allocations/results, and one compiled library invoked over
independent shapes/scalars. These need executable guards, initialization/
ownership proofs and consumer allocation/binding changes. Captured constants
in place of runtime dimensions do not satisfy that gate.

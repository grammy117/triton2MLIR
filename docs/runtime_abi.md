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

## Runtime value recurrences

Runtime `scf.for` bounds now survive for regions containing only registered
standard scalar/tensor computation, including nested value regions. The loaded
tile is immutable tensor state; no new source buffer observation occurs inside
the loop. The adapter clones the original recurrence, bounds, types, assertions
and source locations, without unrolling or adding buffer-state iteration args.
Buffer reads/writes, pointers, opaque operations and CFG branches retain their
existing memory/effect requirements. A loop result used later as an address still
needs its own index/range proof.

This reuses Meta `17f0c234`'s existing SCF conversions (see
`TritonToUnstructuredPass.cpp` and `scf_for_ptr_in_iterargs.mlir`); no local
arithmetic decomposition replaces the upstream body. Both public memory profiles
have native and actual Merlin checks for a tiled runtime recurrence over counts
0, 1 and 5 using exactly one library, including preserved inputs and ragged tails.

Merlin's explicit `reuse_loop_tensor_destinations` compiler feature runs after
elementwise fusion and immediately before bufferization. Upstream fusion's
`RemoveOutsDependency` intentionally replaces unused destinations with empties;
the later rewrite reuses a loop-carried tensor only for a completely overwritten,
positive static, identity-output parallel generic with an unused init scalar.
It preserves arithmetic and source SSA inputs. Native alias analysis determines
physical reuse and the ordinary ownership/lifetime checks remain mandatory.

```bash
python -m t2m shared-demo --kernel runtime-recurrence --memory-profile buffer \
  --out out/runtime_recurrence --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
```

## Guarded runtime read loops

Signed-comparison `scf.for` regions can now load buffers with an entry-guarded runtime upper
bound, constant lower bound and positive constant step. The source-width check
derives a conservative interval for all reachable IVs, including the last IV
update. Tensor memory analysis uses that interval for every access. Executable
loops retain the original runtime SSA bound; the maximum count is proof evidence,
never a replacement trip count. Variable-count recurrences acquire no fictitious
constant final-address fact. Dynamic pointer carriers remain a separate gate.

Row-local masks can use a different coordinate from a flattened physical
address. Their exact clipped expression is retained. Active access bounds use
`physical - mask_origin + mask_limit`; inactive lanes still make no observation.
Fresh-output partition proofs continue to require exact mask/address origins,
and parallel publication refuses unresolved loop-dependent extents. An access
bounds proof alone cannot initialize an output.

This extends the upstream region/pointer normalization rather than adding a
local TTIR compute converter (Meta `17f0c234`, `gather_simple_loop.mlir` and
`TritonToUnstructuredPass.cpp`). Both memory profiles have actual native/Merlin
checks for i32/i64 counts, negative/zero trips, row tails and different counts
through one unchanged library. Undersized allocations and unbounded counts
still refuse. Provenance records minimum/maximum counts
separately from the original executable bound.

```bash
python -m t2m shared-demo --kernel runtime-memory-loop --memory-profile buffer \
  --out out/runtime_memory_loop --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
```

## Conditional runtime write state

Runtime source stores now carry updated logical tensor versions through the
original loop. Readwrite results retain incoming values when the loop has zero
trips or its store branch is not taken. Read/modify/write bodies observe their
previous iteration's version. Both public profiles keep caller inputs immutable;
the mixed-memory profile updates private storage copied from incoming state.
Original tile ownership and access checks remain mandatory.

If zero trips are reachable, post-loop definite initialization retains only the
facts established before the loop. A possible body store cannot initialize a
fresh result. When the guarded minimum trip count is positive, post-body facts
may survive, still subject to branch intersection and coverage checks. Thus a
complete pre-loop initialization can support a fresh output with zero trips;
a conditional store without an initialized false path cannot do so even with
a positive trip count. No sample invocation grants an initialization premise.

Actual Merlin tests cover both profiles, changed counts using one library,
source branches, repeated read/modify/write updates, positive-trip fresh results,
preinitialized fresh results and negative/zero-trip preservation. The accumulator
reference uses ordered binary32 additions, without replacing the recurrence by
a multiplication. Refusal cases retain incomplete initialization and branch
holes. A generalized scalar-fill consumer regression also verifies its source
remains a scalar when the output-slicing optimization does not apply.

```bash
python -m t2m shared-demo --kernel runtime-memory-store --memory-profile buffer \
  --out out/runtime_memory_store --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python
```

Dynamic tensor shapes/strides/grid sizes, general source while/pointer state and
indexed writes remain open. Next: symbolic tensor dimensions and launch extents
linked to actual argument shapes, dynamic allocations/results, and one compiled
library invoked over independent shapes. Those need initialization/ownership
proofs and consumer allocation/binding changes. Allocation capacities above are
static; runtime prefix lengths do not close the dynamic-shape gate.

# Source-specialized packed reads and pointer results

Packed `tt.map_elementwise` regions now retain ordinary source reads under the
explicit initial source-layout specialization and guarded-buffer/source-words
memory profile. This extends [source grouping](source_grouping.md) and
[mapped memory](memory_maps.md); the selected geometry remains semantic input,
not consumer placement or correspondence to a final optimized GPU layout.

## Transformation and boundary

The pointer analysis checks the native packed region signature: each tensor
operand supplies `pack` scalar arguments, and each tensor result receives
`pack` scalar returns. Every scalar argument inherits its original tensor's
possible roots; a pointer tensor result includes all its returned slots' roots.
Mixed-root interfaces carry one selector tensor per pointer field and one
scalar selector per slot. Offsets retain source arithmetic and signed i64
accumulation; selectors identify logical source roots, never physical addresses.

For maps containing reads, source coordinate facts construct a parallel
`linalg.generic` with **one invocation of the complete original scalar region
per logical pack**. Each returned scalar slot has a tensor of group values.
Separate parallel generics restore the original logical tensor coordinates.
Restoration selects already computed values; it does not repeat the original
loads, branches or loops for every output slot. Algorithmic grouping stays
visible and consumer scheduling remains undecided.

Actual Meta/shared event templates and the existing guarded-memory path retain
each load's original position, mask, fill and control. Bounds, initialization,
ownership, snapshots and native lifetime checks remain enabled. Reads captured
before an ordinary write cannot become observations of the changed state.
Pointer-valued results restore every slot's own offset, selector and predicate
before subsequent memory operations.

The pinned Triton verifier forbids writes and volatile reads in map regions.
This extension admits no map stores, atomics, publication or physical-address
observations. Ordinary writes between maps retain their existing contracts.
Arbitrary cross-program conflicting accesses remain a separate dependence
problem; the maintained state fixture bounds recurrence reads to its own
logical program, avoiding an invented global phase between parallel programs.

Source references and reused boundaries:

- [Triton packed signature and memory verifier](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/IR/Ops.cpp).
- [Triton packed multi-result test](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/test/unit/language/test_core.py#L5165).
- [Meta/shared tensor arithmetic patterns](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp).
- The [existing reuse map](triton_shared.md) and [source-word event boundary](source_words.md)
  retain the separate optimizer/version identities and archived Microsoft lineage.

## Measured scope and package differences

Six actual native/Merlin libraries qualify pack two, i32/i64/f32, two/four
source warps, ordinary/selected roots, for/while reads, returned pointers and
per-slot predicates. Each library runs five domains, including empty roots,
one element, exact tiles, multiple programs and ragged tails. Four state
libraries compare all five state/snapshot results, including untouched storage;
two pointer-result libraries check restored root/offset/predicate combinations.
Wide integers exceed the exact range of floating transport. An invalid active
read aborts the existing library with the real bounds diagnostic.
The 95-case existing memory/map/pointer/layout regression suite also passes,
including pack-one maps and native source-write refusals. Guard reports retain
six ordinary read events, or twelve masked selected-root events, and the two
original recurrence loops; result restoration does not multiply those bodies.
The public `shared-tensor`/`qualify-merlin` CLI example is retained at
`out/grouped-read-demo/{capture,merlin}/`; its twenty-member source bundle and
actual Merlin host qualification both verify independently.

The executable payload uses standard tensor/Linalg/SCF and the already declared
memory profile. `group_storage: one-vector-per-result-slot` describes the
source-group expansion; original map identities, pack size, coordinate witness
and event correspondence remain retained. Public tensor/scalar ABI and receipt
schemas are unchanged. Generated compatibility metadata records this extension.

## Complete models and functional target execution

`scripts/capture_grouped_memory_model.py` captures real model2MLIR i64/f32
models with learned 10x32 state, two dependent calls and a third empty call.
All fifteen results remain public, including the first complete state and
snapshots that are still live after the dependent call. Calls specialize
runtime domains `(n, capacity, iterations, delta)` as `(300,320,3,2)`,
`(319,320,1,-3)` and `(0,0,2,5)`. The existing dense value-view policy preserves
matrix/empty public shapes while the kernel owns logical one-dimensional roots.
Source grouping options match the original typed call attributes through the
existing captured-source-layout profile.

The independent evaluator in `t2m/examples/grouped_memory_reference.py` uses
only NumPy, the declared example grouping and program-local observations. It
reads no frontend witness or emitted IR. Forty-eight typed/domain cases compare
it with the separate scalar oracle. This is the explicit initial-stage source
contract; no final GPU-binary equivalence is asserted.

Both complete models pass native LLVM, actual Merlin host compilation and
RISC-V/Spike execution. All thirty results match exact reference words, with
original inputs observed readonly. Weights, manifests, inputs and complete
reference archives remain byte-identical to the parent capture. The target
selection at `out/spike-grouped-memory-models/summary.json` reopens both actual
qualification receipts. Public binding and target CLI artifacts are retained at
`out/grouped-memory-model-demo/`.

To reproduce the i64/two-warp/while model, first capture the shipped TTIR with
the frontend Python:

```python
from pathlib import Path
from t2m.examples.grouped_memory_kernels import grouped_read_state
from t2m.frontend import capture_ttir
out = Path("out/grouped-read-source")
out.mkdir(parents=True, exist_ok=False)
names = ("x", "y", "out_initial", "out_sum", "out_final", "out_final_sum")
(out / "source.ttir").write_text(capture_ttir(grouped_read_state,
    {**{name: "*i64" for name in names},
     **{name: "i32" for name in ("n", "capacity", "iterations", "delta")},
     **{name: "constexpr" for name in ("B", "WHILE", "SELECT")}},
    {"B": 256, "WHILE": True, "SELECT": True}))
```

Then use the [explicit bindings](../examples/grouped_memory_binding.json) and
[kernel ABI](../examples/grouped_memory_abi.json) with the existing environments:

```bash
/path/to/m2m-python scripts/capture_grouped_memory_model.py \
  --out out/grouped-memory-parent --dtype i64 --warps 2
python -m t2m bind-model out/grouped-memory-parent \
  --bindings examples/grouped_memory_binding.json --out out/grouped-memory-bound \
  --python /path/to/native-python --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --merlin-clang /path/to/clang --merlin-host
```

Use the existing [functional target driver](target_execution.md) for the Spike
qualification. Consumer capabilities, ownership/lifetime checks and placement
remain independent of source geometry; no new target-specific frontend case is
required for these models.

This finite qualification does not establish all pack widths, numeric policies,
source versions, pointer-result model/target compositions, general physical or
mutating public views, final GPU correspondence, broader targets or performance.

# Serial scalar memory as tensor data flow

Single-program scalar indexed reads and writes can now remain standard tensor
computation instead of requiring private guard arrays and a memref payload.
This closes the reproduced tensor-admission refusal in the actual shared-view
mutation examples, without changing the original kernel source.

## Reuse and abstraction

The frontend still uses the pinned triton-shared pointer/arithmetic conversion.
The adapter reuses its existing scalar-to-unit-tensor carrier and predicated
gather expansion. Relevant primary references are Meta
[`UnstructuredToMemrefPass.cpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp),
the historical Microsoft
[scalar and gather/scatter converters](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp),
Meta's [masked gather/scatter fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir)
and Microsoft's [defined-fill fixture](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/test/Conversion/UnstructuredToMemref/gather_mask_with_other.mlir).
Those sources distinguish scalar events from tensor lanes and retain flat base
offsets. Their physical memref ABI does not provide our functional tensor-state,
fresh-initialization or program-ownership proofs.

One actual source store lane becomes a standard `tensor.insert` into the
current complete storage value. A mask becomes actual SCF control; active
signed bounds are checked before extraction/insertion, and inactive addresses
are never dereferenced. Gathers observe the current source-ordered tensor
version. Repeated reads are valid after initialization; multiple write lanes
or other logical programs do not inherit scalar ownership permission.

For a multidimensional root used only by scalar-addressed events, complete
linear storage is carried through source loops and its public logical shape
is restored at the boundary. Standard collapse/expand operations are dense
value views. Fresh linear storage is allocated directly, rather than creating
an expanding view on every loop backedge. This also reaches the existing
independent native ownership/lifetime checks; no check is disabled or bypassed.
`prov.serial_tensor_storage` retains each original logical shape and describes
the internal representation. It grants no physical alias or memory effect.

## Source-proved fresh results

Fresh result reads still require covering source stores. A complete serial
loop proof requires all of the following actual source facts:

- One logical program, a signed serial loop starting at zero and stepping by one.
- An unconditional direct scalar store at the loop body's top level.
- The actual store index equals the source induction variable.
- The actual upper bound equals the product of the result's declared dimensions.

The proof reuses `symbolic_layout.expression` and its bounded exact polynomial
analysis, retaining typed intermediate overflow checks. Equal maximum intervals,
symbol spellings and provenance are insufficient. The original executable loop
and runtime bounds remain. A reachable zero trip count earns complete-write
credit only because the same proved element-count expression makes the result
empty. Independently covered constant intervals can establish a small static
one-dimensional result without counting duplicate writes as new coverage.

Facts retain the existing branch intersection and zero-trip restoration rules.
Masks, conditional stores, duplicate indices, a different count with the same
range and pre-initialization reads do not establish the loop overwrite.
Parallel ownership remains a separate proof and cannot be gained by serializing
the source launch.

## Programs and package differences

`tests/test_mutating_view_models.py` keeps the actual overlapping, interleaved,
identical, matrix-column and transposed-clone two-call models. Source loops,
one reused bias load, overlapping write order, learned parameters, prior
snapshots and every returned state remain executable. Automatic selection may
now admit the tensor candidate directly for these five forms.

The rank-three matrix-view form retains its existing guarded source-word path.
Its structured source-width audit cannot yet prove that the divisor is nonzero
only inside the `range(rows * cols)` body when the public scalar range admits
zero. That is a real reachability/range-analysis obligation, not a GPU-specific
operation to discard. Its original complete working interface remains available.

The retained matrix-column comparison has identical complete typed source
operation records after excluding locations: three source loops, four loads and
four stores. Its emitted kernel changes from 49 memref operations to none,
with four standard tensor insertions. This measures the abstraction change;
it is not a timing or allocation-reuse benchmark.

The next proof belongs in `shared_indexing.check_source_addresses`: actual
loop-body reachability can establish positive factors for a nonnegative product
bound. Such facts must stay local to the defining region and must not validate
an earlier, already-executed division outside that region. Preserve valid empty
entry domains, source-width checks and separate caches for distinct contexts;
neither tightening a sample-specific bound nor dropping the quotient/remainder
addresses closes this general obligation.

`t2m/examples/serial_memory_kernels.py` additionally exercises an initialized
state update, a preserved complete snapshot, a dependent read of the completed
fresh snapshot and two scalar observations. Native i64/f64 checks include empty
and inactive accesses, exact large integer words and active bounds failures.

Public tensor/scalar types, receipt schemas, artifact rosters, functional
storage bindings and Merlin signatures are unchanged. Exported tensor MLIR now
uses executable scalar bounds and source-proved initialization instead of a
guarded-memory admission profile for the selected serial forms. Tensor-version
updates do not require physical copies; downstream bufferization decides reuse
and retains its ordinary ownership checks. No accelerator or performance result
is established by retaining this structure.

The unguarded `buffer` continuation retains its earlier static witness scope.
It does not execute a tensor witness's new runtime checks; serial irregular
accesses cannot gain admission there merely because the tensor candidate
passes. An explicit `guarded-buffer` selection retains its executable memory
admission, independently of the new tensor representation.

General parallel irregular write ownership, multi-lane conflicting stores,
partial/fresh initialization, broader reachability, pointer/view/effect
compositions and applicable accelerator qualification remain full-goal work.

## Retained qualification

At implementation commit `1ceb878`, the source-frozen memory selection passes
91 checks and the complete six-model selection passes 30, with overlap. Five
tensor models retain all 44 initial, 132 changed-invocation and 44 actual serial
Spike results. Twenty-five rehashed model claims refuse in both readers and
five original shape domains abort without outputs. The separate rank-three
guarded regression retains eight initial, 24 replay and eight target results,
five refusals and one original-domain abort. Every parent archive remains intact.

Native primitive checks retain 49 exact-word results over thirteen invocations,
including empty/inactive accesses, complete snapshots, dependent fresh reads,
all-unit ranked roots and nonfinite f64 transport. Three further actual runs
abort at the original active gather/scatter bounds checks without outputs.
Initialization negatives retain masked/conditional/duplicate stores, differing
counts with equal ranges, early fresh reads and parallel-conflict refusals.

Evidence reopens under `out/serial-tensor-memory/summary.json`, with source/log
digests, full model references/results, unchanged-source comparison, every
target receipt, refusal/abort outcomes and the separate native selection.
Earlier admission, test-driver and lifetime failures stay in their original
`.scratch` logs; they are not relabeled as successful tensor qualification.
The final logs are `.scratch/serial-models-pinned1.log` and
`.scratch/serial-memory-pinned1.log`. The historical opcode inventory is unchanged.
Reopening passes in `.scratch/serial-tensor-report2.log`; the first report's
duplicate pytest `current` symlink detection remains in its original failed log.

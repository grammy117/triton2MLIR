# Measured coverage and remaining boundaries

The explicit [guarded memory profile](guarded_memory.md) additionally executes
loaded i32/i64 index stores through the full upstream continuation and Merlin's
real host-library interface. It retains masks, source loops, partial incoming
state and an unmapped logical grid. Runtime admission rejects active bounds
violations, duplicate active lanes, cross-program conflicts and fresh-output
holes. This broadens irregular store support beyond affine tile proofs, with
an explicit grid-derived guard cost (normally 16 bytes per output element).
V2 additionally qualifies writable indirect observations in runtime recurrences,
shared readers, initialization and both cross-program read/write orders. Source
gathers become actual memory loads before optimization can merge views; source
loops and payload stores remain. Aliases and general pointer state remain open;
this is not universal TTIR coverage.
Scalar indirect accesses also use a checked one-element-tensor normalization,
retaining source masks/fills and i32/i64 word types. Both scalar and tiled
events reuse the same upstream payload and runtime admission protocol.

The explicit [source-word address policy](source_words.md) additionally retains
original i32 arithmetic before the official i64 pointer-offset accumulator.
Scalar/tiled wraparound, mixed-width pointer chains, runtime `for` pointer
state/final results and nonlinear grid permutations now execute through that
policy. Independent words/shapes and pointer-loop counts reuse Merlin libraries;
real model2MLIR GEMM captures retain contractions and loops. The default
structured range gate is unchanged. Same-base pointer branches and runtime
while state now also execute through Merlin, preserving masks, zero trips,
final pointers and narrow word arithmetic. Nested branches, pointer-returning
helpers and conditional reductions with nonzero initial state also qualify
through unchanged Merlin libraries over independent choices/counts/shapes.
Reduction/init semantics and native alias/lifetime checks remain intact.
Mixed-base joins/backedges refuse;
broader while ownership and aliases remain open. These checks do not imply
universal opcode/attribute coverage.

Use `shared-coverage` to rerun a fixed, bounded native pipeline against actual
upstream fixtures. It inventories operations structurally, retains every input,
output and diagnostic, and records content hashes and the corpus Git revision.
It never executes fixture `RUN` comments. Negative tests and inputs intended for
other pass boundaries stay visible in the denominator.

```bash
python -m t2m shared-coverage /path/to/triton-shared/test/Conversion \
  --out out/coverage --optimizer /path/to/triton-shared-opt \
  --stage tensor-ir --budget 120
```

## Native corpus measurement, 2026-11-07

Corpus: Meta `17f0c234ce474bee334af36e351c62eca7852836`, all 255 files in
`test/Conversion`, split into 339 native MLIR test cases. Optimizer binary SHA-256:
`2af44b4536b2dffbdb90b877b04e97cd2b10fe333020606fa4a1814b3c34926d`.
The [distribution/source pins](triton_shared.md) differ from the corpus revision.

| Observation | Tensor stage | Full upstream stage |
|---|---:|---:|
| Cases measured / discovered | 339 / 339 | 339 / 339 |
| Native-verified source cases | 335 | 335 |
| Successful native stage conversions | 310 | 314 |
| Structurally inventoried cases containing `tt.func` | 323 | 323 |
| Successful stages in that observed TTIR subset | 309 | 306 |
| Source inventories unavailable | 1 | 1 |

The unavailable inventory uses an fp8 type unsupported by xDSL 0.68. The full
upstream stage includes memory/ptr ABI lowering; its successes are not automatically
better tensor-frontend candidates. Each stage has different input assumptions.

These numbers measure a reusable middle layer. They are **not** functional tensor
ABI coverage, numeric equivalence, or Merlin materialization coverage. A whole
opcode is not certified merely because it appeared in a passing composition.
The complete per-case/opcode report is generated as `coverage.json`; intermediate
failures are retained alongside it.

Failures include source/tool version differences (`tt.cat`, structured-state
schemas), pass-specific fixtures containing pointer-tensor forms that this full
stage does not accept, conditional block-pointer conversions, and missing native
dialect interfaces when the inliner is given some already-standard functions.
Changing the expected input boundary is different from adding TTIR support.

## Actual product evidence

The [shared functional tensor path](shared_tensor.md) now executes source-captured
2-D add, masked softmax/reduction helpers, IEEE matmul and ordered multiple-output
stores through standard MLIR and native LLVM. Its op decomposition comes from
triton-shared; the local adapter checks the tensor ABI and memory effects.
The existing reference frontend additionally qualifies rank-one masked tiled
pointwise kernels with an unmapped logical grid and symbolic ownership checks.

Multi-axis affine tiled grids now have an unmapped forall wrapper, bounds/overflow
checks and disjoint ownership. Native tests cover tiled add/matmul plus negative
conflicting-write and cross-tile-read cases. Further product checks now cover fresh
result initialization, permuted accesses, branch memory merges, affine pointer
increments and final pointer values. Actual model2MLIR calls are bound by schema
and scalar specialization, including the original plain/Triton MLP pair. Complete
models match their captured PyTorch goldens and pass Merlin's actual interface
reader and C runtime generation. This does not establish target execution or
universal op/type/attribute coverage. Clipped affine tail masks now additionally
have native checks for ragged M/N/K GEMM, transposed weights, nonzero-fill
reductions, branch-local stores, partial output preservation and surplus inactive
programs. Tiling and K-loop reuse remain executable structure. Fresh outputs use
a symbolic clipped-partition proof rather than nominal tile cardinality.
Legacy `tt.make_tensor_ptr` and `tt.advance` now pass an official pinned-Triton
intake rewrite, followed by the same tensor adapter. Block-pointer ragged GEMM
has native numerical checks with the source grid/K loop retained. Provenance v2
separates original and normalized inventories instead of equating their IDs.
Kernel-local `tt.make_tensor_descriptor`, `tt.descriptor_load` and
`tt.descriptor_store` now use the official descriptor-to-pointer pass. Descriptor
add and ragged GEMM have native checks, and a real descriptor-kernel model is
qualified through unchanged-golden comparison and the Merlin interface. External
descriptors and descriptor gather/scatter/reduce are not covered by these cases.
`tt.scan` / `tt.scan.return` additionally have a parallel tensor doubling
expansion for inlined pure scalar combiners. Native checks cover f32/i32
sum/product/maximum, forward/reverse tails, noncommutative tuple combiners on
two axes and the no-invented-identity case. This does not qualify arbitrary
effectful combiners or all floating-point edge policies.
Real reverse-sum/product and noncommutative tuple-scan models additionally pass
the model2MLIR binding and Merlin interface/runtime path, including consumption
of both kernel results by the enclosing model.
Loaded-index reads now additionally have an
[explicit tensor expansion](indexed_memory.md), with runtime bounds checks in
the active branch, arbitrary boolean lane masks on `tts.gather`, exact fills,
repeated-index reuse and multidimensional logical allocations. The rank-one
structured gather descriptor is also supported. These cases include actual
Merlin compiler/host execution. This does not qualify general scatter,
writable-buffer gathers, undefined-lane propagation or all address arithmetic.
Nonrectangular store masks now also qualify when their addresses form proved
independent dense tiles. Periodic/data-dependent predicates retain their source
loops and inactive initialized values through both Merlin profiles, including
runtime tails and empty domains. This handles regular-address `tts.scatter`;
it does not establish general indirect write conflict semantics. See
[the masked-write proof and representation](masked_writes.md).
Runtime matrix masks additionally qualify projected broadcasts and transposed
storage/traversal, in both profiles over changing and empty two-axis shapes.
[Half tensor execution](half_precision.md) now separately qualifies f16 storage,
f32-intermediate/f16-narrowing kernels, mixed f16/f32 contraction accumulators
and runtime masked matrix state through both Merlin profiles. These examples
cover explicit rounding and inactive-bit cases, not all half operation policies.

The remaining boundaries are:

[Runtime scalar ABI v3](runtime_abi.md) now has f32/f64/i32 payload and i1
source-branch execution cases through Merlin. Runtime values are not specialized;
non-dense stride, general grid and address-domain profiles still require separate work.
Guarded i32/i64 prefix domains (ABI v4) now additionally have native and Merlin
checks, including empty/tail/full domains, preserved outputs and negative limits.
These vary memory domains over static capacities; ABI v5 below separately admits
actual runtime tensor shapes. General indexed writes and broader loop/while
domains remain open.
Runtime value-only `scf.for` recurrences additionally retain their original
standard body and SSA bounds through native/Merlin execution, including zero
trips and repeat invocations through one library in both memory profiles.
Guarded runtime read loops now also prove all reachable source addresses while
retaining runtime bounds and row-local tail masks. Both profiles have i32/i64
Merlin checks including negative/zero trips. Runtime stores additionally retain
conditional tensor state, repeated read/modify/write updates and initial values
on zero trips or a false source branch. Fresh initialization is proved from
pre-loop writes or a guarded positive trip count with complete branch coverage.
General pointer/while state, indexed conflicts and alias/effect profiles remain
separate capabilities.

[Runtime dimension ABI v5](dynamic_shapes.md) now separately qualifies actual
dense input/result dimensions and runtime logical ceil-divided grids. Both
memory profiles reuse one Merlin library across independent shapes including
zero; the payload observes the actual program count. A scalar-only generator
also allocates a result without any tensor input size to infer from. Caller
shape bindings and executable MLIR guards reject inconsistent dimensions, and
output allocation budgets are checked before construction. This does not close
non-dense multidimensional strides, general indexed effects or shape algebra.
Two-axis tiled matrix addition now qualifies changing runtime row strides and
empty/tail shapes through both memory profiles and the actual Merlin library
interface. Exact symbolic offset/stride cancellation preserves the tile axes;
a mismatched dense-layout declaration refuses before memory continuation.
Runtime M/N/K GEMM also qualifies both profiles with direct/transposed RHS,
retained tile contractions and K loop. One library per profile/storage layout
executes empty dimensions, K=0, exact tiles and ragged tails. This extends dense
address/shape coverage; it does not admit arbitrary pointer recurrences or effects.
Read-only indexed gathers now also qualify named runtime matrix allocations and
independent index/result counts in both profiles, with repeated i32/i64 indices
and empty inactive domains. Tensor-profile active checks use the actual extent;
declared maximum allocations and suffix strides only bound the arithmetic proof.

1. Grouped/nonlinear logical grids and general pointer recurrences with established output
   ownership; preserve useful loops and reuse rather than serializing launches.
2. General strided/block descriptors, runtime layout/shape algebra, arbitrary store masks and irregular access with
   defined gather/scatter conflict behavior and supported alias/effect contracts.
3. Mixed-precision, cast, richer combiner/effect and contraction numeric policies. A source
   attribute that upstream drops cannot be repaired by provenance alone.
4. General runtime/mutation/result schemas and staged support for all retained
   payloads. Static standalone kernels and bound models now have actual Merlin
   compiler/host execution checks; those measured cases do not close broader profiles.
5. Explicit effectful/backend profiles for atomics, synchronization, volatile
   observations and hardware intrinsics. Arbitrary hardware-observing assembly
   has no universal equivalent in a pure mathematical tensor ABI.

The archived Microsoft implementation is useful evidence for historical patterns
and regressions. Its monolithic converter imports many of the same patterns as
the staged pipeline, including reductions, matmul, reshape and a cumsum-specific
scan converter. It is not another complete universal converter that removes these
ABI/numeric obligations. See the [pinned source comparison](triton_shared.md).

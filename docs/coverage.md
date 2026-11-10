# Measured coverage and remaining boundaries

[Mixed tensor results and diagnostic streams](mixed_effects.md) now reach
the same Merlin interface with an explicit v2 reference, complete typed result
archives and per-result word/value policies. The 92 new reference/integration
cases pass, including all three memory profiles, exact words for all nine
existing carriers, runtime shapes/loops, one-call observation, reused libraries
and rehashed semantic refusals. Another 51 effect/CLI/registry regressions pass.
Fourteen control-state cases check the upstream prepass selection, and three
native regressions preserve readonly tensor accumulators. Merlin's eight
staging fixtures forbid reads of references and evaluator digests; its docs
and structure gates pass. These transport/effect results do not qualify broader
low-precision arithmetic, pointer diagnostics, external/synchronization effects,
runtime/alias models, the maintained full corpus or target materialization.
The saved mixed demo and actual qualification pass 26/45 artifact checks;
its reused-library invocation passes 25. The 20-package observation report
at `out/mixed-result-coverage/coverage.json` retains all 50 definitions,
with 25 observed and 21 present in captures of host-qualified packages.

[Logical print events](logical_prints.md) now have 53 focused cases through
actual native/Merlin execution, plus seven CLI cases. All three memory
profiles retain signed/unsigned words, floating decimal/hex profiles, logical
coordinates, masks/fills, source control positions and reused libraries.
Physical thread replicas/interleaving and the original-word float-hex software
profile are explicit differences. Another 68 frontend regressions and 84 Merlin
checks pass. The saved hex demo passes 32/51 package/qualification artifact checks.
The explicit 19-package observation report at
`out/logical-print-coverage/coverage.json` retains all 50 pinned definitions:
25 observed, with 21 present in captures of Merlin-qualified packages. These
are selected composition observations, not universal opcode/type support.
Pointer/FP8 print forms and target materialization remain required work.

[Length-aware literal assertions](literal_diagnostics.md) now pass 75 focused
diagnostic/source/effect regression cases through the actual native/Merlin
interface; Merlin's corresponding native/serialization/BF16 checks pass 70.
NUL/UTF-8, rank-two/dynamic conditions, reused libraries, original CF messages,
BF16 and CAS compositions retain exact outcomes and original public signatures.
The saved literal-check package/qualification pass 29/49 artifact checks.
Broader print formatting, external/synchronization effects and target handling
remain open; this adds qualified assertion forms, not universal opcode coverage.
The explicit 18-package report at `out/literal-diagnostic-coverage/coverage.json`
retains all 50 definitions, with 24 observed and 20 present in captures of
Merlin-qualified packages. Adding a message form does not increase the number
of distinct observed definitions.

[Observable-outcome packets](effect_abi.md) now support real void and initially
failing entries through the actual Merlin compiler/library ABI. The 40 focused
cases qualify zero-argument, scalar and readonly dynamic tensor calls in all
memory profiles, exact diagnostics, normal return, source abort and reused
libraries. Rehashed source/outcome and real-result signature refusals remain
checked. The 73 existing producer regressions and 230 Merlin staging tests pass.
Expected failures are explicit observed outcomes, not successful tensor values.

[Source assertions](source_assertions.md) now have actual native/Merlin success
and failure checks, including scalar/tensor conditions, original source regions,
zero trips and complete model2MLIR tuple/state captures in all three profiles.
The focused suite has 24 cases; the diagnostic/guard/consumer regression passes
41. The subsequent explicit 15-package observation report retains all 50
definitions, with 24 observed and 20 present in captures of Merlin-qualified
packages. Broader print diagnostics remain open alongside other external effects
and target materialization; effect-only and initial expected-failure packaging
uses the separate outcome profile above.

`package-coverage` now provides the [receipt-bound observation registry](package_coverage.md)
against all 50 pinned TTIR definitions. Its initial explicit 13-package selection
observes 22 definitions, with 18 present in captures of nine Merlin-qualified
packages. These are selected composition observations; unmeasured definitions,
unknown operations, other dialects and the remaining full corpus/refusal work
remain explicit. This count is not overall opcode support.

[Lifted source constants](lifted_constants.md) now bind exact safetensors bytes
into the actual Merlin interface for f16/bf16/f32/f64, logical Boolean and
signed i8/i16/i32/i64 models in all three memory profiles. Observation and
matmul literals retain source calls, tuple results, original contractions and
nonzero accumulators. [Logical Boolean memory](boolean_memory.md) abstracts
the checked byte-slot convention while retaining masks, fills and source IDs.
Transport qualification does not imply arbitrary arithmetic or noncanonical
Boolean byte support; FP8, scaled-dot, external effects, dynamic/alias model
contracts and accelerator materialization remain open.

[BF16 storage/casts](bfloat_precision.md) now have 20 focused native/Merlin cases
covering complete 16-bit copy/widening domains, RNE boundaries, runtime select
loops and original BF16/F32 contractions. The combined low-precision and floating
atomic regression passes 97 cases. The explicit raw-word NumPy encoding differs
from legacy model2MLIR BF16 archives. [Real BF16 models](bfloat_models.md) now
retain learned state and complete results through all three profiles, using
exact parent words or proved unique legacy embeddings. Their 24 focused cases
and 62 existing model/atomic/half regressions pass. Other numeric policies,
general lifted/dynamic model ABIs, scaled formats and target materialization remain open.

[Integer RMW atomics](atomic_memory.md) now have an explicit private-invocation
profile through actual Merlin execution: all nine integer operations, scalar/
tiled i32/i64, collisions, original old values, initialization and race checks.
The 102 focused cases and the independent 4,663-history ownership oracle do not
close floating atomics, publication/synchronization, aliases or target work.
Integer CAS now additionally qualifies i16/i32/i64 scalar/tiled/rank-two events,
selected buffers, recurrences, collisions and all source orders/scopes through
actual Merlin execution. All 56 focused cases plus the visible-demo test pass;
both atomic model kinds pass ten model cases. Portable input retains a standard
generic atomic region, and the native compatibility requirement is explicit in
[the CAS contract](atomic_memory.md#integer-compareexchange).
An actual model2MLIR atomic model now returns both updated state and old values
through native and Merlin libraries, preserving learned weights and all parent
tuple references. The 32 atomic/existing model regressions and 23 compiler-only
staging tests pass. This closes the flat tensor tuple-reference package gap;
nested results, broader model mutation and dynamic model ABIs remain open.
Region-local atomics now also preserve branch/loop positions, old-value
recurrences and final same/selected-buffer pointers through actual Merlin
execution. All 17 new region cases, 93 staging cases and 90 native consumer
loop/lifetime cases pass; see [the region contract](atomic_memory.md#branches-recurrences-and-selected-buffers).

The pinned [v3.7.1 source inventory](ttir_operations.md) contains all 50 `tt`
operation definitions, extracted with TableGen and partitioned by semantic
family. It establishes an opcode denominator, not universal operation, type,
attribute or composition support. Standard dialects remain additional gates.

[Unsigned high products](integer_products.md) now qualify i32/i64 scalar and
tiled words in all three memory profiles through actual Merlin libraries.
Triton's original Philox rounds also retain runtime seeds/counters and four
exact result streams, checked against an independent published-vector-tested
reference. All 35 focused cases pass; other RNG/numeric policies remain open.

[Actual logical-grid bounds](logical_grid_bounds.md) now prove unmasked dense
unit rows and exact runtime output partitions, including mixed clipped columns.
The initial focused suite passed 87 cases with two strict expected failures.
The [isolated native memory-event bridge](source_words.md#memory-inside-structured-regions)
now closes both conditional-store failures through actual Merlin execution.
New scalar/tiled branch/loop cases also qualify nonuniform fills and observed
writable state. Bounds, ownership and initialization checks remain separate;
this does not establish general irregular memory coverage.

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
Finite mixed-base joins/backedges and lane selection now additionally execute
through [logical selectors](selected_pointers.md), retaining original memory
positions, mutations, masks/fills and paired shape transformations.
Broader while ownership and aliases remain open. These checks do not imply
universal opcode/attribute coverage.

[Tiled online attention](attention.md) now also qualifies through the actual
Merlin library interface with two tile contractions, query reuse and three
running softmax states retained. Empty/exact/ragged query/key domains reuse one
library against an independent finite-f32 reference tolerance; dense reference
scores stay evaluator-only. This does not establish all attention variants,
floating-point policies or accelerator execution.

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

The [scalar map profile](elementwise_map.md) now additionally qualifies
`tt.map_elementwise` and its return with `pack=1`, original scalar branches,
tuple results and runtime recurrences. Fourteen cases include actual Merlin
execution over changing/empty inputs in all three memory profiles. Packed
cross-element groups and mapped source memory remain explicit gaps.
Tile-local [histogram](histogram.md) now retains exact masks, padding values and
indexed counter state in standard MLIR, with 34 native/Merlin cases covering
both i32/i64 inputs and runtime logical tile domains. Global atomic histogram
accumulation and broader types/effects remain separate capabilities.

The source-word selected-buffer path additionally qualifies cat/split and
repeated intra-tile `tt.gather` through actual Merlin libraries. A checked native
intake rewrite bridges cat absent from the installed optimizer; a direct
parallel generic/extract expansion adapts Meta's tile-gather lowering. Original
source graphs, chosen ordering and version boundaries remain explicit in the
[selected-pointer contract](selected_pointers.md).

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

### Typed atomic profile

`typed-atomics` adds bitwise f32/f64 `tt.atomic_rmw xchg` through upstream
unstructured conversion, exactly-once standard `memref.atomic_rmw assign`,
native and actual Merlin execution. Source operation/order/scope and value dtype
are retained under memory admission v4. Integer RMW/CAS remain available; the
v3 integer-only profile is unchanged. Dedicated raw-word tests cover NaN payloads,
signaling NaNs, signed zeros and subnormals; a real model2MLIR capture checks
weights and both tuple results. Floating min/max's integer pointer views and
IEEE f64 addition now have the separate extensions below; other floating-add
numeric policies remain open. See [atomic memory](atomic_memory.md).

### Ordinary logical word views

Equal-width, byte-aligned pointer `tt.bitcast` now works for ordinary loads/stores
on the guarded source-word path. Native and actual Merlin execution retain
f16/i16, f32/i32 and f64/i64 bits through masks/fills, selected roots and
for/while recurrences. Exported MLIR retains original storage tensors and
register bitcasts, with `prov.logical_word_views` annotations. Differing widths,
address-space changes still need further contracts and expansions; atomic CAS
word views now have the explicit typed expansion below. See [logical pointer views](pointer_views.md).

### Atomic logical word views

The typed profile now qualifies all nine i32/i64 RMW operations on equal-width
f32/f64 storage and Triton's floating min/max source algorithm. Native and actual
Merlin raw-word comparisons cover NaN payloads, signed zeros, subnormals, masks
and dynamic empty/tail grids; scalar min/max is included. Real model2MLIR
min/max models preserve weights and both tuple results. The portable interface
retains standard atomic regions and word-view provenance; checked native
legalization emits one integer atomic per primitive without whole-buffer
conversion. This does not qualify every floating numeric policy, external participants
or accelerator execution. See [atomic memory](atomic_memory.md).

Both widths also qualify inside branch/for/while regions with old-value and
pointer recurrences, scalar/vector events and finite uniform/per-lane buffer
selection. Repeated inlined helpers retain separate call contexts through the
actual Merlin interface. Twelve native/Merlin region cases and two additional
Merlin repeated-helper cases pass with independent raw-integer oracles; these
are composition tests rather than an arbitrary-kernel coverage claim.

Integer-word CAS additionally qualifies f16/i16, f32/i32 and f64/i64 views,
including rank-two tiles, scalar/vector events, uniform/lane-selected buffers,
constants and source-region pointer/old-value recurrences. The 42 compositions
pass native comparison and raw-word Merlin checks; 36 producer evidence/template
checks pass. Three demos and 15 real model2MLIR cases preserve public storage
types, weights and tuple results. Native helpers return integer words, avoiding half
floating ABI conversions. The stage-v2 typed-template addition is explicit in
[the package contract](package.md); no physical alias or opaque portable call
is introduced. Differently typed selected roots now have the additional typed
qualification below; arbitrary synchronization/accelerator behavior remains open.

### Selected ordinary storage representations

[Mixed storage selectors](mixed_storage.md) now qualify ordinary memory through
equal-width integer/floating roots without changing their public tensor types.
The 52 focused cases retain exact fills, selected updates, read-after-write
observations and final pointers. Closed two-/three-way selector permutations
carry scalar indices into immutable tiles through original for/while loops.
Actual Merlin libraries vary counts, choices and shapes; three genuine model2MLIR
captures preserve weights and raw results. Another 75 pointer/event regressions
pass. General computed selector recurrences, aliases/layouts and accelerator
materialization remain separate qualification gates.

Selected integer atomics additionally qualify mixed integer/floating roots:
all nine i32/i64 RMW operations and i16/i32/i64 CAS retain actual storage types,
source context and old values. The 76 execution cases, six profile refusals and
five real three-result model captures pass, including independent collision
histories and repeated helpers. The consumer matches actual captured storage
independently of logical CAS formals; 90 new and 200 existing declaration/staging
cases pass. Eighteen existing numeric/profile executions also pass. See
[the precise scope](mixed_storage.md#selected-integer-atomics); numeric floating
views of integer storage and broader effects/target coverage remain open.

### F64 atomic addition

The typed profile admits IEEE f64 FADD with round-to-nearest-even and subnormal
preservation. Native/Merlin scalar/vector tests retain exact finite/zero/old
words and check arithmetic NaN classes separately from payloads. Region
recurrences, finite buffer selection, collision histories without assumed lane
order, and a genuine f64 model2MLIR tuple-result model reach the same Merlin
interface. New captures use standard generic atomic regions with explicit
NaN quieting under `ieee-f64-rne-add-quiet-nans`. The earlier standard-addf
contract remains readable without a quieting guarantee after constant folding.
The selected numeric contract/payload is recorded in provenance and v4
source/access/payload declarations; tensor/scalar ABI and core artifacts are
unchanged. F16/bf16 additions and
accelerator qualification remain open; see [the numeric contract](atomic_memory.md#ieee-f64-atomic-addition).

### F32 atomic addition policies

The typed profile explicitly selects IEEE or sign-preserving input/result FTZ
for f32 FADD. Both retain masks, returned-old words, source recurrences and
finite buffer selection through actual Merlin execution. FTZ's portable atomic
region contains actual bit arithmetic and NaN quieting; the native feature
uses an atomic-load/CAS retry loop with checked borrowing and lifetime gates.
IEEE retains subnormals and explicit NaN quieting in its standard atomic region.
Missing policy is a located
refusal rather than an inferred target choice. A real f32 model2MLIR model
preserves weights and both tuple results. Constant ±0/subnormal/NaN tests retain
canonicalized effects and exact old words. See [the f32 contract](atomic_memory.md#f32-addition-with-explicit-source-policy).

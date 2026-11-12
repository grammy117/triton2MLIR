# triton2MLIR

Read the [overall status](docs/status.md) for progress toward arbitrary TTIR
kernels reaching Merlin, current evidence and the remaining completion gates.

[Protected mixed-precision intake](docs/per_cast_fp8.md#protected-composite-precision)
now expands per-cast FP8 contracts before official composite-dot normalization.
Original and normalized graphs remain distinct, while standard MLIR retains
ragged tiles, runtime K reuse and the original accumulator. The precision
selection covers BF16x3/x6 and logical TF32/x3 through native, Merlin host and
functional Spike execution. A [complete native source reader](docs/native_source_print.md)
also preserves original block-pointer/descriptor types and every cast attribute
before normalization; six mixed-policy cases qualify through native/Merlin/Spike.
Direct external descriptor arguments also qualify selected mixed-policy kernels
through those execution boundaries, retaining mandatory expanded-ABI validation.
Broader source versions, allocation-root/model compositions and accelerator
qualification remain open.

[Source effects on the target](docs/target_effects.md) now preserve actual void
entries, assertions, diagnostics and returning tensors through Merlin/Spike.
Checked invocation/outcome markers and separate source-byte frames distinguish
source failure from setup errors and traps. Original float signs, formatting
and complete tensor/input observations remain explicit.

[Standalone dynamic target invocation](docs/standalone_target.md) now reaches
Merlin's ordinary RISC-V backend with unchanged kernel MLIR and native scalar
arguments. Ten actual Spike images compare eighteen complete results exactly,
including empty domains, signaling NaNs and BF16 source words. Changed runtime
shapes/scalars keep identical kernel-object hashes in the qualified cases;
each image's harness still embeds its invocation. Full arbitrary-kernel,
model-view/effect and accelerator coverage remain open.

[Whole-model target execution](docs/target_execution.md) now carries selected
complete packages through Merlin's ordinary RISC-V build and actual Spike runs.
i32/i64 learned tuple models retain all six results, including empty values;
a lifted-BF16 tiled contraction also passes. Complete output/reference and
readonly-input checks are retained. Broader targets, effects and performance
remain separate qualification requirements.

[Optional source assembly implementations](docs/source_assembly.md) now expose
checked integer programs as standard arithmetic and parallel Linalg. The PTX
decoder is a separate package under `adapters/ptx`; the portable core retains
original source/provider identities and checks its executable value/effect
boundary. Hardware observations and unsupported instructions remain refused.
Real i32/i64 model captures now retain learned words and all six original
results through Merlin; i64 also passes public model CLI binding. Their
computed tensor captures use the consumer's checked destination handling.

The [maintained product corpus](docs/product_corpus.md) now records real
conversion refusals and Merlin execution alongside successful packages. The
combined explicit report observes all 50 pinned TTIR definition names in source
and native/Merlin-host-qualified captures under selected profiles. Broader source
assembly, packed/effectful map compositions and numeric/runtime/target closure remain
open; these counts measure compositions, not universal operation support.

[Mapped source reads](docs/memory_maps.md) now retain logical pointers,
branches/loops and five-result snapshots around ordinary kernel writes through
actual Merlin libraries. The default corpus has eleven qualified cases and two
located refusals; source assembly and packed grouping remain open.
Selected-root map arguments, pointer results and mixed results additionally
qualify through twelve libraries. Real i64/f32 model captures preserve learned
state and every one of fourteen original results through Merlin; i64 also
passes the public model CLI. See [the complete map contract](docs/memory_maps.md).

[Layout-independent packed maps](docs/packed_maps.md) now retain their original
packing annotation while lowering proved slot-local arithmetic to parallel
Linalg. Eight libraries cover pack two/four, i32/i64/f32 and runtime tiled
recurrences. Fourteen additional libraries preserve scalar branches and
per-element for/while trip counts, with 56 further runtime replays. Two real
model captures retain learned weights, dependent calls and all eight results
through Merlin host and actual Spike execution with exact-word comparison.
Coupled/asymmetric slots
now have an explicit [source-specialized grouping path](docs/source_grouping.md)
through seven additional Merlin libraries. The current
[coverage selection](docs/coverage.md) reopens 106 packages and 74 standalone
qualifications. The frontend never infers
adjacency from tensor shape or assigns consumer hardware placement.
Three complete source-specialized model captures additionally preserve learned
weights, dependent calls and all twelve results through native/Merlin host and
actual RISC-V/Spike execution. Semantic source geometry is bound to original
typed model-call scalars; final optimized GPU correspondence remains open.
[Packed mapped reads](docs/grouped_memory.md) now also retain source grouping,
selected roots, pointer results and snapshots across ordinary writes. Six
actual native/Merlin libraries replay five domains each, retaining all results
and existing memory/lifetime checks. Two complete learned-state models now also
preserve original snapshots across dependent calls and all thirty results
through native/Merlin host and actual Spike execution with exact-word comparison.
Returned-pointer model compositions and broader targets remain open.

[Typed owning tensors with byte views](docs/typed_storage.md) now preserve their
public dtype and shape while selected roots use executable internal byte planes.
Exact word transport, byte-level guards and independently derived per-result
scratch costs reach native and Merlin execution, including complete model captures.
[Original changed-width pointer/descriptor parameters](docs/typed_parameters.md)
also share typed roots. [Byte-backed integer atomics](docs/byte_atomic_words.md)
now retain indivisible word updates, byte order, selected for/while state and
old-value tensor recurrences through complete Merlin host packages. Real models
keep learned state and all seven results through API and CLI binding.
The [current completion plan](docs/completion_plan.md) separates these working
paths from remaining full numeric/effect/runtime/target coverage.

Translate Triton TTIR into portable tensor MLIR while retaining the kernel's original dataflow
and intent in annotations and provenance. Examples now include ragged tiled GEMM,
descriptor memory and associative scans; see the
[roadmap](ROADMAP.md) for the path through tiled GEMM, attention and broader kernel semantics,
and [plan.md](plan.md) for the architecture.

Contributor/commit rules are in [AGENTS.md](AGENTS.md). Broader coverage will reuse
[triton-shared's existing passes and analyses](docs/triton_shared.md).

[Indexed descriptors](docs/indexed_descriptors.md) reuse the official pre-layout
pointer expansion for gather, scatter and reducing stores. Loaded indices,
runtime bounds/padding, allocation state and atomic kinds reach standard MLIR;
original descriptor intent remains in provenance with hardware mapping undecided.

[External descriptor parameters](docs/external_descriptors.md) now expand into
typed logical allocation/shape/stride/padding/rounding fields. Actual native and
Merlin execution retain non-contiguous memory, state and two-descriptor tiled
f32/BF16 GEMM with its original K reuse and C accumulator. Public field bindings
and remaining origin/alias/model/target work are explicit.

[Descriptor allocation state](docs/descriptor_state.md) now preserves runtime
origins and shared writable roots through real TTIR calls before inlining.
i64/f32/BF16 libraries, a public demo and an actual model2MLIR tuple capture
execute through Merlin. Original and normalized source bytes travel in a
separately checked compiler capsule; physical views and target work remain open.

[Scaled contractions](docs/scaled_dot.md) now retain FP4 packing, FP8/half
formats, optional per-group scales and original accumulation in standard
tiled MLIR. `shared-demo --kernel scaled-dot` produces an independently
checked Merlin package with its internal-precision choice explicit.

TTIR integration development uses `ttir-experiment-grammy` in both
[Merlin](https://github.com/ucb-bar/merlin/tree/ttir-experiment-grammy) and
[model2MLIR](https://github.com/grammy117/model2MLIR/tree/ttir-experiment-grammy).
This standalone triton2MLIR repository uses `main`. Publish tested changes to
these branches; retain existing branch names and published history.

[Readonly aliases](docs/readonly_aliases.md) now pass through typed package
preparation as exact independent input values; by-value scalar carriers retain
their initial values even when backed by writable tensor storage. Writable
source aliases now have an [explicit allocation-state interface](docs/allocation_state.md),
with runtime offsets, shared updates and one returned root. The generic
`shared-tensor --allocation-bindings` route records its signature/source differences;
undeclared writable overlaps still refuse.

[Typed allocation aliases](docs/allocation_word_views.md) now bind equal-width
f16/BF16/f32/f64 and integer source arguments to one owning root, with actual
read-after-write state through Merlin. Differently typed descriptors and a
complete model2MLIR tuple capture use that same executable binding path.
[Different-width byte-buffer views](docs/byte_aliases.md) now retain typed word
loads/stores, byte updates and explicit byte order through Merlin. Pointer and
descriptor parameters can share one byte root; a real model capture keeps all
five results. Typed-root packing and integer byte atomics have their qualified
profiles above; broader effects and applicable targets remain open.

[Dense model tensor views](docs/model_tensor_views.md) now bind matrix values
to those roots with explicit standard reshapes. Actual i64 and f32/f16/BF16
model captures retain shared updates, learned state, tiling and all results
through Merlin; the v2 binding contract and remaining physical-view work are explicit.

[Source model views](docs/model_source_views.md) now bind actual transposed and
stepped operands to an owning root after independent SSA/coordinate checks.
Three i64/f32/BF16 models retain all seven results each through native, Merlin
host and functional RISC-V/Spike execution. The explicit v3 binding profile
preserves the public model ABI; general physical/mutating views remain open.

[Per-cast FP8 contracts](docs/per_cast_fp8.md) now allow IEEE and both saturation
policies in the same source kernel. Independent standard expansions retain
source loops and helper correspondence; every format/mode pair has selected
native/Merlin/Spike evidence, with earlier normalization and broader model
compositions still open.

The [shared tensor frontend](docs/shared_tensor.md) now reuses those passes for 2-D add,
masked softmax, IEEE matmul and general pure associative scans, emitting functional tensor/linalg IR. Its packages are checked
by **native LLVM host execution** against independent NumPy references. It currently accepts
static shapes and [named dense runtime dimensions](docs/dynamic_shapes.md),
clipped affine tail masks and disjoint output tiles over logical grids;
this is not full arbitrary-kernel coverage.

```bash
python scripts/fetch_shared_opt.py --out .scratch/shared-opt
export TRITON_SHARED_OPT=$PWD/.scratch/shared-opt/triton-shared-opt
python -m t2m shared-demo --kernel softmax --out out/shared_softmax --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel matmul --out out/shared_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel tiled-matmul --out out/shared_tiled_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel loop-matmul --out out/shared_loop_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel ragged-matmul --out out/shared_ragged_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel blockptr-matmul --out out/shared_blockptr_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel descriptor-matmul --out out/shared_descriptor_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel tuple-scan --out out/shared_tuple_scan --python /path/to/native-env/bin/python
python -m t2m verify-bundle out/shared_softmax
```

The optional fetcher extracts a pinned standalone optimizer from a verified distribution;
it builds nothing and installs no target backend. An existing optimizer is also supported.
See [the package differences](docs/package.md) and [native execution scope](docs/native_execution.md).

The [package coverage command](docs/package_coverage.md) indexes verified
captures against the full pinned TTIR inventory, keeping unknown/unmeasured
operations and native/Merlin evidence separate.
`shared-demo --kernel checked-copy` now also preserves
[source assertions](docs/source_assertions.md), including observable failure
diagnostics and original loop/branch placement through Merlin.
`effect-demo` also packages void and initially failing kernels with
[explicit outcome references](docs/effect_abi.md), preserving their real public
signatures through the same compiler and native library interface.
`effect-demo --case literal-check` also exercises embedded NUL and Unicode
messages through the [length-aware diagnostic path](docs/literal_diagnostics.md).
`effect-demo --case print-words` and `print-hex` now also exercise
[logical print events](docs/logical_prints.md), retaining values, source order
and logical coordinates while explicitly abstracting physical print replicas.
`effect-demo --case mixed-results` now returns a complete tensor tuple and
its diagnostic stream from one native/Merlin call, retaining runtime state
recurrence and exact i64 words. See [the explicit mixed reference profile](docs/mixed_effects.md).
`shared-demo --kernel precise-math` also retains both clamp NaN modes and
precise division/sqrt, with [independent IEEE references and BF16 consumer support](docs/scalar_math.md).
`shared-demo --kernel fp8-transport` preserves all four FP8 formats through
[typed byte storage, runtime control and exact word references](docs/float_casts.md).
`shared-demo --kernel fp8-widen` also retains portable casts and qualifies
[exact widening to standard floating types](docs/float_casts.md#executable-widening).
`shared-demo --kernel rounded-cast` preserves explicit RTZ with
[standard-format conversion and caller-intent evidence](docs/float_casts.md#rounding-and-standard-format-conversion).
`shared-demo --kernel fp8-narrow --fp8-overflow satfinite` retains
[per-step FP8 rounding and explicit overflow/infinity contracts](docs/float_casts.md#fp8-narrowing-and-explicit-overflow-contracts).
`shared-demo --kernel fp8-dot` retains
[mixed FP8 contractions, runtime K tiles and original accumulators](docs/fp8_contractions.md).

The [model2MLIR binder](docs/model_integration.md) now replaces registered kernel calls with
translated tensor bodies while preserving the real model ABI, weights and input bindings.
[Lifted source constants](docs/lifted_constants.md) also retain original typed
bytes through Merlin, with explicit compiler-only bindings and evaluator
archive exclusion. Boolean byte slots use checked logical tensor memory.
Complete models pass native comparison against their unchanged PyTorch goldens; optional
checks invoke Merlin's interface/runtime generator and compiler-only staging. The
optional [native Merlin compiler path](docs/merlin_consumer.md) builds and executes
its actual shared library, including the original MLP pair. Derived receipts remain
explicit; accelerator staged/device execution is still unqualified.
Standalone shared tensor packages use `qualify-merlin` and `verify-merlin` for
the same real compiler/host path, with separate qualification evidence and no
Torch wrapper; see [the consumer contract](docs/merlin_consumer.md).
The [indexed-read path](docs/indexed_memory.md) additionally preserves loaded
indices, masked fills and reuse as parallel tensor dataflow, with native and
Merlin host checks; try `shared-demo --kernel indexed-gather`.
Typed runtime scalar inputs also reach Merlin by value. `invoke-merlin` reuses a
qualified library with new scalar values and separate invocation evidence; see
[the runtime ABI](docs/runtime_abi.md).
`shared-demo --kernel runtime-recurrence` preserves a runtime tile recurrence,
including zero trips. The explicit `--memory-profile buffer` also retains the
upstream standard memory payload behind the same public tensor/scalar ABI;
see [its proof scope and package differences](docs/full_memory_bridge.md).
The explicit `--memory-profile guarded-buffer` additionally supports loaded
store indices with executable bounds/conflict/coverage checks through Merlin;
see [the memory contract and metadata cost](docs/guarded_memory.md).
Add `--address-policy source-words` to retain original integer-word arithmetic,
pointer recurrences and nonlinear grids through the official unstructured
converter. Same-base pointer branches and runtime while loops also qualify
through Merlin, retaining their control flow, word math and final pointers.
[The policy and package differences](docs/source_words.md) are explicit.
Finite mixed-buffer branches, lane selections and changing loop bases also
qualify through that path; [selectors keep each original memory event](docs/selected_pointers.md).
`shared-demo --kernel selected-memory --memory-profile guarded-buffer --address-policy source-words`
packages lane-selected reads/writes with three ordered results and source provenance.
`shared-demo --kernel online-attention` retains query-tile reuse, two tile
contractions and running softmax state through Merlin's tensor interface. See
[the measured attention profile](docs/attention.md).
`shared-demo --kernel histogram` retains tile-local integer counts, exact masks
and histogram intent through standard tensor/SCF operations; see
[the expansion and measured profiles](docs/histogram.md).
The explicit `--effect-profile atomics` adds private-invocation integer RMW
and i16/i32/i64 CAS through actual Merlin execution, retaining old values,
selected-buffer effects and source recurrences. `shared-demo --kernel atomic-cas`
and a real model2MLIR tuple-result demo are described in
[atomic memory](docs/atomic_memory.md), including native-toolchain requirements.
The separate `--effect-profile typed-atomics` adds bitwise f32/f64 exchange;
`shared-demo --kernel atomic-exchange` produces a working floating example
with guarded-buffer/source-words memory and the unchanged public tensor ABI.
That profile also retains integer RMWs on floating word views and source floating
min/max; `shared-demo --kernel atomic-min` / `atomic-max` emits examples. Native
word-region qualification uses the explicit `T2M_MERLIN_ROOT` support path.
`shared-demo --kernel atomic-word-cas16` (`32` or `64` also works) retains
integer-word CAS on original floating storage, including exact old bits and
standard atomic regions. Three real model2MLIR captures reach the same Merlin
interface; [the package extension](docs/package.md) records the typed native
template boundary explicitly.
`shared-demo --kernel atomic-fadd` adds IEEE f64 atomic addition with explicit
rounding/subnormal/NaN policy through the same package and actual Merlin path.
New f64 captures retain explicit result NaN quieting in standard atomic regions
and require the `T2M_MERLIN_ROOT` native support path.
F32 addition now selects `--f32-atomic-add ieee` or `ftz` explicitly; try
`shared-demo --kernel atomic-fadd32`. FTZ is executable standard bit arithmetic
inside the atomic region, including NaN quieting, with checked native support
through `T2M_MERLIN_ROOT`. The policy is independent of hardware mapping.
[Logical pointer word views](docs/pointer_views.md) also preserve equal-width
reinterpretation through selected buffers and runtime loops, with register
bitcasts on the original tensor storage types.
[Mixed storage selectors](docs/mixed_storage.md) also retain ordinary memory
through f16/i16, f32/i32 and f64/i64 roots. Closed per-lane selector permutations
carry scalar table indices in the original loops, preserving tiling and reuse
through actual Merlin execution and genuine model2MLIR captures.
The typed atomic profile also preserves integer RMW/CAS through mixed-storage
selectors, with actual root dtypes, original old words and complete three-result
model captures; see [the qualified combinations](docs/mixed_storage.md#selected-integer-atomics).
`shared-demo --kernel runtime-memory-loop` additionally preserves guarded buffer
reads and exact row tails inside a runtime loop.
`shared-demo --kernel runtime-memory-store` retains a read/modify/write recurrence
and incoming output values for zero trips.
`shared-demo --kernel dynamic-count` varies actual tensor/result sizes and logical
launches. `shared-demo --kernel dynamic-matmul` preserves tiled GEMM and its K
loop with runtime M/N/K; both memory profiles execute through Merlin across
changing and empty shapes. See [runtime dimensions](docs/dynamic_shapes.md).
`shared-demo --kernel dynamic-gather` additionally keeps runtime allocation and
index/result extents with masked indexed reads and exact inactive fills.
`shared-demo --kernel masked-store` and `--kernel masked-matrix` preserve
nonrectangular per-lane writes, source loops and transposed matrix layouts;
both have actual Merlin library checks over changing and empty shapes.
[FP16 tensor storage](docs/half_precision.md) also executes through both profiles,
with typed narrowing, mixed FP16/FP32 contractions and preserved masked state.
[BF16 tensor storage](docs/bfloat_precision.md) now also qualifies exact word
transport, RNE casts, runtime selects/loops and BF16/F32 contractions through
Merlin. `shared-demo --kernel bfloat-matmul` emits a visible package; its raw
uint16 archive encoding is explicit. [BF16 model captures](docs/bfloat_models.md)
also retain exact source words, learned state and every tuple reference through
the existing model2MLIR/Merlin interface, preserving the parent's value archives.

The current demo captures a **real Triton kernel without a GPU**, recovers exact-tile tensor
accesses, emits standard `scf`/`tensor`/`linalg` operations, executes that serialized IR with a
reference interpreter, and checks it against an independent NumPy computation.

```bash
uv venv
uv pip install -e '.[triton,dev]'
python -m t2m demo --out out/vector_add
python -m t2m demo --operation multiply-add --out out/multiply_add
python -m t2m demo --size 1000 --out out/masked_add
python -m t2m verify-bundle out/vector_add
python -m pytest -q --basetemp=.scratch/pytest
```

Existing output directories are refused. Use a new path for each demo run. The frontend pins
Triton 3.7.1 and xDSL 0.68.0. There is no triton-cpu, Torch or Merlin runtime dependency.
The optional Triton wheel is needed for capture and saved-TTIR parsing, not just source capture.

```text
out/vector_add/
  model.mlir
  weights.safetensors                 # real empty container for a parameter-free kernel
  weights.safetensors.manifest.json   # tensor argument bindings
  inputs.npz, golden.npy, extra.npz, input_order.json
  capture_receipt.json                # hashes, t2m.capture-receipt.v1
  source.ttir, spec.json, provenance.json, compatibility.json, meta.json
```

These use model2MLIR's core filenames and array conventions. **The receipt/trace schemas and tiled
payload support are explicit differences**, recorded in [the package contract](docs/package.md)
and each bundle's `compatibility.json`. Merlin parser acceptance does not imply staged accelerator
execution. The reference path uses a standard-op interpreter; the shared tensor path has a
separate native host execution gate.

Saved input replay:

```bash
python -m t2m lower out/vector_add/source.ttir --spec out/vector_add/spec.json --out out/replayed
```

This command writes verified MLIR and provenance. It does not fabricate example inputs/goldens
or claim to write a full capture bundle without those artifacts.

Strict native verification and legalization can use an **existing** torch-mlir environment:

```bash
python -m t2m verify-mlir out/masked_add/model.mlir --python /path/to/native-env/bin/python --lower-to-llvm
T2M_NATIVE_PYTHON=/path/to/native-env/bin/python python -m pytest -q --basetemp=.scratch/pytest
```

This checks registered operations with unregistered dialects disabled, then optionally runs standard
passes through bufferization and LLVM legalization. It operates on a separate copy; exported kernel
MLIR stays at the tensor/linalg/SCF boundary. No native compiler is built or installed automatically.
LLVM legalization does not claim native code execution or runtime-helper linking.

An existing triton-shared optimizer can also be used directly:

```bash
TRITON_SHARED_OPT=/path/to/triton-shared-opt python -m t2m shared-lower kernel.ttir --out out/shared --stage structured
```

This writes an inspectable upstream stage, with exact tool/input/output hashes and its boundary
recorded. It does not claim our tensor ABI, provenance correspondence or complete package until
those contracts have been qualified. No native toolchain is built implicitly.

Current support: static f32 rank-one logical tensors, positive-stride storage views and offsets,
independent allocations, rectangular boundary masks, defined scalar/splat fills, one write or
readwrite output, and pointwise add/subtract/multiply. Masked loads with omitted `other` are accepted
only when pointwise definedness proves no stored lane can observe the undefined values. Partial
readwrite results preserve incoming values. Identical nonvolatile reads share an SSA tensor value;
source cache/eviction policies remain provenance instead of selecting physical caches.

Unproved bounds, index overflow, inter-program readwrite dependencies, nonrectangular/data-dependent
masks and nested source regions remain refused. See [R1 evidence and remaining work](docs/r1_memory.md).
Output contains an unmapped `scf.forall`; downstream consumers choose hardware realization.

`prov.family`, `prov.op`, `prov.region_id`, stable source/emission IDs and recovered access facts
retain the source intent as operations decompose. Caller intent is labeled as a description, and
cannot override arithmetic. The interpreter also runs with all `prov.*` attributes stripped.

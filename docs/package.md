# Package contract and explicit model2MLIR differences

[Per-cast FP8 semantics](per_cast_fp8.md) add the explicit software source
selection `fp8_overflow="per-cast"`. Each nonidentity TTIR-to-FP8 cast requires
its own semantic `t2m.fp8_overflow` StringAttr; `prov.*` grants no choice.
All source contracts validate and expand to standard computation before
inlining/folding. Protected composite-dot intake uses elementwise tensor
arithmetic before the native normalizer, then preflights partial contractions
separately. The v2 trace retains the complete typed original cast/dot graph and
the normalized graph; source intent does not grant numerical equivalence.
The original source and per-cast reports remain byte-bound;
public tensor/scalar ABI and receipt schemas are unchanged. Older producers
refuse this selection. Merlin consumes executable casts/clamps, with its fixed
narrowing-before-widening source seam required for combined target lowering.
This profile does not claim an unspecified backend's overflow behavior.

[Native source serialization](native_source_print.md) additionally preserves
original pointer/descriptor types when shared's source-only parser rejects them.
The `t2m.native-source-print.v1` provenance report binds fixed printer flags,
actual stock-library identity, source/output hashes and exact native generic
round-trip verification. Cast validation/expansion precedes official normalization
and shared inlining. Compatibility metadata names the Linux/Itanium export and
LLVM C API dependency. Public tensor/scalar ABI and receipt schemas are unchanged;
older producers refuse this composition. Serialization supplies no numerical,
effect, alias or source-equivalence permission.

Original external descriptor arguments with per-cast policies additionally
receive a source-only operation audit before their one-to-many field expansion.
It grants no signature admission: the actual native field binding and full
normalized TensorABI preflight are mandatory. Existing logical-descriptor
records, public ABI and receipt schemas remain unchanged. This direct-field
extension retains separate allocation-root custody requirements. Protected
allocation source capsule v2 now qualifies selected pointer/descriptor recurrences
and a complete enclosing model without changing the public model ABI. See
[the descriptor contract](external_descriptors.md#original-numeric-contracts-before-field-expansion).

[Protected allocation source](descriptor_state.md#protected-contracts-before-allocation-normalization)
uses `t2m.source-allocation-capsule.v2` when an FP8 overflow selection or source
assembly provider requires expansion before allocation normalization/inlining.
Four additional compiler files bind complete native generic source and printer
identity, executable protected TTIR and original typed contracts. Normalization
must consume the protected source; without normalization, the shared source-only
stage consumes it directly. Producer verification and independent Merlin staging
check the exact roster and byte chain. V1 remains supported, while v1-only
readers refuse v2. Tensor/scalar/model ABI and capture receipt schemas remain
unchanged; custody grants no arithmetic, effects, aliases or source equivalence.
Additive `prov.allocation_source_contracts` and
`prov.allocation_original_dot_precisions` annotations retain pre-wrapper numeric
intent on the entry/named contractions after descriptor expansion. They name
the original source hash and graph separately from bound/normalized records,
grant no semantics, and do not change public ABI or receipt schemas.

[Source model views](model_source_views.md) require
`t2m.model-kernel-binding.v3` with
`source_tensor_view_binding_profile: t2m.captured-source-tensor-views.v1`.
Original parent SSA must establish each extra operand's static transpose/slice
coordinates; captured typed scalars and the executable source root pointer
binding must agree. Existing dense reshapes can coexist under their profile.
Both producer verification and Merlin staging reopen these facts, including
complete schema operand accounting. Older v1/v2 readers refuse v3. Public
model ABI, data/reference bytes and receipt schemas remain unchanged. This
profile grants no physical public alias/mutation or numeric permission.

[Observable target effects](target_effects.md) add the external
`t2m.spike-effect-qualification.v1` receipt with `matched-expected-outcome`.
Actual void/result signatures and source capture/reference schemas stay
unchanged. Process, invocation phase, source-byte frame, outcome, readonly
inputs and every returning tensor remain distinct observations. Expected
assertions publish no fabricated tensor or normal-return evidence. These APIs
require the experimental Merlin consumer; numeric receipts keep their meanings.

[Standalone target invocation](standalone_target.md) adds the separate external
`t2m.spike-kernel-qualification.v1` receipt, with unchanged source capture/ABI
schemas. Optional paired input/reference archives bind a new invocation outside
compiler staging. Merlin's `merlin.runtime-invocation-abi.v1` records checked
runtime dimensions and exact native scalar words without rewriting MLIR.
Private scalar descriptors are C-harness storage only; the actual C interface
passes native typed values. Each image embeds invocation values; selected cases
retain identical kernel program/object hashes across changed inputs. Bound
models retain their existing separate receipt and public ABI. Older consumers
without the runtime-signature/exact-scalar APIs cannot qualify this route.

[Functional RISC-V model qualification](target_execution.md) adds an external
`t2m.spike-model-qualification.v1` receipt, leaving the source bundle unchanged.
Compiler staging excludes evaluator archives. Original runtime input values are
separately embedded by Merlin's bare-metal C harness generator, so this target
image is specific to that invocation; reusable host-library evidence retains
its different runtime contract. References remain outside build invocations.
Complete raw-byte output and readonly-input observations are bound separately.
The zero-element `OUT_BIN v1` extension retains empty shapes; older positive-only
readers reject it. Functional ISA execution grants no vector, accelerator, RTL,
timing, performance or universal source-equivalence qualification.
Source-derived capability selection is shared with the host consumer. Target
build records distinguish requested and effective features and bind the actual
selector bytes; source operation bodies still require independent Merlin
validation. Public tensor/scalar and capture receipt schemas are unchanged.
The optional qualification evaluation field `comparison: words` additionally
requires exact output bits and zero tolerances, including NaN payloads and signed
zero. Existing receipts retain their value comparison; older readers may refuse
the additive stricter policy. It changes no source numeric or memory permission.
Alternatively, optional `word_results: [INDEX, ...]` pins exact bits for selected
results alongside the existing value policy for other results. Indices are
distinct, in range and recorded in sorted order; all-word and subset selections
are mutually exclusive. This evaluator-only field is never a compiler option.
Target execution may additionally emit `runtime-inputs.npz` and
`runtime-inputs.json` (`t2m.runtime-input-projection.v1`) outside the compiler
projection. These derive exact BF16 runtime input words from the existing parent
witness or proved legacy embedding; every other input remains byte-identical.
They change no public model ABI, source capture archive or result reference.
The target verifier replays the derivation; older readers may refuse the changed
runtime-file binding. See [the measured target scope](target_execution.md).

[Optional source assembly implementations](source_assembly.md) add explicit
source-language/provider contracts and installed implementation identities.
Every accepted snippet expands to closed standard scalar computation or
parallel Linalg. Public tensor/scalar ABI and receipt schemas are unchanged;
compatibility metadata names the extension and remaining effect/packing gates.
Provider custody and finite qualifications do not establish universal source
equivalence or applicable target execution.

[Proved layout-independent packed maps](packed_maps.md) retain their original
pack annotation while exposing the same scalar function at every logical
element in standard parallel Linalg. The source graph, slot-independence
evidence and source locations remain explicit; no register/lane mapping or
caller-supplied proof tag is trusted. Public ABI and receipt schemas stay
unchanged. Coupled/asymmetric packing still needs its original grouping facts.
The structured extension records `typed-structured-ssa-congruence-v1` for pure
scalar branches and for/while state. Complete bodies, lexical binders, captures
and nested source correspondence remain executable/explicit. Two real captures
preserve learned weights, both calls and every original result through Merlin
host and functional RISC-V execution. Static public model ABI and source runtime
scalar specialization keep their existing documented differences.

[Source-specialized groups](source_grouping.md) additionally admit coupled and
asymmetric maps after an explicit initial TritonGPU layout selection. The public
ABI and receipts remain unchanged; additive `source_layout_specialization`
provenance binds source/TTGIR bytes, actual native-library identity and checked
coordinate maps. The standard payload computes each logical group once and
restores tensor coordinates. This source choice may change math and is not a
consumer placement hint. Final GPU-binary equivalence and accelerator execution
remain unestablished.
Bound models additionally require `t2m.captured-source-layout.v1` and explicit
`source_layout_arguments` for all source options. Each maps to an original
captured typed scalar with the exact selected value. Bundle verification and
Merlin staging reopen the parent call and reject missing/mismatched claims;
kernel provenance must equal its binding copy. This adds custody requirements
without changing model ABI/data or receipt schemas. Older readers do not supply
these additional checks. See [model source grouping](source_grouping.md).

[Packed reads](grouped_memory.md) extend the same specialization to the existing
guarded/source-word memory profile. Parallel Linalg computes each original
scalar region once per logical group, with one vector per returned slot before
coordinate restoration. Per-slot root selectors, offsets, predicates and read
positions remain executable; original snapshots survive intervening ordinary
writes. Compatibility metadata names this extension. Public tensor/scalar ABI
and receipt schemas stay unchanged. No map-write, physical-address or additional
target permission follows from this profile.
Two real model captures now qualify the same mapped-read state path through
native/Merlin host and functional RISC-V execution, retaining learned weights,
dependent calls, live original snapshots and all fifteen results each. Existing
dense value views and captured runtime/source-option bindings preserve the
static public model ABI and unchanged parent data. No new receipt schema or
numeric/alias/target permission is introduced. See [the qualified compositions](grouped_memory.md).

[Changed-width source parameter bindings](typed_parameters.md) additionally
retain original pointer/descriptor types on typed owning allocations. The
argument configuration schemas remain v1/v2; `typed-root-byte-view-v1` records
both pointer types and byte order. Root origins precede real source casts;
source strides keep their original units. Existing source capsules and typed
storage/cost checks remain mandatory. Both standalone and real-model compiler
staging exclude evaluator values and grant no source-equivalence/target proof.

[Typed storage transport](typed_storage.md) now keeps public i16/i32/i64,
f16/BF16/f32/f64 owning tensors around width-changing source views. Standard
word/byte transport and the actual internal memory ABI are explicit; Merlin
independently checks per-result byte-guard costs. Eighteen native/Merlin cases
cover both byte orders, runtime matrix axes and disjoint bytes within one word.
A real model2MLIR capture retains every result through API and CLI binding.
Remaining effect, physical model-view and target requirements stay open.

Bound models can select `t2m.captured-runtime-arguments.v1`: original typed
custom-call attributes supply runtime kernel constants and proved tensor/result
shape casts. The public model2MLIR signature and all parent data bytes remain
unchanged. Per-call evidence and compatibility metadata state the extension;
Merlin independently reopens parent calls for shape/scalar/scratch custody.
Evaluator values and their digests remain excluded. See
[the exact binding contract](runtime_model_arguments.md).

Baseline inspected: model2MLIR `feat/triton-integration` at `035bb5a`, particularly
`m2m/capture/bundle.py`, `m2m/capture/provenance.py` and `m2m/transforms/externalize.py`.
This document specifies the implemented R0/R1 package, not every future roadmap profile.

[Logical indexed descriptors](indexed_descriptors.md) retain the existing tensor/scalar
ABI and receipt schemas. Original descriptor intent and the actual normalizer
identity are additive provenance; compatibility metadata records the logical
shape/stride/padding/atomic boundary and absence of a physical descriptor ABI.
Generated operations and the selected memory/effect profile retain computation.

[External descriptor fields](external_descriptors.md) add an explicit original
argument-to-expanded-field map under `t2m.logical-descriptor-arguments.v1`.
The public inputs are allocation tensors and ordinary typed scalars; source
positions refer to the actual expanded TTIR signature. Original descriptor
source bytes and AST metadata inputs remain explicit. Existing outer ABI/receipt
schemas and compiler/evaluator custody stay intact; this is a logical field
interface with its physical/origin/alias differences recorded in compatibility
metadata.

New model captures additionally support source-bound
[lifted compiler constants](lifted_constants.md). The original source bytes
live in `weights.safetensors` with explicit constant manifest entries and an
additive `m2m.compiler-constants.v1` receipt profile. Derived packages retain
parent bytes and public ABI; Merlin's checked compiler-only projection is a
separate member. Evaluator extra/golden archives never supply compiler values.

[Source assertions](source_assertions.md) additionally retain their observable
diagnostic/failure effect as standard `cf.assert` operations. Public tensor/scalar
ABI and receipt schemas are unchanged for ordinary tensor packages;
compatibility metadata records the source effect and native stdio transport.
[Effect-only/initial expected-failure packages](effect_abi.md) use explicit
`t2m.effect-kernel-receipt.v1` and `reference.json`, omitting numeric goldens.
They retain actual void or tensor-result signatures, outcome observations and
compiler-only custody rather than fabricating a result.
[Length-aware assertions](literal_diagnostics.md) keep those schemas and actual
signatures, declare the native diagnostic feature, and add a bound
`native/compiler.log` independently of source outcome streams. Original v1
effect packets remain readable. Compiler logs and references remain excluded
from Merlin's compiler projection.
[Logical source prints](logical_prints.md) add standard `vector.print` to the
portable body and declare the logical event/formatting/software profile.
Physical thread replicas are abstracted; public rosters and receipts are
unchanged. [Mixed successful tensor/output-stream packets](mixed_effects.md)
add explicit `t2m.effect-reference.v2` plus a complete `reference.npz` tuple
and per-result word/value policies, retaining the outer effect receipt and
actual tensor ABI. Both references and their individual digests remain outside
compiler staging. This evaluator layout differs from model2MLIR's ordinary
golden/extras layout; compatibility metadata records the extension.
[Scalar math](scalar_math.md) additionally uses the explicit IEEE result policy
for exact non-NaN words without imposing a NaN payload. Pure numeric cases have
empty expected streams. Portable BF16 math stays typed BF16; selected native
materialization records its separate scalar-emulation and word-transport
features. This changes neither the public tensor ABI nor compiler input custody.

## Shared tensor profile (`t2m.capture-receipt.v2`)

Derived model bindings may explicitly select `dense-row-major-v1` tensor
value views under `t2m.model-kernel-binding.v2`. Static caller types and captured
dimensions must establish exact element type/count before standard reshapes
and dynamic kernel shape casts. Core model ABI/data/reference files stay
unchanged. Merlin independently reopens the original calls and scratch facts;
unknown/older binding readers refuse the new version. This supplies no physical
layout or mutation permission. See [dense model tensor views](model_tensor_views.md).

Typed input preparation accepts overlapping readonly tensors by copying their
exact logical values before serialization. Runtime scalar carriers are captured
by value even when their backing bytes overlap tensor storage. The trace names
`readonly-input-value-snapshots-v1`; compiled tensor inputs remain independent.
Undeclared writable tensor overlaps refuse before copying and require an actual
alias/state interface. This changes neither the ABI nor receipt schema; compatibility
metadata records the rule. See [readonly aliases](readonly_aliases.md).

[Allocation-state bindings](allocation_state.md) now provide that explicit
source interface through `shared-tensor --allocation-bindings`. The public entry
takes roots once, passes source pointers/offsets through a real TTIR call and
returns each updated root once. `source.ttir` contains this bound program and
the private source body. New packages retain exact pre-wrapper/native input bytes,
argument configuration and source-only fingerprints in a
[portable source capsule](descriptor_state.md); older pointer-only packages may lack it.
The descriptive binding table records both hashes without granting semantics.
Existing ABI/receipt schemas and compiler/evaluator separation remain unchanged;
compatibility metadata records the source/signature difference.
Equal-width byte-aligned pointer types may share one declared root. Executable
wrapper casts preserve the original source signatures before inlining; register
word views retain one state and the original public storage type. The source
binding table records both types explicitly. This adds no ABI/receipt schema,
byte-order choice, different-width alias or physical-pointer permission; see
[typed allocation aliases](allocation_word_views.md).
Different-width ordinary views may use declared i8 roots with explicit
`byte_order="little"` or `big` (`shared-tensor --byte-order`, or a per-kernel
`bind-model` configuration field). Original pointer/descriptor parameters bind
byte origins before casts; original element strides retain their source units.
One guarded byte plane and standard word pack/unpack preserve masks, fills,
overlap checks and untouched storage. Original word/tile intent remains in
provenance. Existing public ABI/receipt schemas stay unchanged; this supplies
no typed-root byte packing, indivisible byte-backed atomic or physical target
permission. See [the byte-buffer contract](byte_aliases.md).
Typed owning roots and changed-width source parameters use the separately
declared [typed parameter/transport contract](typed_parameters.md).
External descriptors use source-argument configuration v2 to bind shared roots,
runtime origins and actual native logical fields before helper inlining.
This adds an explicit compiler artifact roster, with independent Merlin validation,
without changing the public tensor receipt/ABI schemas. Older readers can refuse
the added roster. Byte custody does not establish source equivalence or grant
alias, numeric or target permission; `source_closure_verified` remains false.

Non-dense source-allocation packages keep owning flat tensors, including
padding, with typed runtime origins, signed element strides and capacities.
The guarded source-word profile executes those addresses without relabeling
dense logical inputs as physical storage. Dense C/results retain their M×N
shape; ABI v5 and receipt schemas stay unchanged. Caller descriptions survive
the rebuilt function as `prov.intent` and still grant no semantics. See
[strided memory](strided_memory.md) for exact scope and remaining view/alias work.
[Strided output state](strided_outputs.md) additionally takes the output
allocation as an incoming readwrite tensor and returns its complete updated
one-dimensional state. Runtime output origin/strides/capacity remain explicit;
all inactive bytes are preserved, including NaN payloads and signed zeros.
This changes the data signature relative to a fresh dense output, without a new
ABI or receipt schema. The actual caller inputs remain immutable under the
functional interface. No cross-allocation alias or target permission follows.

Pure default-library math uses the explicit
`logical-libdevice-standard-math-v1` profile, preserving checked symbol/library,
prototype, purity and source IDs beside standard math operations. Compatibility
metadata distinguishes host libm/intrinsic semantics from GPU approximations.
Native evidence records its actual pass list. Source-word custody admits the
exact added reciprocal-root expansion and retained legacy pipelines; it grants
no numeric permission. See [external math](external_math.md).

F32 BF16x3/x6 normalization reuses the actual native Triton pass while keeping
original and normalized typed graphs distinct. Kernel-level original precision
inventory and caller tags accompany named contractions, with no per-operation
bijection claim. The original C/post-composite order and NaN correction stay
executable. Compatibility metadata states the selected host arithmetic and
separates the native binary hash from its source reference. Public ABI and
receipt schemas are unchanged; see [dot precision](dot_precision.md).
TF32 adds the explicit `logical-tf32-e8m10-rtz-v1` profile to source intent,
normalization reports and compatibility metadata. Integer operations retain
input quantization, gradual underflow and computed NaN class. TF32x3 uses the
actual upstream decomposition and expands only its generated RNA conversion;
original source assembly remains a separate gate. Direct TF32 keeps C inside
the contraction, while composites add it afterward. Physical TF32 encoding,
GPU accumulation behavior and accelerator equivalence remain unqualified.

Scaled dots use the explicit `logical-mx-f64-v1` internal-precision profile:
packed tile unpacking and per-group scaling expand to standard parallel
linalg, original C enters an f64 contraction, and explicit casts round the
declared result. `prov.scaled_dot_contracts` retains that source relationship.
The selected precision differs from Triton's target-dependent software
emulation and does not assert GPU instruction equivalence. ABI and receipt
schemas are unchanged; compatibility metadata records the difference.
See [the numeric and dataflow contract](scaled_dot.md).
Runtime FP4 packages retain logical K and packed/group allocation extents as
distinct scalar/shape bindings with executable source assertions. Per-tile
rounding, loop state and unused-nibble masks remain in the payload. Selected
native materialization may add an independently owned copy of imported private
loop state after its complete proof; no ABI, alias or lifetime rule is waived.

FP8 contractions retain their named matmul and original accumulator in the
portable payload. Derived `prov.contraction_contracts` records source formats,
precision applicability, accumulation bound and intake C identity without
granting arithmetic or hardware permissions. Selected native widening uses
official named-op generalization before scalar cast expansion, retaining
generalized fill transport and every ownership callback. Proved negative zero
and later source additions remain distinct. This changes neither the public
ABI nor package schema; [the implementation and evidence](fp8_contractions.md)
explicitly separate host execution from accelerator qualification.
The cooperating readers preserve complete dense element words when deciding
whether a constant is a splat, including mixed signed zeros. Exact finite
format decoding uses binary exponent scaling. These are correctness repairs
within the existing MLIR and archive representations, without a schema change.

FP8 conversion additionally requires an explicit overflow/infinity contract
(`ieee`, `satfinite`, or `satfinite-nan-inf`). Standard executable clamps,
casts and RTZ correction implement that profile; annotations retain intent.
Native execution selects Merlin's separate `lower_fp8_narrowing` feature
before widening/BF16 legalization. Tensor types, byte storage and receipt
schema are unchanged. The existing v2 `ieee` result policy now admits the
declared FP8 formats, preserving every non-NaN word and NaN class. Older
readers restricted to wider types reject these packages; exact `words` keeps
its byte comparison. See [the FP8 contract](float_casts.md#fp8-narrowing-and-explicit-overflow-contracts).

Selected floating conversions use a deterministic standard MLIR composite,
with explicit source/target types, source IDs and rounding annotations.
RTZ correction, equal-width format bridges and round-to-odd f64→16-bit
bridges retain actual arithmetic rather than relying on tags. This expands
scalar cast internals while retaining the source tile/loop structure; it
changes no public tensor ABI or receipt schema. Compatibility metadata records
the difference. See [numeric contracts and evidence](float_casts.md).
Caller StringAttr intent fields survive rewriting. `prov.intake_intent`
retains every source-ID-to-tags mapping when operations merge, and agreed
`prov.family`, `prov.op`, `prov.region_id` values retain their original names.
These metadata fields supply intent, never numeric or memory permission.

The four pinned FP8 formats retain their distinct MLIR float types and use
raw `uint8` archive carriers. This explicit model2MLIR archive difference is
also written into compatibility metadata for ordinary and mixed-reference
packages. Numeric comparisons decode the declared format; exact byte claims
require word comparison. No by-value FP8 ABI or numeric cast/overflow policy
is inferred from storage. See [the transport contract](float_casts.md).

The [upstream-based frontend](shared_tensor.md) uses the same core filenames with an explicitly
versioned ABI and receipt. `shared-demo` writes complete examples; `shared-tensor` accepts saved
TTIR plus `--abi`, optionally `--inputs` (`in0`, `in1`, ...) and `--reference` (`out0`, `out1`, ...)
NPZ archives. Without both arrays and references it writes IR/provenance only, with no receipt.

In the original v1 tensor ABI every buffer, including writable incoming values, is an argument.
Writable results follow declaration order. The function retains its kernel name. Specialized
integer scalars are recorded in `spec.json` rather than becoming runtime tensor arguments.
`golden.npy` stores result zero; `extra.npz` stores any additional results as `out1`, `out2`, ... .
That use of `extra.npz` is an explicit convention difference, not implied m2m compatibility.

Before writing any bundle, the writer executes the serialized MLIR through the native host LLVM
path and compares every result with the supplied independent references at explicit tolerances.
At zero relative and absolute tolerance, integer results use exact integer comparisons,
including individual i64 bits beyond binary64 precision. Floating results retain the
declared numeric comparison policy; Boolean comparisons concern logical values.
`execution.json` binds those checks to the exact IR hash. It makes no NaN-payload/signed-zero-bit,
universal source-equivalence, original Triton-execution or consumer/device-execution claim.
`verify-bundle` dispatches both receipt versions and verifies all artifact/source/ABI bindings.
The full list of differences is also embedded in each package's `compatibility.json`.

`t2m.shared-tensor-abi.v2` adds a `write` buffer effect for fresh model results.
These buffers become results without incoming function arguments. Conversion checks every
observed load against preceding stores and proves complete coverage from bounded, disjoint
tiles with equal total cardinality, or an exact clipped-prefix partition for ragged tails.
Unwritten holes or initial reads are refused. `readwrite`
continues to preserve incoming storage values. Existing v1 read/readwrite specs retain their
canonical serialization; package receipts continue to describe the actual materialized ABI.

The [bound-model profile](model_integration.md) keeps model2MLIR's actual core
data files, function ABI and learned weights. It uses a distinct
`t2m.bound-model-receipt.v1`, with original capture/FX evidence retained under
`parent/` and explicit TTIR binding/native/consumer evidence for the derived IR.
Optional compiler staging and native Merlin materialization add explicit
`consumer/compiler/` and `consumer/host/` products; the qualification input receipt
is retained separately before the final receipt is augmented. These products
do not change the portable model ABI or claim accelerator execution. See
[the consumer contract](merlin_consumer.md).

The bound compatibility document additionally lists `kernel_profiles` by archived
kernel directory. Each records the same representation differences as a
standalone kernel, including guarded memory, source-word selectors, chosen cat
ordering and dynamic dimensions where present. These additive fields make
kernel-specific changes explicit while the parent model ABI/data remain intact.

`t2m.shared-tensor-provenance.v2` is used when original TTIR needs native-verified
intake rewrite before the shared optimizer can parse it. It records the original
native-verified typed operations, block/region structure and exact source bytes,
plus a separate normalized operation inventory. The native binding does not
expose complete original attribute dictionaries; those remain in `source.ttir`,
and this limitation is explicit. `prov.normalized_ids` / `prov.normalized_ops`
and emission/memory facts reference the normalized graph. They are not presented
as original-operation correspondence. The package receipt/functional ABI remain
unchanged; the difference is source provenance and normalization evidence.

## Shared buffer profile (`t2m.capture-receipt.v2`)

ABI v5 adds [named runtime dimensions and logical launch expressions](dynamic_shapes.md).
Per-lane masked writes additionally retain `prov.source_scatters` and
`memory.source_scatters` observations with exact source IDs and the proof kind;
their functional tensor/scalar ABI and receipt schemas are unchanged.
The existing `f16` tensor element type now has native and Merlin host execution
support using actual NumPy float16 storage. No hidden widening or by-value half
scalar convention is introduced; see [the half profile](half_precision.md).
The existing `bf16` type now additionally uses **raw uint16 words** in standalone
NumPy inputs/references and the unchanged two-byte host descriptors. Numeric
comparisons decode BF16 before applying tolerances. This explicitly differs
from model2MLIR's legacy widened F32 archives. Host qualification requires the
upstream BF16 cast expansion and Merlin's checked scalar LLVM transport feature;
bound BF16 models now preserve their parent's value archives and consume
receipt-bound exact-word witnesses or checked unique non-NaN embeddings instead.
BF16 source weights use Merlin's existing pinned byte reader. These format
differences and measured model scope are explicit in
[the BF16 model contract](bfloat_models.md).
The actual tensor/result allocation extents are dynamic; they are distinct from
ABI v4's static capacities and changing prefixes. Native execution receives an
explicit shape-allocation contract derived from `spec.json`. Static packages
keep their existing ABI and artifact conventions.
Dense multidimensional runtime strides are recovered by exact symbolic
offset/stride cancellation. Public shapes retain every named axis; raw memory
flattening is internal to the buffer profile. No new schema or hidden stride
argument is introduced. Non-dense layouts still require another contract.

`--memory-profile buffer` selects `shared-functional-buffer`, alongside the
default `shared-functional-tensor`. The public tensor/scalar ABI, references and
model binding stay the same. The payload additionally contains standard
`memref` and `bufferization` operations around retained linalg/tensor computation.
Writable state is private; incoming readwrite values are copied before updates.
Fresh outputs still require complete initialization and grid ownership proofs.

The receipt additionally binds `memory-witness.mlir`, `memory-input.mlir`,
`memory-stage.mlir` and `memory-stage.json`. Provenance binds the witness and
continuation hashes; the report binds the continuation input/output bytes.
Bound models retain these files under each kernel's directory. Merlin copies
them into compiler staging without evaluator inputs or references. This initial
profile requires the tensor source-memory proof and does not imply broader
opcode admission. See [the memory boundary](full_memory_bridge.md).

## Shared core artifacts

`--address-policy source-words` with guarded memory selects the explicit
`shared-source-word-buffer` profile. It retains the same tensor/scalar ABI and
guard protocol, while using original integer words with signed i64 pointer
offset accumulation. Accesses use offset tensors instead of recovered slices.
The receipt additionally binds `word-input.mlir`, `word-tensor.mlir` and
`word-stage.json`; admission/provenance bind their policy and hashes. Merlin
also admits the explicit `t2m.source-word-stages.v2` chain for region-local
events. It embeds event input/output, typed source bindings, parent arithmetic
input and both actual native reports inside the same three artifacts. The
producer reconstructs every event placement; Merlin validates declared byte
custody. `memory.word_conversion` and package differences explicitly identify
this version. The tensor/scalar ABI and guarded memory protocol are unchanged.
The provenance pipeline records the two ordered native stages explicitly.

Word CAS adds a typed-template variant within stage v2. `input_types` binds the
actual floating storage capture; `template_input_types` binds an equal-width
integer pointer formal in the isolated native pointer template. All remaining
formals agree. After that native stage, reconstruction restores the original
floating root and its logical integer view at the original event. This avoids
feeding an already structured descriptor through a pointer-bitcast rewrite.
Producer replay reconstructs the complete pair; Merlin refuses a missing or
inconsistent declaration. This adds no public/dummy argument, receipt schema,
artifact name, physical alias or final opaque call.

Byte-backed integer atomics have a separate additive declaration:
`events.byte_order` and each binding's `byte_view` retain logical byte order,
word width, alignment and byte/word address units. `input_types` keeps the
actual i8 root while `template_input_types` declares the isolated integer-word
formal. Producer replay checks native alignment, division, event operands and
returned-old relations. Under explicit typed-atomics and logical byte order,
memory admission v4 adds `atomic.byte_view` and `guard_tile` (original word tile
plus its byte axis), with one indivisible word payload and per-byte guard state.
The independent Merlin reader checks source/storage/width/order/guard declarations;
the producer checks the current complete executable payload recipe. Standard
MLIR and the public tensor/scalar ABI remain unchanged. Native materialization
selects `lower_integer_atomic_regions` through normal ownership/lifetime checks.
This permission is separate from equal-width word CAS and floating policies.
See [the current evidence and remaining gates](byte_atomic_words.md).

Merlin checks/copies these compiler-only members for standalone and static derived
models. Function/grid annotations retain original access intent through model
inlining. Pointer branches/loops normalize region state to i64 offsets before
that native stage; finite mixed bases add logical i32 selectors and predicated
accesses at the original memory event. The transformation report retains source
IDs, selector/base contracts and any upstream inactive-undefined zero refinement.
Compatibility metadata makes these representation differences explicit, with
the same public ABI. Original source/inventory remain separate.
See [source-word semantics and evidence](source_words.md).

Selected ordinary memory may additionally use equal-width roots with different
storage dtypes. Each root keeps its public type and uses register bitcasts at
its original predicated event. Closed selector-tile permutations retain immutable
tiles and carry scalar table indices in the original loop. These internal
representations are explicit in transformation/provenance and compatibility
metadata; artifact names, stage versions and public ABI are unchanged.
See [mixed storage](mixed_storage.md).

Selected integer atomic events additionally retain actual integer or equal-width
floating storage captures under the typed profile. Complete source context and
actual capture dtype distinguish each event's view declaration; an isolated CAS
template's integer pointer formal is checked separately. Primitive contracts and
complete producer replay remain required. Compatibility metadata records this
composition with unchanged public ABI and stage/artifact schemas. Real models
qualify three differently typed tensor results and every original tuple reference.

`--effect-profile atomics` additionally requires guarded-buffer/source-words and
selects the explicit private-invocation integer RMW/CAS contract. Public arguments
and results retain their tensor/scalar ABI. `memory-admission.v3` binds thirty-bit
owner fragments, atomic-compatible participation, actual payload accounting and
source operation/order/scope. Standard RMW events retain old values inside
exactly-once SCF regions; source-undefined inactive old values explicitly refine
to zero. Package differences name this policy, its stronger standard ordering
and grid-derived guard cost. See [atomic memory](atomic_memory.md).
Integer CAS additionally supports i16/i32/i64 through a standard generic atomic
comparison/select region. It remains unmasked, including tail lanes unless
source control flow suppresses them. Finite buffer selection branches inside
parallel tensor tiles. The pinned native ownership toolchain requires Merlin's
checked compare/exchange legalization; this dependency and the executed prelude/
prepared-body hashes are explicit native evidence. Exported MLIR and the public
tensor/scalar ABI remain portable, with no opaque or LLVM helper in the package
source. Existing RMW/v3 guard costs and initialization checks remain mandatory.

`--effect-profile typed-atomics` selects the separate
`private-invocation-typed-atomics-v2` contract with `memory-admission.v4`. It adds
bitwise f32/f64 exchange, IEEE f64 atomic addition, integer word RMWs on equal-width f32/f64 storage and integer word CAS on f16/f32/f64 storage
to integer RMW/CAS while retaining the same public
tensor/scalar ABI, thirty-bit ownership and guard costs. Floating events bind
value dtype and an operation-specific numeric contract in source/access/payload declarations.
Standard `memref.atomic_rmw assign` preserves old and new representations;
inactive undefined old values refine to typed zero. F64 `addf` binds
`ieee-f64-rne-add-quiet-nans`: round-to-nearest-even with subnormals preserved,
exact old words and explicit result quieting with unspecified NaN payload/sign.
It uses a standard generic atomic region and the same checked native helper
boundary as f32 below. The legacy `ieee-f64-rne-add`/standard-addf representation
remains readable, with no guarantee of NaN quieting after constant normalization.
This numeric contract/payload version change leaves core artifacts, receipt
schemas and the tensor/scalar ABI unchanged.
F32 addition requires explicit `f32_atomic_add="ieee"` or `"ftz"` (CLI:
`--f32-atomic-add`). Its selected policy is bound in v4 admission and memory
evidence as well as source/access/payload contracts. Both use standard generic
atomic regions with addition and explicit result NaN quieting. FTZ also retains
executable sign-preserving input/result flushing. Native strict floating ownership additionally
requires Merlin's checked atomic-word feature and default 64-bit descriptor/
index ABI; each complete internal CAS retry helper is restored before lifetime
checks. Portable MLIR has no native helper or LLVM operations. Core artifacts,
receipt schemas and tensor/scalar ABI are unchanged. Other floating arithmetic
needs separate numeric policies.
Compatibility metadata names these
differences for standalone and model packages. Word events additionally bind
storage/value types and unchanged element address units in per-event `word_view`
facts. Original atomic primitive contracts remain shared across call contexts;
exact legacy source-enriched records also remain readable. This allows one
callee to act on integer and floating storage in different calls without
overwriting the original source intent. Their
standard generic atomic regions retain source integer updates and returned bits;
Merlin's explicit native word legalization supplies a checked internal helper
through ownership/lifetime conversion. This is a native-toolchain compatibility
requirement, with no helper in the portable package. Source floating min/max
retains its two sign-conditioned integer atomics, including NaN/zero behavior.
Word CAS compares integer representations, including NaN payloads and signed
zeros, with executable bitcasts/comparison/select. Native helpers return integer
old words to avoid half floating ABI conversions; portable tensor types and
the standard atomic region remain unchanged.

Equal-width ordinary pointer views retain the storage tensor type and insert
register bitcasts at each load/store. `prov.logical_word_views` and source
transformation records retain storage/view dtypes and root positions. Integer
fill bits are restored by a mask select after a zero-filled storage load.
Compatibility metadata names this expansion; the public tensor/scalar ABI and
guard costs are unchanged. See [logical views](pointer_views.md).

The additional explicit `shared-guarded-buffer` profile uses the same public
tensor/scalar ABI with runtime memory admission instead of an affine tensor
witness. `memory-admission.json` replaces `memory-witness.mlir`; the other three
memory continuation artifacts remain. The report and function attribute declare
the grid-derived guard cost (16, 24 or 32 bytes per writable element), included
in standalone evaluator budgets.
Checks retain source masks, loops and payload while rejecting active bounds
violations, conflicting writers and fresh-output holes. This initial profile
supports standalone staging/execution and static derived-model binding. Bound
models retain each kernel's evidence and declare guard scratch summed over
inlined calls, independently checked by Merlin against typed kernel results.
See [the exact memory scope](guarded_memory.md).
V2 admission retains writable observations and complete-write initialization,
with sticky reader/writer tags and actual source-base memory reads before
optimization. Its `memory-effects` continuation boundary is explicit. Existing
v1 evidence retains its original write-only contract and `memory-ir` stage.

| Artifact | MVP convention |
|---|---|
| `model.mlir` | One `forward` with tensor inputs/result, standard dialect operations, `prov.level="linalg-on-tensors"`. Generic assembly preserves discardable annotations. |
| `weights.safetensors` | Valid empty safetensors, because this kernel has no learned/captured constants to bind. |
| `weights.safetensors.manifest.json` | Function-argument ordinal string → `{kind:"input",name:...}`. No output-pointer argument remains. |
| `inputs.npz` | Positional arrays keyed `in0`, `in1`, etc.; f32 shapes agree with the function ABI. |
| `input_order.json` | Manifest input name → positional input index. |
| `golden.npy` | Single output from an independently supplied reference; checked against emitted IR before writing. |
| `extra.npz` | Empty, valid NumPy archive; lifted constants/quantization metadata are not fabricated. |

`prov.weights_file` uses the relative name `weights.safetensors`. An existing output directory is
never overwritten. The receipt is written last. `verify-bundle` checks the recorded artifact bytes
and source/spec/output bindings; hashes establish byte identity, not external authorship or a
complete mathematical proof.

## Differences and additions

| Area | Explicit difference |
|---|---|
| Receipt identity | `t2m.capture-receipt.v1`, not m2m's Torch-specific receipt. Consumers requiring the latter need a declared adapter/schema change. |
| Source graph | `source.ttir` and `provenance.json` describe TTIR SSA, not an FX `frontend-trace.json`. No fake `prov.aten` or Torch framework identity. |
| Source intent | Caller descriptions have `origin="caller-description"` and `grants_semantics=false`. Derived operation family/kind comes from the actual translated opcode. |
| ABI | `spec.json` supplies shapes/effects, positive strides/storage offsets, grid, constexprs and allocation preconditions. The output pointer becomes a result; readwrite outputs also have incoming tensor arguments. |
| Payload | Retained tile grid in `scf.forall`; a consumer must support it or explicitly legalize it. Current Merlin staged support is not certified. |
| xDSL | 0.68.0, versus the surveyed m2m environment's 0.65.0. Local bindings add missing **upstream standard** SCF/tensor ops, not a t2m dialect. |
| Numeric evidence | Serialized-IR reference interpreter versus a supplied NumPy golden. Optional native verification/LLVM legalization reports are separate, hash-bound checks; no native/accelerator execution claim. |
| Parameters/results | MVP has no parameters and exactly one result. General weights, dtype/profile negotiation and multiple results are roadmap work. |
| Source locations | Stable operation/value IDs and TTIR are retained. Original Python file/line reconstruction is deferred; debug paths are excluded from TTIR hashes. |

`compatibility.json` repeats these differences in every complete bundle. `meta.json` keeps
conversion/execution/consumer statuses distinct. None of the additions is necessary to interpret
the arithmetic itself; required ABI preconditions still have to hold at any real call boundary.

Shared `t2m.capture-receipt.v2` packages can be qualified directly through Merlin
with `qualify-merlin`, preserving the source package. The separate
`t2m.merlin-kernel-qualification.v1` directory contains compiler-only staging and
native-host execution products. Entry-symbol adaptation, functional readwrite
results, multi-result reference ordering and provenance projection are explicit;
see [the consumer contract](merlin_consumer.md). This does not rename the source
receipt or assert accelerator qualification in the original capture metadata.

Bound model packages also support all results of a flat tensor tuple/list. The
parent model2MLIR capture declares `m2m.flat-tensor-result-references.v1` in its
metadata and binds `goldens.npz` (`out0`, `out1`, ...) in its receipt. The legacy
`golden.npy` retains the exact first result. Both archives remain byte-identical
in the derived package; native and Merlin execution reports bind their hashes
and compare every result against the complete typed model ABI. The additive
archive is optional for existing single-result packages. It is evaluator-only
and excluded from Merlin compiler staging. Nested/non-tensor results and
general mutation schemas require further contracts; see
[model integration](model_integration.md).

ABI v3 adds typed runtime scalar declarations with explicit source positions.
Exported MLIR keeps plain scalar argument types after logical tensor inputs;
the usual `inputs.npz` keys store zero-dimensional typed values for invocation.
Merlin passes these by value through its existing scalar ABI. This differs from
a tensor-only captured signature; it is not scalar specialization or a rank-zero
tensor boxing transform. See [the runtime ABI](runtime_abi.md).

Reusing an already qualified library produces a separate
`t2m.merlin-kernel-invocation.v1` evidence directory. Its compiler snapshots and
artifact relations identify the exact original qualified library. New inputs,
references and observed results are evaluator artifacts; the source capture and
original compiler/qualification directory remain immutable.

ABI v4 extends runtime scalar declarations with an optional inclusive signed
integer `bounds` interval. The frontend emits entry assertions and scalar/tile
analysis retains symbolic parameters. Merlin invocation signatures retain those
intervals and validate values. This permits runtime memory prefixes over static
logical allocation capacities; readwrite outputs preserve incoming values
outside the active prefix. Tensor shapes/launch sizes are not relabeled dynamic.

## Provenance schema (`t2m.provenance.v1`)

The trace contains the canonical TTIR and spec hashes, tool versions, intent and its origin,
source-operation IDs/opcodes/typed SSA edges, each operation's semantic disposition and recovered
index/access facts, emitted-op IDs with source correspondence, established conditions/preconditions,
numeric profile and the exact output MLIR hash. Bundle writing adds final-byte count and measured
reference-interpreter agreement. User mutation of the converted IR invalidates this trace; a
recorded transform must regenerate provenance before bundle writing can proceed.

The MLIR uses the shared `prov.family` / `prov.op` / `prov.region_id` convention, plus
`prov.source_ids`, `prov.emission_id`, `prov.frontend`, `prov.intent`, `prov.intent_origin`, source/spec
hashes and logical grid/tile facts. Source IDs refer to the trace, not device instruction IDs.
Several emitted ops may implement one source op, and address-only source ops can become analyzed
access facts. The trace distinguishes that from eliminated unused pure values.

## R1 tensor-memory additions

A declared rank-one view maps logical index i to element address
`storage_offset + strides[0] * i`, relative to the source pointer argument. The converter proves
that each access fits that view, then exports its **logical tensor values**, without physical
storage holes or pointer arithmetic. `inputs.npz` materializes those logical values. Binding back
to an original strided buffer is a caller/consumer ABI step, not a hidden hardware operation.
Default contiguous fields are omitted from canonical specs to preserve original R0 spec hashes.

`readwrite` adds the incoming logical tensor to the input manifest and returns its updated value.
One source store is currently supported; reads must precede it and stay in the same program's
owned output tile. Values outside a partial store's predicate are preserved. This standalone
functional ABI does not add mutation support to model2MLIR's custom-kernel registry.

`tensor.extract_slice` + `tensor.pad` represent rectangular masked loads, with an explicit
`tensor.cast` to the proved tile size for xDSL 0.68's pad verifier. Stores insert cropped active
contributions only. Missing `other` is tracked as undefined until a store proves it unobservable;
an implementation fill for those lanes never becomes an observed source value. The trace records
mask/access facts, stored-value definedness, source cache/eviction hints and identical-read reuse.

The optional `verify-mlir` report binds the exact input MLIR hash and distinguishes registered-op
verification, LLVM-dialect legalization and execution. It never modifies an existing capture bundle
or pretends those checks were performed by a bundle's original receipt.

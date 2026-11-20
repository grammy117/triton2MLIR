# Package contract and explicit model2MLIR differences

[Private volatile reads](volatile_reads.md) explicitly select admission v5,
source-word stages v3 and `private-invocation-volatile-loads-v1`. Defined
compiler-only identity calls preserve observations before standard LLVM volatile
leaves are restored, then disappear from exported MLIR. Exact stage pipelines,
source/effect inventories, descriptor SSA and actual volatile flags are checked
independently. These localized LLVM memory leaves differ from pure tensor MLIR;
public tensor/scalar ABI and receipt schemas are unchanged. Older readers reject
the new evidence versions; no external/shared/cache/target permission follows.
Complete volatile models retain original allocation-source capsules, executed
binding v4, receipt v2 and signature v6. Distinct read IDs survive shared enclosing
callsite locations. Every parent archive and all fourteen results remain bound;
host/serial qualification pins the independent reader and LLVM binding. No new
public tensor result schema or mutation/copyback contract is introduced.

[Atomic program phases](atomic_program_phases.md) explicitly select
`t2m.synchronous-program-barrier.v2` with the existing private atomic effect
profile and f32 policy. Original and protected atomic fields are independently
checked; selection must match bound executable memory evidence. Returned old
values, source order and sticky conflict guards remain actual computation.
Public tensor/scalar ABI, receipt schemas and source-capsule artifact rosters
are unchanged. Older barrier-v1 readers refuse v2. No grid-wide, external,
async, new numeric-policy or accelerator permission follows.
Complete i64/f32 atomic-phase models retain the existing executed-call binding
v4, receipt v2 and signature v6, with all original parent archives/results.
Separate host and serial target qualification binds the independent atomic
source reader as well as allocation/phase readers; no new public result schema
or caller copyback contract is introduced.

[Program barriers](program_barriers.md) add the explicit
`t2m.synchronous-program-barrier.v1` source contract. Exact masks/source positions
remain in the original inventory and `prov.program_barriers`; actual standard
SCF phases carry complete tensor versions or synchronous standard accesses.
Compatibility names this abstraction and its single-CTA/private invocation
scope. Public tensor/scalar ABI and receipt schemas are unchanged. This grants
no shared/external/async or cross-program effects. Allocation capsules explicitly
add `barrier_conversions` to source-protection v1; older readers reject the field.
Both readers reopen actual generic/protected source and check scope, payload and
memory order. Bound traces label these IDs `allocation-original`, separately
from wrapper IDs. Complete models retain receipt v2 and signature v6. Selected
host/serial qualification additionally binds the independent source reader.

[Shared mutable views](mutating_views.md) select
`m2m.functional-kernel-storage.v1` / `t2m.functional-kernel-binding.v2` within
binding v4. Whole cloned bases and actual coordinate SSA replace independent
mutable argument values. `mutation_pointer_arguments` records original TTIR
pointer ordinals; both readers check the same readwrite allocation and offset.
The selected automatic memory profile uses guarded structured addresses, with
explicit standard-memory/private admission state. Original public types and
parent archives, receipt v2 and signature v6 retain their meaning. A scalar bias
uses the existing count-checked one-element dense adapter. Complete updated state
remains returned explicitly; no public physical alias/copyback permission follows.

[Functional mutation](functional_mutation.md) explicitly selects
`t2m.functional-kernel-binding.v1` within binding v4. Original schema/base/result
ties must match the retained readwrite buffer in both producer and independent
Merlin reconstruction. Public input/buffer effects use the parent model2MLIR
`m2m.functional-state-results.v1` ports; they add explicit state outputs relative
to the imperative PyTorch invocation. Initial archives and every state/user
reference remain retained. Borrowed entry inputs stay readonly; physical aliases
and automatic caller copyback remain separate invocation work. Receipt v2 and
Merlin signature v6 retain their schemas; compatibility names this extension.

[Rank-reducing source views](rank_reduced_source_views.md) additionally record
`retained_axes` in changed-rank steps under coordinate schema v1/view profile v3.
Both readers derive these axes from original mixed sizes/types. Earlier records
retain their meaning; consumers lacking this form refuse it. Public scalar
results use the existing count-checked dense adapter from one-element ranked
kernel results. Public types, parent archives, receipt v2 and signature v6 remain
unchanged; no public physical alias or mutation permission follows.

[Runtime source views](runtime_source_views.md) explicitly select per-call
`runtime-logical-coordinates-v1` and execution view profile v3 within binding v4.
Runtime shape/stride/offset guards derive from original standard SSA and the
actual source allocation-pointer contract. Canonical extract/empty conditionals
preserve boundary-empty slices. Older readers refuse; parent archives, public
types, receipt v2 and signature v6 are unchanged. Consumer fingerprints additionally
bind the independent `execution_source_coordinates.py` reader. Logical coordinates
grant no public physical alias or mutation permission.

[Executed source grouping](executed_source_layouts.md) reuses captured-source-layout
v1 under binding v4 with exact source option/argument records. Optional
`execution_allocation_policy` on derived receipt v2 selects explicit
`merlin.owned-result-limits.v2`; original defaults and receipt/signature schemas
remain unchanged. Parent archives, full source/SSA checks and runtime guards remain.

[Runtime dense views](runtime_tensor_views.md) explicitly select
`dense-row-major-runtime-v1` / `t2m.execution-checked-call-views.v2` within binding
v4. Runtime source dimension operands and complete count/overflow checks replace
captured-size-only adaptation; older readers refuse. Public model types, original
archives, receipt v2 and Merlin signature v6 remain unchanged. Model2MLIR's
additional `numel_dimension_queries` trace relation accounts for original shape
products and grants no numeric, alias or source-equivalence permission.

[Executed call views](executed_call_views.md) add the explicit
`execution_view_binding_profile: t2m.execution-checked-call-views.v1` extension
to binding v4, with known dense-policy and complete source-coordinate records
per call. Older readers refuse the fields. Parent archives, public tensor ABI,
receipt v2 and Merlin signature v6 retain their existing schemas. Compatibility
metadata names the extension; no public physical alias/mutation is inferred.

[Returned loop ownership](runtime_loop_ownership.md) extends the independently
checked consumer lifetime forms without changing public tensor computation,
signature v6, result/package schemas or parent archives. Native products, saved
programs and selected frontend source fingerprints additionally bind
`dealloc_lifetime.py` and `dealloc_promotions.py`. Actual flags, copies and closed
uses establish ownership; no annotation supplies alias or lifetime permission.

[Execution-owned target packages](execution_owned_target_packages.md) explicitly
select `t2m.spike-owned-result-qualification.v1` with runtime input projection v3
(`positional-execution-owned-source-words`). Source-only signature v6 staging,
the ordinary Merlin builder and actual observed-shape readback retain every
original result. The original parent/package receipts and archives are unchanged;
the separate target receipt binds actual image/process/reference evidence.
Changed model inputs/parameters use complete positional/reference archives
together. Legacy target receipts/readers keep their prior requirements. The
current target selection is serial HTIF; broader executors/effects remain separate.
The target's historical weights-blob section transports the generated positional
invocation bytes rather than the original safetensors layout. Original package
archives and all parameter words remain unchanged.

[Generated owned-result runtime transport](owned_result_driver.md) explicitly
selects `merlin.owned-result-driver.v1` and
`merlin.execution-owned-readback.v1`. A separate bounded invocation blob fixes
runtime-driver input shapes while actual returned descriptors supply dynamic
output extents. Complete shape/payload companions differ from legacy fixed-shape
readback. Public MLIR, parent archives, receipt v2, signature v6 and host
qualification keep their existing meanings; target qualification still requires
its own staged build/execution evidence. Selected consumer fingerprints now bind
the generator, C controller/readback/host adapter and independent decoder.

Selected owned-result consumer fingerprints additionally pin the link-independent
`owned_result_lowering.py`. The separate serial-context/failure-hook build choices
do not change public capture, receipt or signature schemas and are not selected
by source operation names. Their real Spike runtime fixtures are separate from
package qualification; no existing host receipt silently acquires target support.

[Checked C result invocation](checked_c_results.md) adds a reusable generated
bridge API, not a new receipt or qualification backend. Selected consumer-source
fingerprints now include its source-typed generator and C implementation/header;
saved native programs pin them too. The same full model/library retains existing
host qualification, independent Python validation and unchanged parent archives.
Actual C execution evidence does not grant target staging/linking/readback support.

[Compaction](compaction.md) uses the existing selected receipt v2/signature v6
without a new public schema. Per-kernel automatic selection retains tensor
count/copy and guarded source-word packing. The latter's standard memory and
private admission state are an explicit payload difference; every original
model value, archive and actual execution-sized allocation remains retained.

[Automatic memory selection](automatic_memory.md) is explicitly selected with
`--memory-profile auto`; concrete profile defaults are unchanged. A versioned
trace record retains source-only candidate order and refusal diagnostics.
Selected tensors or guarded standard-memory payloads retain their existing
ABI, artifacts and proof/qualification requirements. Numeric/effect/source
policies remain explicit; selection evidence grants no additional semantics.

[Internal computed-size binding](executed_model_binding.md) explicitly selects
`t2m.model-kernel-binding.v4` / `t2m.execution-checked-calls.v1` and
`t2m.bound-model-receipt.v2` through
`write_model_bundle(..., result_mode="execution-owned", merlin_host=True)`.
All parent archives, public tensor ABI and original m2m receipt remain
byte-identical. `kernels/<ordinal>/located.mlir` and byte-bound
`located_implementations` additionally preserve kernel source locations;
stripping locations must reproduce the original implementation digest.
Actual scalar/dimension/cast contracts and complete standard bodies replace
global output-size prediction. Merlin independently reconstructs every call,
surrounding SSA operation, result type, guard and source location before
signature v6 staging. Source correspondence explicitly uses v2.

An unqualified source-only candidate stays under the work directory. Its
compiler staging excludes evaluator arrays, references and execution reports.
Actual host descriptors supply result sizes before original reference
comparison. Only successful complete qualification publishes receipt v2;
failed comparison leaves no output package. Producer verification replays
the derivation and rechecks every retained output word/reference and host
observation. Compatibility v2 explicitly names the Merlin host compiler and
original PyTorch oracle; it claims no separate torch-mlir native execution
backend. Coverage records that distinction. Old/default readers refuse v2
without explicit execution-owned selection. Generated C and the selected serial
target now have separate qualification; a second native evaluator and broader
shape/view/effect/target contracts remain open.

[Execution-produced results](execution_owned_results.md) explicitly select
Merlin signature v6 and its separate allocation/descriptor ABI through
`check_merlin_interface(..., result_mode="execution-owned")`. Actual source
types fix ranks/storage/static dimensions; invoked code produces remaining
extents. Original input constraints and package checks remain, and compiler
staging receives no evaluator arrays or argument paths. Interface v3 and host
compilation/evaluation v2 observations distinguish this route; saved native
program v1 reopens fixed products without compilation. Parent archives and
capture/package receipt schemas are unchanged. Generated C and serial target
execution use explicit additional transport/qualification profiles above;
older runtime readers reject v6 rather than preallocating guessed results.
A second native evaluator and broader execution/effect/shape contracts remain.
This adds no physical alias/mutation or numeric permission.

[Runtime input-word shapes](input_word_shapes.md) select model shape v3,
checked-word arguments, runtime shapes v4, typed program v2 and independent
Merlin signature v5. `word_bindings` records actual public singleton/native
integer reads; invocation values size results only after source-body proof.
Older profiles refuse. Public model arguments, parent archives, receipt schemas
and artifact rosters stay unchanged; selected scalar reads during evaluator
verification do not enter compiler staging.

[Derived model dimensions](derived_model_shapes.md) additionally select
`t2m.symbolic-model-shapes.v2` and `t2m.runtime-shapes.v3`, carrying a bounded
typed `logical-size-program.v1` DAG. Independent source reconstruction admits
`merlin.compile-signature.v4` with that `shape_program`; invocation evaluates
actual public input axes after rechecking the executable body. Earlier schemas
refuse the extension. Public MLIR arguments, receipts and parent archives stay
unchanged; logical relationships grant no physical alias or effect permission.

[Symbolic tensor binding](symbolic_tensor_bindings.md) declares
`t2m.symbolic-model-shapes.v1`, source-derived per-call tensor relationships
and an input-axis runtime shape contract. Actual implementation bodies and the
complete inlined model must reproduce those relationships; evaluator samples
supply none. Symbolic guarded scratch explicitly uses the sum of source shape
upper bounds. Public MLIR types remain dynamic; no scalar ABI inputs are added.
Complete [dynamic model packages](symbolic_model_packages.md) now use the
explicit `merlin.compile-signature.v3` input-axis interface. The evaluator-only
`t2m.model-invocation-inputs.v2` materializer profile and
`t2m.runtime-input-projection.v2` positional archive retain dynamic invocation
shapes and original argument words. They do not enter compiler staging.
Runtime generation preserves captured weight-blob transport and checks supplied
parameter bytes against the original container; reusable host calls can supply
independent parameter values. Existing receipt schemas, parent data and public
MLIR tensor/scalar arguments remain unchanged. Older static consumers refuse;
the new interface is an explicit compatibility requirement.

[Dynamic surrounding arithmetic/slices](dynamic_surrounding.md) additionally
retain standard source computation before/between/after translated kernel calls.
Runtime broadcasting and clipped positive-step slices add no public arguments,
receipt/schema fields, physical alias contract or source specialization. The
real addition/empty-slice model uses the existing independently checked symbolic
profile and v3 compiler signature; general derived expressions remain separate.

[Checked source index arguments](checked_index_arguments.md) select
`t2m.checked-index-runtime-arguments.v1` when dominating executable round trips
and comparisons prove a representable unsigned index conversion. Call records
add the exact checked adapter and `unsigned_integer_range`; producer and Merlin
reconstruct them from the original parent. Source guards remain executable,
older profiles refuse, and no native index width is chosen. This adds no
concrete tensor extent/scratch fact or dynamic whole-model signature. Public
ABIs, artifact rosters and receipt schemas stay unchanged.

[Live runtime arguments](live_runtime_arguments.md) add
`runtime_operands` configuration, optional per-call `runtime_operand_bindings`
and `argument_binding_profile: t2m.live-runtime-arguments.v1` to existing model
binding schemas. Standard element reads and proved index casts retain actual
SSA values; original-parent facts are independently rederived before dimension/
scratch admission. Captured scalar attributes can coexist. Older consumers
without this profile refuse. Public model ABI, parent data archives, compiler
artifact rosters and receipt schemas stay unchanged; this grants no symbolic
public shape, alias, mutation, arithmetic or effect permission.

[Region-local seed recovery](integer_address_regions.md#pointer-seeds-defined-inside-a-region)
records `integer-address-seed-rebase` under the existing closed-pointer
correspondence: actual same-shaped outer SSA roots plus executable source-width
byte recipes. Role/root/divisibility proofs are recomputed before paired state.
`integer-address-seed-retention` names independently validated original typed
pointers when optional recovery lacks a divisor proof; selection follows actual
source seed dependencies, without disabling other recoveries. Unknown consumed
source fields and physical observations cannot authorize retention. Generated
compatibility metadata names both representations. Public tensor/scalar ABI,
artifacts and receipt schemas remain unchanged.

[Integer-address region state](integer_address_regions.md) appends internal
logical pointer fields alongside original integer slots carrying modular byte
offsets. Original for/while/if tuple order, independent while input/result
interfaces, numeric computation and effects stay executable. Complete source
closure, bit-divisibility and SSA scope proofs precede detached publication.
`integer-address-region-state` records and generated compatibility metadata
name this representation; original casts remain inventoried. Provenance grants
no physical address, alignment or alias facts. Public tensor/scalar ABI, artifact
rosters and receipt schemas are unchanged.
Four complete model configurations also retain source state cloning as functional
readwrite results through the existing checked kernel binding. Original model
archives, static public ABI and receipt schemas remain unchanged. This does not
admit argument-mutating source schemas or dynamic public model signatures.

[Same-base pointer differences](pointer_addresses.md#same-base-byte-differences)
add `closed-pointer-integer-difference` transformations under the existing
`prov.closed_pointer_addresses` correspondence. Only actual common SSA roots
and matched lane paths cancel into ordinary modular i64 byte subtraction.
Each addptr keeps its original narrow arithmetic before signed widening and
pointee-byte scaling. Source widths, selections, casts and source identities
remain represented; bounds/ownership and complete source-use checks stay enabled.
Generated compatibility metadata names the cancellation separately from pointer
reconstruction. Public ABI, artifact rosters and receipt schemas are unchanged.
Cross-root/absolute observations and unproved region/effect forms still refuse.

[Numeric region builtins](numeric_regions.md) retain the existing source
contracts and converters inside scalar containers. Nested exact i32 logical
launch queries capture values from their own structured program entry; actual
grid expressions/induction variables supply their standard expansion. Original
query locations/source identities remain represented, surrounding control and
effects stay in place, and the detached expansion publishes only after every
candidate succeeds. `uniform-region-launch-query` records and generated
compatibility metadata name this representation. No physical mapping or six-arg
launcher ABI is introduced; public ABIs, artifact rosters and receipt schemas
are unchanged. Source external purity/prototype checks remain mandatory.

[Local tensor regions](tensor_regions.md) retain checked numeric tensor
intermediates, structured tensor state and mixed nested maps/reductions/scans.
One detached postorder expansion preserves complete refusals; shared remains
responsible for source shape and elementwise conversion. Exact shaped integer
casts use the upstream scalar-in-parallel-Linalg pattern before xDSL's recursive
verification. Updated native/Merlin support gives captured writable destinations
independent invocation storage, preserving used initial values and avoiding data
copies for unused initialization. Ordinary readonly captures and portable MLIR
stay intact. Compatibility metadata records the updated consumer requirement;
public ABI, artifact rosters and receipt schemas remain unchanged.

[Captured tensor lookups](scalar_tile_reads.md) retain immutable SSA snapshots
inside reductions/scans as guarded standard element reads. Shared projections
reuse one value at the original gather position. Shape/axis/source-layout intent
is annotated separately from effects and ownership; source tags cannot recreate
the compiler's internal generated-guard admission. Merlin may construct a
separate proved initial-state copy, including closed padding values. Public ABI,
artifact rosters and receipt schemas remain unchanged; compatibility metadata
records the representation.

[Scalar region composition](scalar_regions.md) expands checked TTIR scalar
math, clamps, casts, numeric bitcasts and existing pure external prototypes in
their original control position before lifting maps/reductions/scans. Existing
exact rounding/high-product expansions precede region admission; unknown effects
and source properties still refuse. BF16-to-f64 widening uses an exact f32
bridge. Transformation/compatibility records describe the representation; public
ABI, artifact rosters and receipt schemas remain unchanged.

[General reductions](reductions.md) retain complete pure scalar combiners,
coupled tuple order/types and structured control in standard Linalg. Each tuple
component starts from its first actual source element; singleton axes project
the original words without executing a combiner. Original reduction/scan helper
effects are audited before normalization or folding. Derived initialization/
body provenance grants no associativity, effects, termination or GPU arithmetic
equivalence. Dynamic state uses Merlin's verified numeric-control destination
path with ordinary lifetime gates. Public ABI, receipts and compiler artifact
rosters are unchanged; compatibility metadata records the representation.

[Exact IEEE source literals](ieee_literals.md) retain scalar/dense
f16/BF16/f32/f64 NaN sign, payload and signaling bits through standard integer
constants and `arith.bitcast`. Original word/type/shape records and primitive
spelling are independently checked in producer verification and Merlin staging.
All parser continuations restore words before executable serialization. Existing
allocation source capsule v2 optionally includes nonempty `literal_conversions`
in its protection report and original entry intent; older strict report readers
refuse that additive field. Original allocation auditing precedes normalization,
including unused helpers. Public ABIs, receipt schemas and artifact rosters are
unchanged. This supplies no computed-NaN or accelerator arithmetic policy.

[Structured scan combiners](scans.md#structured-combiner-qualification) retain
pure scalar if/for/while bodies and uniform captures in standard parallel scan
stages. Original control locations and derived counts remain annotated; tuple
types, operand order and per-combination loop state stay executable. All scan
candidates validate before expansion, including nested unused effects. Public
ABI, receipts and compiler artifact rosters are unchanged. Compatibility metadata
records the representation; annotations grant no associativity, effect, numeric
or target permission. Two complete models additionally retain all fourteen
original results through native/Merlin host/Spike with exact words.

[IEEE f16/BF16 atomic addition](half_atomics.md) extends the v4 typed-atomic
profile with format-specific nearest-even/quiet-NaN contracts. Standard generic
atomic regions retain source storage, masks, numeric updates and old values.
BF16 rounds after each widened addition; native checked load/CAS retry helpers
use i16 captures/results and retain full bodies through ownership/liveness.
Public ABI, receipts and artifact rosters are unchanged. Compatibility metadata
names the extension; older readers lacking these numeric forms refuse. Private
invocation state grants no external publication, physical synchronization or
accelerator implementation.

[Original CFG normalization](cfg_intake.md) adds six byte-bound compiler source
artifacts under the optional `t2m.control-flow-normalization.v1` profile. Typed
function interfaces bridge to the official LLVM pass; original block/successor
and SSA records remain separate from the normalized graph. Merlin independently
checks/copies the fixed stage and artifact roster in standalone and derived
models. Public ABI and receipt schemas are unchanged; older readers need this
additional source-custody support. Provenance and custody grant no source
equivalence, memory/effect or target permission.

[Closed pointer-integer expressions](pointer_addresses.md) are an equivalent
logical representation under checked use closure and byte divisibility: original
bases and modular offsets replace unobserved absolute integers. Source casts,
widths, pairing and locations remain inventoried and annotated; physical address
observations are not granted by this transformation. Scalar gather/scatter
additionally reuses the existing one-element carrier in tensor/buffer profiles,
retaining source masks/fills and all memory gates. Both changes are explicit in
`compatibility.json`; public ABI, source capsule and receipt schemas are unchanged.

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

The complete native no-pass reader now precedes every shared source-parser
fallback, including unselected/default profiles. Explicit global/per-cast FP8
and selected assembly contracts become executable shaped arithmetic before
pointer/descriptor/composite normalization or helper folding. Original and
normalized graphs stay distinct. Assembly reports identify the arithmetic
carrier; provider API, public ABI, source capsule and capture receipt schemas
are unchanged. Older producers may refuse these compositions or lack their
complete original audit. Hardware observations and unimplemented effects remain
located refusals, including in unused private helpers.

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

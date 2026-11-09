# Private-invocation atomic memory through the tensor interface

Status: implemented for the explicit private-invocation integer RMW/CAS profile
and the separately selected typed profile with bitwise f32/f64 exchange and
integer-word RMWs on f32/f64 storage, including Triton floating min/max.
`--memory-profile guarded-buffer --address-policy source-words --effect-profile atomics`
selects it; the default ordinary-memory profile continues to refuse atomics.
The public tensor/scalar ABI is retained with explicit v3 memory admission.
Floating addition, external publication/synchronization and broader
alias/ownership state remain required extensions for the full goal.
The new ownership encoding is implemented and native-checked independently of
admission: all 4,663 bounded access histories agree with declarative identity/
access-category rules, including differences at bit 30 and bit 60. The eighteen
protocol/layout tests also retain every existing ordinary-memory boundary case.

## Source-backed boundary

Meta `17f0c234ce474bee334af36e351c62eca7852836` already supplies:

- `TritonToUnstructuredPass.cpp`: `tt.atomic_rmw` becomes
  `tts.unstructured_atomic_rmw` with exact indices, operation, mask, memory
  semantics and scope.
- `UnstructuredToMemrefPass.cpp`, `UnstructuredAtomicRMWConverter`: a parallel
  tensor `linalg.generic` performs standard `memref.atomic_rmw` and yields its
  old value. `test/Conversion/UnstructuredToMemref/atomic_rmw.mlir` checks
  integer/floating forms, masking and returned values. Structured fixtures
  additionally check chained old-value uses.

The existing optimizer has executed this exact path for a captured Triton
integer indexed atomic-add kernel. Both native stages and standard xDSL
verification pass; source operation, scope and order survive to the TTS stage.
The product now also passes standalone packaging and actual Merlin execution.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`,
`MemRefToLLVM.cpp`, lowers standard RMW atomics with `acq_rel` ordering.
That ordering is at least Triton's relaxed/acquire/release/acq_rel requirement.
The first effect profile owns private invocation buffers with no external
participants: GPU scope closes over logical programs, CTA scope over one
logical program, and system scope adds no externally shared storage to this
ABI. Stronger standard atomics are a refinement; source scope/order remain
recorded for downstream analyses. No physical CTA/thread/warp mapping enters
the frontend. External publication, aliases and progress/synchronization
contracts remain additional work.

## Admission protocol

Keep the ordinary-memory profile unchanged. Explicit atomic admission uses a
versioned contract and the same public tensor/scalar interface. Preserve the
logical forall and parallel payload rather than serializing source programs.

The existing sticky ownership protocol must distinguish atomic participants
from ordinary reads/writes in both arrival orders. Use thirty-bit program
fragments paired with their complements, leaving three flags in each i64 word:
ordinary access, ordinary writer and atomic access. Accumulate them with atomic
OR. An atomic access may see foreign tags only when all previous accesses are
atomic. An ordinary read may see foreign tags only without a writer/atomic
flag. An ordinary store always requires one program owner. A same-program
ordinary observation after an atomic remains sticky and prevents a later
foreign atomic from bypassing that observation. Duplicate atomic lanes are
allowed; duplicate ordinary stores retain their existing rejection.

Atomic RMW requires an already completed/initialized element. Readwrite inputs
provide that state. Fresh state still requires an actual preceding source
write and post-grid coverage; an atomic does not invent its initial value.
Guard storage cost is derived from the new fragment width and actual shapes,
including the boundary where its word count differs from v2. Allocation and
consumer budgets must use the same declared contract.

Upstream masked RMW yields the uninitialized output iter-arg on inactive lanes.
Represent only those undefined old values by an explicit zero refinement,
with no inactive memory effect and no full-output initialization work. Retain
that choice in provenance/package differences. Active old values and every
atomic update remain executable operations.

## Delivery order and evidence

1. Implement and independently enumerate the sticky protocol's state
   transitions, including both atomic/ordinary orders and later observations.
2. Admit indexed integer RMW through the official source-word pipeline;
   instrument bounds/initialization/compatibility without lane uniqueness for
   atomics. Validate actual native payload against the declared RMW operations,
   scope/order and source inventory. Keep all existing memory guards.
3. Version admission/evidence, explicit profile selection and package
   differences. Extend standalone and derived Merlin intake, scalar ABI,
   allocation budgets and source-only staging together.
4. Execute colliding and disjoint i32/i64 add, exchange, bitwise, signed/unsigned
   min/max with masks, tails, loops, old-value chains and fresh initialization.
   Use exact references where ordering is unique; for colliding old values,
   check a valid linearization/permutation instead of imposing a GPU schedule.
   Reuse unchanged Merlin libraries over independent domains and preserve inputs.
5. Region-local RMW and integer CAS now extend the isolated event/pointer-control
   converter. CAS uses an exact standard atomic comparison/select region and
   checked consumer legalization; the upstream unstructured converter has no
   CAS case. Retain a complete, replayable disposition for every source event.
6. Add the float/numeric, descriptor-reduction, synchronization/external-effect
   and target materialization contracts required by the universal objective.

Pure opcode appearances, source-stage success, stronger-order annotations and
host-only execution are distinct evidence. None certifies all effects, all
numeric modes, all targets or the completed all-TTIR goal.

## Qualified representation and results

V3 uses `private-invocation-integer-atomics-v1`, thirty-bit fragments and
`sticky-complemented-atomic-participants-v1`. It retains all nine integer RMW
kinds: add, exchange, bitwise AND/OR/XOR, signed min/max and unsigned min/max.
Actual native payload accounting checks one standard RMW event per admitted
source event, its buffer/type/operation and original old-value result. A masked
inactive output-iter-arg becomes a local zero, avoiding undefined output reads
without initializing a whole tile or touching inactive source memory.

The first Merlin run exposed upstream elementwise fusion cloning an effectful
tensor generic: updates executed twice. Each atomic generic now remains inside
standard `scf.execute_region no_inline`, which executes once and retains the
complete parallel tile. Pure generics outside the event keep ordinary fusion.
No opaque call or global serialization replaces the atomic algorithm. The
boundary and minimum source order/scope remain in provenance; annotations
cannot authorize duplicating effects or weakening the executable ordering.

The 102 focused cases pass across i32/i64, scalar/tiled events and every integer
RMW kind. Checks include active/inactive bounds, all source orders/scopes,
real fresh initialization, colliding updates, old values feeding later atomics,
and ordinary/atomic conflicts in both orders. A reused Merlin ticket-allocation
library checks the returned words as a permutation, including signed wraparound,
without imposing a lane order on the source. The visible disjoint-index demo
uses a deterministic independent reference:

```bash
python -m t2m shared-demo --kernel atomic-rmw \
  --memory-profile guarded-buffer --address-policy source-words \
  --effect-profile atomics --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python --out out/atomic-rmw
```

Existing ordinary guarded/selected-memory/model/event regressions pass all 53
cases. Merlin's 88 standalone/derived intake tests and structure gates pass at
`ab2c892e3823457c871300989283859f5ba37517`. Its staging validates source-only
custody, v3 cost/protocol and declared atomic facts, excluding evaluator values.
Actual source/guard/linearization and accelerator proofs remain separate.

A real model2MLIR tuple-result atomic capture now preserves both updated state
and old values through native and actual Merlin execution. Its integer state
parameter, index buffer, inputs and every independent PyTorch reference remain
byte-identical to the parent capture. Model2MLIR's additive `goldens.npz` archive
captures all flat tensor results while retaining the legacy `golden.npy` first
result. Both execution reports bind the complete reference set. Merlin compiler
staging explicitly excludes the archive and all evaluator values.

The 32 atomic/selected-memory/source-event/model-bundle regression cases pass,
including rejection of missing, unbound, reshaped or mistyped result references
before execution and rehashed execution reports that omit reference custody.
Merlin's 23 compiler-staging tests also pass with tuple-reference access blocked.
See [model integration](model_integration.md) for the additive package contract.

## Branches, recurrences and selected buffers

RMW events now reuse the same isolated native templates as loads/stores.
Source-word pointer normalization retains same-base and finite selected-base
pointer state through `if`, `for` and `while`; each event stays at its source
position. Uniform and lane-varying buffer selection emits disjoint predicated
atomic events and merges their original old values. Operation, scope and order
are checked against the source after native event conversion, including during
evidence replay. These are semantic selectors, with no physical address spaces
or target launch mapping introduced.

The actual Merlin compiler now recognizes a closed atomic snapshot dependency
in its explicit loop-destination reuse pass. A retained exactly-once region,
unused-init tensor event, independently synthesized seeds and actual private
atomic targets permit the downstream pure accumulator to reuse its destination.
The event itself stays intact. External targets, unknown effects and shared
seeds grant no permission; native bufferization and lifetime checks stay enabled.
The original failure was caught by the lifetime check, rather than bypassed.

All 17 new region cases pass: i32/i64 scalar/tiled loops, branch-local add/XOR,
old-value accumulation, same/selected bases, final-pointer updates, zero trips,
empty/tail domains, inactive invalid pointers and active bounds failures.
Twelve combinations execute natively and reuse actual Merlin libraries over
five independent invocations each. Three rehashed event tests reject changes
to operation/order/scope. Merlin's 93 staging and 90 native loop/lifetime cases
pass, including refusal of unknown/external atomic snapshot dependencies.

The subsequent ordinary-memory/pointer/event/atomic/model regression run passes
all 145 cases with existing native/model2MLIR/Merlin environments enabled.
Together with the 17 new region cases, this qualifies the changed paths without
asserting a new complete-suite or universal-coverage result. Merlin's loop reuse
and staging changes are at `6a13a1a617244ad69e869c49a37e937dbdcd0739`;
its documentation and structure gates pass. The visible `out/atomic-rmw/`
package passes native comparison and all 20 artifact checks. Its separate
`out/atomic-rmw-merlin/` qualification passes actual Merlin host execution,
unchanged-input/reference comparison and all 45 artifact checks. Accelerator
execution remains unperformed.

## Integer compare/exchange

CAS now qualifies i16/i32/i64, matching Triton v3.7.1's public integer width
domain. All four source orders and three scopes retain their original facts.
It reuses the official rank-one gather descriptor and the existing base/offset,
region-event, initialization and atomic participation analyses. Scalar and
rank-N events retain value order through checked unit/reshape normalization.
Finite selected bases branch inside an ordinary parallel tensor generic;
only the selected buffer participates. No frontend loop serializes tile lanes.
Every enclosing effectful generic retains an exactly-once region to prevent
tensor fusion from cloning an atomic nested in a scalar branch.

The pinned optimizer does not register the `tts.atomic_cas` found in the current
Meta sources. The adapted event therefore retains the original `tt.atomic_cas`
over its official logical descriptor until expansion. The exported operation is
standard `memref.generic_atomic_rmw`: compare the current integer word with the
original comparison, select the original new word or unchanged current word,
and return the old value. Guard and payload inventories check operation/type,
source IDs, source scope/order and exact comparison/select/yield edges.

CAS has no source mask. An output-store mask cannot make an invalid CAS lane
inactive; active bounds and completed initialization remain mandatory. Actual
branch placement can suppress an event, and selected buffers use executable
branches rather than inventing a CAS mask or dropping an unselected effect.

The pinned native ownership pass cannot process MemRef's scalar-result atomic
region. Native qualification therefore requires the explicitly supplied Merlin
compiler support (`T2M_MERLIN_ROOT`; model binding also accepts `--merlin-root`).
Merlin's `lower_integer_compare_exchange` legalization uses a private synchronous
helper with one LLVM cmpxchg, acq_rel success/acquire failure ordering, no pointer
escape and no allocation. Its complete body is retained while ownership sees
the typed declaration, then restored with unchanged declaration attributes before
liveness verification and final LLVM conversion. Borrowing grants reconstruct
the current body, types and symbol scope. Portable `model.mlir` contains the
standard atomic region and no LLVM operations or opaque helper calls. These
native-toolchain requirements are explicit package differences.
The currently qualified native pointer/index ABI is the default 64-bit host
ABI. Narrower index conversion and explicit layout/triple overrides refuse;
portable atomic semantics remain independent of this native materializer limit.

The 56 focused profile cases and visible-demo test pass. They cover integer
widths, scalar/tiled and rank-two events, both selected-base forms, recurrences,
final pointers, zero trips, collisions/old values, every source order/scope,
active-address/initialization failures and rehashed scope/order changes.
Both RMW and CAS model captures also pass all ten model cases with original
weights, buffers and complete tuple references preserved. Merlin's 207 native
legalization/lifetime/staging regressions pass at
`9d066f6ff155e0105d0d444424d5b469c66932aa`. These measurements do not close float
policies, external synchronization, aliases or accelerator materialization.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel atomic-cas \
  --memory-profile guarded-buffer --address-policy source-words \
  --effect-profile atomics --optimizer /path/to/triton-shared-opt \
  --python /path/to/native-env/bin/python --out out/atomic-cas
python -m t2m qualify-merlin out/atomic-cas --out out/atomic-cas-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
python -m t2m verify-merlin out/atomic-cas-merlin
```

Final validation at Merlin `a735821a6a0f8ba86931f8ca7491fa4260a1a2a6` passes
all 215 compiler/staging/lifetime cases. The 167 existing producer pointer,
event, RMW, region and model regressions also pass. After the native ABI check,
thirteen focused producer cases pass; the final helper-linkage suite passes
all 23 cases and inspects actual converted LLVM linkage. These are scoped
regressions, not a newly claimed complete producer-suite result.

`out/atomic-cas-merlin-native/` passes host execution and all 45 qualification
artifact checks. `out/cas-model-native/` passes complete native/model/Merlin
comparison and all 80 derived-package artifact checks. Both actual LLVM modules
contain internal CAS helpers; portable model source remains standard tensor/
SCF/memory IR. Earlier qualification directories remain unchanged. The source
capture's original weights and all model result references stay byte-identical.

## Typed profile and floating exchange

`--effect-profile typed-atomics` selects
`private-invocation-typed-atomics-v2` with `t2m.memory-admission.v4`. It retains
the existing integer RMW/CAS contract and additionally admits f32/f64 exchange.
`shared-demo --kernel atomic-exchange --memory-profile guarded-buffer
--address-policy source-words --effect-profile typed-atomics` emits a standalone
example; `scripts/capture_atomic_model.py --kind exch` captures the tuple-result
model2MLIR source for later binding.
Each floating event binds its actual value dtype and `bitwise-exchange` numeric
contract in source inventory, access, adaptation and staged intent. The v3
integer profile still refuses floats. V4 retains the same thirty-bit ownership
protocol, initialization requirements, scopes, ordering and allocation costs.

Meta's existing unstructured atomic converter supplies `memref.atomic_rmw
assign`; the frontend adds typed inactive zero and an exactly-once SCF boundary.
There is no floating arithmetic, whole-buffer representation copy, LLVM operation
or opaque helper in exported MLIR. Native and Merlin execution tests compare
raw words for f32/f64 quiet/signaling NaNs with payloads, signed zeros,
infinities, subnormals and maximal finite values. They check both returned old
values and final state across scalar/vector programs, masked invalid addresses
and dynamic empty/tail grids. Repeated Merlin invocation reuses the same library.
A real model2MLIR capture additionally retains weights and both tuple results.
The general package comparator uses numerical tolerance with equal-NaN semantics;
these dedicated word comparisons establish the stronger exchange bit contract.

Floating min/max now use the typed word-region extension below. Triton 3.7.1
`language/semantic.py` implements them using same-width integer pointer bitcasts
and signed/unsigned atomic min/max selected by the input sign bit. Preserve that
source algorithm rather than substituting floating min/max: NaN payloads and
signed zero are part of its word behavior. Meta's pointer-offset propagation
supplies the equal-storage-byte foundation. Changed storage widths still need
explicit rescaling/bounds/alias reasoning, and floating add separately needs a
rounding/subnormal/NaN contract.

Measured exchange delivery: 43 focused exchange/model/CAS-compatibility cases
passed in `.scratch/typed-atomic-final.log`. The broader run passed all 116
existing memory/schema/guard cases; its new float-model negative test initially
used an unchanged dtype and was corrected/rechecked in the focused run. Merlin
passed 99 staging tests plus its documentation/structure gates. Native and
actual Merlin evidence are separately verified in `out/atomic-exchange/`
(20 artifacts), `out/atomic-exchange-merlin/` (45), and
`out/exchange-model-native/` (80). The model's weights and both result references
are preserved. No accelerator execution is claimed.

[Ordinary logical word views](pointer_views.md) now supply the load/store
foundation described above; atomic view admission and region/native legalization
remain additional work.

## Integer word updates and floating min/max

The v4 `typed-atomics` profile now additionally retains all nine i32/i64 RMW
operations on equal-width f32/f64 storage. Each source operation, access and
payload binds a `word_view` with storage/value dtypes, element bits and
`equal-width-elements` address units. Legacy integer-only admission still refuses
these views. No floating-add rounding policy is inferred.

The address analysis keeps the original pointer type and an event-specific
storage/view type. Ordinary loads/stores may use storage/register bitcasts while
an atomic retains its source integer view. Isolated native event templates
capture the actual floating root and reconstruct the view internally; their
retained input/output and replay bind that distinction. Runtime guards validate
bounds, ownership and completed initialization against the actual storage buffer.

After guards, each word update expands to a parallel standard tensor generic.
Its `memref.generic_atomic_rmw` region bitcasts the current floating word to an
integer, applies the original integer operation and bitcasts the update back.
Exchange uses the original outside integer word. The returned old floating word
is bitcast back to the exact source result bits; masked undefined old values
retain the explicit integer-zero refinement. Provenance includes logical word
views, source scope/order and the source-to-standard event correspondence.

Merlin's explicit `lower_atomic_word_views` feature recognizes these closed
regions and emits one integer LLVM atomic RMW per primitive, with internal
linkage and a typed synchronous floating-memref borrowing boundary. It uses the
same retained-body ownership suspension/restoration and actual-body liveness
checks as integer CAS. The default 64-bit descriptor/index ABI remains required.
Exported MLIR has no LLVM helper, physical reinterpretation or whole-buffer
representation copy. The feature also handles ordinary integer CAS, and both
explicit atomic features compose at one pre-bufferization seam.

Triton's f32/f64 min/max now use this path without changing their source
signed/unsigned, sign-conditioned algorithm. Both source RMWs, original masks
and old values remain represented; substituting generic floating min/max could
change NaNs and signed zero. Independent raw-integer oracles check all nine
word operations and floating min/max, including quiet/signaling NaN payloads,
zeros, infinities, subnormals, empty/tail grids and scalar/vector min/max.
Actual Merlin libraries are reused across domains. Real model2MLIR min/max
captures retain learned weights and both state/old-value tuple results.

Run `shared-demo --kernel atomic-min` or `atomic-max` with guarded-buffer,
source-words and typed-atomics. Native qualification additionally requires the
explicit `T2M_MERLIN_ROOT` compiler-support path.
`scripts/capture_atomic_model.py --kind min` / `--kind max` captures the model
source for binding. Word CAS views, differently typed selected roots and broader
address/alias contracts remain further work; floating addition requires its own
rounding/subnormal/NaN contract. Accelerator execution remains separate.

### Canonical atomic identities and assignments

Merlin's pre-bufferization passes can simplify an update without removing its
atomic effect. For example, integer `add 0` or `and -1` can leave only inverse
current-word bitcasts; signed extrema can become current-independent writes.
The checked native legalization recognizes exact current identities, inverse
bitcasts, scalarized integer constants, outside floating assignments and actual
floating constant assignments. Identities still perform a real integer RMW;
assignments retain their old result. Constants are cloned as native operations,
so signaling-NaN payloads never pass through Python float conversion. Unknown
extra arithmetic/effects and changed borrowing declarations remain refusals.

Measured word delivery: 26 native numerical cases and 22 actual Merlin cases
pass, with independent raw-word oracles for all nine integer updates and floating
min/max. Thirteen evidence/demo cases and ten constant-normalization cases also
pass. The 15 existing atomic model cases and ten new min/max model cases pass
across their measured runs. The 131 pointer/view/exchange/CAS/region/schema
regressions pass; 242 final Merlin compiler/staging/loop cases and the
code-derived documentation/structure gates pass. None of these finite suites is
universal TTIR or accelerator coverage.

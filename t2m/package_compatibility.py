"""Explicit kernel representation differences shared by both package writers."""

from .memory_contract import BUFFER_PROFILE, GUARDED_PROFILES, TENSOR_PROFILE, WORD_PROFILE


def kernel_representation_differences(trace, *, dynamic=False):
    profile = trace.get("profile", TENSOR_PROFILE)
    differences = []
    if profile == BUFFER_PROFILE:
        differences.append("explicit mixed tensor/memref payload with archived semantic witness and memory continuation; the public tensor/scalar ABI is retained")
    if profile in GUARDED_PROFILES:
        differences.append("explicit guarded standard-memory payload with an independent runtime admission report; active bounds, read/write ownership, completed initialization and per-store lane uniqueness are executable; source-base gathers retain memory effects before optimization; grid-derived metadata cost is explicit; no pure tensor witness is asserted")
    if profile == WORD_PROFILE:
        differences.append("explicit source-word address policy retains source integer arithmetic before signed i64 pointer-offset accumulation; raw offset tensors replace structured pointer views; word-stage input/output evidence is retained")
        if trace.get("memory", {}).get("word_conversion") == "native-isolated-events-v2":
            differences.append("region-local memory uses isolated native event templates reinserted at original locations and arithmetic-only parent conversion; word-stage v2 archives intermediate IR and both actual native reports; typed event reconstruction is checked; tensor/scalar ABI is unchanged")
    transformations = trace.get("transformations", [])
    if any(t.get("kind") == "source-fail-stop-assertions" for t in transformations):
        differences.append("source assertions use upstream scalar/lane-wise standard cf.assert at original control positions, with source condition/message IDs retained; failing native invocations print literal diagnostics and abort using the evaluator's libc stdout/setbuf transport; the public tensor/scalar ABI is unchanged")
    if any(t.get("kind") == "logical-boolean-byte-memory" for t in transformations):
        differences.append("exclusive Triton Boolean byte loads observed by nonzero tests and stores of zero-extended i1 values become logical i1 tensor memory; masks, fills and original source IDs remain explicit; raw byte observations and changed byte addressing require a separate storage ABI")
    if any(t.get("kind") == "logical-pointer-word-view" for t in transformations):
        differences.append("equal-width logical pointer views retain original storage tensor types and source-word offsets; loads/stores use register bitcasts rather than physical aliases or whole-buffer representation copies; source integer fill bits are selected after a masked zero-filled storage load; word-view provenance is retained")
    if any(t.get("kind") == "logical-atomic-word-view" for t in transformations):
        differences.append("integer word atomics retain floating tensor storage and exact source integer updates in standard generic atomic regions: RMW on f32/f64 and CAS on f16/f32/f64; word-view dtypes and unchanged element address units are explicit v4 event/access/payload facts, while original primitive contracts remain shared across call contexts; exact legacy source-enriched records remain readable; old words are preserved with typed inactive zero refinement for masked RMW; the native ownership toolchain requires Merlin's checked atomic-word-view legalization and default 64-bit descriptor/index ABI; portable MLIR contains no physical alias or opaque helper")
    if any(p.get("operation") == "cas" and p.get("word_view") for p in trace.get("memory", {}).get("atomic_payload", [])):
        differences.append("word CAS uses an additive source-word stage v2 typed-template variant: input_types bind actual floating storage captures while template_input_types bind integer pointer formals only at the isolated native pointer-analysis boundary; producer replay reconstructs the complete typed event and consumer staging checks the declared pair; public tensors, artifact names and receipt schemas are unchanged; native CAS helpers return integer old words to avoid half floating ABI conversions")
    selected = [t for t in transformations if t.get("kind") == "selected-pointer-memory"]
    if selected:
        differences.append("finite mixed-base pointers use internal logical-buffer selectors paired with source-word offsets; predicated accesses stay at the original memory event; the public tensor/scalar ABI is unchanged")
        payload = trace.get("memory", {}).get("atomic_payload", [])
        viewed = {tuple(sorted(p["source_ids"])) for p in payload if p.get("word_view")}
        ordinary = {tuple(sorted(p["source_ids"])) for p in payload if not p.get("word_view")}
        if viewed & ordinary:
            differences.append("selected integer atomics retain each root's actual integer or equal-width floating storage type; one complete source context can bind both plain and word-view events; compiler staging matches their actual storage captures independently of logical CAS template formals, while primitive contracts and complete producer event reconstruction remain mandatory; stage/artifact schemas and public tensor/scalar ABI are unchanged")
        if any(t.get("expansion") == "per-root-original-storage-memory-event-with-register-bitcasts" for t in transformations):
            differences.append("selected ordinary memory retains each root's own equal-width storage dtype; predicated loads/stores perform register bitcasts at the original event and preserve source fills after loading; shared element offsets do not imply a shared storage representation; artifact and public tensor/scalar ABI schemas are unchanged")
        if any(t["missing_other_zero_refinement"] for t in selected):
            differences.append("inactive undefined source load values use the pinned upstream gather's zero refinement; this does not assert that inactive source memory was initialized or read")
    if any(t.get("kind") == "logical-selector-permutation" for t in transformations):
        differences.append("closed selector permutations retain immutable initial selector tiles and carry scalar i32 table indices through the original loop; logical per-lane choices are decoded at their uses, while original source-word offset recurrences remain executable; this changes internal control representation without a GPU mapping, new public argument or native lifetime exemption")
    if any("t2m-expand-cat-preserve-order" in t.get("pipeline", []) for t in transformations):
        differences.append("TTIR cat uses a checked pre-layout join/transpose/reshape expansion in lhs-then-rhs order, choosing a permitted ordering; original native source and normalized graphs remain distinct")
    if dynamic:
        differences.append("ABI v5 named runtime tensor dimensions and logical ceil-divided grids; dynamic result allocation uses scalar bindings; Merlin staging uses compile-signature v2")
    if trace.get("memory", {}).get("effect_profile") == "private-invocation-integer-atomics-v1":
        differences.append("explicit private-invocation integer atomics use standard memref RMW operations and original old values; v3 thirty-bit owner fragments permit atomic collisions while rejecting ordinary races; source scope/order are retained with stronger standard acq_rel semantics; inactive undefined old values refine to zero; tensor/scalar ABI is unchanged")
    if trace.get("memory", {}).get("effect_profile") == "private-invocation-typed-atomics-v2":
        differences.append("explicit private-invocation typed atomics retain integer RMW/CAS, bitwise f32/f64 exchange and IEEE f64 round-to-nearest-even addition with subnormals preserved and unspecified arithmetic NaN payloads; v4 memory admission retains v3 thirty-bit ownership, initialization and guard costs; floating exchange preserves all old/new bits without arithmetic or rounding; source scope/order use stronger standard acq_rel semantics; inactive undefined old values refine to typed zero; tensor/scalar ABI is unchanged")
    numeric = {event.get("numeric_contract") for event in trace.get("memory", {}).get("atomic_payload", [])}
    if "ieee-f64-rne-add-quiet-nans" in numeric:
        differences.append("strict IEEE f64 atomic addition preserves subnormals, round-to-nearest-even, exact old words and explicit arithmetic NaN quieting in standard generic atomic regions; the versioned quiet-nans contract distinguishes it from the legacy standard-addf profile, which does not guarantee NaN quieting after constant normalization; native ownership requires Merlin's checked atomic-word feature, complete internal CAS retry helper and default 64-bit descriptor/index ABI; core artifacts and tensor/scalar ABI are unchanged")
    if "f32-rne-sign-preserving-ftz-add" in numeric:
        differences.append("explicit f32 atomic addition policy flushes subnormal inputs/results to sign-preserving zero with round-to-nearest-even; arithmetic NaN payload/sign is unspecified and old words are exact; portable generic atomic regions contain the executable bit arithmetic; native ownership requires Merlin's checked atomic-word feature and default 64-bit descriptor/index ABI, preserving the complete internal CAS retry helper through lifetime checks; no native helper appears in portable MLIR")
    if "ieee-f32-rne-add" in numeric:
        differences.append("explicit IEEE f32 atomic addition policy preserves subnormal inputs/results with round-to-nearest-even, exact old words and unspecified arithmetic NaN payload/sign; portable generic atomic regions retain addition and explicit result NaN quieting through canonicalization; native ownership requires Merlin's checked atomic-word feature and complete internal CAS retry helper under the default 64-bit descriptor/index ABI; package artifacts and tensor/scalar ABI are unchanged")
    if any(item.get("operation") == "cas" for item in trace.get("memory", {}).get("atomic_payload", [])):
        differences.append("compare/exchange uses a standard generic atomic comparison/select region, retaining success/failure values and the original old word; selected bases branch inside parallel tiles; the pinned native ownership toolchain additionally requires Merlin's checked compare/exchange or atomic-word-view helper legalization and qualified default 64-bit descriptor/index ABI (T2M_MERLIN_ROOT for native qualification); portable model.mlir retains no LLVM operations or opaque helper calls")
    return differences

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
    selected = [t for t in transformations if t.get("kind") == "selected-pointer-memory"]
    if selected:
        differences.append("finite mixed-base pointers use internal logical-buffer selectors paired with source-word offsets; predicated accesses stay at the original memory event; the public tensor/scalar ABI is unchanged")
        if any(t["missing_other_zero_refinement"] for t in selected):
            differences.append("inactive undefined source load values use the pinned upstream gather's zero refinement; this does not assert that inactive source memory was initialized or read")
    if any("t2m-expand-cat-preserve-order" in t.get("pipeline", []) for t in transformations):
        differences.append("TTIR cat uses a checked pre-layout join/transpose/reshape expansion in lhs-then-rhs order, choosing a permitted ordering; original native source and normalized graphs remain distinct")
    if dynamic:
        differences.append("ABI v5 named runtime tensor dimensions and logical ceil-divided grids; dynamic result allocation uses scalar bindings; Merlin staging uses compile-signature v2")
    return differences

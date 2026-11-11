# Package observations against every TTIR definition

`t2m.frontend.inspect_ttir(source, work)` now supplies a separate original-source
inventory even when product conversion refuses. It uses the pinned native
parser/verifier in a bounded subprocess, retains exact input bytes and logs,
and performs no normalization or target compilation. Invalid source creates no
successful inventory. The successful-package registry below stays unchanged;
source inspection alone grants no product or execution coverage.

Each explicit selection admits at most 256 packages and 256 standalone
qualifications. The earlier 64-entry limit prevented the expanded corpus from
being reported together. Processing remains serial with the existing receipt,
per-JSON size and per-case operation budgets; increasing the selection limit
grants no operation coverage or source/target authority.

`package-coverage` joins verified shared-tensor and bound-model packages to
the [complete pinned 50-op inventory](ttir_operations.md). It reopens retained
evidence and executes no compiler or kernel. The report is
`t2m.package-coverage.v1`, saved as `coverage.json` in a fresh output directory.

```bash
python -m t2m package-coverage \
  out/lifted-observe-model/qualified out/lifted-matmul-model/qualified \
  out/bfloat-matmul \
  --qualification out/bfloat-matmul-merlin \
  --out out/package-observations
```

Every known definition remains in `ttir`, including unmeasured operations.
Definition metadata records the release, source revision/hash, TableGen record
and source coordinate. `ttir_definitions.py` contains extracted metadata, not
an implementation allowlist. Unknown `tt.*` names remain in `unknown_ttir`;
all other dialect operations remain in `other_operations`, without assuming
those dialects are standard or supported.

Allocation packages with a normalized original-source capsule use its native
`allocation-inventory.json` as the original graph. Case records identify
`pre-wrapper-original-source-operations`, pin that inventory and retain separate
bound-source hashes/counts. Pointer-only capsules without an original native
inventory retain `allocation-bound-source-operations`; the report never invents
an original inventory. Capsule/receipt validation precedes indexing. This
records byte-bound producer observations, not independent source equivalence.

Rows retain exact declared input/output types and available properties/attributes,
with form hashes and source IDs. The native original-graph binding currently
does not expose complete attribute dictionaries. Those fields remain explicit
nulls with `attribute_evidence: not-exposed-by-source-inventory; consult-exact-source`,
rather than invented empty dictionaries. Exact source bytes remain archived.
Original source graphs supply the denominator;
normalized graph counts remain separate, so descriptor normalization cannot
erase original descriptor operations from coverage. Cases retain actual
source/payload/trace hashes, semantic ABI and memory profile. The registry
checks receipt membership and bytes before collecting an observation; changed
payloads, foreign paths, symlinks and duplicate selections refuse.

Native package comparison and Merlin qualification are separate observations.
Bound models use their receipt-bound host reports. Standalone kernels require
an explicitly supplied qualification, matched by the exact original source
receipt digest; its historical `source_directory` hint is never followed.
An unrelated, changed or incomplete qualification refuses. Merlin observations
require whole-body lowering, compiler staging, reference comparison and
evaluator-input exclusion. Numeric tolerances remain explicit. Missing host
evidence never inherits success from native execution.

The final local report at `out/package-coverage/coverage.json` selects 13
packages and 13 kernel implementations. It observes 22 of the 50 definitions;
18 occur in captures of its nine Merlin-host-qualified packages. These counts
describe that selection, not repository-wide support or the number of executed
operations. Branch activity and per-operation execution coverage are not
measured. Source inventories remain byte-bound producer declarations, not
independent source-equivalence proofs. Name matching does not prove source
version compatibility or universal type/attribute semantics.

Tests use real descriptor normalization and actual native/Merlin qualification,
retain the existing demo's declared floating tolerance, check unknown names
separately, and reject changed bytes and symlinked parents.
All ten registry/upstream-corpus regression cases pass.

This closes the missing package-to-definition observation adapter, not the
maintained-coverage workstream. Failed/refused kernels are not complete
packages and remain outside this selection. Keep them in separately measured
corpus/refusal reports; `shared-coverage` continues to measure upstream stages
without granting product coverage. A maintained full corpus, refusal matrix,
numeric/effect closure and applicable target evidence remain completion gates.

Observable-outcome receipts/qualifications are also admitted as explicit
`reference_kind: observable-effects`. Their normal-return or assertion-failure
outcome remains recorded separately from numeric tolerances. A matched source
failure does not imply every inventoried operation executed or that a tensor
result was produced; source and per-operation execution coverage stay distinct.
Mixed returning packets use `reference_kind: observable-effects-and-tensors`.
Their qualified host observations retain both the exact stream/termination
policy and the complete per-result word/value policies. This remains source
composition evidence, without inferring that every source operation executed.

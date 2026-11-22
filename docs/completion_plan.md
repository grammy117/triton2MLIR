# Finite closure plan: portable TTIR to the Merlin interface

Updated 2026-11-21. This is the governing v1 release plan. The latest scope
requires every generic/useful TTIR computation to have an executable lowering;
exclusions apply to assembly, hardware observations and missing external
semantics, not to difficult generic computation. Use a broader logical
load/store, SSA or standard control/effect expansion where a specialized pattern
cannot be recovered. Preserve actual semantics and disclose lost structure.

This plan separates generic computation closure from arbitrary device semantics
and qualification of every possible target. Historical measurements in the roadmap,
coverage and handoff documents retain their original scope; they are evidence,
not an automatically expanding backlog.

## What we will finish

Ship a versioned, usable TTIR frontend that preserves useful tensor tiles,
fusion, reuse, recurrences and effects through Merlin's public input interface.
For every source form in its pinned domain, return either a qualified portable
package or a precise, actionable disposition. Do not promise portable execution
of arbitrary physical observations or undefined source behavior.

**Closure means complete accounting plus qualified support for a declared
profile. It does not mean every kernel becomes portable MLIR.** Unsupported
forms remain unsupported in the published numbers. Generic operation gaps
remain release work; physical/GPU-specific behavior has explicit limitations.

A successful portable result contains executable standard MLIR, the actual ABI,
source/emission provenance, numeric/effect policies and required compiler
members. It reaches the same public `StagedProgram` boundary as model2MLIR and
executes through Merlin. Any package differences remain explicit in
[the package contract](package.md). Intent annotations never replace computation.

## Freeze the scope

| Axis | v1 boundary |
| --- | --- |
| Source | Triton 3.7.1, revision `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`; its 50 TableGen `tt` definitions plus explicitly admitted supporting dialect operations. New versions require a separate compatibility change. |
| Input | Verified pre-layout TTIR or a captured source specialization, with a semantic ABI. Post-layout GPU dialects, GPU binaries and recovering arbitrary Python launchers are excluded. |
| Kernel domain | Finite logical buffers, explicit input/state/output bindings, declared shape/scalar domains, checked address/initialization/alias rules and named numeric/effect profiles. The classifier specifies admitted types, attributes and compositions; operation names alone grant no support. |
| Portable consumer | The current public Merlin kernel and bound-model staging/preparation interface, whole-body host lowering and reusable host execution. No retired namespace fallback. |
| State | Explicit logical roots and functional state/results. Physical alias identity, caller mutation/copyback and externally participating storage are supported only where an existing qualified adapter states that contract; otherwise excluded from v1. |
| Other targets | Keep existing serial-target results as regression evidence. A fresh bounded serial smoke selection is required. New accelerator scheduling, correctness and performance campaigns are separate releases and cannot block portable-interface v1. |
| Performance | Preserve and inspect algorithmic tiles, reuse and recurrence on the structured route. Generic fallbacks must disclose lost structure and extra storage. No GPU speedup or backend-equivalence claim is required. |

Freeze producer revisions, tool hashes, policy rules and the expanded acceptance
manifest in C0 below. Starting observations are triton2MLIR `b579ada9675aa7d7f0f2033c5fd03568274e24c1`,
Merlin `28e4d702fa1d384a4c6c66cac0ebced9ed7108c6` and model2MLIR
`f3f0de1ae130faebbd27bd36374d11e1e211daa2` on their existing branches.
Final evidence names the actual final commits, installed tools
and artifact bytes. Freeze dependencies used by that run; building a universal
supply-chain/equivalence proof system is outside this release.

## Dispositions and honest counters

Each classified source form has one route and separate evidence status. These
are the planned release-policy values, not new implemented CLI/schema claims.

| Route | Meaning | Counts as portable support? |
| --- | --- | --- |
| `portable` | Qualified structured tensor/standard MLIR path. | Yes, within its exact profile. |
| `portable-with-contract` | The same, with explicit extra caller information and enforced preconditions. | Yes, conditional on that contract. |
| `portable-fallback` | A disclosed, qualified generic standard-MLIR implementation with the same observable semantics. | Yes; report fallback use separately. |
| `backend-required` | Behavior needs a physical device/runtime or an unavailable implementation. Report the requirements; do not emit an opaque successful portable package. | No. No new original-GPU execution runtime is required for v1. |
| `unsupported` | A valid source form explicitly excluded from this release, with a reason and remediation. | No. |
| `invalid-source` | The pinned verifier rejects it. | No. |

Evidence is independently `qualified`, `expected-refusal`, `failed` or
`not-run`. A selected positive case with `failed`/`not-run` blocks release.
A missing dependency is `not-run`, not a pass or an exclusion. Count all 50
operation policy entries, admitted typed profiles, portable cases, contract
cases, fallback cases and exclusions separately. Never report refusal coverage
as portable op coverage.

A malformed/contradictory ABI is an invalid-contract diagnostic, not evidence
that the underlying TTIR is invalid. A missing required contract produces an
actionable request for the named bindings; it does not count as a portable pass.

Every op entry needs a predicate for admitted forms and an explicit complement
rule. Supporting operations, types, semantic attributes, regions and their
compositions need the same admission checks. Unknown names or unavailable
attribute evidence cannot inherit a neighboring form's success. New kernels use
the same general lowering rules; there is no kernel-name or fixture allowlist.
A valid generic refusal is a tracked gap even outside the frozen corpus. Finite
tests do not prove universal translation or classification correctness.

## Decide the hard boundaries now

The defaults below are release decisions, rather than invitations to build a
general solver, alias engine or GPU runtime. Reuse the existing mechanisms.

| Gap | First route | v1 stopping decision |
| --- | --- | --- |
| Irregular/loaded addresses, while-pointer state, nonlinear indexing | Existing structured analysis; otherwise explicit root/extent/stride bindings and guarded logical memory. | Retain bounds, initialization, source order and write-conflict checks. Broaden the logical memory/control expansion when specialized inference fails; valid generic gaps remain open. |
| Aliases, views and mutation | Explicit storage roots, offsets, layouts, overlap and functional state/results; use existing qualified adapters. | Never assert no-alias to silence a conflict. Unrepresentable physical alias identity or unimplemented copyback is unsupported, with a functional-state adapter as remediation. |
| Runtime shapes/scalars and value-dependent output sizes | Declared scalar/dimension bindings; reconstruct actual shape dependencies or use existing execution-owned results with finite allocation limits. | No guessed sizes or implicit specialization. Supply missing bindings or extend generic runtime allocation/shape transport; valid generic gaps remain open. |
| Numeric formats, precision, reductions and RNG | Named, qualified profiles with their rounding/order/overflow rules. | Missing defined generic arithmetic needs an executable implementation. Undefined source enumerants and GPU-specific exact-bit claims need actual semantics. No silent precision relaxation. |
| Packed maps/assembly | Existing independence/layout proofs and optional typed source provider. A generic expansion may be slower if its semantics are qualified. | General source-coupled packing, partial registers and unimplemented float/ISA forms are excluded unless a qualified standard-MLIR implementation is supplied. Preserve original pack intent and disclose sharing loss. |
| The current packet recurrence ownership failure | Fix the generic Merlin ownership/destination proof within C1, or keep the packet route opt-in and retain the previously qualified scalar recipe as a disclosed fallback. | Do not enable the failing path by default or weaken lifetime checks. A scalar recipe does not claim packet sharing. |
| Barriers, atomics, volatile and async | Keep qualified invocation-local phases/effects. Abstract physical scheduling only when actual dependencies and observations permit it. | External participants, general cross-program publication, race-sensitive behavior and device completion need an explicit execution contract/provider; otherwise backend-required. Serialization is not automatically equivalent. |
| Inline assembly, extern calls and physical address observations | Known source provider or an explicit typed standard-MLIR implementation with its dependencies and trust boundary. | Unknown instructions, hardware clocks/IDs, escaping physical addresses, impure externs and descriptor bytes are backend-required/unsupported. Provenance is not their implementation. |
| Remaining CFG/function/ABI forms | Existing official normalization and public bindings, or broader standard func/CF/SSA. | Valid generic entry/control forms require a working ABI/lowering; missing support remains a blocker. Preserve located diagnostics while implementing it. |
| Target scheduling/materialization | Existing Merlin host path and bounded serial smoke cases. | Missing accelerator providers become an explicit target capability limit. Do not expand the frontend project to finish every backend. |

A trusted caller declaration and an established source fact remain distinct.
Contracts express the intended input domain; they never prove arbitrary program
properties. A false declaration must not allow unchecked memory access.

## Extra information users can supply

Extend the current ABI/binding mechanisms rather than introduce a parallel API.
C2 freezes the exact supported fields and package differences. Required forms:

- **Storage:** allocation roots and byte extents; element types/byte order;
  offsets/strides; overlap relationships; readonly/readwrite roles; initial and
  final state ports. Guards must dominate affected memory operations.
- **Runtime domain:** dimension/scalar sources, finite ranges, logical grid,
  layout constraints and output-allocation limits. Shape functions must use
  declared available inputs; input-word reads still need bounds checks.
- **Numeric intent:** selection of an already implemented rounding/precision,
  reduction/atomic or RNG profile. Changed semantics require an explicit new
  profile and new evidence, not an annotation claiming equivalence.
- **Missing source implementation:** a typed portable MLIR implementation and
  its effect/dependency contract, or a selected existing provider. Bind it to the
  source operation and implementation bytes. Qualify it against an independent
  reference; label source equivalence as provider-supplied where it is trusted,
  rather than universally proved by tests or by the caller's assertion.

For example, a loaded-index gather can use a declared logical root and checked
extents instead of discovering a symbolic closed form for every index. A
value-sized compaction result can use execution-owned bounded allocation instead
of predicting its length from a sample. Unknown assembly needs an implementation;
adding `pure=true` does not supply one.

## Fallback policy

Try structured tensor MLIR first. If its proof does not apply, select an existing
qualified generic tensor or guarded logical-memory expansion only when its
preconditions hold. It may keep explicit serial control where source semantics
permit it; it cannot invent a deterministic order for an externally observable
race. An optional semantic provider must emit actual supported computation.
Otherwise return backend-required or unsupported.

Record the selected route, reason, preconditions, numeric/effect policy and
structural impact beside source provenance. A fallback must not silently
materialize a dense attention score matrix, discard source recurrence, round in
a different place or erase an effect. Document any supported slower alternative
as a separate profile. External runtime fallback is a distinct target-dependent
product, not portable lowering and not required to finish v1.

## Remaining execution sequence

Five bounded work packages replace the previous open-ended sequence. The effort
ceilings below are planning limits for one implementer, not promised calendar
dates or a request for a long compute campaign. **Ten focused engineering days
is the proposed implementation ceiling.** Each package ends in a reviewable
commit, updated dispositions and exact evidence. Do not start a sixth package
by finding another interesting variant.

Planned authored deliverables are `docs/v1_policy.json` (profile/route rules),
`tests/data/v1_acceptance.json` (exact case and invocation roster) and
`scripts/check_v1_closure.py` (the release gate). Generated run records belong
under `out/v1-closure/<run>/`, including `summary.json`. These files are C0/C4
implementation work, not artifacts already supplied by this planning update.

| Package | Ceiling | Concrete output and exit |
| --- | --- | --- |
| C0: freeze | 1 day | A tracked policy ledger covering all 50 `tt` definitions and admitted supporting forms, plus a fixed expanded manifest selected from existing fixtures. Each row has profile predicates, route, owner, evidence/counterexample and a reason for exclusion. Freeze the baseline tests and consumer/tool pins. No untriaged row. |
| C1: safe default/public interface | 2 days | Resolve the uncommitted packet path safely; qualify default kernel and full-model staging, immutable weights, live buffer/scalar ports, every result and artifact-only execution. Explicitly record any slower fallback. No lifetime, alias or payload-accounting bypass. |
| C2: contracts and memory/runtime escape routes | 2 days | Document/enforce the existing supported root/view/scalar/shape contracts; add only missing adapters needed by the frozen accepted cases. Check a good and a violating contract, dynamic replay, empty/zero-trip paths and allocation limits. Classify other relationships now. |
| C3: numeric/effect/provider disposition | 2 days | Close the finite admitted profile list using existing conversions/providers; test their boundaries. Freeze backend-required/unsupported profiles for shared/async/physical/unknown-source behavior. No new ISA family, math library or theorem-prover project. |
| C4: release qualification | 3 days | Run the frozen acceptance selection and relevant baseline regressions from final commits; publish complete counts, logs, package differences, limitations and reproduction instructions. Host execution plus bounded fresh serial smoke; independently report unavailable optional targets. |

A work item gets one bounded implementation attempt and one corrective attempt.
If it exceeds its ceiling, choose a supported contract, qualified fallback or
explicit exclusion **for hardware-specific behavior or missing external
semantics**. Difficult generic computation must use a broader executable
representation or remain an open blocker. A regression in previously
qualified baseline behavior cannot be relabeled unsupported to get a green
report. Fix it, retain the prior qualified default, or stop with a concrete
release blocker and ask for an explicit scope/time decision. Do not silently
extend the deadline or claim success.

New kernels found after C0 are new-version requests unless they demonstrate a
bug in an already-admitted profile. Existing regression suites remain required;
the fixed release manifest limits new acceptance development, not regression
obligations. A real accepted-profile bug may block release, but it must not
restart an unlimited semantic-coverage campaign.

## Frozen acceptance selection

C0 maps these obligations to exact existing fixtures and reference archives.
Prefer existing captures/tests. Freeze at most 128 expanded release cases,
including contract/refusal cases; freeze additional invocation values too.
The full regression snapshot stays separate. If all 50 policy entries need more
inventory-only records, record those separately without claiming execution.

| ID | Required workflow |
| --- | --- |
| K1 | Pointwise/masked and strided tiles: exact fills, untouched output, inactive out-of-bounds lanes. |
| K2 | Tiled GEMM: original nonzero C, runtime K loop, reused tiles, empty K and ragged M/N/K. |
| K3 | Online-softmax attention: carried accumulator/max/normalizer and bounded tiles; no dense-score replacement. |
| K4 | Tuple reduce/scan/histogram and nested maps: actual combiner, source ordering and loaded-value snapshots. |
| K5 | Indexed/selected/while memory: checked active indices, root selection and a defined conflict policy. |
| K6 | Logical storage/views/state: supported offsets/strides/byte aliases, preserved initial words and all final state. |
| K7 | Runtime control/CFG: if/for/while, negative/zero trips, nontrivial final state and source-width arithmetic. |
| K8 | Block pointers and logical descriptors: original descriptor inventory, runtime origins/fields and masked edges. |
| K9 | Named numeric profiles: casts/low precision/scaled dot, exceptional values and explicit tolerances/word rules. |
| K10 | Packed maps/source assembly: admitted provider/proof, retained structure or disclosed scalar fallback, refusal of unimplemented groups. |
| K11 | Local effects: atomic old/final values, volatile observations, barrier phases and exact print/assert outcomes. |
| K12 | Public ABI: dynamic shapes/scalars/grid, execution-owned and void outputs; unsupported entry forms refuse explicitly. |
| M1 | Learned immutable parameters and dependent kernel calls; every tuple result and snapshots. |
| M2 | Symbolic shapes plus live scalar controls; one compiled artifact reused on changed and empty inputs. |
| M3 | Internally sized/bounded execution-owned results, including empty outputs and an actual allocation-domain rejection. |
| M4 | Functional buffer/user state and admitted view/alias bindings; all state observations retained. |
| M5 | Mixed numeric/tiled model composition, including lifted constants and source precision/reuse. |
| M6 | Mixed tensor/effect or void model: exact normal/failing outcome and all returning results. |
| M7 | Two back-to-back full transformer blocks: ordinary PyTorch first, Triton second, actual launch discovered and integrated automatically; no hand-authored binding table or replacement with the CPU reference. |

For each positive workflow, require independent whole-result/effect comparison,
changed-input replay, source/payload accounting and actual public Merlin staging
and host execution. Exercise boundary values relevant to its profile; do not
expand every workflow into an unbounded Cartesian product. Integers/storage are
exact; floats use the named source policy. Compiler inputs exclude evaluator
arrays/goldens. A numerical comparison alone does not establish useful structure.

Negative cases must cover bad bounds/shape/scalar bindings, invalid alias or
initialization claims, allocation overflow, an unknown numeric/source profile,
physical/external behavior and changed source/ABI/policy/artifact bytes.
Check located diagnostics and remediation, not just a nonzero exit code.
Baseline accepted forms may not be reclassified to avoid a failing test.

Serial smoke selects existing supported integer-memory, contraction and
owned-result/model cases; it is a separate result. The existing source replay's
two prerequisite skips and incoming unrelated Merlin gate debt remain explicit.
A new required-case skip cannot hide behind those historical exceptions.

## Release check and handoff

C0/C4 will add a small policy validator/report around existing
`product-coverage`, `package-coverage` and public staging tests. These commands
already exist; a release-policy validator does not exist yet. Do not call the
old package observation count a closure result. The validator must check:

1. The pinned definition set exactly equals the 50 policy entries; admitted
   supporting forms, attribute predicates and default exclusions are explicit.
2. Every frozen case has its expected route and final evidence. No unresolved,
   failed or not-run required case; no implicit skipped family or sample-based
   shape specialization.
3. Every claimed portable route has original verified source, complete standard
   payload, ABI/shape/effect checks, provenance, compiler-only staging, whole-body
   lowering and all required host results. Provider trust and numeric policies
   remain visible. Expected refusals produce no successful portable artifact.
4. Default portable artifacts reopen from only their declared publication
   members and required installed dependencies. Final producer/consumer commits,
   tool identities, source rosters, policy/manifest hashes and package differences
   are recorded. This is reproducibility evidence, not universal equivalence.
5. Required baseline regressions and bounded serial smoke pass, or a predeclared
   unrelated/prerequisite exception is independently reported. Unsupported and
   optional target counts are visible beside portable counts.

The final report states **v1 scope closed** only if all five gates pass. It also
lists unsupported/backend-required profiles, requested user contracts, fallback
costs and later target work. Each limitation explains why the source instruction
is used, why portable lowering cannot implement it, the workloads where it
appears and the available remediation. The two-transformer demo must pass
complete results and changed-input replay through public Merlin compilation.
Then stop this roadmap. No implicit requirement to
support new source releases, arbitrary physical semantics or every accelerator
remains attached to v1. Generic computation requirements remain mandatory.

Until those gates pass, the release is open. This document changes the completion
contract and execution order; it does not itself complete the implementation.

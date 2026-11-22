# Finite closure plan: portable TTIR to the Merlin interface

Updated 2026-11-22. This is the governing v1 release plan. The requested scope
requires every generic/useful TTIR computation to have an executable lowering;
exclusions apply to assembly, hardware observations and missing external
semantics, not to difficult generic computation. Use a broader logical
load/store, SSA or standard control/effect expansion where a specialized pattern
cannot be recovered. Preserve actual semantics and disclose lost structure.

This plan separates generic computation closure from arbitrary device semantics
and qualification of every possible target. Historical measurements in the roadmap,
coverage and handoff documents retain their original scope; they are evidence,
not an automatically expanding backlog.

**Execution decision:** use the fixed five-day remaining effort budget below,
then deliver either a qualified release or an explicitly incomplete handoff.
Do not keep implementing indefinitely. Checked user contracts, executable
fallbacks and documented non-support are available decisions. Excluding a
generic form reduces the delivered scope; it does not satisfy or disappear
from the original full-generic-coverage requirement.

## Where we are now

| Deliverable | Observed state | What remains |
| --- | --- | --- |
| Public Merlin interface | Kernel and bound-model staging, preparation and whole-body host execution exist. | Qualify the final selected packages from final revisions; retain explicit model2MLIR package differences. |
| Required two-transformer demo | Implemented: ordinary PyTorch block followed by one real Triton block, actual launch discovered automatically. Native/public host and artifact-only ragged replay have passed. | Include the existing demo in final acceptance and publish its reproduction command, source, MLIR and limitations. No new model architecture is required. |
| Source/profile inventory | All 50 pinned definitions have admission, semantic fields, caller contracts and complement rules in the reviewed ledger. Low-precision printing, unpacked i4 storage, FP8 arithmetic/accumulators and FP8 atomic addition are qualified. All five named items are focused-qualified. One new native-valid noinline-helper finding remains in the existing correction reserve. | Close the remaining named items and join final typed/profile evidence to actual source-to-Merlin packages/results. The reviewed ledger is not an execution or universal coverage verdict. |
| Frozen integration selection | All 107 preceding cases passed together with no skips/failures at `2290fb5`; retained under `out/v1-closure/acceptance-107/`. The source review then found narrow histogram/gather admission gaps. Their fix passed eighteen new public-interface checks and ten affected regressions. | The updated compiler and 128-case selection still need final whole-selection qualification after the profile audit. Retain the 107-case result as the preceding baseline, not the final candidate verdict. |
| Known numeric regression | Signaling-NaN min/max correction passed all 12 scalar format/memory-profile checks and the Merlin replay: 2,600 passed, two pre-existing prerequisite skips. | Qualify the final public Python transport; keep separately selected stock-CLI transport outside this claim. |
| Release checker | Implemented CLI with six reader/refusal checks. It rejects incomplete execution, absent semantic qualification and missing baseline custody. | Populate actual final evidence and complete semantic qualification. The checker is not itself a completed release audit. |

The demo is two small complete single-head inference blocks, without dropout
or a causal mask. It is an integration/correctness demonstration, not a GPU
performance, training or arbitrary-model claim. See [the demo](transformer_demo.md).

### The finite evidence gap

The earlier partial-run package audit observed 29 generic TTIR names in
Merlin-qualified packages. The following 20 missing witnesses are now recovered
from existing implementations: the maintained corpus and twelve targeted tests
passed serially. The combined verified selection has 67 unique packages and
59 standalone qualifications, observing all 49 generic definitions in
Merlin-executed packages. Retained evidence:
`out/v1-closure/family-witnesses/package-observations.json`,
`.scratch/v1-family-corpus.log` and `.scratch/v1-family-witnesses.log`.

This closes the **operation-witness gap**, not the full semantic/profile audit.
The table retains the original gap and the material reused; it is not a new
twenty-operation implementation backlog.

| Family | Missing original-TTIR witnesses | First existing material to reuse |
| --- | --- | --- |
| Shape/value transforms | `tt.cat`, `tt.join`, `tt.split`, `tt.unsplat` | `t2m/examples/product_corpus.py`, `tests/test_pointer_unsplat.py` |
| Arithmetic/formats | `tt.fp_to_fp`, `tt.dot_scaled`, `tt.mulhiui` | `tests/test_float_casts.py`, `tests/test_scaled_dot.py`, `tests/test_integer_products.py` |
| Logical pointers | `tt.advance`, `tt.make_tensor_ptr`, `tt.ptr_to_int`, `tt.int_to_ptr` | `tests/test_block_pointers.py`, `tests/test_pointer_differences.py` |
| Logical descriptors | `tt.make_tensor_descriptor`, `tt.descriptor_gather`, `tt.descriptor_scatter`, `tt.descriptor_reduce`, `tt.descriptor_store` | `tests/test_indexed_descriptors.py`, existing descriptor source fixtures |
| Indexed tensor computation | `tt.gather`, `tt.histogram` | Maintained product corpus, `tests/test_histogram.py` |
| Logical atomic effects | `tt.atomic_cas`, `tt.atomic_rmw` | `tests/test_atomic_memory.py`, `tests/test_atomic_word_cas.py` |

For each row, record one of: existing implementation qualified; missing caller
contract; broader fallback needed; missing implementation; or explicit
non-support. Bind successful evidence to original TTIR, executable payload and
Merlin results. A normalized operation appearing in a report is insufficient.
This list is the immediate audit backlog, not a claim that unseen compositions
are correct. A new counterexample inside an admitted profile remains a bug.

### Resolved demonstrated generic blocker

The source review found native-verified i8/i16 `tt.atomic_rmw add` rejected
by producer and consumer width checks. The existing logical atomic path now
retains all nine integer RMW kinds, successful/failed CAS, every old/final word,
guards and source scope/order for byte-sized integer words. Three complete
public-interface programs qualify raw i8/i16 and mixed f16/byte-backed i16
views; 41 affected regressions pass. See [the exact scope](atomic_memory.md#byte-sized-integer-source-words).
The direct floating-exchange gap is now implemented and qualified for f16,
BF16 and all four pinned FP8 formats. Six complete public-interface programs
preserve exact words through tiled/scalar effects, runtime control, changed
and empty inputs and bounds failures. Three native format cases additionally
check constant signaling-NaN exchange and reject altered helper address math.
See [the qualification scope](atomic_memory.md#low-precision-floating-exchange).
The original refusals and their resolution remain in the
[tracked gap ledger](../tests/data/v1_generic_gaps.json).

The i1 form now passes a complete public Merlin program with all nine RMW
kinds, both CAS outcomes, every old/final value and changed/empty/bounds replays.
The existing byte-CAS mechanism preserves logical width at every event. Native
i4 updates also pass. The public unpacked i4 storage extension now qualifies
both result ABIs, with canonical uint8 cells, all sixteen words and nineteen
replays per library. This does not infer packed-nibble storage or scalar ABI.

The admission review now records five bounded work items: public i4 storage,
FP8 arithmetic/accumulators, FP8 atomic addition, low-precision diagnostics and
entry value bindings. Low-precision diagnostics now pass four complete public
programs with exact values, words and streams. Unpacked i4 storage now passes
both public result ABIs. FP8 arithmetic/accumulators additionally pass thirteen
public programs with 69 successful library replays. FP8 atomic addition now
qualifies eight public programs with 36 invocations. Entry value ports now
qualify nineteen public programs, 87 public calls and 35 contract checks. The
five original items are focused-qualified; a native-valid noinline-helper
finding is tracked in the existing correction reserve. The
[review and attempt ceilings](semantic_profiles.md) retain the original eight
native-verified witnesses, current contracts and concrete reuse paths. Exact
sources, bindings, verifier identities and failures remain in the
ledger. These are generic numeric/storage/interface gaps, not GPU exclusions.
Reuse checked carriers and executable expansions; a final cast cannot replace
per-update source-width arithmetic. If unresolved at the original effort cap,
publish exact non-support and report incomplete full closure.

The final whole-selection run and semantic qualification are still open. The
admission review is complete; it is not a universal atomic or composition claim.

### Current correction-reserve finding

The five original items are implemented and focused-qualified. The source review
then verified a private helper with `noinline=true` that fails the native
TTIR-to-linalg continuation after its call remains uninlined. This is defined
portable computation with a source performance hint, not an allowed GPU/assembly
exclusion. Its exact source, native verifier evidence and failure remain in the
same gap ledger as `source-noinline-hints`.

Use the existing five-hour shared correction reserve: preserve original Boolean
call-grouping intent and source identity, expose the real body through the
existing upstream inliner, and qualify values/effects plus explicit provenance.
Disclose call-boundary, code-size and register-pressure changes. No new
interprocedural compiler subsystem, backend or corpus expansion is authorized
by this finding. The original forty-hour cap and final 128-case roster remain
unchanged. If unresolved, retain exact non-support and report incomplete closure.

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
| The packet recurrence ownership experiment | The failing optional assembly experiment is preserved outside the active frontend; the qualified scalar recipe records its sharing loss explicitly. | No failing path is selected by default and no lifetime check is weakened. Packet preservation is deferred optional optimization, separate from generic map computation. |
| Barriers, atomics, volatile and async | Keep qualified invocation-local phases/effects. Abstract physical scheduling only when actual dependencies and observations permit it. | External participants, general cross-program publication, race-sensitive behavior and device completion need an explicit execution contract/provider; otherwise backend-required. Serialization is not automatically equivalent. |
| Inline assembly, extern calls and physical address observations | Known source provider or an explicit typed standard-MLIR implementation with its dependencies and trust boundary. | Unknown instructions, hardware clocks/IDs, escaping physical addresses, impure externs and descriptor bytes are backend-required/unsupported. Provenance is not their implementation. |
| Remaining CFG/function/ABI forms | Existing official normalization and public bindings, or broader standard func/CF/SSA. | Valid generic entry/control forms require a working ABI/lowering; missing support remains a blocker. Preserve located diagnostics while implementing it. |
| Target scheduling/materialization | Existing Merlin host path and bounded serial smoke cases. | Missing accelerator providers become an explicit target capability limit. Do not expand the frontend project to finish every backend. |

A trusted caller declaration and an established source fact remain distinct.
Contracts express the intended input domain; they never prove arbitrary program
properties. A false declaration must not allow unchecked memory access.

## Extra information users can supply

Extend the current ABI/binding mechanisms rather than introduce a parallel API.
At the first checkpoint, name the actual supported fields, enforcement sites
and remaining adapters in the existing contract documentation. The list below
describes information to express, not a claim that every field combination is
already a supported API:

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

Use this decision order for every difficult case, once:

1. Reuse an existing structured lowering or supported upstream normalization.
2. If information is missing, request the smallest storage/shape/numeric/source
   contract through the existing interface. Validate it or guard the assumed
   domain; do not ask users to assert arbitrary safety or equivalence.
3. If only pattern recovery fails, use broader standard computation: indexed
   tensor operations, checked logical loads/stores, explicit SSA state or
   control flow. Keep guards, fills, initial values and effects executable.
4. If implementation semantics are external, accept an explicit typed provider
   only under the existing provider contract. Record what is trusted and what
   was measured. Do not start implementing an ISA or vendor library.
5. Otherwise issue a located refusal and record the exact unsupported form,
   affected workloads and remedy. Hardware-specific or unspecified external
   semantics are release exclusions. An unimplemented generic form is a
   visible scope reduction/open full-coverage item, never a supported op.

Performance intent is part of the decision. Retain the original tile/reuse/
recurrence provenance on broader paths and report what structure survived.
Do not quietly replace online attention with dense scores or change observable
ordering. If only a materially different algorithm is available, offer it as an
explicit alternative profile rather than claiming the original was preserved.

## Remaining execution sequence

**Cap the remaining work at five focused engineering days (40 hours) for one
implementer.**
This replaces the earlier ten-day proposal; it is an effort cap, not a promised
calendar delivery date or authorization for heavy compute. Preserve work already
done: do not restart C0-C4 or rebuild the demo. Record actual effort and decisions
in the release notes; a newly discovered problem does not reset the budget.

The [authored policy ledger](v1_policy.json) now partitions all 50 definitions:
49 mandatory generic operations and one optional source-assembly operation.
The [source admission review](semantic_profiles.json) covers all definitions,
supporting forms, actual semantic fields and complement rules. Final semantic
qualification remains open; neither ledger is a production allowlist.
The [acceptance manifest](../tests/data/v1_acceptance.json) now names 128
cases across K1-K12, M1-M7, negative boundaries and serial smoke, including
hashes of 35 fixture source files. It retains all original 94 cases, the thirteen
missing-family checks, eighteen narrow-index checks and three complete narrow-atomic
programs.
The reviewed changes and qualification scope are in the manifest. The preceding
107-case selection passed, including the transformer demo and serial smoke;
the new candidate has not run all 128 together. Existing broader regressions
remain required. The hard cap is 128 cases; the roster is full. Do not
add more variants without a reviewed replacement. The admission review is done;
close the named gaps and finish the final semantic evidence joins before the rerun.

Use `scripts/check_v1_closure.py`; do not create a second release tool.
Generated run records belong under `out/v1-closure/<run>/`.
Do not infer a release pass from a manifest or operation-name counts.
The checker independently rejects any unresolved entry in the tracked generic
gap ledger, even if all acceptance cases pass or an audit claims closure.

| Checkpoint | Remaining effort | Concrete output and exit |
| --- | --- | --- |
| 1. Finish the inventory and freeze | Day 1 | Admission review, layout correction and all twenty missing operation witnesses are done; 128 cases are frozen. The finite backlog began with five named items/eight witnesses; printing, unpacked i4 storage, FP8 arithmetic/accumulators and FP8 atomic addition are qualified. Entry value ports now qualify as well. All five original items are focused-qualified; one new native-valid noinline-helper finding uses the existing correction reserve. Final evidence joins remain open; no speculative new analyses. |
| 2. Close only named gaps | Days 2-3 | Use the per-item ceilings in the reviewed backlog, existing analysis, checked caller contracts or broader standard MLIR. Add a valid and violating case when introducing a contract. Retain immutable parameters, live state, all results and useful tiling/reuse. Resolve hardware/unknown-source cases by explicit disposition. |
| 3. Run final qualification | Day 4 | Run every final acceptance case, the relevant baseline regression selection and the fixed serial smoke from recorded revisions. Reopen the two-transformer publication without evaluator files; compare both outputs on original, changed and zero inputs using one library. Record every failure, skip, fallback and refusal separately. |
| 4. Correct once and deliver | Day 5 | Fix only release findings within the remaining budget. Rerun affected checks; rerun the full frozen selection if compiler/fixture changes invalidate its evidence. Publish the release check, gap/limitation inventory, packages and reproduction instructions. Deliver a closed release or an explicitly incomplete handoff, then stop expanding this roadmap. |

These checkpoints finish the existing C0-C4 obligations; they do not introduce
new implementation tracks. A named gap gets one implementation attempt and one
corrective attempt, each within the shared budget. If it cannot be closed,
choose a qualified contract/fallback or explicitly record non-support. Do not
spend another cycle polishing the same unqualified path.

For a generic gap, non-support must name the exact form and its practical
impact, and the final report must say full generic closure was **not achieved**.
This permits a usable bounded deliverable without changing the truth of the
original goal. A previously qualified regression cannot be relabeled excluded
to obtain a release pass. Keep its qualified default or report the blocker.

New kernels do not automatically become new acceptance projects. A bug in an
admitted profile is a release finding and uses the remaining budget; a new
source version or excluded profile belongs in a separate proposal. Do not
start a new backend, alias solver, general wrapper tracer, ISA provider or
performance campaign to finish this release. If budget runs out, report what
failed and the specific decision needed; do not silently extend the deadline.

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

The policy validator joins existing `product-coverage`, `package-coverage`
and public staging evidence. Its successful reader tests are not compiler
qualification. Do not call a package observation count a closure result. The
validator and final source/profile review must establish:

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

Run the checker in the frontend environment with the existing consumer roots
configured, after producing the final evidence directory:

```bash
python scripts/check_v1_closure.py \
  --run out/v1-closure/final \
  --merlin-replay /path/to/merlin-replay/replay.json \
  --output out/v1-closure/final/closure.json
```

It reads the run's exact `manifest.json`, `command.json`, `process.json`,
`pytest.xml` and `semantic-audit.json`; the replay path is a real Merlin replay
record, not a handwritten success label. Exit zero requires all four machine
gates; a missing/failed/skipped required case, stale fixture or absent audit
keeps the result open. The source/profile review binds policy, limitations,
implementation bytes and all 49 generic entries to actual verified packages,
qualified case IDs and Merlin execution. It must describe precise profiles and
remaining forms truthfully. The checker verifies this evidence's custody and
joins; it does not prove arbitrary source equivalence from an author's label
or turn an operation-name observation into semantic coverage.

The final report states **v1 scope closed** only if all five gates pass. It also
lists unsupported/backend-required profiles, requested user contracts, fallback
costs and later target work. Each limitation explains why the source instruction
is used, why portable lowering cannot implement it, the workloads where it
appears and the available remediation. The two-transformer demo must pass
complete results and changed-input replay through public Merlin compilation.
Then stop this roadmap. No implicit requirement to support new source releases,
arbitrary physical semantics or every accelerator remains attached to v1.
Generic computation requirements remain mandatory for the full-closure claim.

If the effort cap arrives first, deliver the same artifacts with an
**incomplete** status and a finite remaining-gap table. Do not claim v1 scope
closed while a required gate fails. Each generic gap needs the source form,
reproducer, failing stage, last attempt, available workaround, lost capability
and next bounded action. Any proposal to reduce the original requested scope
must name those differences explicitly.

The handoff consists of:

- Exact commits/tool pins and one reproducible command for the demo and checks.
- Original TTIR, standard MLIR, source/intent provenance, ABI and complete
  compiler-only packages, with model2MLIR differences stated.
- A per-profile support table: direct, checked contract, fallback, excluded or
  unfinished; separate execution evidence and counts, with no refusal-as-pass.
- Complete logs/result summaries and original/changed/empty boundary evidence.
- The limitation inventory: why used, why unsupported, likely workloads,
  user-supplied information or workaround, and any performance-structure loss.

Until those gates pass, the release is open. This document changes the completion
contract and execution order; it does not itself complete the implementation.

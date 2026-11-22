# Source admission review and bounded generic closure work

The [source/profile ledger](semantic_profiles.json) reviews all fifty pinned
definitions: forty-nine required generic operations and optional source assembly.
It records family admission, actual caller contracts, complement rules,
supporting-dialect obligations and current implementation hashes. Operation
attribute names and argument constraints come from the existing pinned TableGen
dump; custom verifier behavior was also checked against pinned `Ops.cpp` and
native source verification. This is a reviewed inventory, not a universal
translation proof or a final execution verdict.

The earlier package observation report contains all forty-nine generic names in
selected Merlin-executed packages. Its packages remain prior evidence. The
current producer/consumer revisions still need the final 128-case qualification
and actual package/result joins in `semantic-audit.json`. Neither an operation
name nor native verification establishes that every type or composition works.

## Remaining implementation work

The admission checkpoint identified **five named work items and eight exact
native-verified witnesses** in the [generic-gap ledger](../tests/data/v1_generic_gaps.json).
Low-precision diagnostics qualify through four complete public programs and
unpacked i4 storage qualifies through both public result ABIs. FP8 arithmetic
and scaled-dot accumulators now qualify through thirteen public programs;
**two work items and three witnesses remain open**. These are generic
computation/interface failures. They cannot be removed by
classifying them as GPU instructions or by counting their refusals as support.

| Work item | Demonstrated failure | Existing material and bounded next action | Attempt ceiling |
| --- | --- | --- | --- |
| Public i4 storage — qualified | The original direct i4 atomic-add ABI refusal remains in the resolved ledger. | Both public result ABIs now keep actual i4 updates and all old/final values using one canonical uint8 cell per value, checked as 0..15. Nineteen replays per library include all words, tails and empty inputs; invalid padding fails before invocation. Packed pairs and by-value scalars remain separate. | Closed within the 3-hour attempt |
| FP8 arithmetic and accumulator profiles — qualified | Both original native-valid failures remain in the resolved ledger. | Thirteen public programs qualify all four formats, exhaustive binary/sign-source pairs, exact words, per-operation/per-dot rounding, original C and wider inputs. Official f64 emulation and exact casts retain source control/reuse. Caller and execution-owned paths pass, including strict owned-control/dimension guards. See [the exact scope](fp8_arithmetic.md). | Closed within the 4-hour attempt |
| FP8 atomic addition | Native-valid `tt.atomic_rmw fadd` on FP8 is refused without an implemented update policy. | Reuse half/BF16 per-event numeric profiles and checked integer/byte-CAS helpers. Implement actual FP8 rounding, overflow and NaN behavior; retain every old word. | 4 hours |
| Low-precision diagnostic values — qualified | The original internal i4/FP8 print refusals remain in the resolved ledger. | Existing logical formatting and exact widening now preserve all words, streams, signedness, source control and both outputs through four public libraries. The native pure tensor carrier is also qualified. General FP8 arithmetic remains separate. | Closed within the 1-hour attempt |
| Entry value bindings | A selected value-returning i32 entry is refused. A void entry with a by-value f16 input cannot construct its runtime scalar binding. | Extend the existing checked source wrapper/allocation interface with explicit scalar-storage inputs and returned-value output ports. Reuse actual helper bodies, native inlining and the existing tensor ABI; avoid assuming a low-precision native by-value ABI. | 3 hours |

These are ceilings within the original forty-hour budget, not new budgets or
promised completion times. The five attempts total at most fifteen hours.
Reserve one shared correction cycle of at most five hours, at most eight hours
for final qualification and two hours for the deliverable. Work already done,
review and remaining regressions all charge to the same original cap. Finish
earlier when the gates pass; do not spend the cap merely because it exists.

After these attempts, do not start another compiler subsystem. Register a new
valid generic counterexample with exact evidence, charge any necessary repair
to the existing correction/reserve time, and retain it as open if unresolved.
An explicit unsupported v1 form is a scope reduction and must be reported as
incomplete full generic closure. It does not become supported by changing its
classification or by adding a caller assertion without an implementation.

## What callers can actually supply today

The existing interfaces already accept the following information:

| Information | Existing interface | Enforcement |
| --- | --- | --- |
| Logical tensors, mutation and dimensions | `TensorBuffer`, `TensorABI`, `RuntimeScalar`, `GridExtent` | Exact source argument/type joins, dimension/range guards, complete result/state ports and invocation budgets. |
| Shared allocation roots and origins | `PointerArgument`, `ScalarArgument`, source allocation arguments v1 | Source-order typed wrapper calls, actual one-root state, bounds and independent consumer replay. A declaration cannot manufacture physical alias identity. |
| External logical descriptor fields | `DescriptorArgument`, source allocation arguments v2 | Actual root, shape, stride, offset, padding and rounding fields; declared scalar domains and native descriptor normalization. Physical descriptor bytes are not accepted as logical fields. |
| Source-coupled register groups | `source_layout` | Actual source specialization and checked coordinate maps, or a separate independence proof. This does not choose the consumer's hardware mapping. |
| Numeric behavior | Existing FP8 overflow/per-cast, dot/scaled-dot and atomic profile options | Implemented arithmetic, original source attributes and independently checked payloads. A new policy label alone grants no implementation. |
| Optional assembly semantics | Existing selected source-language provider contract | Actual emitted standard computation, types, effects, dependencies and stated provider trust. Missing assembly semantics remain an explicit limitation. |

Public i4 carriers now use the existing `TensorBuffer` dtype field with the
explicit [unpacked cell contract](atomic_memory.md#unpacked-four-bit-integer-cells).
General scalar-storage/result entry bindings remain a **proposed extension**,
not existing fields users can pass today. The source/profile ledger keeps that
distinction explicit. By-value f16/BF16/FP8/i4 input gaps do not imply that those
values are unsupported inside private helpers or ordinary tensor computation.

## Qualification and stopping rule

Keep the frozen 128-case roster. New focused checks for the five named fixes
remain separately reported; do not hide them behind the earlier 107-case pass
or silently enlarge the acceptance manifest. Requalify affected regressions,
then run the complete final selection from recorded source/tool revisions.
The two-transformer publication must reopen using compiler-only artifacts and
compare both blocks' outputs on original, changed and zero inputs using one
library. Serial smoke remains a separate measured result.

Use `scripts/check_v1_closure.py`. Finalization still requires exact source
custody, actual executable packages/results, a completed semantic audit, no
unresolved required generic gaps and the final Merlin baseline replay. Expected
refusals, missing tools and unmeasured profiles are distinct from portable passes.

At the original effort cap, deliver the packages, demo, commands, exact pins,
logs and limitation inventory. State either qualified closure or explicitly
incomplete closure with the remaining exact forms. Explain why each limitation
arises, where it occurs, which extra information or workaround is available and
what numeric/performance capability that workaround changes. No new source
release, backend campaign or arbitrary device observation silently extends v1.

# Checked memory continuation across native pointer/control limits

General source branches and while-carried pointers can reach the standard
tensor interface even when the native structured pass cannot reconstruct their
pointer/control tuples. The normal structured path remains first. A recorded
native failure now permits the existing checked memory-event continuation only
after the exact retained Python IR is serialized and independently verified by
the native source reader. Invalid source and failed verification still refuse.

The original for/while/if regions, scalar arithmetic, tile values and memory-event
positions remain executable. Finite logical root selectors and source offsets
reuse `pointer_control` / `pointer_select`; each actual memory event uses the
existing shared converter. The parent then uses its arithmetic/SCF conversion.
No physical address, target instruction, runtime scheduling choice or opaque
kernel call is introduced. The normal bounds, initialization, ownership, effect
and complete-output checks still decide admission afterward.

The failed stage, its exact input/output/diagnostic artifacts and the separate
source-verification record remain in `native-structured-memory-residual`.
Successful continuation is distinguished from successful structured conversion.
The generated compatibility record names the difference. Public tensor/scalar
ABI and receipt schemas are unchanged.

## Complete clipped writes without invented prior state

An isolated masked scatter can describe an entire clipped output tile. Requiring
an incoming value in that case wrongly rejects a fresh output. Conversely, a
bound implied by a mask does not prove that the mask writes every in-bounds lane.

`scatter_facts.covers_clipped_tile` first uses the independently established
dense scatter geometry. It checks the actual Boolean SSA, proving every needed
comparison from the destination's in-bounds coordinates. Conjunction needs both
operands; disjunction needs a sufficient operand. Unknown/data predicates and
unsupported relations decline. No lanes or programs are enumerated by the proof.

Only a proved complete clipped mask can omit the prior-value merge. The emitted
tensor insertion writes exactly that slice; outside values remain untouched.
Partial predicates still need initialized state, and fresh incomplete outputs
still refuse. Tail clipping supplies no parallel ownership or alias permission.

## Source-backed reuse

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` provides
`PtrAnalysis::getIfPtrState`, `reconcileIfPtrState`, `rewriteIfOp`, structured
root/offset/stride state and residual pointer conversion. Its
`conditional_ptr_as_src.mlir` and `if_preincrement.mlir` fixtures retain selected
bases and nested offsets. `MaskState::parseAnd` / `parseCmp` supply the existing
rectangular-mask boundary. The local extension proves the additional converse
needed for a complete tensor write; it does not turn necessary bounds into
initialization evidence.

The existing historical Microsoft source-word/event conversion remains reused.
Reference source pins and the actual optimizer binary identity remain separate.
No native stack or Triton CPU backend is built.

## Real kernel and model obligations

`t2m/examples/root_state_kernels.py` contains three real source forms:

- Branch-selected reads from differently sized roots inside a numeric loop.
- Pointer state that changes roots across the actual loop yield and final read.
- While-carried pointers with selected positive/zero/negative memory progress,
  separately retained numeric position, masks and final observations.

Their source controls remain independent of memory progress. The kernel ABI uses
symbolic widths, dynamic grids, empty/ragged domains and i64 data. Tensor, buffer
and both guarded address policies have separate qualification. Tensor output
retains logical `scf.forall`, the original for/while control and tensor/Linalg
computation, with no memref/GPU operations. Independent NumPy references cover
all results; reusable host libraries and serial objects retain changed sizes
and scalar values. Partial fresh writes remain negative witnesses.

`scripts/capture_root_state_model.py` captures two dependent calls with learned
matrix/table state, surrounding broadcast computation, original snapshots, live
i64 controls and an empty result. All thirteen model results remain in the
original public order. Original parents and source correspondence remain bound;
runtime references and golden outputs stay outside compiler staging. The model
selection uses a source-bound width specialization independently of the dynamic
standalone-kernel qualification.

The initial model reference exposed a source-fixture alias error: the CPU custom
operator returned two aliased results despite its two independent source output
allocations. Explicit result copies correct that reference contract. The next
negative-test run exposed a manifest lookup that incorrectly required a `name`
on weight entries; the corrected lookup preserves the actual domain-abort test.
A further i64 source fixture had mismatched integer branch yields; an explicit
source-width constant corrects the invalid Triton source without changing the
valid i32 computation. Those failed logs remain retained and do not count as
passing evidence.

## Completion scope

The integrated kernel/source/refusal selection passes 29 checks in
`.scratch/checked-memory-current2.log`; related mask/matrix/branch/selected-pointer
regressions pass 78 in `.scratch/checked-memory-regression1.log`. After correcting
the invalid i64 source fixture, the while kernel selection passes six checks in
`.scratch/checked-memory-while-width.log`. Two complete models pass eight checks
in `.scratch/root-state-model-final.log`; the corrected while model passes four
in `.scratch/root-state-while-model-final.log`. These selections overlap.

`out/checked-memory-continuation/summary.json` reopens three kernel and three
complete model libraries: 45 initial and 159 replay result arrays, 90 exact-word
serial results across twelve actual images, and three original-domain aborts.
All parent archives and both model callsites remain checked. Kernel images for
empty and ragged domains retain identical computation-object hashes. The
reopening log is `.scratch/checked-memory-report1.log`. Source hashes and artifact
custody do not establish a complete build dependency closure.

Reproduce with the existing source/native/model/Merlin/cross/Spike selections:

```bash
python -m pytest -q tests/test_checked_memory_continuation.py \
  tests/test_root_state_model.py --basetemp=.scratch/pytest-checked-memory
```

These are general pointer/control and complete-write mechanisms, qualified on
specific source forms. They do not establish arbitrary nonlinear/loaded address
relations, general view/alias projections, every write image, shared/asynchronous
participation, numeric/source-version closure or accelerator performance.
Provenance and retained structure supply no unmeasured target verdict. The full
goal and its other workstreams remain active in [the completion plan](completion_plan.md).

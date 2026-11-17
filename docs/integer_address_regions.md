# Logical address state across structured control

Integer addresses carried by original `scf.for`, `scf.while` and `scf.if`
interfaces can remain portable when their complete use closure never observes
a physical address. This extends the [straight-line address contract](pointer_addresses.md)
without choosing a GPU address space, physical allocation or launch mapping.

For example, the maintained real Triton source carries an address and writes
state at every original loop iteration:

```python
address = x.to(tl.uint64) + byte_offset
for _ in range(steps):
    value = tl.load(address.to(tl.pointer_type(x.dtype.element_ty)), mask, other=-17)
    total = total ^ value
    tl.store(state + lane, total, mask)
    address += stride.to(tl.int64) * word_bytes
distance = (address - initial_address).to(tl.int64)
```

The internal representation is an actual logical pointer base paired with an
executable i64 byte displacement, representing `base + displacement` modulo
2^64. An original integer state slot carries its displacement and an appended
pointer field carries the matching base. Every original tuple slot, branch,
loop edge and effect stays in its original order. Source pointer reconstruction
uses the existing checked element-width division and ordinary memory lowering.
Same-base differences become executable byte subtraction only after an actual
SSA root proof. No absolute pointer integer is materialized.

## Analysis and publication

`t2m/integer_address_regions.py` uses finite monotone dataflow over actual SSA
edges. It classifies ordinary integers separately from closed addresses,
propagates original pointer seeds, and computes low-zero-bit facts for byte
divisibility. Ordinary numeric loop state participates in that fixed point;
relative-address facts never become facts about physical address bits.

While entry/before inputs and condition/after/results have independent
interfaces. Pairing one side does not add a field to the other unless its own
source values need it. A branch may select either root while preserving the
matching displacement, including inside a loop. A preceding branch's actual
result can also be an invariant anchor for a following loop. Folding requires
that the candidate SSA base dominate the carrier and every actual use; a
before-region-local pointer cannot escape into a post-loop difference.

Classification and pairing happen on a detached module. Complete closure,
source field checks, divisibility, scope and deferred difference proofs must
succeed before its body replaces the caller's module. Any refusal leaves the
original operation graph and SSA uses unchanged. Budgets bound integer values
(65,536), address values (8,192), propagated seed memberships (262,144), seeds
per value (4,096), and each difference's generated recipe (32,768 operations).
The existing bounded straight-line recipe and divisibility limits still apply.

## Reuse boundary

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`,
`lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp`, rewrites
pointer-valued for iterargs into base/offset computation. Its
`scf_for_ptr_in_iterargs.mlir` and `ridiculously_nested_loops.mlir` tests retain
actual original entry/backedges and mixed numeric state. The local adapter
reuses that boundary and the existing `pointer_control.py`/`pointer_select.py`
edge conventions. Integer-address classification, paired changing roots,
independent while ports and transactional publication supply the missing
closed-address contract. No upstream source or fixture was copied.

Recovered pointers still pass through the installed shared converters and the
existing tensor/buffer memory gates. Bounds, initialization, write ownership,
alias/view checks, native lifetime checks and complete payload accounting stay
enabled. Source annotations supply no alignment, root equality or effect facts.

## Package and current qualification

`integer-address-region-state` transformations record source IDs, control kind,
paired representation and unobserved physical addresses. Original casts and
regions remain inventoried; `prov.closed_pointer_addresses` retains emission
correspondence. Generated compatibility metadata explicitly names the internal
pointer fields and the preserved source numeric/control boundary. Public
tensor/scalar ABI, artifacts and receipt schemas are unchanged.

`tests/test_integer_address_regions.py` captures real i8/i16/i32/i64 Triton
sources for a for loop, while loop, branch-selected loop and alternating-root
while loop. Sixteen libraries and actual Spike images retain both returned
results as exact words, with eighty additional unchanged-library host runtime
invocations. These cover empty domains, zero trips, ragged tiles, source word
overflow, both stride signs and branch choices. Each source writes readwrite
state at every iteration; the reference uses independent logical indices and
integer-word arithmetic. Native, Merlin-host and target outcomes stay separate.

One additional native-verified saved-TTIR while fixture has an ordinary numeric
entry/before input and an address-valued condition/after/result. Its paired
output interface has a different arity from its input. The actual loop load
supplies the next signed condition input; source control is retained without
assuming general termination. Native/Merlin host/Spike compare its complete
result exactly, and three further host invocations cover zero trips and
different loaded words. Four existing i32 control-form libraries additionally
abort on an invalid active address, publishing no success receipt or output.

`out/integer-address-regions/summary.json` reopens the selected **17 libraries
and actual target images, 33 exact-word results, 83 further host invocations and
four bounds refusals**. `out/integer-address-region-inventory/coverage.json`
reopens **327 packages, 268 standalone qualifications and fourteen product
cases**. All fifty pinned operation names remain observed under selected native/
Merlin-host profiles; this is not universal operation/type/attribute coverage.
The 57-case existing address/model/contract regression selection passes, as do
the updated 23 focused contracts. Counts overlap and are not unique coverage.

`tests/test_integer_address_region_contract.py` checks original tuple/while
interfaces, retained memory effects and complete rollback on located refusals.
Physical observations, address-derived bounds, truncation, mixed numeric/address
phis, unproved byte divisibility and poisoning overflow flags still refuse.
Provenance and serialization do not grant admission.

This qualifies the listed closed structured forms. Correlated nontrivial pointer
phis, more general root equivalence, address state in other region interfaces,
broader arithmetic/alias/effect compositions and applicable accelerators remain
completion work. General physical observations and synchronization need concrete
semantics; they cannot be erased because address conversion is difficult.

## Complete PyTorch model qualification

`scripts/capture_integer_address_model.py` captures four genuine model2MLIR
models: i32 for/branch and i64 while/alternating-root forms. Each registered
`torch.library.triton_op` clones its incoming state before the original Triton
kernel updates it; the translated kernel's functional readwrite result expresses
that same value update. The operation's schema declares no argument mutation.
Its CPU implementation independently uses logical tensor slices and typed XOR,
without compiler pointer helpers, physical addresses or GPU execution.

```python
original = x + self.weight
first, first_distance = address_state(original, self.weight, self.initial,
                                      19, 128, 4, 80, -3, 1)
follow, follow_distance = address_state(original, self.weight, first,
                                        19, 128, 2, 31, -8, -1)
```

The actual models additionally call the same kernel on empty data and with
zero trips plus an unobserved invalid origin. All eleven original results stay
live, including the first state after its dependent update, both calls' byte
distances, empty/zero-trip values, input-derived snapshot, surrounding model
math and original saved initial state. Every original result matches exact words
through native, Merlin host and actual Spike: **four complete models and 44
target results**. Parent weights, manifests, inputs and complete reference
archives remain byte-identical. `tests/test_integer_address_model.py` reproduces
the selection; `out/integer-address-models/summary.json` reopens its receipts.
The latest `out/integer-address-model-inventory/coverage.json` reopens **331
packages, 268 standalone qualifications and fourteen product cases**. It retains
all fifty observed names under selected native/Merlin-host profiles, with the
same explicit limits on universal and target coverage.

Source runtime scalars bind from the existing typed call attributes. The public
model ABI remains static and unchanged. This supplies no general argument
mutation, dynamic public signature, root-equivalence or accelerator permission.

## Pointer seeds defined inside a region

A before region can compute `local = outer + increment` and forward
`local.to(tl.uint64)` through `scf.condition`. Treating `local` as an opaque
base loses the relation to `outer` outside that region. Moving `local` outside
would also violate SSA scope when its increment depends on before-region state.

`pointer_address_seeds.py` instead reuses the bounded byte recipe from
`pointer_differences.py`: actual addptr operands supply their original integer
increments, each signed-widened only after its source computation and scaled
by that addptr's input pointee width. Pointer bitcasts preserve accumulated
bytes. Same-root selects retain their actual condition and both offsets. Shape
operations remain exact SSA anchors; no lane permutation is inferred or moved.
An existing shaped pointer outside the control can therefore anchor a tiled
region-local addptr without manufacturing a new pointer shape.

The seed becomes an actual same-shaped outer pointer integer plus the executable
byte recipe on the detached graph. Address classification, original seed roots
and byte-divisibility fixed points are recomputed before paired fields are
prepared. A previously zero displacement relative to `local` says nothing
about the new displacement relative to `outer`; carrying that old fact forward
would miscompile unaligned byte views. The original narrow arithmetic, condition,
load/store effects and source locations stay executable. No physical address
or provenance-derived alignment is introduced.

Rebasing is optional for each actual source seed. If its new byte representation
cannot establish a required reconstruction divisor, the frontend retains that
seed's original exact typed pointer and independently validates the complete
paired lowering again. Other seeds can still recover their outer roots. The
choice follows actual SSA seed dependencies, never names, provenance or sample
values. This preserves an unaligned byte-view path without granting a false
alignment fact or disabling a separate required rebase. Physical observations,
unknown consumed source fields and incomplete source closure cannot trigger
this alternative. Every failed attempt is detached and discarded before the
original module can change; bounded retries and recipe budgets still apply.
Recovery permits sixteen monotone retention refinements, 32,768 generated
operations per seed and 65,536 across a pass. Exceeding a budget is a located
transactional refusal, not an unbounded retry or an inferred permission.

`integer-address-seed-rebase` records name the recovered source IDs and exact
byte boundary under the existing `prov.closed_pointer_addresses` correspondence.
Generated compatibility metadata explicitly names the representation. Public
tensor/scalar ABI, artifacts and receipt schemas remain unchanged. Source
budgets, actual-root proofs and complete transactional refusal still apply.

This adapter uses the previously reviewed Meta/shared base/offset boundary and
Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7`,
`third_party/ascend/lib/TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.cpp`
(`parseAddPtr`, `parseBitcast`) with its independent
`third_party/ascend/unittest/Analysis/PtrAnalysisBitcastTest.cpp` byte-address
oracle. No upstream code or fixture was copied. Shared's ordinary shaped
arithmetic conversion still handles the executable standard casts; xDSL's
scalar-only cast binding is not treated as a native MLIR restriction.

### Seed qualification

`tests/test_pointer_address_seeds.py` uses native-verified saved TTIR with
independent while input/result ports. Scalar cases retain a loaded signed word
as their next condition input and original i32 multiply/add wrap before byte
widening. Tile cases retain an actual outer shaped pointer, region-local offset
arithmetic and observable four-word updates. All four i8/i16/i32/i64 widths
reach native, Merlin host and actual Spike. Runtime replays include zero trips,
empty unused roots, different loaded words and signed i32 extremes.

Two additional little/big logical byte-view cases retain an original unaligned
typed pointer. A word load crosses an owning word boundary and its signed result
controls the actual recurrence. Independent byte references use integer codecs,
not compiler address helpers. A one-word allocation fails at the real compiled
whole-word bounds guard, with no success receipt or output archive. No assumed
alignment or physical byte-order equivalence is granted.

`out/pointer-address-seeds/summary.json` reopens **ten libraries/actual target
images, fourteen exact-word results, 48 additional unchanged-library host
invocations and four actual bounds aborts**. The latest combined
`out/pointer-address-seed-inventory/coverage.json` reopens **341 packages,
278 standalone qualifications and fourteen product cases**. All fifty names
remain observed under selected native/Merlin-host profiles; source-name breadth
does not establish universal coverage or accelerator execution.

The focused contracts additionally preserve original SSA uses on refusal,
reject false byte-divisibility facts and unknown consumed source fields, retain
independent seed choices after serialization, and enforce the refinement budget.
These are saved-TTIR examples, not additional Python AST captures, GPU execution
correspondence, new complete model variants or a performance measurement. General
root/alias/region/effect/source combinations, dynamic public model signatures
and applicable accelerators remain completion work.

The current 52-case contract/address-region/complete-model selection passes
with no skips (twelve other storage-width region cases were deliberately
deselected). It includes 33 focused contracts, all ten new libraries, five
existing region configurations and all four complete address-state model
regressions. `.scratch/pointer-address-seeds4.log` records the selection. Counts
overlap earlier evidence and are not additional unique coverage.

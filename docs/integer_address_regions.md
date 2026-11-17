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

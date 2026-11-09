# Bounds from the actual logical grid

Status: implemented for matching-rank dense affine tiles. The tensor and
buffer profiles now admit unmasked unit rows using the actual launch bounds,
without extra memory guards or redundant source masks. Bounds, ownership and
fresh-output coverage remain independent gates.

## Observed gap and two independent gates

A histogram can store one complete row per logical program into
`out: tensor<tiles x bins x i32>`, launched with `GridExtent("tiles", 1)`.
Its unmasked row coordinate is `program_id(0)`. The generated forall enforces
`0 <= program < tiles`. Previously `views.axis_permutation` compared independent
worst-case intervals for program and guarded `tiles`, losing that relation.
`launch_domains.affine_bounds` now retains it for matching launch dimensions.

Bounds are only the first gate. `DefiniteWrites.finish` also proves exact
runtime partitions, including an unmasked unit row paired with clipped column
tiles. Sparse readwrite updates preserve incoming holes; the same stores into
a fresh output still refuse.

Flat rank-one pointers into a dynamic matrix remain a separate descriptor-rank
limit. A matching-rank descriptor with one singleton zero-stride axis can now
recover a combined uniform origin: exact polynomial factors repartition it
along declared dense strides, with unique stride correspondence and exact
address reconstruction. No runtime division is introduced. All existing
access, mask and ownership checks still apply; recovery evidence records the
coordinates and source IDs.

## General proof rule

Derive facts from the actual wrapper's launch expression and entry-checked
dimension bindings. For an axis with `GridExtent(D, q)`, a real program satisfies
`0 <= p < ceil(D / q)`. Every nonempty domain therefore has `q*p <= D-1`.
For a positive affine program coefficient `k*q`, replace that term by
`k*(D-1)` when computing an upper bound. For a negative multiple use the same
inequality to compute a lower bound. Retain the existing independent interval
bound as well, taking the tighter established bound. Unsupported coefficients
keep the old conservative rule. An empty launch makes no memory observation.

Use the scalar/shape expression bound to the generated loop, not a caller's
provenance assertion, an example extent, or a source/kernel name. Constant grids
and mismatched dimension identities grant no dynamic correlation. Source-word
overflow checks remain separate; an in-bounds mathematical coordinate cannot
repair arithmetic that overflowed earlier.

For fresh results, extend the existing partition analysis with exact runtime
tiling. Require zero-based affine tile origins, one distinct program axis per
varying storage axis, exact tile strides, complete static axes, and matching
launch/dimension bindings. A divisor-one axis covers its runtime dimension
with unit tiles. Wider unclipped tiles also require a genuine divisibility
fact; a ceil-divided launch alone leaves a partial final tile. Readwrite partial
updates may preserve incoming holes, while a fresh output must cover them.

## Implementation and qualification

1. Add the bounded affine grid-correlation helper; wire it from the same ABI
   facts that construct forall launch bounds. Keep no-correlation behavior.
2. Extend matching-rank access validation and exact runtime partition coverage.
   Keep source stores, loop/grid structure and all numeric/effect checks.
3. Check soundness against an independently enumerated small-domain oracle,
   including empty domains, multiple axes, negative coefficients and divisors.
4. Execute the unchanged unmasked histogram row kernel in tensor/buffer/guarded
   profiles over independent shapes. Verify fresh results and input preservation
   through one actual Merlin library per profile.
5. Retain negative cases for shifted/out-of-bounds rows, mismatched or constant
   launch dimensions, missing fresh-output coverage and unclipped ragged tiles.
   Run the existing dynamic-shape/matrix/mask/initialization regressions.
6. Record the launch relation and partition evidence with the existing
   source/ABI/IR bindings. Do not introduce a target opcode, opaque call or
   semantic meaning for provenance tags.

The initial focused suite passed **87 cases**, with **2 strict expected failures**
for branch-local conditional stores in the source-word profile. The subsequent
[isolated native memory-event bridge](source_words.md#memory-inside-structured-regions)
now makes both conditional cases pass through native and actual Merlin execution;
the expected-failure markers are removed. The suite includes native execution and
actual Merlin libraries reused over empty, ragged and surplus launch domains,
unmasked histogram rows in all three profiles, and the negative cases above.
An independent finite-domain affine oracle covers 3,484 scenarios. Package ABI
is unchanged; logical-grid and dense-origin proof metadata are additive.
The existing dynamic-shape, dynamic-matrix, tail-mask and symbolic-layout
regressions also pass all 43 cases with native/Merlin environments enabled.

Source boundary: LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`,
`mlir/lib/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.cpp` and its
`scf_forall` fixture model bounds from the actual lower/upper/step operands.
Meta `17f0c234ce474bee334af36e351c62eca7852836` instead exposes program
IDs/counts as launcher arguments. This adapter owns the whole-launch forall,
so it establishes the correlation from the same checked ABI constructing it.

This closes a reusable proof gap and improves the emitted abstraction without
discarding the source's tiling. General aliases, non-dense layouts, packed maps,
mapped memory, broader numeric/effect profiles and target execution remain
separate requirements for universal coverage.

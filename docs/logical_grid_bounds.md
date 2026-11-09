# Next tensor-proof extension: bounds from the actual logical grid

Status: planned, not implemented. The guarded profile already executes the
observed kernels; this work admits their unchanged source into the tensor
profile without extra memory guards or redundant source masks.

An isolated prototype now establishes the next implementation's feasibility.
`.scratch/grid_correlation_probe.py` checks 3,484 affine scenarios against
independently enumerated actual launch domains, including multiple axes,
coefficient signs, divisors and empty cases. `.scratch/grid_histogram_probe.py`
combines the candidate access rule with an exact unit-row partition in a child
process. The unchanged unmasked histogram row source then passes native and
actual Merlin compilation plus five independent invocations, including empty
and surplus domains. These probes do not modify production modules or ship an
admission rule. Integrate the general proof and the negative cases below before
changing the status of this proposal.

## Observed gap and two independent gates

A histogram can store one complete row per logical program into
`out: tensor<tiles x bins x i32>`, launched with `GridExtent("tiles", 1)`.
Its unmasked row coordinate is `program_id(0)`. The generated forall already
enforces `0 <= program < tiles`. Nevertheless `views.axis_permutation` compares
independent worst-case intervals for program and guarded `tiles`, so it cannot
prove `program + 1 <= tiles`. Adding `rows < tiles` to the source lets the
existing clipped-prefix proof qualify the row view; the guarded profile also
qualifies the unchanged unmasked form.

Bounds are only the first gate. `DefiniteWrites.finish` currently requires a
clipped partition for varying grids or runtime result dimensions. Simply
relaxing the access check would still fail fresh-output coverage. Both gates
must use the actual launch/shape relation, with independent ownership checks
and existing initialization rules preserved.

Flat rank-one pointers into a dynamic matrix are a separate descriptor-rank
limit. They already execute through the guarded profile; this proposal does
not conflate that representation issue with unmasked matching-rank bounds.

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

## Implementation and qualification order

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

This closes a reusable proof gap and improves the emitted abstraction without
discarding the source's tiling. General aliases, non-dense layouts, packed maps,
mapped memory, broader numeric/effect profiles and target execution remain
separate requirements for universal coverage.

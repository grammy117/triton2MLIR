# General reductions at the model2MLIR/Merlin boundary

`tt.reduce` now retains general pure scalar combiners, including coupled tuples,
numeric captures, branches and for/while state. Inputs keep their source widths
and actual initial values. The portable payload uses standard Tensor/Linalg/SCF,
with no warp algorithm, hardware memory class or opaque reducer implementation.

## Source-backed transformation

Meta/shared's
[`ReduceConverter`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp)
handles a selected single-operation roster and supplies neutral constants.
Its separate argmin/max patterns recognize selected tuple bodies. The archived
Microsoft converter follows that pattern; neither supplies the general
structured/coupled region needed here. A neutral constant also requires a real
identity: integer AND cannot start at one, and maxnum cannot use negative infinity
as an identity for an all-NaN source tile.

The adapter follows standard Linalg reduction semantics while retaining the
complete original combiner. It extracts the first source element along the
reduced axis as each output's initialization. A second slice contains every
remaining element. Single-component reductions use named `linalg.reduce`;
coupled tuples use one `linalg.generic` with identity input maps, projected
output maps and a reduction iterator on the original axis. This keeps mixed
types and cross-component dependencies together despite xDSL's single-input/
output convenience class for named reductions.

Linalg's scalar input/accumulator order differs from the source combiner's
left/right order. Explicit SSA mapping preserves source-left as accumulator and
source-right as current input. Original operations, properties, captures,
control regions, yields and locations are cloned without replacing a difficult
body by a mathematical name. The native target still chooses its legal reduction
implementation; no GPU tree or bitwise floating reduction equivalence is claimed.
Ordinary surrounding arithmetic gains no reassociation permission.

A singleton axis returns the original element and never executes the combiner.
All original candidates still pass complete body checks first, including
singleton/dead bodies. No neutral value is synthesized, no accumulator format is
widened implicitly and no extra singleton axes are dropped. Scalar TTIR results
use rank-zero internal tensor transport before extraction. Empty runtime domains
retain the original launch/mask semantics; a zero-size static reduction axis
has no invented result.

The scalar checker is shared with [scans](scans.md). It permits pure registered
arithmetic/math and complete numeric structured control. The narrow xDSL
unsigned-remainder trait omission is handled by its exact arithmetic class,
without a speculation or nonzero-divisor grant. Memory, calls, unknown effects,
tensor captures and remaining unexpanded TTIR scalar builtins still need their
own supported semantics. These are actual remaining forms, not opaque successes.

Original reduction/scan regions also receive an effect audit before any
normalization or inlining. It follows complete nonrecursive helper definitions,
including unused reads and diagnostics. A parent operation's native Pure trait
cannot authorize dropping a nested effect during folding. The later inlined
scalar/type/control checks remain mandatory. Located source refusals cover both
direct and helper-hidden effects, singleton axes and unused whole reductions.

`prov.family`, `prov.op`, reduction composite identity, original axis and
`prov.reduction_initialization` retain intent in MLIR. The
`ttir-first-element-reduction` transformation records tuple types, original
source IDs, body correspondence and derived control counts. Provenance grants
no omitted arithmetic, associativity, numeric, memory or termination facts.
Public tensor/scalar/model ABI, receipts and compiler artifact rosters are
unchanged. Generated compatibility metadata records this representation.

## Merlin destination and lifetime handling

Dynamic tensor recurrences containing scalar GCD/branch combiners exposed an
existing destination-planning gap. Merlin now verifies numeric `scf.if`,
`scf.for` and `scf.while` containers natively before accepting their values in a
complete producer. It inspects every descendant and captured read independently,
including unused calls, writes and tensors. Scalar state grants no buffer
ownership, fresh destination, capture alias or termination premise. Original
control stays in place; a completed result may then feed the existing checked
value copy/reuse path. Borrowing, bufferization and post-deallocation lifetime
checks remain enabled. The 121-check native destination/refusal selection passes.

## Valid source boundary

Pinned Triton explicitly forbids memory writes inside `map_elementwise`, including
volatile loads modeled as writes. Those refusals belong to invalid-source
coverage for this version. They must not be counted as an unimplemented portable
map-store profile. Valid ordinary writes outside maps, volatile observations
outside that restriction, external state and remaining region combinations keep
their separate completion requirements. The
[pinned native verifier](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/IR/Ops.cpp)
and existing native refusal tests retain this distinction; do not bypass it to
claim broader coverage.

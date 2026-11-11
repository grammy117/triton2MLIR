# Packed maps with layout-independent scalar computation

Packed maps now lower automatically when the actual inlined scalar SSA graph
establishes that every slot computes the same function of its own input slots.
That function may now contain pure scalar `scf.if`, `scf.for` and `scf.while`
regions, including data-dependent trip counts and carried values.
The result is a parallel `linalg.generic`, with the original pack size retained
in `prov.map_pack` and the checked abstraction named by
`prov.map_grouping="proved-slot-independent-v1"`. No GPU register/lane layout,
new public argument, opaque operation or package schema is introduced.

## Source boundary and upstream reuse

Triton 3.7.1 groups arguments and results by input/output tensor, then by packed
slot. Its LLVM conversion packs consecutive **per-thread register values**,
not necessarily consecutive logical tensor coordinates. See the pinned
[map conversion](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L572-L640)
and [Python interface](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/triton/language/core.py#L2847-L2919).
Pre-layout tensor shape alone does not establish a cross-slot grouping.

Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and archived Microsoft's
converter have no map-region expansion. Ascend
`8f0ade66580245e30ad97093526a4018e4e18fc7` has one in
`third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp`, with tests in
`third_party/ascend/unittest/pytest_ut/test_map_elementwise_ops.py`. Its packed
div/mod fixture computes the same independent function in each slot. Promoting
every slot argument to an entire input tensor is valid for that fixture but
does not establish cross-slot or asymmetric cases. This adapter proves the
required independence before reusing the existing standard Linalg expansion,
upstream helper inlining/arithmetic conversion and Merlin compilation.

## Checked abstraction

For every returned scalar slot, the adapter checks two separate facts:

1. Its SSA dependencies are restricted to the corresponding slot of the input
   tensors and scalar captures uniform across the invocation.
2. Its typed expression is congruent to every other slot of that output:
   opcode, operand order, result ordinal, full types, executable properties
   and attributes must agree, including floating literal words (signed zeros
   and NaN payloads). Only descriptive `prov.*` metadata and locations
   are excluded from semantic equality.

Terms are interned iteratively, so repeated expressions do not require recursive
expression expansion or element enumeration. A shared normalized term does
not erase the separate slot-dependency check: `a0+a1` cannot become `2*a`.
Known effect-free scalar arithmetic/math DAGs use the original proof;
structured scalar regions use its extension described next. The pinned xDSL
binding omits the effect trait on `arith.extsi`; that exact registered class is
admitted using upstream MLIR's `Arith_CastOp` purity contract. Unknown effects,
regions and memory reads do not inherit that exception.

### Structured scalar regions

`typed-structured-ssa-congruence-v1` extends the same proof to known scalar SCF
operations. Region arguments use lexical binders instead of SSA identities;
typed operands, captures, result positions, executable attributes/properties,
region order, terminators and complete ordered bodies remain part of the key.
An inner induction variable cannot be substituted for an outer one. Interned
region terms avoid expanding nested expressions into duplicated trees.

Slot-dependency analysis includes every captured region value as well as direct
operation operands. A guard or hidden body capture from a different slot still
refuses even when its normalized expression matches. Representative liveness
retains the producers of captured values. Corresponding nested source locations
are fused at their original positions; provenance does not supply the proof.

The actual branches and loops are cloned into the scalar Linalg body. They are
not converted to eager selects, unrolled, or assigned to physical threads. Loop
bounds may differ between logical elements. Complete region bodies remain
executable. Unused structured control currently refuses rather than being
discarded as dead arithmetic: unknown termination is observable even when a
loop has no memory effects. Packed memory and physical grouping remain separate
requirements.

The emitted scalar body contains a representative output dependency closure.
Unused pure arithmetic is removed before scalar arguments are identified,
preventing discarded cross-slot divisions from becoming speculative work.
Equivalent source locations are retained together; the transformation report
records the original pack, checked grouping, scalar operation counts and output
source IDs. Caller annotations cannot substitute for any check.

Because every output is the same independent scalar function, any legal
original register grouping gives the same logical tensor result. This proof
does not choose a physical grouping or certify device-specific numeric behavior.
Tiling, source outer recurrences, output order and arithmetic remain executable;
the retained packing annotation can inform downstream optimization.

## Qualification and remaining coverage

Eight actual Merlin libraries qualify pack sizes two/four, i32/i64/f32,
rank-one/rank-two tiles and source runtime recurrences. Tensor, buffer and
guarded source-word profiles are included. Each library replays four additional
domains: empty input, one element, zero trips and multiprogram ragged tails.
Wide i64 values exceed exact floating transport. Independent references compare
both results and functional input preservation. Negative cases retain coupled
slots, asymmetric arithmetic and differing fastmath attributes, including a
forged claimed grouping annotation.

Thirteen packed-map checks, 46 existing map/memory regressions and the refreshed
maintained-corpus check pass. Word-level congruence tests reject signed-zero and
NaN-payload differences; dead cross-slot arithmetic is neither executed nor
attributed to an equivalent live expression.

The maintained product corpus also contains a successful rank-two i64 packed-map
case with two results and three independent runtime domains. The asymmetric
`packed-map` case remains refused, alongside source assembly. The current
verified corpus has thirteen cases and eleven actual Merlin libraries.

Fourteen additional native/Merlin libraries now qualify structured packed maps:
pack two/four, i32/i64/f32, rank-one/two tiles, zero-divisor branches and for/while
loops whose per-element trip counts come from an input tensor plus a runtime
scalar. Each unchanged library replays four more domains (56 replays total),
including empty/scalar inputs, zero/negative trip domains and ragged tails.
Independent integer references retain values above 2^53; every complete result
and functional input preservation is checked. Eleven direct proof checks cover
hidden cross-slot guards/captures, lexical binders, result ordering, nested
fastmath differences, producer/source-location preservation and unused loops.
The 59 existing packed, unpacked and mapped-memory regressions also pass.

This closes the qualified layout-independent scalar class, not arbitrary packed maps.
Coupled/asymmetric slots need an established mapping from original register
groups to logical coordinates, carried out of a supported layout IR as dataflow
facts. Other regions, mapped reads and broader numeric forms need their own checked
expansions. Inferring adjacency, duplicating packed inputs or dropping slots
would change computation and remains forbidden. Applicable target execution
and performance are separate completion gates.

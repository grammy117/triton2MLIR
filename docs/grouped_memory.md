# Source-specialized packed reads and pointer results

Packed `tt.map_elementwise` regions now retain ordinary source reads under the
explicit initial source-layout specialization and guarded-buffer/source-words
memory profile. This extends [source grouping](source_grouping.md) and
[mapped memory](memory_maps.md); the selected geometry remains semantic input,
not consumer placement or correspondence to a final optimized GPU layout.

## Transformation and boundary

The pointer analysis checks the native packed region signature: each tensor
operand supplies `pack` scalar arguments, and each tensor result receives
`pack` scalar returns. Every scalar argument inherits its original tensor's
possible roots; a pointer tensor result includes all its returned slots' roots.
Mixed-root interfaces carry one selector tensor per pointer field and one
scalar selector per slot. Offsets retain source arithmetic and signed i64
accumulation; selectors identify logical source roots, never physical addresses.

For maps containing reads, source coordinate facts construct a parallel
`linalg.generic` with **one invocation of the complete original scalar region
per logical pack**. Each returned scalar slot has a tensor of group values.
Separate parallel generics restore the original logical tensor coordinates.
Restoration selects already computed values; it does not repeat the original
loads, branches or loops for every output slot. Algorithmic grouping stays
visible and consumer scheduling remains undecided.

Actual Meta/shared event templates and the existing guarded-memory path retain
each load's original position, mask, fill and control. Bounds, initialization,
ownership, snapshots and native lifetime checks remain enabled. Reads captured
before an ordinary write cannot become observations of the changed state.
Pointer-valued results restore every slot's own offset, selector and predicate
before subsequent memory operations.

The pinned Triton verifier forbids writes and volatile reads in map regions.
This extension admits no map stores, atomics, publication or physical-address
observations. Ordinary writes between maps retain their existing contracts.
Arbitrary cross-program conflicting accesses remain a separate dependence
problem; the maintained state fixture bounds recurrence reads to its own
logical program, avoiding an invented global phase between parallel programs.

Source references and reused boundaries:

- [Triton packed signature and memory verifier](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/IR/Ops.cpp).
- [Triton packed multi-result test](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/test/unit/language/test_core.py#L5165).
- [Meta/shared tensor arithmetic patterns](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp).
- The [existing reuse map](triton_shared.md) and [source-word event boundary](source_words.md)
  retain the separate optimizer/version identities and archived Microsoft lineage.

## Measured scope and package differences

Six actual native/Merlin libraries qualify pack two, i32/i64/f32, two/four
source warps, ordinary/selected roots, for/while reads, returned pointers and
per-slot predicates. Each library runs five domains, including empty roots,
one element, exact tiles, multiple programs and ragged tails. Four state
libraries compare all five state/snapshot results, including untouched storage;
two pointer-result libraries check restored root/offset/predicate combinations.
Wide integers exceed the exact range of floating transport. An invalid active
read aborts the existing library with the real bounds diagnostic.
The 95-case existing memory/map/pointer/layout regression suite also passes,
including pack-one maps and native source-write refusals. Guard reports retain
six ordinary read events, or twelve masked selected-root events, and the two
original recurrence loops; result restoration does not multiply those bodies.
The public `shared-tensor`/`qualify-merlin` CLI example is retained at
`out/grouped-read-demo/{capture,merlin}/`; its twenty-member source bundle and
actual Merlin host qualification both verify independently.

The executable payload uses standard tensor/Linalg/SCF and the already declared
memory profile. `group_storage: one-vector-per-result-slot` describes the
source-group expansion; original map identities, pack size, coordinate witness
and event correspondence remain retained. Public tensor/scalar ABI and receipt
schemas are unchanged. Generated compatibility metadata records this extension.

This finite qualification does not establish all pack widths, numeric policies,
source versions, complete-model integration, final GPU correspondence, broader
targets or performance. Source-specialized pure model target qualifications do
not qualify these new memory compositions.

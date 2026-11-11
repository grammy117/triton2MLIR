# Mapped source reads through the Merlin interface

Pack-one maps now admit logical pointer inputs and source reads under the
explicit guarded-buffer/source-words profile. The source map edge connects each
tensor pointer operand to its scalar block argument; pointer results have the
corresponding scalar-to-tensor edge. The existing finite-base analysis retains
the owning root and original signed word offsets. No physical pointer integer,
lane identity or hardware storage class is introduced.

After pointer normalization the map becomes a parallel `linalg.generic` with
its original scalar region. The existing Meta/shared source-word templates
convert each read at its original position. Masks, nonzero fills, branches and
for/while loops remain executable. Guarded memory independently checks actual
active bounds, initialization and ownership. Source annotations cannot supply
those facts or authorize an event to move.

Mapped read snapshots remain values when ordinary kernel writes occur between
map invocations. Original reads and their consumers are preserved even when
the original snapshots are only consumed after mutation. The functional public
input stays unchanged; declared writable source state returns its whole
allocation, including every untouched word.

The pinned Triton verifier explicitly forbids memory writes inside map regions:
[MapElementwiseOp::verifyRegions at f797708c](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/IR/Ops.cpp#L614-L642).
It also models volatile loads as writes. The adapter does not bypass that
verifier, even when an original helper call temporarily hides a store before
inlining. Native refusal tests retain this source-version boundary. Valid
shared-state kernels can place ordinary writes outside read maps; those writes
retain their existing memory/effect admission.

Thirty-four map/source-memory regression cases pass. Seven actual libraries
cover the basic pointer map and i32/i64/f32 read maps with source for/while
control, original nonzero fills and all five returned state/snapshot values.
The six state libraries replay five domains each, including zero-size roots,
zero trips, tails and untouched storage. Wide i64 values exceed the exact
range of floating transport. An actual out-of-bounds invocation aborts its
existing library without a fabricated numeric reference. The existing pure
map profiles and packing/profile refusals remain checked.

The maintained product corpus now supplies independent references for its
memory-map case. Packed grouping, broader pointer/base compositions, numeric
forms and applicable target execution retain their separate qualification
requirements in the [completion plan](completion_plan.md).

## Finite selected roots on map interfaces

Maps now also carry a logical root selector alongside each mixed-base pointer
argument/result. The parent field is an i32 tensor with the original pointer
shape; each scalar invocation receives or yields one i32 tag. All original
interface slots remain in order. Fields identify source owning-buffer ordinals,
not physical addresses. Pointer offsets retain their original source arithmetic
and signed i64 accumulation independently of those tags.

This handles selection before a map, selection inside its scalar region,
pointer-valued results, and mixed pointer/numeric results whose memory is also
read inside the map. The unchanged source-word event templates retain actual
read positions and per-root masks. Pure pointer-return maps also select the
event-isolation path before upstream pointer conversion; no layout or lane
mapping is needed to interpret the result.

Twelve new actual libraries cover all four compositions for i32/i64/f32. Each
replays five domains, including either unselected root being empty and both
roots being empty on an inactive domain. Wide source words and every result
remain exact. Thirty-four existing selector/shape/value regressions also pass.
This extends measured base compositions; packed groups, broader numeric/forms
and applicable targets remain open.

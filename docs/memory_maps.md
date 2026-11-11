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

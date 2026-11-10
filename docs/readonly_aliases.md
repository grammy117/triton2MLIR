# Readonly aliases at the functional input boundary

Typed package writers now accept overlapping readonly tensor arguments. Each
argument supplies a logical value; before serialization, possible readonly
overlaps are copied to independent arrays with the same dtype, shape and every
input word. This supports identical arguments and partially overlapping source
allocation spans without changing the kernel's tiling, reuse or recurrence.

Runtime scalars are loaded by value before invocation. A scalar's zero-dimensional
NumPy carrier may therefore share backing bytes with a tensor, including writable
state. Input preparation copies that scalar carrier alone, preserving its initial
value and the tensor's storage. The callee receives an ordinary scalar value.

## Scope and checks

`validate_inputs` first checks the complete typed input roster, size budget,
runtime shapes, scalar bounds and Boolean storage. It then checks every pair in
the original overlap graph before making any copy. Two readonly tensors with
possibly overlapping byte bounds are conservatively snapshotted. Potential
overlap involving a writable tensor and another tensor requires exact disjointness;
the bounded NumPy check refuses uncertainty as well as confirmed overlap.

This ordering prevents copying a readonly pair from hiding a later overlap with
writable state. Interleaved disjoint views remain accepted when exact analysis
proves that their observed bytes are disjoint. See NumPy's
[`shares_memory`](https://numpy.org/doc/stable/reference/generated/numpy.shares_memory.html)
contract for bounded exact analysis and `TooHardError`.

Source intake separately refuses physical pointer identity and volatile
observations in this profile. Inputs must remain stable during preparation.
Snapshots supply no synchronization semantics. Writable tensor aliases require
shared state and source ordering in an explicit alias interface; they still
refuse before native execution or package creation.

The descriptive memory field is
`input_preparation: readonly-input-value-snapshots-v1`. Compiled and serialized
tensor inputs still satisfy `independent-storage-ABI-precondition`. Package
compatibility metadata records this preparation rule; ABI and receipt schemas
are unchanged. The generic raw-MLIR native executor retains its strict input
independence requirement because it has no typed source effect contract.

Preparation can allocate extra host copies. It grants no target memory placement
or optimization permission. The original grid, masks, source addresses, tiles,
K reuse, accumulator and guarded output state remain executable in standard MLIR.
All bounds, initialization, write-conflict, ownership and lifetime checks remain.

## Qualified evidence

Twenty input-preparation cases check identical/partial overlaps, exact input
words across seven tensor carriers, scalar carriers, disjoint interleaved views,
bounded-analysis refusal and complete-graph validation. Nine native/Merlin
integration cases qualify both package writers: matrix addition across the
three memory profiles; f32/f16/BF16 strided tiled GEMM with independent writable
output state; and by-value scalar carriers overlapping writable i64 state across
the three profiles. These 29 cases pass, as do 100 relevant package, effect,
native and runtime-scalar regressions.

The nine compiled libraries also pass 33 additional invocations: identical,
partial and disjoint readonly spans; reversed/broadcast source reads; empty and
zero-K domains; and changing scalar carriers. Writable tensor overlaps refuse
before native work or a package directory exists. Complete returned allocation
words, original input preservation and actual library reuse are checked.

The visible BF16 package and qualification verify 33/52 artifacts. The explicit
report at `out/readonly-alias-coverage/coverage.json` selects 39 packages and
retains 31/50 TTIR definitions observed, with 27 in host-qualified captures.
These are observations for that selection; maintained corpus and universal
type/attribute coverage remain open.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel readonly-alias-bfloat-dot \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/readonly-alias-bfloat --python /path/to/native-python
python -m t2m qualify-merlin out/readonly-alias-bfloat \
  --out out/readonly-alias-bfloat-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`readonly-alias-dot` and `readonly-alias-half-dot` select f32/f16 operands.
The visible demo passes live overlapping operands to the writer; the saved
archive contains their independent values. A serialized NPZ carries no physical
alias relationship. Writable tensor aliases, strided output-C reads, logical-view
model adapters and applicable targets remain in the [completion plan](completion_plan.md).
Declared writable source sharing now has an [allocation-state entry](allocation_state.md):
callers pass one root and explicit offsets rather than overlapping independent
tensor values. The preparation rule and its undeclared-overlap refusal stay intact.

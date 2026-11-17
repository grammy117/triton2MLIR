# Dynamic arithmetic and slices around Triton calls

The three previously failing surrounding-computation model profiles now pass
through complete packages, native execution, Merlin host and actual Spike.
The real PyTorch model in `scripts/capture_live_scalar_model.py --extent` is:

```python
original = x + self.weight
first = operation(original, x.shape[0])
second = operation(first, first.shape[0])
empty = operation(x[:0], 0)
return second, first, empty, original, self.weight
```

The public input retains a dynamic length with the original exported `[2, 31]`
domain. The saved scalar parameter is argument zero; `x` is argument one. All
five results remain present, including the empty result and incoming parameter.
The actual source addition and slice replace the former unresolved ordinary
ATen calls; registered calls retain their real translated kernel bodies.

## Source computation and consumer contract

Model2MLIR
[`9e2b646`](https://github.com/grammy117/model2MLIR/commit/9e2b646ce3cff822fe245832c45d81e8827141e7)
adds runtime binary broadcasting and positive-step slicing in standard
Linalg/Tensor/arith/cf. Actual source dimensions supply extents. Static singleton
projections retain affine maps; ambiguous runtime singletons use explicit
indexed tensor reads with compatibility assertions. Endpoint clipping reuses
checked interval-sized shape arithmetic. A proved nonnegative logical-i64
length narrows before quotient/remainder, avoiding overflow and unnecessary
wide division runtime helpers. The existing half scalar opmath policy remains.

The enclosing model uses the existing symbolic profile and
`merlin.compile-signature.v3`. Producer verification and Merlin independently
reconstruct the addition's actual destination shape and the empty slice's fixed
size. No sample, saved parameter value or annotation supplies a compiler shape.
Original guards, kernel tile/grid structure, masks and every public result stay
executable. Public ABI, parent archives, artifact rosters and receipt schemas
are unchanged; this adds no physical view-alias or mutation permission.

## Measured selection

`tests/test_live_scalar_model.py` checks all three tensor/buffer/guarded-buffer
configurations through native, Merlin host and Spike. All fifteen complete
initial results compare exact words. Twelve further host calls reuse the same
libraries/programs at lengths 2, 3, 7 and 31 while changing scalar parameter
words. All sixty further results and functional inputs are checked. Six actual
calls at lengths 1 and 32 abort through the source-domain guard without outputs.
The test now requires checked symbolic adapters and rejects a sampled constant
13 as a dimension fact.

`tests/test_dynamic_pointwise_execution.py` separately executes seventeen typed
ATen-emitter cases: scalar/row/column/runtime-singleton add/sub/mul, clipped/
stepped slices, signed-i64 extreme bounds/step and incompatible-broadcast aborts.
The primitive fixtures use explicit unused test-driver output-extent scalars to
size descriptors; these establish computation, not general compiler shape facts.
The real enclosing model above has no such extra inputs and independently
derives its Merlin signature.

The final primitive selection passes 17 in
`.scratch/dynamic-pointwise-final5.log`. Model2MLIR's broader selection passes
274 with sixteen skipped checks; its final symbolic/domain selection passes
46. Counts overlap. The earlier wide-division and same-width truncation failures
remain recorded separately and are superseded by the final primitive selection.
The final complete-model selection passes three with seven unrelated cases
deselected in `.scratch/dynamic-surrounding-models-final2.log`; its receipts are
under `.scratch/pytest-dynamic-surrounding-models-final2/`.
`out/dynamic-surrounding/summary.json` reopens every target receipt, original
parent archive, native/host result, replay and guard outcome. The separate
broader `.scratch/dynamic-surrounding-final1.log` retains 160 passes, one skip
and its early, now-fixed extreme-slice failure; it is not relabeled all green.

## Remaining dynamic coverage

[Derived model dimensions](derived_model_shapes.md) now separately qualify
conditional broadcast sizes and clipped slice counts with independently checked
signature v4 programs. More general expressions, runtime-supplied slice bounds, dynamic value/source views,
resizing state, data-dependent output allocation and arbitrary effect/model
compositions remain open. New producer emitters alone do not close those
consumer interfaces. Accelerator placement, GPU numeric correspondence and
performance require separate qualification. Full arbitrary-TTIR coverage
remains active and incomplete.

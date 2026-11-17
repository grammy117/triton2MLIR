# Results discovered during execution

Merlin now has an explicit serial host mechanism,
`merlin.execution-owned-results.v1`, for tensor results whose dimensions are
computed by the executed body. The
[Merlin implementation](https://github.com/ucb-bar/merlin/commit/434748c9809e3584ea12d70c0fa04bbcc5ce5eb7)
exposes `merlin.llvmlower.owned_results.compile_owned_results`; it is not yet selected by
triton2MLIR packages, model binding, the native evaluator or the Spike harness.
Existing compile signatures through v5 retain their source-proved preallocation
contracts. No package ABI or TTIR coverage count changes in this milestone.

## The concrete boundary

The input-word model already supports:

```python
n = count.item()                  # an actual public singleton input word
first = prefix(x, n, x.shape[0])
```

Its v5 contract independently proves a size expression before allocation.
That expression does not represent general vector computation such as:

```python
selected = x[x > 0]               # size discovered from vector contents
return selected, selected, x
```

The new consumer mechanism executes standard MLIR for positive-value
compaction, obtains the actual result descriptors afterward, and copies every
result to independent caller storage. It preserves the source's actual count
and pack loops rather than invoking a separate size predictor. Duplicate
results and returned snapshots of input values retain their full contents.
This is an actual standard-MLIR execution test, not a claim that the PyTorch
indexing syntax above already imports through model2MLIR.

## Reused upstream behavior and resource contract

The selection retains Merlin's ordinary bufferization, ownership, deallocation
and independent lifetime checks. It omits result-to-out-parameter conversion.
Upstream ownership transfers returned allocations to the caller and inserts
copies where borrowed arguments otherwise escape. The official clone lowering
then emits those copies in standard memory operations. A generated C bridge
uses upstream's first-result-pointer convention; C owns tuple descriptor layout.

Official generic MemRefToLLVM callbacks select an invocation-local allocator.
Actual allocation records bound capacities, peak charged bytes and allocation
count, including alignment/tracking overhead. Additional executable checks on
actual dimensions precede byte-product arithmetic, so a negative or overflowing
extent cannot wrap to a small allocation request. The initial resource profile
covers supported identity-layout heap allocations in a native 64-bit host ABI;
stack/nonidentity allocations and other native widths need further contracts.

Every result is checked against a live allocation before payload copying.
Checks retain static dimensions, signed strides, alignment, accessible byte
footprints and a total output-byte limit. Empty views perform no access.
Duplicate owners release once. Cleanup frees recorded owners rather than
trusting pointers supplied by descriptors. Unreturned live allocations refuse.
Private loaded images and locally bound functions retain the selected program;
source/prepared/LLVM/bridge/library and implementation bytes are checked.

Provenance and source locations survive preparation. Neither annotations nor
the descriptor transport grant numeric, memory, alias or ownership permissions.
The portable model still returns tensors; the selected host consumer owns this
lower-level allocation/descriptor ABI.

## Qualification and remaining work

The final Merlin regression selection passes **133** execution/allocation,
codegen, pipeline and lifetime checks in
`out/build/owned-result-final3.log`. After the last source changes, **36** focused
native/result/allocator checks pass in `out/build/owned-result-final4.log`.
Counts overlap. Documentation and structure gates pass. All new modules and
changed support/test modules pass Ruff; `pipeline.py` retains its 71 pre-existing
diagnostics with no new diagnostics. Earlier failing logs remain distinct.

Merlin's focused tests exercise real content-derived output sizes, empty and
duplicate results, input snapshots, mixed result ranks, exact floating words,
native scalars, source/build substitutions, corrupted descriptors and actual
allocation failures. The allocator tests separately exercise aligned/empty
storage, byte/count bounds and invalid releases. Host correctness does not
qualify Spike or accelerators, and no new Triton source operation is counted.

The fresh resizing-loop experiment remains a located lifetime refusal.
Merlin's generation checker currently separates replaced carried buffers only
for positive static trip counts with a final same-block release. A runtime or
zero-trip loop can return its initial buffer, while positive trips return a
new generation. Conditional cloning/retention and returned ownership need a
closed proof before the checker can admit that case. The new transport keeps
the refusal; it does not disable the gate.

The next integration requirements are:

1. Add an explicit independently source-checked staged result-mode signature.
   Retain exact public ranks/dtypes and reject downgrade/rehashed mismatches;
   do not reinterpret v3/v4/v5 preallocation contracts.
2. Bind real model/kernel results computed from internal data. Extend the
   necessary model2MLIR emitters and per-call shape/resource relationships;
   never substitute saved invocation values for executable source computation.
3. Carry the selected mode through triton2MLIR package verification, compiler-only
   staging, native evaluation and complete result/reference receipts. Preserve
   source/weight archives and keep evaluator arrays out of compiler staging.
4. Extend the independent generation proof for zero/runtime-trip resizing
   state, conditional final owners and transfer to function results. Close
   initial/fresh/final aliases and every use/release edge; preserve negative
   lifetime tests.
5. Materialize the same semantics in the applicable target harnesses with real
   resource/lifetime contracts and complete execution evidence.

These remain part of the full arbitrary-TTIR objective, alongside broader
memory/effect/numeric/source compositions and accelerator scheduling/execution.

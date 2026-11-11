# Logical indexed tensor descriptors

In-kernel `tt.make_tensor_descriptor` now has an executed path for
`tt.descriptor_gather`, `tt.descriptor_scatter` and `tt.descriptor_reduce` through
the model2MLIR-like kernel package and actual Merlin compiler/host interface.
The implementation reuses Triton's installed pre-layout
`triton-rewrite-tensor-descriptor-to-pointer` pass, followed by the existing
shared arithmetic and checked source-word memory pipeline. No second descriptor
lowering or target launch interface is introduced.

## Upstream boundary

Reviewed source and tests:

- Triton v3.7.1, `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`:
  [pointer rewrite](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp),
  SHA256 `7082dfc76bad6996625c16d7d3cdd9ce823da7dbe183958a403faeda34f17c45`,
  and [descriptor tests](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/test/Triton/rewrite-tensor-descriptor-to-pointer.mlir),
  SHA256 `10a1a0a2b5b89c096b76aefe95fc1a76a461d2cd9a0ef0c30a591589dd7afa31`.
  Its gather/scatter patterns construct indexed pointer tensors, signed
  per-dimension masks and source padding. Reducing stores translate to typed
  atomic RMW operations, including signed/unsigned min/max and bitwise kinds.
- Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7`:
  [descriptor conversion](https://github.com/triton-lang/triton-ascend/blob/8f0ade66580245e30ad97093526a4018e4e18fc7/third_party/ascend/lib/TritonToLinalg/DescriptorConverter.cpp)
  and its `third_party/ascend/unittest/Conversion/General/TritonToLinalg/`
  `test_parse_descriptor_{gather,scatter,reduce}.mlir` tests. These independently
  demonstrate logical row loops and descriptor-to-atomic decomposition. The
  native upstream pass supplies the reusable implementation here, so Ascend's
  physical loop annotations and target dialect do not enter our payload.
- The pinned Meta shared source `17f0c234ce474bee334af36e351c62eca7852836`
  and archived Microsoft shared conversion sources were checked for descriptor
  converters. The existing installed shared reader does not parse this source
  descriptor vocabulary. Native verification/normalization precedes the shared
  pipeline; all subsequent shared analyses and arithmetic conversions remain in use.

The normalizer records installed version, actual binary size/hash, exact input
and normalized-output hashes, selected pass pipeline and an independent source
reference revision. The source reference is explicitly **not** a build receipt
for that installed binary.

## Computation and provenance

The original native-verified graph and normalized graph remain distinct.
`prov.kernel_original_descriptors` retains the original descriptor operations,
typed operands/results, graph IDs and profile on the emitted function. The
`logical-tensor-descriptors-v1` report leaves hardware mapping undecided and
grants no semantics. Existing source IDs and normalized IDs keep their separate
meanings; this kernel-level intent record is not a fabricated exact descendant
mapping for every operation introduced by upstream.
[External entry descriptors](external_descriptors.md) now additionally expose
the actual upstream logical field interface with original/expanded ordinals;
device descriptor bytes and nonzero external origins remain separate gates.

Executable pointer arithmetic, loaded row indices, masks, zero/NaN padding,
stores and atomic kinds supply computation. Gather permits repeated rows;
scatter retains per-store lane uniqueness and cross-program ownership checks.
Reducing stores use the existing explicit private-invocation atomic profile,
retaining source scope/order with its stronger standard semantics. Original
loop placement, runtime repeat count, prior allocation state and untouched
padding words survive. No ordinary scatter race is silently serialized.

The public ABI is owning allocation tensors and declared shape/stride/offset
scalars. Writable allocation state returns once as a complete tensor. Existing
ABI v5 and effect receipt/reference schemas remain unchanged. Compatibility
metadata explicitly distinguishes this logical view from a physical TMA
descriptor, device alignment/placement constraints, async transfer, launch ABI
or accelerator-equivalence claim. External descriptor parameters use the
separately declared logical field representation and its binding checks.

## Reproduction and qualification

Use the existing shared optimizer/native environment; no new compiler build:

```bash
python -m t2m shared-demo --kernel descriptor-gather \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/indexed-descriptor-gather --python /path/to/native/python
python -m t2m shared-demo --kernel descriptor-scatter \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/indexed-descriptor-scatter --python /path/to/native/python
python -m t2m shared-demo --kernel descriptor-reduce \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/indexed-descriptor-reduce --python /path/to/native/python
python -m t2m qualify-merlin out/indexed-descriptor-gather \
  --out out/indexed-descriptor-gather-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend/python --python /path/to/native/python \
  --merlin-clang /path/to/clang
```

`tests/test_indexed_descriptors.py` carries independent NumPy row/state oracles
and real native/Merlin execution. Twenty-four semantic cases qualify 24 libraries,
144 additional successful invocations and ten real scatter conflict refusals.
The matrix covers transfer i16/i32/f16/BF16/f32,
floating zero/NaN padding, all six integer reduction kinds, signed i64 min/max,
unsigned i32 min/max and explicitly selected IEEE f32 addition. Each library is
reused with changed shapes/scalars, empty domains, tails and negative offsets.
Transfer references retain NaN payloads and signed-zero words; integer oracles
retain modular arithmetic and values above 2^53. Both same-tile and cross-program
scatter collisions must abort the actual compiled invocation without a success
receipt. Reference arrays and their digests remain excluded from compiler inputs.

NaN padding exposed a real literal-bridge gap. Canonical positive quiet-NaN
words now pass typed dense parsing only after checking the original bits and
exact codec round-trip. Encoded-byte dense attributes receive the same check.
Noncanonical, signaling and negative NaN payloads retain located refusals;
scalar NaN constants retain their existing separate gate. Twenty independent
literal checks cover repeated parse/serialization and native constant words in
f16/BF16/f32/f64, including signed zero. Input/load transport still preserves
arbitrary NaN words under its existing memory contract.

The pinned TTIR verifier rejects gather/scatter element widths above 32 bits;
i64 gather/scatter is invalid source, while valid i64 reducing stores are a
separate case. This gate is preserved rather than bypassed. The measured matrix
does not prove every permitted rank/type/attribute composition, external
descriptor bindings, floating min/max, half/BF16 reducing stores, external
synchronization or applicable accelerator materialization. Those remain required
parts of the full-coverage goal.

Visible packages are `out/indexed-descriptor-{gather,scatter,reduce}` (33 verified
artifacts each) and their separate `-merlin` qualifications (52 each).
`out/indexed-descriptor-coverage/coverage.json` joins them to the prior explicit
corpus: 44 packages, 34/50 definitions observed and 30 in host-qualified captures.
This observation count is separate from universal opcode/type/attribute coverage.

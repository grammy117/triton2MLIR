# External logical descriptor arguments

Kernels with actual `!tt.tensordesc` entry parameters now reach the Merlin
compiler/host interface. The installed pinned Triton descriptor-to-pointer pass
supplies the executable one-to-many conversion. Its actual original and
normalized native parameter signatures are compared before binding the public
tensor/scalar ABI. No device descriptor allocation, byte decoder or launch
backend is introduced.

The source and tests reviewed for [indexed descriptors](indexed_descriptors.md)
also define this interface. In
[the pinned rewrite](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp),
`unpackDescriptor`, the function type converter and structural arith/SCF
conversions establish the field order below. The installed binary fingerprint,
source reference, exact source and normalized bytes remain separately recorded;
the reference revision is not a build receipt.

## Field interface

Each descriptor parameter expands, in its original parameter position, to:

| Field | Native type / public representation | Computation |
|---|---|---|
| Base | Signless element pointer / owning allocation tensor | Logical allocation origin at element zero |
| Shape, once per axis | Signed i64 scalars | Per-dimension logical bounds; declared nonnegative scalar bounds required |
| Stride, once per axis | Signed i64 scalars, in elements | Actual indexed address arithmetic, including non-unit/column-major/broadcast reads |
| NaN padding | i1 scalar | Floating out-of-bounds loads select canonical quiet NaN or zero; integer padding remains zero |
| F32-to-TF32 rounding | i1 scalar | F32 loads use the upstream nearest-even E8M10 rounding, retaining special-value words; other element types ignore this field |

All ordinary parameters retain their order after expanding preceding
descriptors. Triton's real AST descriptor parameters additionally carry ordinary
shape/stride metadata parameters. Those are preserved as independent source
inputs, even when unused; they are not silently merged with descriptor fields.
The caller supplies consistent values when reproducing a Python descriptor
object. Multiple descriptors and private helper signatures use the same typed
conversion. Unknown layouts or unexpected actual signature changes refuse.

`RuntimeScalar.source_arg` indexes the **expanded** TTIR signature. The additive
`t2m.logical-descriptor-arguments.v1` record maps every original argument ordinal
to its expanded fields and declared allocation/scalar/constant bindings.
Descriptor block shapes, original type spellings and native field types remain
explicit. `prov.kernel_descriptor_arguments` carries that original intent on
the function; the original native graph and normalized graph retain separate
IDs. This metadata grants no arithmetic, memory, alias or rounding permission.

The public ABI remains TensorABI v5 where runtime dimensions are used, and
existing capture/effect receipts remain unchanged. Compatibility metadata states
the logical-field extension explicitly. Original `source.ttir` still contains
the descriptor entry parameters, and the package retains the exact original
source rather than substituting a normalized graph under that name.

## Runtime memory and useful structure

The whole-function shared pointer pass left chained runtime-stride pointer
expressions unresolved in an actual external-descriptor probe. Isolation now
selects those expressions structurally, following actual nested pointer
offsets through shape operations. Existing normalization flattens their logical
base/offset state; the same native shared converter handles typed event
templates, substituted at the original control location. The parent receives
arithmetic conversion only. No source annotation or kernel name selects this
path, and pointer conversion is not repeated on already converted events.

Single rank-one offsets with runtime products now select that same event path.
The whole-function pipeline had scalarized F32 runtime-stride pointers into
generic Linalg before establishing structured state. The selector follows the
actual pointer-offset DAG and reuses the native base/offset templates; it does
not replace signed arithmetic, masks or original control positions. Static
literal-stride offsets retain their existing route. Rank-one F32 load/store
libraries preserve every output word and readonly input through native/Merlin
host and actual Spike execution, including NaN words and source padding. Each
host library also replays empty roots, changed strides and runtime rounding
flags, and rejects invalid active bounds.

The existing bounds, initialization, ordinary ownership/lane uniqueness and
private atomic gates remain executable. Runtime field values drive actual
loads, padding, stores and reducing operations. Non-contiguous tile transfers
and full incoming allocation state survive; conflicting ordinary scatter
refuses. Reducing stores retain their source kind/order/scope under the
existing private-invocation atomic profile.

Real f32 and BF16 tiled GEMM additionally qualifies **two** external descriptor
parameters, original `16x16x32` tiles, the K loop and its original C accumulator.
Named linalg contractions remain inside the source recurrence. Independent
ordered references cover padded and column-major allocations, broadcast reads,
ragged domains, zero K, empty outputs and a rounding-sensitive initial C that
detects reassociation of the contraction. Logical grid mapping stays undecided.

## Evidence and reproduction

The semantic matrix includes indexed gather/scatter with f32/BF16 words,
all six i32 reducing kinds, i64 min and explicitly selected IEEE f32 addition;
rank-one i64 and rank-three BF16 block loads/stores; and the two tiled GEMM
types above. Each library is reused with changed descriptor fields, shapes,
strides, offsets, flags and domains. Independent references check every result
and untouched state word, including canonical fill and transported NaN words.
Capacity failures and ordinary lane conflicts must abort actual compiled calls
without a success receipt. References and their digests stay outside compilation.

Six CLI workflows expose real captured descriptor signatures:

```bash
python -m t2m shared-demo --kernel external-descriptor-gather \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/external-descriptor-gather --python /path/to/native/python
python -m t2m qualify-merlin out/external-descriptor-gather \
  --out out/external-descriptor-gather-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend/python --python /path/to/native/python \
  --merlin-clang /path/to/clang
```

The other kernel choices are `external-descriptor-scatter`,
`external-descriptor-reduce`, `external-descriptor-load`,
`external-descriptor-store` and `external-descriptor-matmul`. The latter retains
runtime ragged `17x19x37` f32 GEMM; the API/tests also qualify BF16. Generated
`spec.json` and the descriptor argument record provide complete public field
names/types/source positions. Generic `shared-tensor --abi` uses that same
interface for other source kernels.

Thirty-five focused checks pass: 18 semantic library cases, six actual CLI
workflows and 11 protocol checks. The semantic libraries replay 100 additional
valid domains and reject 20 real capacity/scatter violations. The relevant
existing memory, model-binding and descriptor suite passes 128 checks. Six
visible packages under `out/external-descriptor-*` each verify 33 artifacts;
their separate `-merlin` qualifications each verify 52.
`out/external-descriptor-coverage/coverage.json` selects 50 packages and 36
standalone qualifications, observing 34/50 definitions and 30 in host-qualified
captures. This is measured composition evidence, not universal coverage.

## Remaining full-coverage requirements

[Descriptor allocation bindings](descriptor_state.md) now extend the field ABI
with nonzero runtime origins, shared writable roots, real derived-model captures
and portable pre-wrapper source custody. Direct logical fields still start at
element zero; the allocation wrapper supplies executable origin arithmetic.
Mixed-type aliases and physical/mutating model views remain open. Neither route
supplies physical GPU descriptor identity.

Broader element/rounding/effect combinations, external synchronization,
physical descriptor byte interfaces and applicable target materialization also
remain open. The logical TF32 descriptor-load field has its own upstream RNE
semantics; the separate dot-precision profiles retain their declared contracts.
Neither these finite execution cases nor the operation observation count proves
universal arbitrary-kernel coverage.

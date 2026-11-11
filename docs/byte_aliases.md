# Byte aliases: implementation and remaining plan

Different-width ordinary views of declared i8 byte buffers now execute through
Merlin. `convert_shared(..., memory_profile="guarded-buffer",
address_policy="source-words", byte_order="little")` selects the explicit
logical byte representation; `big` is also supported. Equal-width typed-root
aliases retain [their existing contract](allocation_word_views.md). The full
TTIR-to-Merlin goal remains open for typed-root byte packing, byte-backed atomic
effects, physical views and applicable target materialization.

## Implemented byte-buffer path

Each source addptr keeps its own pointee size after the original integer
computation is sign-extended to i64. Pointer casts preserve the byte address.
The existing finite-root/control analysis therefore carries bytes for these
roots through source region edges, including signed word advances after casts.
No offset is rounded or divided to invent alignment.

A wide ordinary load/store becomes one byte-plane event with a final static
word-byte dimension. Standard parallel linalg pack/unpack retains the original
word type and explicit byte order. One store plane exposes overlap across both
source lanes and byte positions to the existing uniqueness guard; splitting
each byte into a separate store would miss that conflict. Byte storage keeps
the existing guard protocol at actual byte granularity. Masks, original fills,
whole-tile snapshots, narrow stores and untouched byte state remain executable.

`logical-byte-buffer-word-event` records and `prov.logical_byte_views` retain
the source IDs, word widths, byte order, original tiles, byte tiles and roots.
`prov.byte_word_event` marks the standard word pack/unpack. These are dataflow
and source-intent records, not source-equivalence or target proof. Existing
tensor/scalar ABI, receipts and compiler/evaluator separation remain unchanged;
compatibility metadata explicitly describes the logical representation.

Sixteen actual native/Merlin libraries cover i16/i32/i64/f16/BF16/f32/f64,
both byte orders and scalar/tiled events. They retain source NaN/payload words,
byte and word stores, signed strides/advances, tails and empty domains over
96 additional runtime replays. Forty-eight actual guard invocations reject a
missing last byte, overlapping words and cross-program conflicts. Ten protocol
checks retain default refusals, address-space/root restrictions and indivisible
atomic requirements. This finite matrix does not prove arbitrary-kernel coverage.

## Reuse the existing analyses at an explicit boundary

Meta triton-shared `17f0c234ce474bee334af36e351c62eca7852836` rejects
different-stride pointer bitcasts in `TritonToUnstructuredPass.cpp` and
`bitcast_ptr_unstructured_invalid.mlir`: element offsets cannot simply survive
a change in pointee size. The archived Microsoft conversion is not a newer
implementation of that missing source-word behavior.

Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7` supplies useful existing work:

| Source | Behavior to retain |
|---|---|
| `TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.cpp`, `parseBitcast` | Equal-width offsets pass through; narrowing multiplies offsets; widening requires an exact multiple of the ratio. Divide the complete accumulated offset, preserving proved axis information. |
| `unittest/Analysis/PtrAnalysisBitcastTest.cpp` | Independent byte-address interpretation, scalar cast boundaries, combined offsets, half-width sub-axes and failed divisibility cases. |
| `TritonToUnstructure/UnstructureConversionPass.cpp` | Concrete byte-buffer-to-f16/BF16/f32/i32 preparation and scalar addptr accumulation. Its physical ptr/int conversions are not a portable logical-root representation. |
| `unittest/Conversion/General/TritonToLinalg/pointer_bitcast_byte_buffer.mlir` and corresponding pytest fixture | Byte-buffer tensor offsets, repeated scalar addptr, loop-local casts and paged KV-cache load/store compositions. |

Use the offset/decomposition analysis and these independent fixtures. Do not
copy target pointer casts or physical integer addresses into the frontend.
No Triton-CPU backend or implicit upstream build is required.

## Execution order

1. **Define the logical storage contract.** Keep public owning tensors and
   their original dtypes/shapes. Declare source byte order explicitly for
   width-changing observations. Derive byte capacity with checked extent/size
   multiplication; preserve original whole-root input and output words.
2. **Normalize pointer state in byte units.** Each original addptr contributes
   `sext_i64(source_offset) * sizeof(its source pointee)`. Preserve the arithmetic
   producing that offset before extending it. Pointer casts preserve byte
   addresses; selections and region backedges retain the existing finite-root
   fixed point. Only divide offsets after a complete exact-divisibility proof.
3. **Retain typed memory events.** Represent a whole original word load/store
   with its width, byte origin, mask, fill and root. Give that composite an
   executable standard expansion with integer packing/unpacking. Preserve a
   whole-tile load snapshot before following writes, and source store order.
   Keep original event/tiling provenance so a consumer can recover aligned wide
   operations; byte decomposition must not erase useful vector memory intent.
4. **Extend guards at byte granularity.** Bounds cover every active byte.
   Ownership/initialization distinguish disjoint bytes within one wider root
   element, while overlapping wide/narrow observations retain dependencies.
   A word-sized guard would incorrectly refuse independent byte writes; merely
   disabling it would miss real conflicts. Partial stores preserve other bytes.
5. **Compose effects explicitly.** A wide atomic requires an indivisible
   logical word update even when storage is represented as bytes. Separate
   byte atomics cannot implement it. Preserve returned-old values, scope,
   ordering and mixed ordinary/atomic conflicts through an actual consumer
   implementation. External publication and physical observations retain their
   independent dependency/target requirements.
6. **Finish package/model/target delivery.** Retain original, normalized and
   bound source custody; declare byte-order/representation differences in the
   ABI/compatibility records. Stage only compiler data in Merlin. Qualify real
   model calls, all results and applicable target materialization while keeping
   accelerator mapping downstream.

## Acceptance evidence

Use one compiled implementation over multiple shapes, origins, signed strides
and loop counts. Cover all byte-aligned supported TTIR element widths, widen
and narrow cast chains, partial/unaligned origins, source-word wraparound and
wide offset cancellation, helper calls, branches, pointer recurrences, selected
roots, masked invalid lanes and nonzero fills. Include Ascend's byte-buffer
and paged-cache shapes with independent Python integer byte-address references.
Check exact NaN/payload/uncomputed words, untouched bytes, empty domains,
capacity overflow, overlap conflicts and indivisible atomics at execution.

Acceptance requires original-to-consumer operation accounting and retained
tiling/reuse independently of output agreement. No opaque fallback, provenance
claim, smaller opcode denominator or unexecuted target package proves coverage.

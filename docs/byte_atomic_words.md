# Byte-backed atomic words: verified mechanism and remaining intake work

The standard-MLIR mechanism and Merlin host lowering now execute. **TTIR
byte-atomic admission remains refused:** pointer-control reconstruction, full-word
byte guards and source/payload accounting still need integration. Isolated word
templates now retain and check the byte-address boundary. These component
tests do not increase TTIR opcode coverage or claim a runnable Triton package.

## Executable mechanism

`t2m.byte_atomic_words` emits a standard `memref.view` of private, aligned byte
storage as i16/i32/i64 words. The view aliases the original bytes; it creates
no shadow word buffer. Each active lane performs one
`memref.generic_atomic_rmw`. Integer RMWs and exact compare/exchange return the
original old word. A following ordinary byte store is visible to the next
word event, including partial-word changes and untouched trailing bytes.

A raw word view alone would make source byte order depend on the target.
Instead, a small private literal-byte allocation establishes the standard
view's local byte representation. Pure integer operations decode/re-encode
the word inside the indivisible update region. The probe is executable MLIR,
not a caller annotation, source-data observation or target-name choice. A
consumer may fold it when its data layout establishes the same facts. The
source byte order stays explicit; no host endian constant enters the frontend.

Bounds cover the entire word without overflowing an end-address sum. Alignment
and bounds checks occur only for active lanes. Inactive extreme addresses and
empty storage perform no word read and return the declared zero refinement.
The caller supplies an aligned private allocation and retains whole-event
ownership, initialization and ordinary/atomic conflict checks. The probe uses
normal ownership-based deallocation, with no manual-free bypass.

`prov.byte_word_atomic` retains the operation, width, byte order, byte-address
units and indivisible event intent on both the tensor event and atomic region.
`prov.byte_order_probe` identifies compiler-owned literal transport. These
annotations grant no arithmetic, alias or source-equivalence permission.

## Merlin compiler boundary

`lower_integer_atomic_regions` is a separate default-off native feature. It
validates bounded, live, pure scalar integer bodies and their typed captures
before edits. It retains the complete update in an internal scalar function
and uses an atomic acquire load plus acq_rel/acquire compare/exchange retry.
The existing ownership seam retains/restores both helper bodies and verifies
their current declarations before liveness and LLVM conversion.

Borrowing reconstructs the current complete pointer helper and checks its
actual private, integer-only scalar callee. Names and provenance grant nothing;
changed linkage, visibility, pointer stride, memory ordering or unqualified
effects lose permission. Numeric/source equivalence remains independent of
this no-escape check. The feature shares one seam with existing CAS/word
lowering; selecting legacy CAS keeps its original closed profile.

The native `execute_native` / `observe_native` API requires explicit
`integer_atomic_regions=True` plus the existing Merlin compiler-support path.
Its feature file and adapter report distinguish this path. Portable input
remains standard MLIR. The present native helper requires the qualified default
64-bit host ABI; accelerator materialization retains its own requirements.

## Verified evidence

- 42 native integer RMW/CAS cases cover both declared byte orders, duplicate
  indices, exact old values, masked extreme addresses and intervening byte stores.
- Six actual Merlin libraries cover i16/i32/i64 word storage, both byte orders,
  repeated invocation, all results and immutable source arrays.
- Three runtime refusals cover unaligned, negative and partially out-of-bounds
  words; a separate case retains the explicit native-feature gate.
- Pure codec cases execute both local-representation branches, including every
  i16 word, against independent Python byte encoding. Empty byte storage also
  retains inactive-address behavior.
- Merlin's ownership/LLVM, adversarial borrowing and legacy atomic regressions
  pass independently. This supplies no TTIR/source-equivalence or device proof.

## Checked source template boundary

`word_events.templates(..., byte_order=...)` and `convert_events` now handle
actual byte-address `tt.addptr` followed by integer-word `tt.bitcast` at RMW/CAS
events. Native templates have a word pointer formal while their parent capture
remains an i8 pointer. Source byte addresses stay signed i64. A standard SCF
check asserts divisibility only on active lanes; standard scalar/linalg integer
division then supplies word indices to the unchanged upstream pointer converter.
No memory operation is split into byte effects or hoisted from its source region.

The cast position establishes units from actual IR: an addptr after a word cast
has word units and cannot inherit this byte-address recipe. Bindings separately
retain actual and template types, byte order, word size and both address units.
The parent cast is restored only at substitution. Reconstruction checks the
complete current alignment/division prefix, source value/mask and native event
index, plus the original returned-old-value relation. Native constant pooling
may reorder literal definitions; their full typed values and all remaining SSA
edges still have to match. Caller tags grant no permission.

The source-word v2 component chain retains these actual template bytes and
one consistent explicit byte order. Producer replay regenerates every template
and substitutes the checked native output. Real Triton pointer kernels exercise
both native event and arithmetic stages; altered capture types, divisions,
alignment, masks, indices and old-value returns refuse. This is native source
conversion/replay evidence, not complete byte ownership, source equivalence,
consumer admission or runnable-kernel qualification. The main frontend and the
independent Merlin reader retain their current refusal until the remaining
integration gates below are implemented together.

The focused run passes 57 source-boundary/alignment/replay cases and 100
existing byte-refusal, region-event and word-CAS regressions (157 total), using
the pinned shared optimizer and actual native/Merlin environments. Package
coverage counts remain unchanged.

## Remaining source integration, in order

1. Connect the checked templates to pointer-control/finite-root reconstruction.
   After lowering original pointer state to byte offsets, reconstruct a byte
   addptr and then its word cast. A word addptr receiving a byte offset would
   misrepresent source units even if a later converter happened to compensate.
   Preserve pointer selections, original source integer arithmetic and region state.
2. Extend the independent Merlin source-word reader for this explicit typed
   byte-storage capture boundary. Preserve original primitive/order/scope and
   exact stage custody. Producer replay already supports the component chain;
   current consumer refusal is required until the byte effect relation is checked.
3. Guard every byte of each word as one atomic participant event. Preserve
   initialization, returned-old semantics and ordinary/atomic conflicts.
   Several byte atomics cannot implement one source word atomic.
4. Expand admitted events through this mechanism and account for the actual
   alias, complete update region and old-value decoding. Producer/consumer
   evidence must check the changed storage relation without granting semantics
   from metadata or dropping difficult payloads.
5. Qualify original kernels, selected roots, loops, mixed ordinary/atomic state,
   public CLI packages and complete model captures through the real Merlin
   interface. Floating numeric profiles and applicable targets stay explicit.

# Byte-backed atomic words through the Merlin interface

The explicit `typed-atomics` effect profile with `--byte-order little|big` now
admits integer TTIR atomics on i8 owning storage through complete native and
Merlin host packages. All nine i32/i64 RMW kinds and i16/i32/i64 CAS retain their
word update, original old values, masks and complete returned byte state.
Source conversion, byte guards, executable payload matching and independent
consumer declarations are integrated. Broader compositions, floating numeric
profiles and applicable targets remain completion requirements.

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
The frontend supplies an aligned private allocation and checks whole-event
ownership, initialization and ordinary/atomic conflicts for every touched byte.
The complete-word bound precedes scaling a word index back to bytes. A byte
iteration supplies guard metadata only; the payload remains one word atomic.
The probe uses
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

The raw native `execute_native` / `observe_native` API requires explicit
`integer_atomic_regions=True` plus the existing Merlin compiler-support path.
Its feature file and adapter report distinguish this path. Portable input
remains standard MLIR. The present native helper requires the qualified default
64-bit host ABI; accelerator materialization retains its own requirements.
Package writers and the Merlin compiler worker select this capability from
actual integer atomic regions on standard byte views. The worker composes it
with loop destination reuse at the same checked bufferization seam. No
provenance label selects the update's numerical behavior.

The qualified Merlin branch is `ttir-experiment-grammy` at `8a0ed43b7` or a
compatible successor. Older readers retain their refusal of the additional
byte-word contract; the base integer-region lowering was introduced at `22d9cf702`.

## Verified evidence

The integrated source/payload matrix passes 102 cases: actual source RMW/CAS
kernels, scalar/tiled events, both declared orders, all nine RMW kinds, every
integer CAS width, changing runtime counts and reused Merlin libraries. Actual
runtime refusals cover alignment, negative/partial-word addresses and invalid
unmasked CAS tails. Six additional checks retain strict integer evidence fields,
uninitialized byte refusal and last-byte ordinary/atomic conflicts in both
cross-program orders. Existing component evidence remains useful separately:

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

The source-word v2 chain retains these actual template bytes and
one consistent explicit byte order. Producer replay regenerates every template
and substitutes the checked native output. Real Triton pointer kernels exercise
both native event and arithmetic stages; altered capture types, divisions,
alignment, masks, indices and old-value returns refuse. Pointer normalization
reconstructs a byte addptr before the word cast, preserving original pointee
scales through source arithmetic. Producer replay and the independent Merlin
reader bind actual i8 captures, isolated word formals, source primitive/order/
scope, logical byte order and original-word/per-byte guard tiles.

The earlier focused run passes 57 source-boundary/alignment/replay cases and 100
existing byte-refusal, region-event and word-CAS regressions (157 total), using
the pinned shared optimizer and actual native/Merlin environments. Package
coverage counts did not change from those component tests alone.

## Current payload verification

`byte_atomic_match` rebuilds the executable recipe and compares its effect/
control skeleton and typed integer expression DAGs against the current payload.
Only literal folding and exact neutral integer identities are normalized.
Word alias, bounds/alignment assertions, active branch, complete update and
old-word decoder must match; dead extra arithmetic and unknown effects refuse.
The endian flag must arise from the actual private literal allocation, stores,
word view/load and representation assertion. Tags may be removed without changing
permission; altered tags cannot authorize a different body.

The checked byte import uses standard zero-offset, unit-stride reinterpretation
with the exact guard-ABI extent. This avoids upstream LLVM's
`ViewOpMemrefCastFolder`, which folds an unranked-to-ranked `memref.cast` into
`memref.view` without preserving its ranked source requirement. Native per-pass
verification localized that assertion in the pinned `triton-ptr-to-memref`
canonicalizer. No shadow storage, unchecked physical pointer or toolchain build
is introduced. Normal ownership/deallocation remains mandatory.

The independent reader checks declarations and byte custody. Recipe checks,
runtime guards and measured libraries are separate evidence; universal source
equivalence and `source_closure_verified` are not established by these records.

## Public saved-source workflow

The visible i64 unsigned-max demo uses big logical byte order, collisions, a
masked tail and untouched padding. Its source/ABI/independent arrays are at
`out/byte-atomic-rmw-inputs/`; the public commands produce a source package and
a separate Merlin qualification:

```bash
python -m t2m shared-tensor out/byte-atomic-rmw-inputs/source.ttir \
  --abi out/byte-atomic-rmw-inputs/abi.json --out out/byte-atomic-rmw \
  --memory-profile guarded-buffer --address-policy source-words \
  --effect-profile typed-atomics --byte-order big \
  --inputs out/byte-atomic-rmw-inputs/inputs.npz \
  --reference out/byte-atomic-rmw-inputs/reference.npz \
  --optimizer "$TRITON_SHARED_OPT" --python "$T2M_NATIVE_PYTHON"
python -m t2m qualify-merlin out/byte-atomic-rmw --out out/byte-atomic-rmw-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python "$T2M_MERLIN_PYTHON" \
  --python "$T2M_NATIVE_PYTHON" --merlin-clang "$T2M_MERLIN_CLANG"
```

These generated development artifacts verify 20/45 members. They are ignored
outputs; source kernels and reproducible integration tests are tracked.

## Remaining full-scope work

Qualify byte-word events with finite root selection, source for/while state,
mixed ordinary updates, typed owning-root bindings and complete model captures.
Those generic paths are connected, but their broader compositions require their
own execution evidence. Floating byte-backed arithmetic needs explicit numeric
policies. External participants, synchronization/publication, physical model
views, complete corpus coverage and applicable accelerator materialization
remain open in the full [completion plan](completion_plan.md).

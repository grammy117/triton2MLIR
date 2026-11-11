# Logical pointer word views

The guarded-buffer/source-words path supports byte-aligned, equal-bit-width
`tt.bitcast` pointer views for ordinary loads and stores. Public tensors keep
their original element type. Memory events use that storage type, with
`arith.bitcast` on loaded/stored values. Source offsets remain element offsets;
there is no physical alias, address-space selection, whole-buffer representation
copy or new public ABI. Tiling, finite base selection, masks, source fills and
for/while pointer recurrences remain executable.

## Source and transformation boundary

Meta triton-shared `17f0c234ce474bee334af36e351c62eca7852836`,
`lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp`, propagates
base/offset state through pointer bitcasts when storage strides agree.
`test/Conversion/TritonToUnstructured/bitcast_ptr_unstructured.mlir` covers
chains and loop state; the invalid fixture rejects differing strides. Our
existing finite-root pointer analysis supplies the region edges and reuses
upstream conversion at the original memory events.

This extension requires equal **bit** widths and byte-aligned scalar types.
That is stricter than equal storage bytes: i1/i8 representations require further
padding-bit semantics. Differing element widths need explicit address rescaling,
byte-range bounds and overlap/alias reasoning. Explicit address-space changes
remain located refusals. Selected roots retain their own equal-width storage
types at each original memory event.
The pinned unregistered TT pointer binding exposes its type parameter body;
that parameter is parsed with the existing typed parser and checked against the
supported scalar ABI. No operation semantics are recovered from printed IR lines.

The analysis first validates all pointer edges and memory events. Pointer
bitcasts become identity edges in the signed i64 offset graph. Reconstruction
uses each actual source root's storage type. A load then bitcasts its result
into the source view type; a store bitcasts its value into storage. Existing
guards still check actual bounds, ownership, initialization and store uniqueness.
Selection branches at the original event; pointer recurrences retain their
original loop order, conditions and zero-trip behavior.

## Fills and provenance

A source integer fill can encode a signaling NaN in floating storage. Bitcasting
that constant into a floating fill before upstream folding can lose its payload
or quiet bit. Instead the storage load uses a typed zero fill; after bitcasting
the loaded value, a source-mask select restores the original integer fill bits.
This does not read inactive memory or change a defined source fill.

`logical-pointer-word-view` transformation records bind the source event, root
positions, storage/view dtypes and element width. `prov.logical_word_views`
retains that intent on the exported function and corresponding scalar bitcasts.
The exact source inventory and the standard executable operations remain in the
package. These tags supply traceability, not new aliasing or rewrite permission.
Compatibility metadata explicitly records the logical view and fill expansion.
The existing mixed standard-memory payload and public tensor/scalar interface
are unchanged; this is not a claim of pure tensor memory or accelerator execution.

## Qualification and atomic extension

[Explicit allocation bindings](allocation_word_views.md) now connect distinct
typed source arguments to the same root before inlining, including integer and
floating descriptor parameters. Runtime origins, shared state and all model
results survive the same register-view conversion and source custody checks.

Native and actual Merlin tests compare raw words for f16/i16, f32/i32 and
f64/i64 views. They cover scalar storage of arbitrary words, NaNs, signed zeros,
subnormals, masked invalid tail addresses, nonzero bit-pattern fills, finite
per-lane root selection, casts before/after selection, for/while recurrences,
zero trips, dynamic empty shapes and preserved incoming output rows. Merlin
reuses each compiled library across inputs/shapes. Changed-width casts remain
source-located refusals.

[Atomic word RMWs](atomic_memory.md#integer-word-updates-and-floating-minmax)
now have that separate standard region expansion on f32/f64 storage. All nine
i32/i64 operations and Triton's floating min/max retain the current word, exact
integer update, masks and returned old values through native and Merlin execution.
Ordinary and atomic uses keep event-specific types even when they share a source
pointer. Atomic views use the explicitly selected typed v4 profile and checked
native legalization. [Word CAS](atomic_memory.md#integer-word-compareexchange-on-floating-storage)
additionally qualifies i16/i32/i64 comparisons on f16/f32/f64 storage, preserving
old bits through standard regions and integer native helper results.
Ordinary [mixed storage selectors](mixed_storage.md) now retain each root's
actual equal-width dtype. [Selected integer atomics](mixed_storage.md#selected-integer-atomics)
now also retain integer/floating roots under the typed profile. Changed-width
views and broader numeric/effect contracts remain further extensions;
provenance alone grants none of them.

Measured delivery: 59 new/existing pointer cases passed in
`.scratch/logical-word-views-final.log`; the final standalone view suite passes
22 cases including scalar programs, changed-width refusals and an explicit
physical-address observation refusal. Eight focused existing atomic cases also
pass. `out/word-views/` contains the verified 20-artifact package and
`out/word-views-merlin/` contains the verified 45-artifact Merlin qualification.
The visible Merlin output additionally matches the independent reference as raw
i32 words. These are bounded source/native/host checks, not universal TTIR or
accelerator coverage.

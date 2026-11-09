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
and selected roots with different storage element types remain located refusals.
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

## Qualification and remaining atomic work

Native and actual Merlin tests compare raw words for f16/i16, f32/i32 and
f64/i64 views. They cover scalar storage of arbitrary words, NaNs, signed zeros,
subnormals, masked invalid tail addresses, nonzero bit-pattern fills, finite
per-lane root selection, casts before/after selection, for/while recurrences,
zero trips, dynamic empty shapes and preserved incoming output rows. Merlin
reuses each compiled library across inputs/shapes. Changed-width casts remain
source-located refusals.

Atomic views require a separate standard atomic-region expansion. Triton's
floating min/max algorithm uses signed/unsigned integer RMWs on these views;
rewriting those operations as floating min/max is not an established equivalence.
Retain the original integer update on the current word of the floating buffer,
then bitcast the updated word back inside the same atomic region. Expand only
the event, preserve its returned old word, and legalize it in the native compiler
without changing the public buffer type. Integer RMW/CAS and floating exchange
continue through their existing qualified profiles. See [atomic memory](atomic_memory.md).

Measured delivery: 59 new/existing pointer cases passed in
`.scratch/logical-word-views-final.log`; the final standalone view suite passes
22 cases including scalar programs, changed-width refusals and an explicit
physical-address observation refusal. Eight focused existing atomic cases also
pass. `out/word-views/` contains the verified 20-artifact package and
`out/word-views-merlin/` contains the verified 45-artifact Merlin qualification.
The visible Merlin output additionally matches the independent reference as raw
i32 words. These are bounded source/native/host checks, not universal TTIR or
accelerator coverage.

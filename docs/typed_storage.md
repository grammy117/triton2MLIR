# Typed owning tensors with byte views

`convert_shared(..., memory_profile="guarded-buffer",
address_policy="source-words", byte_order="little")` now preserves a typed
owning tensor while source pointers observe narrower or wider words. `big`
selects the other explicit logical byte order. The public tensor shapes and
dtypes, runtime scalars, grid and functional result ordering stay unchanged.

The frontend reuses the finite-root pointer/control analysis to select roots
that actually require width-changing storage, including connected alternatives
of a selected pointer. Unrelated ordinary roots keep their existing path.
For a selected root of shape `S` and word size `W`, internal storage is an
`i8` tensor of shape `S + [W]`. The appended static byte axis preserves every
public dimension without adding a runtime scalar or specializing an extent.

The actual source entry receives a byte pointer and casts back to its original
pointer type before its original uses. Each original `addptr` therefore keeps
its source pointee size and source integer arithmetic. Existing byte-plane
events, masks, fills, byte bounds, ownership and initialization checks remain
executable. This is a logical storage convention, not physical pointer identity.

Standard parallel linalg regions unpack inputs and pack returned roots using
integer bitcasts, shifts and bitwise operations. No floating conversion or
arithmetic occurs in transport. Untouched signaling NaNs, signed zeros and
other word patterns survive. A partial byte store preserves all other bytes.
Original memory-word/event provenance remains available to a downstream
optimizer; whole-root transport is the present correctness path, and its
elimination or recovery of aligned wide accesses needs its own legality checks.

## Package and Merlin contract

`typed_storage_transport` in provenance and `prov.typed_storage_transport` on
the function declare `t2m.typed-storage-transport.v1`. Each root names its
public dtype/shape, byte dtype/shape, word size and effect. Records are in
public buffer order, with explicit byte order and `grants_semantics=false`.
The public `spec.json` remains typed; memory admission binds the actual private
storage ABI. Original source inventories and source custody remain distinct
from this executable storage adaptation.

Guard metadata is allocated per storage byte. Consequently its per-public-result
cost is the existing admission cost multiplied by `W` for a transported root,
and the existing cost for other results. The function carries exact
`prov.memory_guard_bytes_by_result_element` plus a conservative scalar maximum.
Merlin independently derives storage shapes and these costs, checks the exact
function/profile declarations, and stages per-result costs in its signature.
Bound models derive total scratch from the actual public call shapes. Native
and Merlin evaluators budget that storage before allocating result arrays.
The declarations grant custody and budget consistency, not source equivalence,
alias permission, atomic implementation or target qualification.

## Evidence and public workflow

Fourteen native/Merlin libraries qualify i16/i32/i64/f16/BF16/f32/f64 in both
byte orders, plus 70 reused-library invocations covering signed source strides,
pointer advances, masks, empty roots and zero trips. Additional BF16/f64 matrix
cases retain runtime public axes. Separate programs can update disjoint bytes
within one public word. A real model2MLIR i32-root capture retains three calls,
learned state and all five results through API and CLI binding and actual Merlin
host execution. These are measured compositions, not universal memory coverage.

```bash
python -m t2m shared-demo --kernel typed-root-byte-buffer \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/typed-root-byte-buffer --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/typed-root-byte-buffer \
  --out out/typed-root-byte-buffer-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
```

`typed-root-byte-buffer-big` selects big byte order. Saved-source conversion
uses `shared-tensor --byte-order little|big`. The model capture script
`scripts/capture_byte_buffer_model.py --typed-root --out ...` supplies the
matching real typed-parent example; its GPU path remains unqualified.

Visible artifacts are `out/typed-root-byte-buffer` (33 verified artifacts),
its separate `-merlin` qualification (52), the generic saved-source package
`out/typed-root-byte-buffer-generic` (20), and the copied verified complete
model `out/typed-root-byte-model` (80). The refreshed explicit registry at
`out/typed-storage-coverage/coverage.json` contains 60 packages and 41 standalone
qualifications. It still observes 34/50 TTIR definitions, with 30 represented
in host-qualified captures. This is deeper storage/composition coverage, not
new universal opcode or accelerator coverage.

Changed-width [pointer and descriptor parameter bindings](typed_parameters.md)
now use the same executable transport. Byte-backed indivisible atomics,
physical/mutating model views, independent numerical/effect closure and
applicable target materialization remain separate requirements.
Default conversion still requires the existing width contract unless explicit
byte order is supplied. The full arbitrary-TTIR goal remains open.

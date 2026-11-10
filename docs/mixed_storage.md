# Selected buffers with different storage types

The guarded source-word path now retains ordinary loads/stores through a finite
selection of equal-width integer and floating buffers. Each public tensor keeps
its own storage dtype: f16/i16, f32/i32 and f64/i64 qualify. Pointer selection
continues to carry a logical buffer identity and an element offset. A common
source pointer view does not force every root into the same storage type.

## Memory and dataflow

Validate every root before rewriting an event. At each selected ordinary event,
construct a predicated access using that root's actual storage pointer type.
Bitcast the loaded register value into the original logical type, or bitcast a
stored logical value into storage. Retain original event order, including reads
after selected writes. No whole-buffer representation copy is added.

Inactive loads use a storage zero internally; the original fill is selected
after restoring the logical value. This preserves integer fills that encode
signaling NaNs in floating storage. Bounds, initialization and ownership guards
use each actual root. Dormant invalid pointers remain dormant; activating the
same pointer triggers the existing bounds check. Changed element widths and
physical-address observations remain refused.

The reuse boundary remains Meta `17f0c234ce474bee334af36e351c62eca7852836`:
`TritonToUnstructuredPass.cpp` propagates equal-storage-byte pointer offsets,
with `bitcast_ptr_unstructured.mlir` fixtures. Its structured branch analysis
and `if_returning_mixed_structured_ptrs.mlir` separate bases from offsets.
The local adapter keeps the functional tensor ABI and actual storage dtypes,
then reuses the native memory conversion and Merlin lowering.

## Immutable selector permutations

Loop-carried pointer swaps can permute immutable per-lane selector tiles.
Carrying those tiles as mutable native buffers introduces unnecessary copying
and difficult ownership cycles. The frontend now recognizes closed forwarding
cycles through addptr/bitcast. It retains the original initial selector tiles
and carries scalar i32 table indices through the existing for/while loop.
Each use decodes the original tile selected by that index. Offsets retain
their original source-word arithmetic and recurrences.

This representation preserves zero trips, both directions of three-way cycles,
final pointers, runtime choices and per-lane root choices. It selects no physical
memory layout or execution hierarchy. Computed backedges and modified while
condition state do not inherit this rule. General dynamic ownership remains a
separate compiler frontier; no lifetime check was weakened or disabled.

## Package and evidence

`logical-pointer-word-view` reports identify each reinterpreted root and storage
dtype. `logical-selector-permutation` identifies original loop state positions,
immutable selector tiles and the scalar index representation. Source graphs,
loop operations and register bitcasts retain provenance. Compatibility metadata
describes both transformations. Artifact names, stage-v2 custody and the public
tensor/scalar ABI are unchanged.

The focused suite qualifies 52 cases: 12 read-only compositions, 24 selected
update compositions, 12 three-way cycle compositions and four negative/bounds
cases. It covers all three widths, scalar/vector events, uniform/per-lane
selection, for/while loops, read-after-write observations, empty/tail domains,
inactive state and final pointers. Actual Merlin libraries are reused over
independent choices/counts/shapes with raw-word comparisons and preserved
functional inputs. Another 75 existing pointer/event regressions pass.

Three genuine model2MLIR captures additionally retain actual learned weights,
initialized output state and raw results through native and Merlin execution.
Run `scripts/capture_mixed_view_model.py --width 16` (32/64 also work) with
`--out` to capture one; `tests/test_mixed_view_model.py` shows the checked binding.
Source references use independent PyTorch integer operations, without a Triton
CPU backend. These ordinary-memory cases do not establish universal alias/layout
or accelerator coverage. Integer atomics have the additional qualification below.

Visible evidence: `out/mixed-storage/` verifies all 20 package artifacts;
`out/mixed-storage-merlin/` verifies all 45 qualification artifacts and actual
host execution. All five results also pass explicit raw-word comparisons.
`out/mixed-view-model32/model/` verifies all 79 model artifacts, including
unchanged weights and native/Merlin raw outputs. Generated artifacts are ignored
by Git; compiler sources, tests and contracts are versioned.

## Selected integer atomics

The explicit `typed-atomics` profile now also admits integer primitives whose
logical pointer selects ordinary integer storage or an equal-width floating
word view. All nine RMW operations qualify for i32/f32 and i64/f64; CAS qualifies
for i16/f16, i32/f32 and i64/f64. Each root retains its public storage dtype.
The existing standard integer atomic or floating-storage atomic region executes
the source operation, including original old words. Floating representations
are bit data here; no floating arithmetic or rounding policy is inferred.

Selected roots share the original source event and complete inlined call
context. Their stage bindings retain different actual storage captures.
Merlin now matches both context and captured storage dtype before checking
word-view agreement. A CAS template's integer pointer formal remains distinct
from its floating storage capture. This fixes the previous refusal caused by
comparing both roots' view declarations as if they described one storage type.
Primitive operation/order/scope checks and complete producer reconstruction
remain mandatory; this custody check is not a source-equivalence proof.

Artifact/stage schemas, public tensors/scalars and the private-invocation
contract remain unchanged. Compatibility metadata explicitly records the
mixed-storage atomic representation. Native legalization still uses checked
typed helpers under the default 64-bit descriptor/index ABI; portable MLIR
contains standard executable regions. Bounds, initialization, masks, original
source regions and native lifetime checks remain intact.

The focused suite passes 76 execution compositions plus six explicit-profile
refusals. It covers all nine RMW operations, loaded indices, uniform/per-lane
selection, scalar/vector events, for/while recurrences, final pointers and
repeated helpers. Four collision cases check independent complete word histories
for both roots, including signaling NaNs and wraparound, without assuming lane
order. Five real model2MLIR captures preserve integer/floating learned weights
and all three tuple results with exact native/Merlin word comparisons. Eighteen
existing numeric/profile executions also pass. The consumer fix at
`83da7091b7b0fbbabb513f8cd9943076ed11b987` passes 90 new declaration checks,
200 existing staging checks and documentation/structure gates.

Use `scripts/capture_mixed_atomic_model.py --kind cas --width 16 --out ...`
or `--kind rmw --width 32`; CAS also supports 32/64 and RMW supports 64. The
checked three-result binding is in `tests/test_mixed_atomic_model.py`.
Numeric floating operations through integer storage, wider formats, external
participants, synchronization and accelerator materialization remain separate
implementation and qualification requirements.

Visible mixed-atomic evidence: `out/mixed-atomic-cas16/` and
`out/mixed-atomic-rmw64/` each verify 20 artifacts, with separate `-merlin/`
qualifications verifying 45 artifacts and exact host results. The corresponding
`-model/model/` packages each verify 80 artifacts, unchanged weights and all
three exact native/Merlin results. The RMW kernel example retains unsigned
maximum with loaded indices; the model example retains integer XOR.

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
CPU backend. These cases do not establish universal alias/layout, mixed-storage
atomic or accelerator coverage.

Visible evidence: `out/mixed-storage/` verifies all 20 package artifacts;
`out/mixed-storage-merlin/` verifies all 45 qualification artifacts and actual
host execution. All five results also pass explicit raw-word comparisons.
`out/mixed-view-model32/model/` verifies all 79 model artifacts, including
unchanged weights and native/Merlin raw outputs. Generated artifacts are ignored
by Git; compiler sources, tests and contracts are versioned.

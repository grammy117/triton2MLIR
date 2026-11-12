# IEEE f16 and BF16 atomic addition through Merlin

The existing `guarded-buffer`, `source-words`, `typed-atomics` path now admits
`tt.atomic_rmw fadd` on f16 and BF16 storage. Both formats keep their original
two-byte tensors, masks, indices, logical parallel events, returned old values
and source order/scope annotations. This closes a missing numeric/effect family;
arbitrary numeric/effect compositions remain completion work.

## Executable contract

| Format | Numeric contract | Portable update |
|---|---|---|
| f16 | `ieee-f16-rne-add-quiet-nans` | f16 `arith.addf`, then explicit integer NaN classification/quieting. |
| BF16 | `ieee-bf16-rne-add-quiet-nans` | Exact f32 extensions, `arith.addf`, one BF16 truncation per event, then NaN quieting. |

Each update rounds nearest even and preserves input/result subnormals. Computed
NaNs are quiet with unspecified payload/sign. Old results and untouched state
retain original words, including signaling NaNs and signed zero. Inactive
undefined old values retain the existing typed-zero refinement. Numeric tests
use the independent exact-rational `float_cast_reference.add_word`, comparing
finite/zero/infinity words exactly and classifying only computed NaNs.

BF16 widening never defers rounding across events or a source recurrence.
Nearby BF16 operands have a significant sum that fits in f32; a farther operand
lost by f32 rounding cannot move a BF16-representable larger operand across its
BF16 midpoint. Narrow after each addition, as in Merlin's existing BF16 scalar
addition emulation. These are software contracts of the selected typed profile.
The [PTX atomic specification](https://docs.nvidia.com/cuda/archive/13.2.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
also specifies nearest-even and `.noftz` for scalar f16/BF16 addition; this does
not qualify an unspecified backend or GPU binary. F32 retains its separate
explicit IEEE/FTZ selection.

## Reuse and consumer implementation

Event/mask/parallel old-value conversion reuses Meta's
[`UnstructuredToMemref` implementation](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp)
and its [atomic fixtures](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/UnstructuredToMemref/atomic_rmw.mlir).
The older Microsoft direct converter has no half numeric-policy implementation
to supply this contract. The [reuse map](triton_shared.md) records versions.

The adapter refines each admitted atomic event into an executable standard
`memref.generic_atomic_rmw` region. Parallel tile structure and source indices
remain intact. Merlin's existing `lower_atomic_word_views` seam recognizes the
complete arithmetic/quieting region, actual format masks, captured operand,
neutral flags and default/explicit nearest-even rounding. Its internal
atomic-load/CAS retry helper retains its complete body through ownership checks
and restores it before liveness and LLVM conversion. Increment/old-result call
transport is i16; typed memrefs retain their original element format. Borrowing
reconstructs the entire current helper. Names and provenance grant no permission.

RISC-V linking exposed a missing generic `__truncsfhf2` symbol. Merlin now
supplies the pinned LLVM compiler-rt binary32-to-binary16 specialization alongside
its existing half extension. Integer rounding handles subnormal, midpoint,
overflow and NaN cases without recursive half conversion. Tests check every
positive/negative adjacent-half rounding boundary, all 65,536 original half words
and the typed helper ABI. Portable MLIR still contains standard regions and no
opaque native helper. A consumer can select native floating atomics when their
numeric, ordering and old-value semantics satisfy this contract.

## Package interface and evidence

Public tensor/scalar ABIs, receipts and compiler artifact rosters are unchanged.
V4 atomic declarations bind actual storage, exact integer widths, format-specific
contracts and payload recipes independently in producer verification and Merlin
staging. Older readers lacking these contracts refuse. Generated compatibility
metadata records the extension. BF16 archives keep their existing raw-word
representation and model word witnesses.

Four scalar/tensor f16/BF16 libraries pass native, Merlin host and actual Spike.
Sixteen additional host invocations reuse unchanged libraries over empty,
singleton and ragged domains with rotated special values. Native collisions
also check complete old-value histories without imposing lane order. Two real
model2MLIR captures preserve learned state, dependent calls, an empty-domain call,
the original snapshot and all six results through native/Merlin host/Spike.
All twelve model results match exact words.

`out/half-atomics/summary.json` reopens six target receipts and twenty complete
results, plus a public original-source CLI capture. The refreshed
`out/half-atomic-inventory/coverage.json` reopens 129 packages, 94 standalone
qualifications and fourteen product cases, retaining all fifty observed names
under selected profiles. This is not universal type/attribute/kernel coverage.
The selected frontend native/evidence regressions pass 110 checks; Merlin's
atomic/BF16/runtime suite passes 269. Structure/docs gates pass separately.

With existing configured tools:

```bash
python -m pytest -q tests/test_half_atomic_add.py tests/test_half_atomic_model.py \
  --basetemp=.scratch/pytest-half-atomics
python scripts/capture_half_atomic_model.py --dtype bf16 --out out/half-model-parent
```

The capture script runs in model2MLIR's environment. Frontend tests use the
supported Triton/shared/native/Merlin environment variables. Public `shared-tensor`
also accepts original TTIR, its ABI and independent input/effect-reference
archives with the three profiles above.

External participants/publication, physical synchronization, half exchange/
min/max, broader alias/region/model combinations and accelerator correctness/
performance require further implementations and qualification. An annotation
cannot replace these behaviors.

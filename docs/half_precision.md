# Half tensor storage through the Merlin interface

The existing `f16` tensor ABI now executes through the native runner and actual
Merlin host libraries in both memory profiles. Tensor storage remains IEEE half:
NPZ arrays use NumPy `float16`, tensor arguments/results retain `f16`, and native
descriptors address two-byte elements. Model manifest materialization also
preserves float16 inputs and safetensors weights without widening their storage.
The materialization fixture is a protocol check, not a fabricated FX capture.

A real model2MLIR selected-pointer model now also qualifies f16 learned weights,
inputs, two kernel calls and surrounding model arithmetic through actual Merlin
compilation/execution. Its parent weights, inputs and golden bytes remain intact.
This exposed and fixed a separate size-budget gate in the bound-model writer:
f16 arguments now count as two bytes instead of being rejected before the
already-supported manifest loader. The two real-model dtype cases and existing
half regression pass all nine tests. `capture_selected_memory_model.py --dtype
f16` supplies the genuine FX capture; the earlier protocol fixture remains a
separate materialization check.

The installed torch-mlir runtime already supports half memref descriptors.
The missing boundary was the runners' dtype tables. Connecting those tables
does not rewrite arithmetic, change the public signature or add a target
backend. The existing generic HostModel memref ABI needs no half scalar punning.
By-value `f16` scalars remain unsupported; their C calling convention is a
separate contract.

Triton's typed promotions and narrowing operations remain executable. A real
half-input/half-output affine kernel carries f32 scalar operands and f32
intermediates before its f16 store conversion. One Merlin library handles
changing scalar values. Native and actual Merlin checks cover halfway ties to
even, signed zero, minimum half subnormals, finite overflow to infinity, and NaN
classification. Bit comparisons cover non-NaN results; no universal computed-NaN
payload preservation is claimed.

Mixed contractions retain half A/B operands and the original nonzero f32
accumulator. Integer-valued test data makes every product/sum exact, so the
independent NumPy reference does not rely on a particular backend reduction
tree. The existing source-accumulator repair still runs before either memory
continuation; half storage does not grant a different contraction policy.

Half masked matrix kernels additionally retain runtime dimensions, transposed
storage/traversal and inactive values through the same library over empty and
ragged shapes. These cases check inactive signed-zero and NaN payload bits
without applying floating arithmetic to the preserved state.

No receipt or tensor ABI schema changes are needed. `f16` was already a declared
tensor element type; this adds execution evidence and the missing runner path.
BF16 now has a separate [raw-word storage/cast profile](bfloat_precision.md).
Complete BF16 model archives, fp8 storage, alternative rounding profiles, by-value
half scalars and universal numeric equivalence remain open. A dtype appearing
in this profile does not certify every operation or attribute combination.

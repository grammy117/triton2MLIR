# Allocation roots inside captured model tensor values

`KernelBinding.tensor_views="dense-row-major-v1"` explicitly permits a model
tensor value to supply a differently ranked kernel allocation root with the
same element type, count and logical row-major order. This joins the
[allocation-state interface](allocation_state.md) to actual model2MLIR calls:
a matrix can supply a flat root, and an updated root can return as a matrix.
The source tiling, reuse, recurrence and memory guards remain executable.

## Shape bridge and provenance

The binder resolves the kernel's concrete extents from captured typed scalar
attributes and checks the caller's static element count and type. Kernel
dimensions and scalars still satisfy the declared ABI bounds. Equal count permits standard
`tensor.collapse_shape`/`tensor.expand_shape` reassociation. Dynamic kernel
types receive `tensor.cast` only after this concrete check. Result bridges
perform the inverse steps, keeping every source result and its declared order.

Rank-zero values have one element; zero extents have zero elements. The
rank-zero input bridge uses `tensor.extract` and `tensor.from_elements` because
xDSL's expand-shape verifier rejects upstream's valid empty reassociation.
These are standard operations with the same unit-value semantics. No numeric
conversion changes the element words. The implementation follows LLVM
`a47bddcc` TensorOps.cpp and its `tensor/ops.mlir` reshape fixtures.

This is **logical tensor value** reshaping. It supplies no physical stride,
storage-offset, alias, ownership, mutation or target-layout permission.
Encodings, dynamic caller dimensions, unequal counts and changed element types
refuse. Without the explicit policy, the existing exact-shape binding applies.
Physical model views and mutating operator schemas remain separate work.

Adapter operations have their own callsite provenance and binding roles.
Original FX bytes remain under `parent/`; each kernel retains its TTIR,
source inventory and standard MLIR body. Reshapes do not impersonate source
ATen or TTIR operations. The public model signature, parameters, inputs and
all references remain unchanged from the parent capture.

## Explicit binding version and independent consumer checks

Selecting this policy emits `t2m.model-kernel-binding.v2`, with
`tensor_view_binding_profile: t2m.captured-dense-tensor-views.v1` and per-call
view records. Legacy bindings retain v1. Old Merlin readers refuse v2 instead
of ignoring the additional shape contract. Package verification rejects unknown
profiles, schema downgrades and inconsistent payload declarations.

Merlin independently reopens the original parent call by ordinal. It checks
identity, schema operands/results, captured scalar types/ranges, unencoded
static caller types, exact element types/counts and complete reshape records.
This also applies to static kernels without runtime scalars. Guarded scratch
is derived from the actual kernel dimensions; empty calls contribute zero.
Evaluator inputs, weights-as-example-shapes, goldens and their digests supply
none of these facts. Staging establishes custody/shape consistency; numeric,
effect, ownership, lifetime and target gates remain separate.

The CLI accepts `tensor_views` and optional `allocation_bindings` in each
`t2m.kernel-bindings.v1` configuration item. The latter uses the strict
[source argument configuration](allocation_state.md) and may be inline or a
relative JSON filename. Identical source bytes, ABI, lowering options and source-allocation bindings
reuse one conversion across captured scalar specializations. Per-call scalar
values remain binding facts. This avoids separately regenerated wrapper source
identities for one implementation and reduces compiler work; a changed source
file refuses before a cached implementation can be published under its old pin.

## Qualified model compositions

Four real model2MLIR captures now pass native and actual Merlin execution:

- i64 state updates retain three source pointers into one allocation, partial
  overlaps, a carried loop, multiple calls and an empty result. Values above
  `2^53` remain exact; returning an earlier state catches accidental mutation.
- f32/f16/BF16 GEMM captures flatten matrix inputs/parameters into allocation
  roots and return shared C/output state as matrices. Each model uses one
  implementation for nonzero and zero K, retaining the source K loop,
  16×16×32 tiling and named contractions. Sensitive C and all tuple results
  match independent ordered CPU references word for word.

Both the API and public CLI preserve parent weights, input archives, original
references and BF16 word witnesses. The visible BF16 CLI model records one
kernel source for its two captured calls. Source argument binding has the
explicit bound-program/public-root signature difference documented separately;
pre-wrapper source custody is not strengthened by this reshape profile.

The 28 direct view checks cover i64/f32/BF16 word transport across matrix,
flattened, scalar and empty shapes, plus type/count/encoding/dynamic refusals.
Four model checks cover the API, CLI, explicit opt-in and rehashed profile
refusals. All 32 focused cases and 71 legacy model/binding regressions pass.
Merlin's 267 staging checks include independent rehashed type/count/profile
refusals and evaluator read barriers.

The visible BF16 parent and bound model verify 11/81 artifacts. The explicit
report at `out/allocation-model-view-coverage/coverage.json` selects 41 packages;
31/50 TTIR definitions remain observed, with 27 in host-qualified captures.
Those observations do not establish universal opcode/type/attribute coverage.

## Reproduce the BF16 model

First produce the standalone source/spec as in [allocation state](allocation_state.md).
Use the same existing shared optimizer and compatible compiler environments:

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel allocation-state-bfloat-dot \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/allocation-state-bfloat --python /path/to/native-python
/path/to/m2m-python scripts/capture_allocation_state_model.py \
  --kind bfloat-dot --out out/allocation-view-bfloat-model/capture
python -m t2m bind-model out/allocation-view-bfloat-model/capture \
  --bindings examples/allocation_state_bfloat_model_binding.json \
  --out out/allocation-view-bfloat-model/bound --python /path/to/native-python \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/frontend-python \
  --merlin-clang /path/to/compatible-clang --merlin-host
```

The example paths are relative to the configuration file. The capture script
also accepts `updates`, `dot` and `half-dot`. Their API qualification is covered
by the model tests. This closes a dense logical-value bridge; mixed-type aliases,
physical/non-dense model views, mutation, broader numeric/effect profiles and
applicable targets remain in the [completion plan](completion_plan.md).

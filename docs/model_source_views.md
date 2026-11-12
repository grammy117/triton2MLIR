# Source tensor views bound to owning model state

`KernelBinding.source_views` binds an extra captured model tensor operand to
an existing kernel allocation root after recovering its element coordinates
from the original executable model SSA. This connects transposed and stepped
model views to the allocation-state frontend while retaining the complete root,
untouched words, source tiling, runtime loops and memory guards.

## Executable meaning and provenance

There are two separate requirements:

1. The parent model must actually compute the view from the declared root.
   Recover static coordinates through canonical `linalg.transpose`, positive
   rank-preserving `tensor.extract_slice` and identical-type `tensor.cast`.
   Transpose regions must yield the original input element without arithmetic.
2. The allocation-bound TTIR must actually pass that root plus the declared
   element offset at the configured original pointer argument. Captured integer
   kernel arguments must equal the recovered shape, strides and offset.

Logical tensor equality does not authorize physical aliasing. Sharing inside
the kernel comes from its executable allocation-bound pointers and state.
The enclosing model remains functional: each updated owning root is a new
result, and earlier roots/views remain observable snapshots. No public physical
model alias, mutation or layout permission follows from a provenance tag.

Supported paths are static, unencoded, same-element-type tensor values of
rank at most eight, with at most 32 view steps. Empty slices retain their real
shape/offset; they do not fabricate data or execute inactive programs. Dynamic
coordinates, rank-reducing slices, encoded tensors, word-changing bodies,
unrelated copies/roots and negative source slice steps need further explicit
implementations. Existing standalone signed-stride support retains its separate
scope. Direct mutating custom-op schemas and general physical model-view ABIs
remain open.

Each kernel's original TTIR, allocation capsule, source inventory and executable
standard body remain present. Outer callsite operations additionally retain
`prov.kernel_source_view_profile` and `prov.kernel_view_arguments`. The binding
record retains the original root/view types and exact recovery steps; these
records explain the derivation and grant no additional semantics.

## Explicit interface difference

The existing `t2m.kernel-bindings.v1` API/CLI configuration accepts:

```json
{
  "operands": {"state": "owner"},
  "tensor_views": "dense-row-major-v1",
  "source_views": {
    "view": {
      "root": "state",
      "source_arg": 1,
      "shape": ["rows", "cols"],
      "strides": ["s0", "s1"],
      "offset": "origin"
    }
  }
}
```

Here `owner` and `view` are original custom-call schema tensor operand names;
`state` is the kernel input allocation name. `source_arg` is the original TTIR
pointer argument ordinal. Coordinate names identify actual captured typed
runtime kernel arguments, also bound through `runtime_arguments`. Constants
in an annotation cannot substitute for the original captured arguments.
This profile requires an exact same-element-type root-plus-offset pointer
binding; mixed-width byte-pointer model views remain separate work.

The derived correspondence uses **`t2m.model-kernel-binding.v3`**, with
`source_tensor_view_binding_profile: t2m.captured-source-tensor-views.v1` and
per-call `source_tensor_views`. Existing dense value reshapes can coexist under
their explicit profile. Bindings without source views retain v1/v2. Older
readers refuse v3. The public model signature, weight bytes, runtime inputs,
result order and all parent references remain unchanged.

Bundle verification and Merlin independently recover the original parent SSA
relationship, reopen complete call attributes, and compare the source pointer
binding. Every original schema tensor operand must be accounted for. Rehashed
coordinate/pointer records, missing call records, profile deletion and schema
downgrades refuse. Compiler staging reads only compiler members; original
inputs, references, exact-word archives and their digests remain excluded.
These checks establish source/binding consistency, with numerical execution,
memory guards and native ownership/lifetime checks retained separately.

## Qualified complete models

`scripts/capture_source_view_model.py` captures a real registered Triton op
with owning-root and non-dense view operands. The example uses a 6x16 learned
matrix, a transposed stepped view, a dependent row/column stepped view and an
empty view. Three calls reuse one translated implementation. Seven model
results retain full updated roots, selected values, the original non-dense view
and earlier snapshots kept live across later updates.

The i64 case includes values above 2^53. F32 and BF16 use the same source
structure; BF16 retains each source rounding boundary. Native and actual Merlin
host execution compare all seven results as exact words against original CPU
capture references. Functional RISC-V/Spike qualification uses the same
standard payload, independent readonly-input readback and exact-word results.
The i64 case additionally exercises the public `bind-model` CLI and rehashed
producer/consumer refusals. No accelerator performance claim follows.

The focused producer/model suite passes 19 checks, including all three actual
target images and the i64 CLI. Another 103 legacy model/binding checks and both
specialization-order regressions pass. Selected operand mappings travel with
their candidate through body inlining, source-view proof and correspondence.
Merlin's 304-case intake selection checks source/runtime/dense-view staging;
its 87-case view/BF16 selection also checks exact cast execution and unchanged
refusals. The target receipts are independently reopened in
`out/spike-source-view-models/summary.json`. Structural checks retain three
unmapped logical grids, source loops/access provenance and standard payloads.

The BF16 model exposed an integer-to-BF16 consumer gap. Merlin now expands
neutral RTNE signed/unsigned i1/i8/i16/i32/i64 conversions directly with integer
significand arithmetic and a final word bitcast. A float intermediate can
double-round wide inputs. Exact native integer-oracle tests cover complete
small-word domains, wide midpoint neighbors and integer extrema; unsupported
flags/shapes/widths refuse before any source edits.

Reproduce with the existing tool environments documented in
[runtime model arguments](runtime_model_arguments.md):

```bash
"$T2M_MODEL2MLIR_PYTHON" scripts/capture_source_view_model.py \
  --dtype i64 --out out/source-view-parent
python -m pytest -q tests/test_model_source_views.py tests/test_source_view_model.py \
  --basetemp=.scratch/pytest-source-model-views
```

The model tests require the existing shared optimizer, native compiler,
model2MLIR and Merlin environment variables. Spike checks run when the existing
RISC-V toolchain variables are available; host execution alone is recorded
separately. General model mutation, dynamic public ABI, broader source/numeric
compositions and accelerator materialization remain in the completion scope.

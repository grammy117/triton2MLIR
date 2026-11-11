# Shared allocation state for aliased source pointers

`convert_allocations` and `shared-tensor --allocation-bindings` accept an
explicit mapping from source arguments to owning allocation tensors. Multiple
source pointers may refer to the same root, with signed constant or runtime
element offsets. An initialized writable root has one incoming value and one
returned updated value; fresh write-only roots retain their complete initialization
checks. This preserves source pointer sharing through the functional
Merlin interface, including reads that observe earlier writes.

## Executable binding and hardware boundary

The adapter builds a fresh public TTIR entry. It retains the original kernel
body as a private function and calls it with the declared root pointers,
`tt.addptr` offsets, typed scalars and explicit integer constants. Native
shared parsing verifies the source and bound program. The existing upstream
`--inline --symbol-dce` stage then substitutes those operands **before** pointer,
mask, memory-effect and ownership analysis. Every aliased access consequently
uses the same root, version and guard state. No array sample, pointer address
or provenance label selects the lowering.

This reuses TTIR function/call handling, rather than implementing another
memory model. The surveyed Meta
[`call` fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonPtrToMemref/call.mlir)
and [`visibility` fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonToLinalgExperimental/preserve_func_visibility.mlir)
document that boundary. Installed optimizer bytes and pipeline evidence retain
their own pins; the survey is not a build receipt.

The resulting MLIR keeps the original computation, tiled loops and source
ordering. Each root is privately materialized once, and all its untouched
words remain in the returned allocation. The logical grid remains unmapped.
Active bounds, lane conflicts, initialization, cross-program ownership and
native lifetime checks all remain required. Conflicting accesses by separate
logical programs are refused; an arbitrary sequential program order cannot
make a source race valid.

## Interface and provenance differences

The existing tensor ABI/receipt schemas are unchanged. The public signature
now describes roots and runtime parameters, replacing the original pointer
argument roster. Callers supply those roots explicitly. Separate input tensors
with undeclared writable overlaps still refuse during package preparation.
The complete updated root is returned once; duplicated source pointer arguments
do not become duplicated independent state results.

`source.ttir` in the package is the bound program, including the private
original body and its helpers. Its complete source inventory includes the real
wrapper call and offset operations; no helper is hidden as an opaque native
implementation. New packages additionally retain the exact pre-wrapper bytes,
generic native input, argument configuration and source-stage fingerprints in
a [portable source capsule](descriptor_state.md). Legacy packages without that
capsule retain pre-wrapper bytes only in conversion work. `source_argument_binding` records both byte hashes and the
descriptive `t2m.source-allocation-bindings.v1` table with
`grants_semantics=false`. Arithmetic, sharing and effects come from executable
TTIR; the table grants no rewrite permission or independent source proof.

Pointers sharing a root may also use equal-width byte-aligned scalar views.
The wrapper applies offsets to the declared root and emits real pointer
bitcasts before native inlining; existing [word-view normalization](pointer_views.md)
then preserves one state version with register casts at each source event.
Explicit [byte-buffer bindings](byte_aliases.md) also admit wider source
arguments from i8 roots with declared byte order. Typed-root byte packing,
sub-byte representations and automatic recovery of arbitrary NumPy view
ownership remain open. The source entry must be void, contain one block and be accepted
by the installed shared source reader. Original helpers remain subject to complete inlining and
operation/effect checks. Kernel/model view adapters and applicable target
materialization remain separate completion gates.

## Python and CLI

```python
from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations

# root_abi.entry is a fresh symbol; root_abi declares owning roots and scalars.
result = convert_allocations(
    "kernel.ttir", root_abi,
    (PointerArgument("state"), PointerArgument("state", "origin"), ScalarArgument("n")),
    "out/conversion", source_entry="kernel",
    memory_profile="guarded-buffer", address_policy="source-words")
```

The saved CLI configuration uses a strict schema and source argument order:

```json
{
  "schema": "t2m.source-allocation-arguments.v1",
  "source_entry": "kernel",
  "arguments": [
    {"kind": "pointer", "allocation": "state"},
    {"kind": "pointer", "allocation": "state", "offset": "origin"},
    {"kind": "scalar", "name": "n"}
  ]
}
```

```bash
python -m t2m shared-tensor kernel.ttir --abi root-abi.json \
  --allocation-bindings source-arguments.json \
  --memory-profile guarded-buffer --address-policy source-words \
  --inputs inputs.npz --reference reference.npz \
  --out out/bound-kernel --python /path/to/native-python
```

Offsets are measured in the root element type. `kind: constant` with an integer
`value` specializes a source integer scalar; floating and other runtime scalars
use their exact declared `ScalarArgument` type. Unknown fields, missing arguments,
unknown roots/scalars, type mismatches and overflowing constants refuse.

## Consumer correctness and qualified compositions

An actual partial-overlap replay exposed an existing Merlin fusion bug: moving
a tensor producer's captured memref reads into a following overlapping store
changed a full-tile snapshot into dependent scalar updates. Direct native
execution passed while Merlin's result differed. Merlin now preserves the
effectful producer boundary before upstream greedy fusion, including initially
multiple uses whose unused destination dependency the pass later removes.
Nested bodies and unknown effects retain that boundary even with the optional
broadcast cost guard disabled. Pure arithmetic fusion remains available;
temporary compiler pins are erased before bufferization. The 109 relevant
Merlin fusion, option and destination regressions pass.

Runtime i64 source pointers now share one root through loops, partial/exact
overlaps, changed offsets, empty/zero-trip domains and multiple logical programs.
Exact words above `2^53`, input preservation and compiled-library reuse are
checked. Separate compiled invocations reject active out-of-bounds addresses
and cross-program dependencies with no result archive or successful receipt.
Identical pointer bindings also qualify through tensor, buffer and guarded
memory profiles. The generic CLI produces a checked native package with explicit
source constant bindings.

Three additional f32/f16/BF16 tiled GEMM libraries bind source C and output into
one f32 allocation. Source C is a dense view at a runtime origin; the output may
use signed/padded/transposed strides. Partial overlaps in one logical tile and
exact C/output aliases across programs retain the original K loop, one named
16×16 contraction and 16×16×32 tiling. Full-allocation word comparisons include
untouched NaN payloads and signed zeros, zero K, empty shapes and sensitive C.
Independently strided C views and broader model compositions still need their
own qualification. Extra snapshot materialization has no performance claim.

All 23 focused producer/consumer cases and 35 existing readonly/strided-memory
regressions pass. The visible BF16 package and qualification verify 33/52
artifacts. `out/allocation-state-coverage/coverage.json` selects 40 packages;
31/50 TTIR definitions remain observed and 27 occur in host-qualified captures.
These counts describe that selection, without universal type/attribute coverage.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel allocation-state-bfloat-dot \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/allocation-state-bfloat --python /path/to/native-python
python -m t2m qualify-merlin out/allocation-state-bfloat \
  --out out/allocation-state-bfloat-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`allocation-state-dot` and `allocation-state-half-dot` select f32/f16 inputs.
This is measured alias/state composition coverage. The full
[completion plan](completion_plan.md) remains active.
An [explicit dense model value bridge](model_tensor_views.md) now additionally
joins these roots to matrix-shaped model2MLIR calls with original weights and
all result references retained. It grants no physical view or mutation semantics.

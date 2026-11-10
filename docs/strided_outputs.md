# Strided writable output allocations

The guarded source-word interface now qualifies tiled GEMM into a runtime
strided output allocation for f32, f16 and BF16 inputs, with f32 C/output.
It retains original source addresses, masks, logical tiles, the K recurrence
and the full incoming allocation state. No layout ABI or device opcode is added.

## Public data and mutation boundary

Unlike a fresh dense result, the output allocation is an additional incoming
tensor with `readwrite` intent. Runtime output capacity, origin and signed row/
column element strides describe the addressed view. The returned one-dimensional
tensor contains the **entire updated allocation**, including prefix, suffix
and holes between active coordinates. Dense source C retains its M×N shape.

The existing functional interface creates private output state from the
incoming allocation. It keeps caller inputs immutable and returns the updated
state explicitly. Source stores update only their actual active addresses;
padding is neither discarded nor initialized with invented values. Consumers
that need a logical M×N view must retain those explicit origin/stride bindings.
This differs from the previous dense-result signature, while ABI v5 and the
package/receipt schemas remain unchanged.

The original one named 16×16 contraction remains inside its source K loop,
with a 16×16 f32 carried accumulator and 16×16×32 tiling. Actual shared pointer,
gather/scatter and standard-memory conversions remain responsible for lowering.
Reviewed against Meta's
[`gather_scatter_all_mask` test](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/UnstructuredToMemref/gather_scatter_all_mask.mlir);
the installed optimizer is the separately content-pinned `0c27923` distribution.
No source-reference pin is presented as a native build receipt.

## Evidence and guards

Three new typed packages pass native and actual Merlin execution, checking
the full result word for word. Independent source-allocation views and ordered
f32 arithmetic supply the expected values; uncomputed padding includes
noncanonical NaN payloads, signed zeros and finite words. Numerical comparison
alone cannot substitute for preservation of those untouched bytes.

Each compiled Merlin library replays nine additional layouts/domains:
column-major, reversed/mixed-sign, padded, repeated readonly inputs, multiple
tiles, a C=`2^24` case, zero K and empty M/N. Empty logical outputs preserve
nonempty incoming allocation storage even when the unused output origin lies
outside it. Zero K still stores original C into the active output view.

Three invalid store configurations per type remain refused invocations:
duplicate active lanes in one store, a collision between distinct logical
programs without a lane collision, and an active address beyond the allocation.
The real compiled guard aborts; no output archive or successful invocation
receipt is published. Such failures cannot become successful source-assertion
outcomes. Input/output bounds, initialization, ownership and lifetime gates
remain required; provenance supplies no permission to bypass them.

The six-case combined output/read-stride matrix passes. The new output slice
accounts for 27 positive replays and nine guard refusals beyond initial package
qualification. This is measured composition coverage, not universal stride,
alias, opcode or target qualification.
The visible BF16 output package and Merlin qualification verify 33/52 artifacts.
The explicit 38-package observation report remains at 31/50 TTIR definitions
observed and 27 in host-qualified captures; this extends memory composition
evidence without adding an unsupported universal-coverage claim.

```bash
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m shared-demo --kernel strided-bfloat-output-dot \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/strided-bfloat-output --python /path/to/native-python
python -m t2m qualify-merlin out/strided-bfloat-output \
  --out out/strided-bfloat-output-merlin \
  --merlin-root "$T2M_MERLIN_ROOT" --merlin-python /path/to/frontend-python \
  --python /path/to/native-python --merlin-clang /path/to/compatible-clang
```

`strided-output-dot` and `strided-half-output-dot` select the other inputs.
Aliases between declared source allocations, C loaded from the strided output
itself, logical-view model adapters, broader effects and applicable target
materialization remain in the full [completion plan](completion_plan.md).
An [explicit allocation-state entry](allocation_state.md) now additionally
binds dense source C and strided output into one root, preserving source sharing
and returning its full updated state. Independently strided C and model view
adapters remain separate qualifications.

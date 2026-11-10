# Binding translated kernels into model2MLIR

`t2m.model_binding.bind_model(model_mlir, [KernelBinding(...)])` consumes the
actual model2MLIR `prov.kernel` contract. It replaces typed custom-kernel calls
with translated standard tensor bodies and preserves the enclosing model ABI.
No backend kernel table, CPU Triton backend or GPU runtime is needed.

Binding checks the call/declaration identity, complete schema operand names,
tensor argument/result types and every scalar specialization. Source buffer names
can map explicitly to schema names; source results can map explicitly to return
order. Each call must select exactly one implementation. Missing/ambiguous
bindings, mismatched scalars and incomplete operand/result mappings fail.

Inlined operations retain their source provenance, with callsite-prefixed region
and emission IDs. A binding sidecar includes the original call attributes and
kernel provenance. Resolved private declarations are removed. Math outside the
kernel call remains ordinary model computation. Binding itself establishes typed
SSA validity; native numerical and consumer execution remain separate checks.

`scripts/capture_model_binding_demo.py` creates a genuine model2MLIR capture of
a `torch.library.triton_op` GEMM plus surrounding model math and learned weights.
Run it with an existing model2MLIR environment. Its registered PyTorch CPU
implementation supplies the independent golden; the actual Triton body is
captured and translated separately with triton2MLIR's pinned frontend. This
avoids invoking a Triton target backend and keeps both environments independent.

The model2MLIR registry currently rejects argument-mutating custom operations.
Fresh `write` results satisfy its functional interface after initialization
checks. General mutation/schema negotiation and downstream staged payload
support still need explicit capabilities; the binder does not silently change
model2MLIR's registry rules or certify a target's materializer.

## Checked model bundle and Merlin interface

The `bind-model` command reads `t2m.kernel-bindings.v1` JSON descriptions, translates
the supplied TTIR specializations and binds every registered call. It retains the
parent's real weights, manifest, runtime inputs, constants, input order and golden
byte-for-byte. A native check materializes arguments by **function argument index**,
using the manifest and `input_order.json` rather than treating weight positions as
runtime input positions, and compares the complete model result with the unchanged
PyTorch references. Flat tensor tuple results use model2MLIR's additive,
receipt-bound `goldens.npz` archive in result order. The original first-result
`golden.npy` remains unchanged. Missing, unbound, reshaped or mistyped references
fail before native execution; both native and Merlin reports bind every result.

```bash
# Existing, independent environments; no GPU or Triton CPU backend required.
/path/to/m2m-env/bin/python scripts/capture_model_binding_demo.py --out out/model_capture
python -m t2m shared-demo --kernel loop-matmul --out out/shared_loop_matmul --python /path/to/native-env/bin/python
python -m t2m bind-model out/model_capture --bindings examples/model_binding.json \
  --out out/bound_model --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --rtol 1e-5 --atol 5e-6
python -m t2m verify-bundle out/bound_model
```

Set `TRITON_SHARED_OPT` to an existing optimizer for both conversion commands.
The optional Merlin check invokes its actual `forward_signature` reader and
`c_runtime.generate`: `model_gen.h`, `model_io.h`, `model_call.c` and `weights.bin`
are retained under `consumer/runtime/` and hash-bound by the derived receipt.
This measures real interface ingestion/runtime generation; target execution and
staged accelerator materialization are separate unperformed checks.

The new receipt is **`t2m.bound-model-receipt.v1`**. The original m2m receipt, model,
FX trace and metadata remain under `parent/`; unchanged data artifacts satisfy
the parent receipt without duplicating weight files. The transformed MLIR has its
own binding/native evidence and bundled TTIR implementations. Its parent FX trace
is never relabeled as correspondence for changed IR. Merlin paths requiring a
Torch-only capture receipt retain their own admission gates. Merlin's compiler-only
stager now admits this derived receipt with separate original/derived source
correspondence. Optional `--merlin-host` compiles and executes the staged whole
program; see [the measured consumer path](merlin_consumer.md).

## Integer atomic model demo

`scripts/capture_atomic_model.py --kind cas` captures an actual functional
`torch.library.triton_op` call returning updated state and original old values.
The model has an integer state parameter and comparison buffer; its registered
CPU reference supplies both independent PyTorch results. Original weights,
buffers, inputs, `golden.npy` and the complete `goldens.npz` archive remain
byte-identical in the bound package. The existing default `--kind add` demo
continues to qualify the RMW path.

After creating `out/atomic-cas/` as described in [atomic memory](atomic_memory.md),
the checked [binding configuration](../examples/atomic_cas_binding.json) reuses
its source with the model's explicit 17-element state ABI. The source accesses
the first sixteen elements and returns the remaining incoming state unchanged;
the standalone demo's state extent is sixteen. Comparison/value inputs and old
results each contain sixteen elements. Source names map explicitly to model
schema names, and both model results cross the interface in their declared order.

```bash
/path/to/m2m-env/bin/python scripts/capture_atomic_model.py \
  --kind cas --out out/cas_model_capture
python -m t2m bind-model out/cas_model_capture \
  --bindings examples/atomic_cas_binding.json --out out/cas_model \
  --optimizer /path/to/triton-shared-opt --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --merlin-host --merlin-clang /path/to/clang
python -m t2m verify-bundle out/cas_model
```

The supplied Merlin root also supplies the pinned native CAS ownership
compatibility adapter. Compiler-only staging excludes every evaluator input and
reference. Whole-model native and actual Merlin execution compare both original
results and retain functional inputs; accelerator execution remains unperformed.

## Ragged tiled model demo

The same model capture and binding path now covers a 35×37 by 37×21 GEMM.
Its six logical programs retain 16×16 tiles and three K iterations. Dynamic
slice extents and exact zero padding implement the ragged M/N/K boundaries;
the source kernel is not replaced with a whole-tensor matmul.

```bash
/path/to/m2m-env/bin/python scripts/capture_model_binding_demo.py --ragged --out out/ragged_model_capture
python -m t2m shared-demo --kernel ragged-matmul --out out/shared_ragged_matmul --python /path/to/native-env/bin/python
python -m t2m bind-model out/ragged_model_capture --bindings examples/model_ragged_binding.json \
  --out out/merlin_ragged_model --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --rtol 1e-5 --atol 6e-6
python -m t2m verify-bundle out/merlin_ragged_model
```

Tests compare the complete model, including learned weights and the scalar
epilogue, against the unchanged captured PyTorch golden, then invoke Merlin's
actual interface reader/runtime generator. The derived receipt and target
execution boundaries are the same as for the exact-tile model above.

For the legacy block-pointer form, use
`scripts/capture_model_binding_demo.py --block-pointers --out out/blockptr_model_capture`,
`shared-demo --kernel blockptr-matmul --out out/shared_blockptr_matmul`, and
`examples/model_blockptr_binding.json` with `bind-model`. Supply the same native
and Merlin environment arguments as above. This captures real `make_block_ptr`
and `advance`, retains separate original/normalized provenance, and qualifies
the complete model against its unchanged PyTorch golden and Merlin interface.

The current tensor-descriptor form uses `--tensor-descriptors`,
`shared-demo --kernel descriptor-matmul --out out/shared_descriptor_matmul`,
and `examples/model_descriptor_binding.json`. It captures a 35×36 by 36×20
GEMM with `make_tensor_descriptor` / descriptor loads/stores. Complete-model native
comparison and Merlin interface/runtime generation use the same package and
qualification path, with the original descriptor source retained in provenance.

## Scan models and tuple kernel results

`scripts/capture_scan_model.py` captures actual sum, product and two-result
affine-composition `triton_op` models. A registered eager PyTorch CPU reference
supplies the parent golden; no Triton target backend executes. The tuple model
consumes both kernel results in surrounding model math and retains learned weights.
Tests bind every result and compare the complete model before Merlin interface/
runtime generation. A multiple-result component does not require a multiple-output
model package; the existing single-model-output receipt boundary stays explicit.

```bash
/path/to/m2m-env/bin/python scripts/capture_scan_model.py --kind tuple --reverse --out out/tuple_scan_model_capture
python -m t2m shared-demo --kernel tuple-scan --out out/shared_tuple_scan --python /path/to/native-env/bin/python
python -m t2m bind-model out/tuple_scan_model_capture --bindings examples/model_tuple_scan_binding.json \
  --out out/merlin_tuple_scan_model --python /path/to/native-env/bin/python \
  --merlin-root /path/to/merlin --merlin-python /path/to/merlin-env/bin/python \
  --rtol 1e-5 --atol 3e-6
python -m t2m verify-bundle out/merlin_tuple_scan_model
```

For a reverse sum, use `--kind sum --reverse`, `shared-demo --kernel scan`, and
`examples/model_scan_binding.json`. The retained scan stage/composite annotations
describe parallel tensor computation; source tuple ordering and reverse traversal
remain executable semantics.

## Original MLP pair

`scripts/capture_mlp_pair.py` captures model2MLIR's existing `mlp_basic` and
`mlp_basic_triton` workloads, retaining their actual initialization and weights.
The test extracts the original standalone JIT function through Python AST,
captures it with the pinned Triton frontend, and binds it by schema names into
the captured model. The plain and Triton captures have byte-identical weights
and identical eager goldens. Both complete models pass native execution against
those goldens and the optional Merlin interface/runtime-generation check.

This tests the original two-layer fused MLP plus the surrounding third layer,
rather than only substituting a standalone matmul result for a model output.

## Selected-memory calls

`scripts/capture_selected_memory_model.py` captures a real model with learned
weights, two `t2m_demo::selected_gather` calls with different scalar
specializations, and surrounding arithmetic. Independent eager PyTorch code
supplies the complete model golden. The source kernels concatenate pointers
from two input buffers and gather them with repeated intra-tile indices before
the selected loads. Both implementations retain source-word selectors, tile
dataflow and source provenance when bound through the existing registry.

The maintained integration test checks the original weights, inputs and golden
bytes, both call bindings, native model execution, compiler-only Merlin staging
and actual Merlin host execution. Neither a GPU nor Triton CPU backend supplies
the reference. This profile has fresh results and readonly inputs; the existing
model2MLIR prohibition on argument-mutating custom operations still applies.
The capture script's `--dtype f16` option separately qualifies real half
weights/inputs and model arithmetic through the same compiler/native interface.
The bound-model materialization budget now admits two-byte f16 arguments; no
ABI widening or schema change is needed.

Bound `compatibility.json` now includes `kernel_profiles`, keyed by each archived
kernel directory. It lists the same representation differences as standalone
packages, including guarded memory, source-word offsets, logical selectors,
chosen cat ordering and dynamic dimensions when present. The parent model ABI
and data files remain byte-identical. This is additive metadata in the existing
bound-model compatibility schema; it does not manufacture an m2m FX trace for
the derived program.

[BF16 models](bfloat_models.md) now retain exact inputs/results and learned
safetensors state through native and actual Merlin execution. The optional
model2MLIR `tensor_words.npz` remains evaluator data and is excluded from compiler
staging with its digests. Parent value archives remain unchanged; uniquely
embedded non-NaN legacy BF16 values may also be recovered with explicit evidence.
Visible observation and matmul captures retain six tuple results and the
original BF16/F32 contraction respectively. Source-bound
[lifted constants](lifted_constants.md) now additionally reach Merlin through
exact safetensors storage and a checked compiler-only profile. Dynamic/alias
model contracts remain separate completion gates.

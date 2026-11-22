# Two transformer blocks with automatic Triton integration

The demo is a complete pair of inference transformer encoder blocks, not a
Triton pointwise operation inserted into an otherwise ordinary model. The first
block is ordinary PyTorch. The second block is one real Triton JIT kernel.
Both have pre-norm layer normalization, Q/K/V projections, scaled self-attention,
output projection, residuals, a second normalization and a ReLU feed-forward
network. Each has its own learned parameters. The model returns both block
outputs so the boundary and final answer are independently observable.

The example uses one attention head, embedding width 16 and feed-forward width
32, without dropout or a causal mask. Sequence lengths 13, 16 and 32 are exposed
as separate static specializations; this is not a dynamic-length or performance
benchmark. The Triton body keeps normalized values, attention and FFN tiles in
SSA and uses eight IEEE f32 `tt.dot` operations. All block computation comes
from that source. Its CPU implementation uses independent PyTorch operators only
to supply eager references, never as the translated Triton implementation.

## Run

Use the existing separately pinned frontend/native/model2MLIR/Merlin environments
described in [the consumer contract](merlin_consumer.md) and
[the refactored public interface](repository_layout.md). Set
`TRITON_SHARED_OPT`, `T2M_MODEL2MLIR_PYTHON`, `T2M_NATIVE_PYTHON`,
`T2M_MERLIN_ROOT`, `T2M_MERLIN_PYTHON` and `T2M_MERLIN_CLANG` explicitly.
The invoking Python environment needs the current triton2MLIR and Merlin source
packages; it does not need Torch. No new toolchain build is performed.

```bash
python scripts/run_transformer_demo.py --out out/two-transformer-demo --tokens 13
```

The output directory must be fresh. `summary.json` records each complete result
comparison, the actual library, launch metadata and numeric policy. Intermediate
source and standard MLIR remain inspectable under `launch-translation/`,
`bound/` and `compiled/`. Compiler-only stages exclude input/reference arrays.
The runner invokes an artifact-only copy under `published/` on the captured,
changed and zero-valued inputs, reusing one library. Float comparisons use
`rtol=3e-4`, `atol=3e-5`; this is not GPU bitwise equivalence.

## Automatic path

1. `torch_launch_capture.discover` exports the actual model and finds real
   `triton_op` nodes. Fake tracing of each op's launch body exposes the actual
   JIT function, logical grid, pointer arguments, constant specialization and
   returned allocations. No tensor contents or CPU reference computation select
   the binding. The adapter temporarily registers those identities with m2m;
   users do not write a kernel registry or `KernelBinding` table.
2. Ordinary model2MLIR conversion lowers the first block's ATEN operations and
   retains the second block's actual typed call. The original capture and its
   complete parameter/reference artifacts remain unchanged.
3. `automatic_model.translate_launches` checks the file-backed source identity,
   captures verified pre-layout TTIR using the pinned Triton frontend, converts
   it with the normal tensor/memory pipeline and derives checked operand/result
   bindings. The existing binder independently checks the actual model call.
4. The bound model executes through native MLIR qualification, then public
   Merlin `StagedProgram` compilation with `MerlinHostBackend`. Immutable model
   parameters become exact compiler constants. Only the input tensor remains
   in the runtime ABI. Every returning result is compared with PyTorch.

Torch discovery currently uses the probed Torch 2.10 adapter. The existing Torch
environment has Triton 3.6; it discovers Python launch metadata, not TTIR. The
separate converter captures that file-backed source with pinned Triton 3.7.1.
Exact source file identity and actual resulting TTIR are retained. This records
the version boundary; it does not prove arbitrary source-version equivalence.

## Evidence and limits

The first 16-token run is retained under `out/two-transformer-demo-initial/`:
native comparison plus three public-host invocations, both output matrices
retained, maximum observed absolute error below `1.2e-6`. The original run
preceded artifact-only publication; it does not establish that later boundary.
The fresh ragged integration test additionally checks wrapper rejection, source
identity, original eight contractions, first-block provenance, complete results
and the public one-input runtime signature. See
`tests/test_automatic_transformer.py`; the final publication selection is recorded
separately in its test log.

Automatic discovery currently admits a pure schema, one launch, static dense
zero-origin tensor pointers, fresh returned outputs and literal specialization
arguments. It rejects extra wrapper arithmetic/views, multiple launches,
hidden scratch, runtime symbolic launch bindings, mutation/aliases, unresolved
autotuning and physical descriptor metadata. Those need broader wrapper-graph
composition or the existing explicit binding/storage interfaces. These are
automatic-adapter limits, not evidence that their generic computation cannot
be represented in MLIR. No original GPU kernel or accelerator is executed.
The [limitations inventory](limitations.md) distinguishes hardware exclusions
from generic work that still blocks closure.

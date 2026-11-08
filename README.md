# triton2MLIR

Translate Triton TTIR into portable tensor MLIR while retaining the kernel's original dataflow
and intent in annotations and provenance. Exact tiles and the first tensor-memory milestone are implemented; see the
[roadmap](ROADMAP.md) for the path through tiled GEMM, attention and broader kernel semantics,
and [plan.md](plan.md) for the architecture.

Contributor/commit rules are in [AGENTS.md](AGENTS.md). Broader coverage will reuse
[triton-shared's existing passes and analyses](docs/triton_shared.md).

The [shared tensor frontend](docs/shared_tensor.md) now reuses those passes for 2-D add,
masked softmax and IEEE matmul, emitting functional tensor/linalg IR. Its packages are checked
by **native LLVM host execution** against independent NumPy references. It currently accepts
static structured memory and affine disjoint output tiles over logical grids;
this is not full arbitrary-kernel coverage.

```bash
python scripts/fetch_shared_opt.py --out .scratch/shared-opt
export TRITON_SHARED_OPT=$PWD/.scratch/shared-opt/triton-shared-opt
python -m t2m shared-demo --kernel softmax --out out/shared_softmax --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel matmul --out out/shared_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel tiled-matmul --out out/shared_tiled_matmul --python /path/to/native-env/bin/python
python -m t2m shared-demo --kernel loop-matmul --out out/shared_loop_matmul --python /path/to/native-env/bin/python
python -m t2m verify-bundle out/shared_softmax
```

The optional fetcher extracts a pinned standalone optimizer from a verified distribution;
it builds nothing and installs no target backend. An existing optimizer is also supported.
See [the package differences](docs/package.md) and [native execution scope](docs/native_execution.md).

The [model2MLIR binder](docs/model_integration.md) now replaces registered kernel calls with
translated tensor bodies while preserving the real model ABI, weights and input bindings.
Complete models pass native comparison against their unchanged PyTorch goldens; optional
checks invoke Merlin's actual interface reader and C runtime generator. The original plain
and Triton MLP workloads pass through this path. Derived receipts remain explicit, and
staged/device execution is still unqualified.

The current demo captures a **real Triton kernel without a GPU**, recovers exact-tile tensor
accesses, emits standard `scf`/`tensor`/`linalg` operations, executes that serialized IR with a
reference interpreter, and checks it against an independent NumPy computation.

```bash
uv venv
uv pip install -e '.[triton,dev]'
python -m t2m demo --out out/vector_add
python -m t2m demo --operation multiply-add --out out/multiply_add
python -m t2m demo --size 1000 --out out/masked_add
python -m t2m verify-bundle out/vector_add
python -m pytest -q --basetemp=.scratch/pytest
```

Existing output directories are refused. Use a new path for each demo run. The frontend pins
Triton 3.7.1 and xDSL 0.68.0. There is no triton-cpu, Torch or Merlin runtime dependency.
The optional Triton wheel is needed for capture and saved-TTIR parsing, not just source capture.

```text
out/vector_add/
  model.mlir
  weights.safetensors                 # real empty container for a parameter-free kernel
  weights.safetensors.manifest.json   # tensor argument bindings
  inputs.npz, golden.npy, extra.npz, input_order.json
  capture_receipt.json                # hashes, t2m.capture-receipt.v1
  source.ttir, spec.json, provenance.json, compatibility.json, meta.json
```

These use model2MLIR's core filenames and array conventions. **The receipt/trace schemas and tiled
payload support are explicit differences**, recorded in [the package contract](docs/package.md)
and each bundle's `compatibility.json`. Merlin parser acceptance does not imply staged accelerator
execution. The reference path uses a standard-op interpreter; the shared tensor path has a
separate native host execution gate.

Saved input replay:

```bash
python -m t2m lower out/vector_add/source.ttir --spec out/vector_add/spec.json --out out/replayed
```

This command writes verified MLIR and provenance. It does not fabricate example inputs/goldens
or claim to write a full capture bundle without those artifacts.

Strict native verification and legalization can use an **existing** torch-mlir environment:

```bash
python -m t2m verify-mlir out/masked_add/model.mlir --python /path/to/native-env/bin/python --lower-to-llvm
T2M_NATIVE_PYTHON=/path/to/native-env/bin/python python -m pytest -q --basetemp=.scratch/pytest
```

This checks registered operations with unregistered dialects disabled, then optionally runs standard
passes through bufferization and LLVM legalization. It operates on a separate copy; exported kernel
MLIR stays at the tensor/linalg/SCF boundary. No native compiler is built or installed automatically.
LLVM legalization does not claim native code execution or runtime-helper linking.

An existing triton-shared optimizer can also be used directly:

```bash
TRITON_SHARED_OPT=/path/to/triton-shared-opt python -m t2m shared-lower kernel.ttir --out out/shared --stage structured
```

This writes an inspectable upstream stage, with exact tool/input/output hashes and its boundary
recorded. It does not claim our tensor ABI, provenance correspondence or complete package until
those contracts have been qualified. No native toolchain is built implicitly.

Current support: static f32 rank-one logical tensors, positive-stride storage views and offsets,
independent allocations, rectangular boundary masks, defined scalar/splat fills, one write or
readwrite output, and pointwise add/subtract/multiply. Masked loads with omitted `other` are accepted
only when pointwise definedness proves no stored lane can observe the undefined values. Partial
readwrite results preserve incoming values. Identical nonvolatile reads share an SSA tensor value;
source cache/eviction policies remain provenance instead of selecting physical caches.

Unproved bounds, index overflow, inter-program readwrite dependencies, nonrectangular/data-dependent
masks and nested source regions remain refused. See [R1 evidence and remaining work](docs/r1_memory.md).
Output contains an unmapped `scf.forall`; downstream consumers choose hardware realization.

`prov.family`, `prov.op`, `prov.region_id`, stable source/emission IDs and recovered access facts
retain the source intent as operations decompose. Caller intent is labeled as a description, and
cannot override arithmetic. The interpreter also runs with all `prov.*` attributes stripped.

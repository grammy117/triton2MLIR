# MVP implementation evidence

**2026-11-07.** Local development used the existing read-only tool environment
`/scratch/grammy/blast/merlin/.venv/bin/python` (Triton 3.7.1, xDSL 0.68.0, NumPy 2.4.6,
pytest 9.0.3). No existing environment was modified, no native compiler was built, and no GPU
kernel was executed. The normal isolated install/run commands are in [README.md](../README.md).
The Python wheel build is also checked, including the packaged Triton example sources.

The default demo captures `t2m/examples/vector_add.py` with N=1024, BLOCK=256 and grid=4.
It inventories 38 TTIR operations, recovers the tile accesses, eliminates unused pure overflow
guard chains by backward liveness, and emits a four-program **unmapped `scf.forall`** containing
tensor slices and `linalg.generic`. The source output pointer becomes the returned tensor.

The standard-op reference interpreter executes the serialized output and matches a separately
computed NumPy vector sum. The `multiply_add` kernel exercises a distinct two-operation graph.
The interpreter never reads provenance to decide arithmetic. Tests strip every `prov.*` attribute
and still execute the same computation; source intent is not an alternate semantic definition.

`t2m demo --out ...` writes 12 hashed artifacts plus `capture_receipt.json`. The package keeps
the model2MLIR core filenames and array conventions, and explicitly records different receipt/
trace schemas, the limited tensor ABI, xDSL pin and unqualified consumer execution status.

Additional integration observations:

- The model2MLIR environment's independent `safetensors.numpy.load_file` loads the generated empty
  weights container as `{}`.
- The IDE Merlin checkout's `frontends.linalg_mlir.make_context()` parser accepts and verifies
  the generated 19-operation vector-add module. Its permissive parser can retain unknown standard
  operations; this does **not** certify native upstream op verification or staged materialization.
- `merlin.capture.bundle.CaptureBundle` resolves all seven core bundle artifacts at the expected
  paths. Filename/path acceptance does not certify the distinct t2m receipt schema.

The R0 checkpoint passed **25 tests**; current progress is recorded in [r1_memory.md](r1_memory.md).
R0 automated coverage includes source and saved TTIR;
numerical add and multiply-add at different
tile sizes; provenance/SSA correspondence and round trips; incorrect caller intent; source argument
order/effects; bounds, output coverage and integer-overflow refusals; a 2²⁴-element shape with no
element enumeration; unsupported live operations; duplicate stores; ABI checks; stale provenance,
incorrect goldens and corrupt artifact checks. All use one pytest process and small actual arrays.

Remaining limitations are documented in [ROADMAP.md](../ROADMAP.md) and [package.md](package.md).
In particular, masked vector-add tails and nested TTIR regions are **refused**, not implemented;
native LLVM/device execution and arbitrary-kernel support remain later gates.

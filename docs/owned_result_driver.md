# Generated invocation and observed result shapes

Merlin now provides a source-typed signature-v6 driver generator, synchronous C
controller, hosted main and complete observed-shape decoder. The production host
adapter executes the real three-kernel compaction model LLVM; no handwritten
per-model C descriptors or test-only observation format is required.

`merlin.llvmlower.owned_result_driver.generate(source, signature, arrays, out)`
reconstructs the staged source/input contract and writes typed `model_owned.c`,
separate `invocation.bin`, the checked signature and a versioned driver record.
The fixed input shapes belong to this runtime driver, not the model compiler.
Input contents and native scalars can change without rebuilding either product;
executable scalar bounds, dimension equalities and canonical Boolean checks still
run. Invocation values and references remain outside compiler staging. Reopening
reconstructs complete artifacts and named implementation identities rather than
following serialized paths.

The shared C controller invokes the actual model once, validates all returned
descriptors, makes complete independent logical copies and releases tracked source
owners before calling observers. Allocation/descriptor failures publish no
results; observer failures stop publication and clean every copy. Busy contexts
remain owned by their original caller. The hosted main retains source assertion
diagnostics and writes DONE only after successful complete observation.

The explicit `merlin.execution-owned-readback.v1` profile uses OUT_BIN v1 pairs
`inN_shape`/`inN` and `outN_shape`/`outN`. Actual returned descriptors supply
dynamic output extents. The independent decoder checks source rank/static
dimensions/storage, shape products and output budgets, every input byte, complete
result rosters and terminal completion. IEEE/BF16/FP8 words stay raw. This profile
differs explicitly from legacy fixed-shape output and readonly-input frames;
existing readers and package schemas do not silently acquire it.

`tests/test_compact_model_driver.py` links the actual qualified compaction LLVM
and generated C with the production hosted main/runtime. Three input capacities
each reuse one model/driver executable across four input/parameter patterns.
All 84 results match exact independent words, including empty/full selections,
overflow, the count, snapshots and stepped result slices. Two invalid capacities
abort through original exported source guards before any result frame or DONE.
The focused Merlin driver/readback/legacy transport selection additionally passes
63 checks. Generated evidence is retained in
`out/compact-model-driver/summary.json`; these observations add no opcode coverage.

This establishes production hosted transport for the selected real TTIR model.
The [execution-owned target package](execution_owned_target_packages.md) now adds
normal v6 staging/build integration, a bare-metal console adapter and separate
actual target qualification. A successful decoder establishes
transport/input observations; full independent reference comparison and actual
process success remain separate qualification requirements. General resizing
ownership, physical/mutating views, effects and accelerator performance remain open.

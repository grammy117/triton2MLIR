# Functional whole-model target execution

`scripts/qualify_spike_model.py` carries an existing bound model package through
Merlin's ordinary whole-model RISC-V backend and actual Spike execution. This
is an explicit execution edge outside TTIR conversion. Portable source MLIR,
logical tiling and provenance stay unchanged; the selected consumer owns CPU
mapping, bufferization and code generation.

## Measured scope

Real i32/i64 source-assembly models retain learned parameters, dependent calls,
runtime kernel argument specializations and all six original results, including
two empty tensors. A real lifted-BF16 tiled contraction retains its original
input/weight words, F32 accumulation and complete result. These cases execute
the linked RISC-V images and compare all values with the existing independent
model references. They are finite functional CPU qualifications, not universal
TTIR, vector-instruction, accelerator, RTL, timing or performance evidence.

Retained local receipts are under
`out/spike-qualified-models/{i64,i32,bf16-matmul}/qualification.json`.
Each directory contains its actual image, raw output and complete evaluation;
generated artifacts remain outside Git.

The Merlin consumer additionally fixes two general reader issues exposed by
that contraction: native implicit named-region casts and standard custom syntax
for attributed `bufferization.to_buffer`. Named contractions and their actual
mixed types/maps/provenance remain present. Native implicit casts preserve the
source accumulator precision.

Host and target materialization now share `t2m.consumer_features`: actual BF16,
FP8, diagnostic and atomic operations select the existing consumer capabilities.
The target records both caller-requested and effective features. Provenance,
model names and reference values never select these implementations; Merlin
still checks every body's legality and ownership independently.

Additional measured models retain fourteen mapped-memory results, three
descriptor-state results and all three results of mixed-storage RMW64/CAS16.
The latter exposed an order-dependent atomic preparation marker in the ordinary
scalar pipeline. Destination reuse now precedes the complete atomic/ownership
suffix in either feature application order. Byte-backed atomic recurrences
also qualify after preserving source execute-region attributes through the
consumer's custom reader. Four actual byte-backed models retain selected roots,
while-carried old values, learned state and all seven results, including empty
domains. RMW32/RMW64 and CAS16/CAS32 keep one indivisible word event and their
original byte representation; the lifetime and ownership gates remain enabled.

The eight state-model integrations compare all 51 results bit for bit and
recheck original input bytes. Their retained test receipts live under
`.scratch/pytest-spike-state-words/test_complete_model_executes_a*/target/`.
These extend the measured compositions; they do not establish general external
participant, synchronization, physical-address or target coverage.
The original integer tuple models and lifted-BF16 contraction also pass the
same strict policy, bringing this selection to eleven models and 64 complete
results. The BF16 consumer feature now supplies its upstream cast-expansion
prerequisite in the ordinary default pipeline as well as the explicit host path.
The local aggregate `out/spike-state-models/summary.json` pins these eleven
reverified qualification receipts and records the remaining coverage limits.

### Exact BF16 runtime inputs

Public BF16 inputs now use their receipt-bound source-word witness, or the
existing proved legacy embedding without NaNs. A separate runtime-only archive
projects each word into the high bits of an F32 container. Merlin's existing C
generator emits the original BF16 storage from that exact embedding; every one
of the 65,536 encodings passes an actual C storage/readback test. Other input
arrays retain their original dtype, shape and bytes. Weights keep their ordinary
compiler-owned safetensors path.

This closes a real difference from host materialization: a valid archive with
canonical legacy NaN values and unchanged exact source words previously built
and ran on Spike but returned incorrect integer word observations. A fresh real
model capture with that explicit archive projection now passes host and target
execution with all six results, including exact BF16 selections and integer
views. A finite legacy observation model and a BF16-input contraction also pass.
The three complete-model tests retain thirteen results under
`.scratch/pytest-bfloat-runtime-model/`.

Computed F32 NaN payloads may differ between host and RISC-V under this numeric
profile. The exceptional observation case uses value comparison for those
computations and separately checks storage/word results exactly. Finite legacy
and contraction cases pass the all-word policy. No source numeric permission is
broadened to make a target result pass.

`runtime-inputs.npz` and its `t2m.runtime-input-projection.v1` derivation record
live beside the qualification, outside compiler staging. Only the projected
input archive reaches the build worker; the mixed input/reference word archive
and result references do not. Verification re-derives every projected byte from
the unchanged source bundle and rejects changed values even after outer hashes
and the build's archive digest have been rebound. Existing qualifications with
no projection retain their original input-file binding.

## Reproduce with existing tools

Select the existing tools through Merlin's normal environment variables:
`MERLIN_COMPILER_PYTHON`, `MERLIN_CLANG`, `MERLIN_MLIR_INSTALL`,
`MERLIN_RISCV_GCC`, `MERLIN_SPIKE` and `MERLIN_NM`. The metadata reader may be
the selected cross-toolchain's `nm`; no LLVM rebuild is required. Use the frontend
environment for the command and an existing Merlin environment for `--python`.

```bash
python scripts/qualify_spike_model.py \
  --capture out/source-assembly-learned-i64 \
  --out out/spike-model-i64 \
  --merlin-root /path/to/merlin \
  --python /path/to/merlin-env/bin/python --comparison words
python scripts/qualify_spike_model.py --verify \
  --out out/spike-model-i64 \
  --merlin-root /path/to/merlin \
  --python /path/to/merlin-env/bin/python
```

The command never installs or builds a toolchain. Each worker has a sixty-second
wall-time limit; the simulator has a thirty-second limit. The image requests a
64 MiB DRAM span, 4 MiB allocator arena and 2 MiB code reservation. Entry values
have a conservative 64 MiB bound before generation. Cases exceeding these
budgets require a deliberate follow-up, rather than an unbounded allocation.

The optional integration test takes a colon-separated explicit selection in
`T2M_SPIKE_MODEL_CAPTURES`, plus `T2M_MERLIN_ROOT` and `T2M_MERLIN_PYTHON`, and
runs `python -m pytest tests/test_spike_model.py`. No selected models/tools means
the ISA test skips; compiler-directory integrity tests still run. The existing
source-assembly and lifted-constant model tests document how to produce inputs.

## Boundaries and evidence

1. Verify the complete immutable source bundle and independent model result
   roster. The current execution edge requires `t2m.bound-model-receipt.v1` and
   a static ranked tensor public model ABI. Dynamic kernel scalars already
   specialized by model binding retain that source contract.
2. Call Merlin's actual `stage_compile_inputs`. Its directory contains no
   runtime input/reference archives or their individual digests. Reopen every
   declared member and reject undeclared files or symlinks.
3. Copy that compiler projection and add only the byte-identical `model.mlir`
   filename alias required by the whole-model builder. Use ordinary
   `spike_model.build(backend="scalar", full_readback=True)` with checked loop
   destination reuse and source-derived consumer capabilities. `--feature`
   additionally selects an explicitly requested existing native capability.
4. Supply original runtime inputs separately to the C harness generator,
   using the checked exact-word projection above for public BF16 arguments.
   **Bare-metal images embed those runtime values.** This differs from the
   reusable host-library invocation path: the build worker receives runtime
   inputs for harness generation, although compiler staging excludes them.
   Reference values and their paths never enter the build invocation. Original
   weights/constants remain legitimate compiler parameters.
5. Execute the actual linked ELF. Preserve raw binary stdout, typed complete
   outputs and metrics. Every original argument is observed readonly after
   forward and before state commit. This checks end-of-call bytes; it supplies
   no proof about transient writes or synchronization with external agents.
6. Compare every result outside compilation. The default `values` policy keeps
   exact integers, including values beyond F64's exact range, and floating
   tolerances (zero by default) with the existing NaN policy. Explicit
   `--comparison words` requires zero tolerances and retains every output bit,
   including signed zeros and NaN payloads. Use it for exact storage/state
   profiles. BF16 retains raw source-word archive storage. Empty tensors keep
   their original shapes. No reference policy changes compilation or source
   numeric permissions.
7. Publish `t2m.spike-model-qualification.v1` only after all gates pass. Pin the
   source receipt, compiler projection, tools, relevant consumer sources, ELF,
   raw output, observer records, output archive and references. A complete
   dependency-source closure remains unestablished and explicitly labeled.

Verification reopens source/custody relations, re-decodes the actual wire bytes
through an explicitly selected Merlin and repeats all result comparisons. It
does not run Spike or execute tools selected by saved JSON. Tests change output
and readonly-input payloads with valid recomputed checksums and updated outer
pins; both must still refuse. Injected evaluator data in compiler staging also
refuses. Full readback uses the additive zero-element `OUT_BIN v1` extension;
older positive-only readers reject those frames.

The v1 qualification's additive evaluation field `comparison: words` records
the stricter policy. Existing receipts without it retain their original value
comparison; verification never silently upgrades historical evidence. Older
readers that do not implement the stricter field refuse these qualifications.

## Remaining target work

Broader kernel/type/effect combinations, dynamic public kernel invocation,
physical/mutating model views and applicable accelerator materialization remain
open. Numeric-only model qualification refuses source prints without their own
effect reference. The host opcode registry does not infer target coverage from
these model runs. Keep target qualification separate from source equivalence,
algorithm-structure checks and performance measurements.

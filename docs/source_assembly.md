# Source assembly as explicit portable computation

`shared-tensor --assembly-provider NAME` selects an installed optional source
implementation from `triton2mlir.source_assembly.v1`. The provider decodes the
complete selected snippet and returns executable standard scalar arithmetic or
math. Tensor forms become parallel Linalg computation. Original tiles,
outer control flow, recurrence and result ordering remain visible to Merlin.
The core pipeline contains no ISA instruction cases. Source language selection
is independent of output hardware selection.

The separately packaged `adapters/ptx` provider, `ptx-integer-v1`, implements
checked PTX integer programs. No GPU, CUDA driver or native compiler build is
needed to use its standard MLIR output. Both API and CLI default to refusing
source assembly without a selected implementation. `bind-model` also accepts
the option globally or per kernel binding; complete-model qualifications are
recorded separately from standalone execution.

## Source and consumer boundary

The original native-verified TTIR, assembly string, register constraints,
purity/packing fields and typed SSA interface remain in the source graph.
Every original snippet is decoded before helper inlining or greedy rewrites;
a caller's purity marker cannot hide unsupported instructions as dead code.
The existing Meta/shared passes still own helper inlining, arithmetic/memory
conversion and subsequent normalization.

The provider interface supplies only source text/types and scalar SSA operands.
It receives no example inputs or reference outputs. The core independently
checks a closed, bounded, effect-free standard scalar DAG, complete typed
results and matching tile shapes. Unknown operations, addresses, free values,
parented operations and effects refuse. Source `pure=false` and packed forms
need further explicit contracts. There is no provider call or assembly opcode
in executable `model.mlir`.

Provider entry point, distribution/version and installed Python/metadata file
hashes are retained with every source expansion. Changes after import refuse
instead of relabeling cached code; files are reopened after expansion too.
These are implementation-custody records. They do not independently prove that
an arbitrary provider implements its source language correctly. Source
equivalence remains limited to its named specification/contract and qualified
cases. Native verification, numeric comparisons and applicable target execution
remain separate gates.

Public tensor/scalar ABI and receipt schemas are unchanged. Generated
`compatibility.json` explicitly records the source implementation boundary and
remaining effects/packing obligations. Original source IDs attribute the
one-to-many standard expansion; `prov.composite_family="source_assembly"`
retains the grouped computation on tensor regions.

## Native normalization and the arithmetic carrier

Stock Triton's descriptor/pointer normalizers do not register Tensor/Linalg.
An actual descriptor load → decoded tensor program → descriptor store failed
while parsing `tensor.empty`, before any descriptor rewrite. Protected source
now lifts the same checked scalar recipe to shaped `arith`/`math`, dense
constants and typed `tt.splat` broadcasts for captured scalars. The shared
arithmetic pass later supplies parallel Linalg. Multiple inputs/results,
predicate/select values, mixed integer widths and all original fields/SSA
edges retain their meanings; no decoder or source numeric policy changes.

The common `t2m.tensor_dag` carrier also serves early FP8 recipes. Explicit
global/per-cast FP8 choices and source implementations expand before native
normalization or helper folding. A complete source audit now precedes every
source-parser fallback, including unselected/default effects. When shared's
source-only parser rejects original types, the stock native no-pass reader
retains every field first. Composite precision receives the same original
snapshot and protected normalization before shared inlining. An unused private
helper cannot hide an unimplemented hardware observation behind a purity flag.

This reuses the existing upstream arithmetic conversion rather than a new
arithmetic backend. Reviewed implementation/tests:

- [Meta arithmetic conversion and legal dialects, 17f0c234](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp).
- [Meta typed splat/scalar conversion patterns](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp).
- [Meta two-dimensional arithmetic fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonArithToLinalg/convert_2d_elemwise_arith_binary.mlir).
- [Archived Microsoft registration, 050b93ae](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToLinalg/TritonToLinalg.cpp).

Provider API v1, source profiles, public ABI and capsule/receipt schemas stay
unchanged. Reports explicitly name `standard-elementwise-arith` where that
carrier is used. Original operation IDs and `prov.composite_family` retain the
algorithmic group; the source capsule/entry annotation retains complete original
provider contracts. Hardware observations, memory/synchronization, broader
packing/instructions and accelerator qualification remain open.

`tests/test_assembly_descriptor_intake.py` now qualifies six i32/i64 standalone
kernels: four owning-allocation recurrences and two original external-descriptor
field interfaces. The tuple programs retain mixed-width shift counts, signed
predicates, initialized predicated updates, two ordered results and full-width
word arithmetic. Runtime aliases, reversed strides, ragged/empty domains,
zero steps and oversized/negative-as-unsigned counts replay unchanged libraries
26 times. Unknown hardware observations refuse before native passes, including
unused private helpers and the unselected/default source path.

`scripts/capture_assembly_descriptor_model.py` and
`tests/test_assembly_descriptor_model.py` additionally qualify genuine i32/i64
model2MLIR captures. Learned word tensors, two dependent updates, an empty call
and the live original snapshot retain all four model results each. Original
weights, inputs, complete reference archives and source capsules remain
byte-identical. All six kernels and both models execute through native, Merlin
host and actual RISC-V/Spike; every one of the fourteen target results matches
exact words. Independently reopened receipts and the public original-source
allocation CLI capture are selected in `out/spike-assembly-descriptors/summary.json`.
This finite selection leaves the older 106-package inventory unchanged and
does not qualify broader instructions, effects, packing or accelerators.

## Optional PTX integer profile

Semantics follow [NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html):
integer arithmetic, comparison/selection, logic and shift sections. Input/output
words are i32/i64 with matching `r`/`l` constraints. Pack size is one. Checked
local register declarations and SSA register updates preserve source order;
initialized predicated writes become value selects.

| Source computation | Executable standard representation |
|---|---|
| Add/subtract, low multiply/multiply-add, signed negate/absolute | Modular integer arithmetic with original word width and no saturation/overflow assumptions. |
| Signed/unsigned min/max | Corresponding typed signed/unsigned arithmetic. |
| Bitwise AND/OR/XOR/NOT and word moves | Integer word operations; no floating transport. |
| Integer predicates and selection | Typed comparisons, i1 predicates and value selects. |
| Left/right shifts | Explicit unsigned-i32 count checks/clamping, safe shifts and source-correct zero/sign fill. No oversized MLIR shift is executed. |

The adapter lexes assembly structurally, including comments, declarations and
instruction separators. It does not parse printed MLIR or select behavior by
kernel/example names. Tokens, statements, local registers and generated payload
have explicit budgets. It checks the whole snippet, including trailing and
otherwise unused instructions. Clock/lane/device state, memory, synchronization,
unknown modifiers, carries, saturation, division, floating policies, clobbers,
tied operands, packed registers and general assembly control flow remain
located refusals. Predicated writes need an initialized destination.

The pinned Meta/shared and archived Microsoft converters provide no portable
assembly re-raising. Ascend `8f0ade66580245e30ad97093526a4018e4e18fc7`,
`third_party/ascend/lib/TritonToLLVM/TritonToLLVM.cpp`, preserves source assembly
in `LLVM::InlineAsmOp`, including packing and side-effect flags. Its
`test_asm.py` and `test_asm_scalar.py` separately exercise numeric ADD and a
physical system counter. The latter requires real hardware semantics and is
not arithmetic that may be discarded. This adapter reuses the existing
conversion pipeline after decoding specified pure source computation.

## Reproduction and qualification

Build the small optional wheel from a source copy under `.scratch`, using the
existing frontend environment's setuptools/build tooling. Install that wheel
into the xDSL 0.68 frontend environment, with dependencies already installed.
No Triton/LLVM source build is involved. Keep wheels/build products in ignored
scratch/output directories.

```bash
python -m t2m shared-tensor source.ttir --abi abi.json --out out/assembly \
  --assembly-provider ptx-integer-v1 --optimizer "$TRITON_SHARED_OPT" \
  --memory-profile guarded-buffer --address-policy source-words \
  --inputs inputs.npz --reference reference.npz --python "$T2M_NATIVE_PYTHON"
python -m t2m.examples.product_corpus --out out/assembly-corpus-inputs \
  --assembly-provider ptx-integer-v1
```

Sixteen actual word-program libraries qualify i32/i64 programs, rank-one/rank-two
tiles and runtime recurrences: fourteen use guarded source-word memory and
two i32 arithmetic cases additionally qualify tensor/buffer profiles. Each library replays
four additional domains, including empty input, zero trips and ragged multiple
tiles. Independent Python word references retain full-width overflow, extreme
signed values and oversized/negative-as-unsigned shift counts. The public CLI
also compiles a scalar source program through native and actual Merlin
execution. Provider boundary/refusal tests cover unknown effects/state,
initialization, instructions, malformed source and changed identities.

The optional maintained selection adds a qualified assembly case while keeping
the original default assembly refusal and asymmetric packed-map refusal. It
does not convert a refusal into success by changing a reported status. Source
opcode observations, instruction/profile coverage and applicable target
execution remain different measurements. Arbitrary assembly is still part of
the [full completion scope](completion_plan.md).

The retained i64 shift demo at `out/source-assembly-word64/` verifies its capture
and separate Merlin qualification. The selected corpus and coverage reports
live at `out/source-assembly-product-qualified/` and
`out/source-assembly-inventory-coverage/`. A separate development build receipt
at `out/source-assembly-provider/build-receipt.json` checks identical decoder
bytes in the checkout, built wheel and installed package. This Python-wheel
receipt does not establish source/build correspondence for the native
Triton/shared binaries.

## Complete model captures

`scripts/capture_assembly_model.py` captures genuine i32/i64 model2MLIR models
with nonzero public inputs, learned word tensors and typed shift counts. Two
dependent kernel calls retain the original assembly-defined recurrence; a third
call retains empty matrix values. All six original results preserve dtype,
shape and exact words through native and actual Merlin execution. Every parent
weight, manifest, input and complete reference archive remains byte-identical.
The i64 model additionally qualifies through public `bind-model`, exercising
both global and per-binding provider selection. The existing explicit dense
value-view policy preserves the public matrix/empty ABI.

The integration exposed a consumer destination-proof gap for scalar reads of a
preceding computed model tensor. [Merlin 5d3a62d](https://github.com/ucb-bar/merlin/commit/5d3a62dcad5be54faf7fada0706dd143ead4b74d)
recursively checks complete pure read producers before applying its existing
destination/copy mechanism. Unknown/effectful/init-dependent/library-backed
producers still refuse; source reads and all native alias/lifetime gates stay
intact. Its 107 destination and 34 ownership checks pass, together with the
structure/documentation gates. Both models were requalified against that
published consumer revision.

The verified bundles at `out/source-assembly-learned-i64/` and
`out/source-assembly-learned-i32/` each bind eighty artifacts and retain every
parent result. The explicit
report at `out/source-assembly-model-inventory/coverage.json` reopens 72
packages, 47 standalone qualifications and fourteen corpus cases. All 50
definition names remain observed in native/Merlin-qualified captures under
selected profiles. This adds complete-model composition evidence; it supplies
no arbitrary-assembly, universal numeric/effect or target qualification.

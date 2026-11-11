# Source assembly as explicit portable computation

`shared-tensor --assembly-provider NAME` selects an installed optional source
implementation from `triton2mlir.source_assembly.v1`. The provider decodes the
complete selected snippet and returns executable standard scalar arithmetic or
math. Tensor forms become parallel `linalg.generic` regions. Original tiles,
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

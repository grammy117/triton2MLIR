# Complete native source serialization before normalization

`t2m.frontend.read_ttir_generic(source, work)` uses the pinned stock Triton
library to verify and print original TTIR in generic syntax. It runs no compiler
pass. Native reparse, verification and an exact generic print round trip must
succeed before the caller can inspect operations, properties, attributes,
regions, block edges, types or constant spellings.

This closes a source-version gap in the installed shared optimizer: it rejects
original block-pointer types and the stock wheel's older descriptor spelling
before producing generic IR. Its eventual pointer/tensor transformations remain
useful. The new reader exposes the complete original graph first, so per-cast
FP8 contracts can expand into standard elementwise arithmetic before those
official transformations run. Missing or malformed source fields cannot vanish
through an earlier canonicalizer or an incomplete native attribute getter.

## Reused native APIs and bounded binding

Triton's Python `module.str()` binding constructs MLIR `OpPrintingFlags` without
exposing generic-format options. MLIR's constructor also reads its registered
printer options. The worker therefore invokes the loaded library's no-argument
`mlir::registerAsmPrinterCLOptions()` export, then LLVM's C API
`LLVMParseCommandLineOptions` with exactly these flags:

```text
--mlir-print-op-generic
--mlir-print-local-scope
--mlir-print-debuginfo
```

The registration uses its pin-probed Linux/Itanium exported name. No C++ object
layout, source operation pointer or foreign MLIR context crosses ctypes. Both
functions come from the actual loaded stock library. Missing exports, rejected
flags, nongeneric printing or failed verification/round-trip refuse. This is a
version-specific binding, not an unrestricted compiler-option interface.

Printer options are global inside LLVM, so only a fresh isolated worker selects
them. The caller's printers remain unchanged. Input is limited to 4 MiB, source
graphs to 200,000 operations, each output file to 16 MiB and execution to 30
seconds. Compiler scratch stays under the requested project work directory.
The `t2m.native-source-print.v1` report binds original/output hashes, flags,
verification outcomes and actual native-library bytes/hash. Surveyed source
revisions are explicitly not binary build receipts. No LLVM/Triton build or
Triton-CPU backend is required.

Primary implementation references:

- [Stock Triton printer binding, f797708c](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/src/ir.cc).
- [Its surveyed LLVM pin](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/cmake/llvm-hash.txt).
- [MLIR printer options and default flags, 1f126a6d](https://github.com/llvm/llvm-project/blob/1f126a6dea50d185c0781743a667390037ae88bd/mlir/lib/IR/AsmPrinter.cpp).
- [LLVM C argument-parser declaration](https://github.com/llvm/llvm-project/blob/1f126a6dea50d185c0781743a667390037ae88bd/llvm/include/llvm-c/Support.h).

## Package boundary and qualified scope

The per-cast converter selects this reader when shared's source-only parser
cannot accept the original source. It retains the complete original preflight
graph, validates every source cast, and expands the existing arithmetic recipe.
Official pointer/descriptor normalization then precedes shared helper inlining.
Normalized operations receive a separate preflight and ID namespace. Original
descriptors, masks, logical grid and numerical intent remain explicit.

Compatibility metadata records the native reader separately from the shared
optimizer. The native serialization report appears in provenance; the original
source, complete typed graph and executable program remain receipt-bound.
Public tensor/scalar ABI and package receipt schemas are unchanged. Older
producers refuse these per-cast/source-version compositions. Serialization
establishes a checked reader boundary, not numerical/source equivalence, memory
permission or target qualification.

Six original-pointer/descriptor cases combine three independent FP8 policies
and mixed RTNE/RTZ casts across tensor, buffer and guarded/source-word routes.
Ragged and completely inactive programs preserve source boundary checks; all
six computed results qualify through native, Merlin host and actual Spike.
Two additional ragged tiled GEMMs combine the same original source forms with
BF16x3 or TF32x3, retaining the K loop, pointer advances, partial contractions,
original cast/dot intent and all active output tiles through the same three
execution boundaries. The eight target receipts are independently reopened in
`out/spike-native-source-fp8/summary.json`; a descriptor case also passes the
public `shared-tensor` CLI. This selection leaves the older 106-package
inventory unchanged.
Separate tests preserve nested dictionaries, exact word constants and malformed
semantic fields, reject missing exports/invalid native IR, and keep caller
printing unchanged. Malformed casts refuse before any normalization. The fifteen
reader/pointer/contraction cases and 55 source/cast/descriptor regressions pass.

Broader source versions, external descriptor argument/binding compositions,
enclosing model captures, general numeric/effect closure and accelerator
correspondence remain requirements of the full completion plan. Logical
descriptor support supplies no physical descriptor bytes or hardware placement.

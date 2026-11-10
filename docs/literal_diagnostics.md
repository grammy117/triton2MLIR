# Length-aware literal diagnostics

Source `tt.assert` and `cf.assert` now retain complete UTF-8 messages containing
embedded NUL. Scalar and tensor conditions still use the existing Meta
triton-shared `17f0c234ce474bee334af36e351c62eca7852836` AssertConverter;
the archived Microsoft converter registers that same conversion. Portable
`model.mlir` retains ordinary `cf.assert` at its original control positions,
with source/emission correspondence. No GPU printf ABI or physical placement
enters the frontend.

The ordinary upstream LLVM assertion lowering calls `puts`, which truncates
at a NUL. Merlin `c79a5717a8aad31eb7c7578adae913dfa078046a` provides the explicit
`lower_length_aware_diagnostics` host feature. It runs immediately before CF
conversion, after normal ownership/lifetime checks, and expands actual standard
assertions into an internal check and byte-counted `putchar` loop followed by
`abort`. Private literal globals contain the full UTF-8 message plus linefeed.
The helper body remains in native LLVM; an opaque external effect does not
replace the computation. Incompatible runtime declarations refuse before edits.

The native evaluator and Merlin compiler select this feature from actual
standard assertions containing NUL. Annotations, expected streams and example
values grant no lowering semantics. Native verification requires the explicit
`T2M_MERLIN_ROOT` support checkout, just as other optional native adapters do.
The existing owned evaluator stdout policy makes bytes observable before abort.
Usable host stdio is a runtime precondition; accelerator diagnostics remain open.

Native MLIR prints non-ASCII string bytes as escapes. Both intake parsers
normalize such attribute tokens to `StringAttr` only after lossless UTF-8
decoding. Embedded NUL remains part of the string. Non-UTF-8 byte attributes
and raw dense tensor storage keep their existing byte representations.

## Package and interface differences

Public tensor/scalar/void signatures, portable operations and receipt schemas
are unchanged. Effect JSON references already encode NUL and Unicode losslessly.
Expected streams still come from the caller and remain excluded from compiler
inputs. Initially failing BF16 or atomic kernels retain real declared results
and publish no values.

New observable-outcome packets additionally bind `native/compiler.log`.
Native adapter progress is written there, independently of source stdout/stderr.
This matters for integer CAS: compiler progress cannot become a kernel
diagnostic. The log is additive compiler evidence; original v1 packets remain
readable, and no reference value or compiler log enters Merlin's staged input.
Compatibility metadata explicitly declares the selected native support feature.

## Running the visible example

```bash
T2M_MERLIN_ROOT=/path/to/merlin python -m t2m effect-demo \
  --case literal-check --out out/length-aware-diagnostics \
  --optimizer .scratch/toolchain/triton-shared-opt --python /path/to/native-python \
  --memory-profile guarded-buffer --address-policy source-words
python -m t2m qualify-merlin out/length-aware-diagnostics \
  --out out/length-aware-diagnostics-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

The source message contains two NULs, Unicode, percent signs, quotes and a
newline. The independent expected outcome is assertion failure, not normal
return. The serialized standard MLIR executes separately through native LLVM
and a freshly compiled Merlin library.

The nine end-to-end cases cover all three memory profiles, scalar source loops
and branch positions, rank-two tensor checks, dynamic empty/tail domains, exact
failure bytes, reused passing/failing libraries, original CF messages and a
truncated-reference refusal. BF16 transport and integer CAS also compose with
the diagnostic seam while retaining actual result signatures and ownership.
Merlin's 70 native, serialization and BF16 regression cases pass, including
runtime ABI refusals, helper-name collisions, exact dense bytes and source
provenance retention.
The combined source/effect/diagnostic regression passes 75 cases. The saved
`out/length-aware-diagnostics` package and separate Merlin qualification pass
29 and 49 artifact checks respectively; the original 28-artifact v1 packet also
still verifies.
The explicit 18-package observation report retains the pinned 50-op denominator:
24 definitions are observed and 20 are present in captures of Merlin-qualified
packages. These remain composition observations, not per-op execution or
universal type/attribute coverage.

This closes literal UTF-8 assertion transport. A separate
[logical print profile](logical_prints.md) now covers ordinary numeric and
literal source events. Broader `tt.print` formatting
and stream ordering, arbitrary binary diagnostics, external/assembly effects,
synchronization and accelerator materialization still need their own contracts.

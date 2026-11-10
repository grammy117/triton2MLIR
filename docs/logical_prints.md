# Logical source print events

`tt.print` now lowers into standard `vector.print`, `tensor.extract`, arithmetic
and structured loops. Actual printed values, signedness, hex word decoding,
literal prefixes, source control positions and logical coordinates remain
executable. `prov.source_prints` and source/emission IDs retain the composite
intent; removing provenance does not change the output.

## Reuse and the hardware boundary

Triton `f797708c0626e5f9840ca5b0a98790e2c7cb09ad` supplies the primary semantics:

| Source | SHA-256 |
|---|---|
| `lib/Conversion/TritonGPUToLLVM/PrintOpToLLVM.cpp` | `d0b7252243ae60d36bd53437e781e5eeb8512af20d861cdabd0be01f8c29d6f1` |
| `python/test/unit/language/print_helper.py` | `a322d49713f0aa219b989d0738143d74cd99fd1003d3da9c5c5a9c4a29f2cd7f` |
| `python/test/unit/language/test_subprocess.py` | `9da034673a1e159dc76eb4c3dfc8f3d498cade8691da3536a2422d0453d43ad8` |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp` | `11318c5ddc1136cc1f52c8e4e86596c6883d3378bedc76ea20600bb2cc7e13d6` |

Those tests check per-operand tensor coordinates, signed/unsigned values,
fixed decimal floats and fixed-width integer hex. They also expose physical
replication: scalar/no-argument events can repeat once per GPU thread. The
pinned interpreter explicitly uses a different format. Ascend
`8f0ade66580245e30ad97093526a4018e4e18fc7` DevicePrintConverter and its device
tests preserve a separately implemented print interface; their declaration is
not a portable implementation that Merlin can execute.

The Meta shared optimizer preserves registered `tt.print` through its tensor
and memory stages. We retain that verified operation through the existing
passes, then expand at the logical tensor/memory ABI boundary. No dialect
verification flag is relaxed, no temporary opaque function is accepted as the
final implementation, and all surrounding computation/memory still uses the
actual upstream passes.

## Declared portable profile

Emit **one event per logical program, operand and tensor element**, in operand
and row-major element order within a program. Include the three logical
program coordinates, tensor-local indices, literal prefix, optional operand
number and value. A scalar has an empty index tuple. A no-argument operation
prints its prefix once per logical program. Absent grid axes have coordinate
zero; the grid remains an unmapped iteration space.
The coordinate list is complete; it does not inherit NVIDIA's printf
argument-count truncation.

Physical thread replicas and hardware interleaving are abstracted explicitly.
The selected host consumer serializes the logical grid; the frontend does not
bind a warp, thread/core count, target instruction or physical printf buffer.
Print-before-assert order and zero-trip/untaken source regions remain executable.

Signed integers widen with their source sign. Unsigned64 decimal conversion
uses bounded modular unsigned division with no float conversion; hex uses
explicit word bitcasts/shifts and literal digits. Floating hex retains the
original f16/BF16/f32/f64 words, including NaN payloads and signed zero.
Decimal f16/BF16/f32 widen to f64; the selected software profile prints six
fixed decimal places, infinities and signed NaNs in the C numeric locale.
This fixed formatting is an explicit implementation profile of standard
human-readable `vector.print`; the stock MLIR runner-utils float formatter
uses a different decimal representation.

Floating hex is an explicit **original-typed-word software profile**. The
pinned GPU tests cover integer hex; NVIDIA's implementation promotes narrow
floats before the hex-format varargs path. We retain original float words
without claiming byte-identical device varargs behavior. The native GPU
execution path is not qualified by these software observations.

Source prefixes are literal data, including percent characters. Saved TTIR
may contain NUL/UTF-8 prefixes, which use byte-counted output. The source
Python Triton wrapper has its own printable-prefix restrictions.
Physical pointer values still need a declared logical-address contract.
Packed FP8/other carriers, effectful combiner regions and accelerator
diagnostic handling remain separate gates.

## Native implementation and package differences

Merlin's explicit `lower_scalar_diagnostics` feature materializes scalar/string
debug operations after ordinary buffer ownership/lifetime checks. Complete
internal LLVM helper bodies use byte-counted `putchar` for literals and
constant, typed `printf` formats for numbers. Source prefixes never become
format strings. The feature includes length-aware assertion lowering, refuses
incompatible runtime definitions before edits, and preserves source locations.
Both native evaluators select it from actual operations and require the
explicit `T2M_MERLIN_ROOT` support path.
The required Merlin implementation is
`41d2a1ed5639b0dd6c8fb5fc013256265b1a40d6`.

Public signatures and receipt schemas are unchanged. Compatibility metadata
declares the logical event/formatting/runtime profile. References and native
observations remain excluded from Merlin's compiler inputs. Native entry
spelling is safely renamed when it collides with `printf`/`putchar`/`abort`;
the portable source/ABI stays unchanged.

Void/read-only packets now qualify returning print effects; initially failing
tensor kernels retain their real result declarations and exact print/failure
streams. A complete returning-tensor packet still needs a mixed numeric and
observable-output reference contract. Existing numeric-only comparisons do not
qualify diagnostic streams. This remains an explicit next package gate.

## Visible demo and evidence

```bash
T2M_MERLIN_ROOT=/path/to/merlin python -m t2m effect-demo \
  --case print-hex --out out/logical-print-hex \
  --optimizer .scratch/toolchain/triton-shared-opt --python /path/to/native-python \
  --memory-profile guarded-buffer --address-policy source-words
python -m t2m qualify-merlin out/logical-print-hex \
  --out out/logical-print-hex-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
```

`print-words` uses decimal output. Both real kernels have a masked readonly
input, runtime dimensions/loop/control arguments and a logical grid. Their CPU
references use independent Python integer/string operations, not the compiler's
word decoder.

The 53 focused cases pass across all three memory profiles: i1/i8/i16/i32/i64
signedness/casts and u64 extrema, decimal/hex f16/BF16/f32/f64, subnormals,
NaNs/infinities/signed zeros, scalar/multiple/rank-two arguments, index padding,
multi-axis grids, empty/tail/full domains, loops/branches and exact source
failure order. A compiled library is reused with new inputs. Additional cases
check literal NUL/Unicode/percent bytes, physical-pointer/schema refusals,
runtime symbol collisions, empty static operands with zero events,
integer CAS with real result declarations, and
execution with all provenance stripped. Another 68 existing frontend/effect/
coverage regressions, seven public CLI cases and 84 Merlin binding/native/
staging regressions pass. The visible hex package/qualification pass 32/51
artifact checks.
The explicit 19-package registry retains all 50 definitions, with 25 observed
and 21 present in captures of Merlin-qualified packages. It records source
compositions and exact outcome policies, without per-op execution or universal
type/attribute claims.

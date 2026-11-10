# Tensor results and observable effects from one kernel call

A returning Triton kernel can now produce every functional tensor result and
its logical diagnostic stream through the same native/Merlin interface. The
source program, public tensor/scalar ABI, tiling, masks, recurrence and reuse
remain executable. A source readwrite buffer becomes an incoming readonly
tensor version and a returned updated version. Provenance preserves the
source intent; it does not replace any computation or effect.

The visible demo updates a tiled i64 state in a runtime loop, prints the
updated state inside an alternating branch, then returns the state and a
second computed tensor. Values above `2^53` retain their low bits. It uses
independent CPU arithmetic and formatting for its references.

```bash
export TRITON_SHARED_OPT=/path/to/triton-shared-opt
export T2M_MERLIN_ROOT=/path/to/merlin
python -m t2m effect-demo --case mixed-results \
  --out out/mixed-results --python /path/to/native-python
python -m t2m qualify-merlin out/mixed-results \
  --out out/mixed-results-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/frontend-python --python /path/to/native-python \
  --merlin-clang /path/to/compatible-clang
python -m t2m verify-bundle out/mixed-results
python -m t2m verify-merlin out/mixed-results-merlin
```

## Explicit reference version and package difference

`t2m.effect-reference.v2` adds a complete typed numeric reference to the exact
UTF-8 stream and termination reference:

```json
{
  "schema": "t2m.effect-reference.v2",
  "termination": "return",
  "stdout": "independently computed exact logical stream\n",
  "stderr": "",
  "results": {
    "archive": "reference.npz",
    "comparisons": [
      {"kind": "words"},
      {"kind": "values", "rtol": 0, "atol": 0.000001}
    ]
  }
}
```

The adjacent `reference.npz` contains exactly `out0`, `out1`, ... in actual
ABI result order. Every slot must match its declared dtype and resolved runtime
shape. The archive is limited to 64 MiB of primitive tensor data; ZIP members,
NPY headers, shapes and uncompressed byte counts are checked before allocation.
Object arrays, duplicate/extra/missing slots and noncanonical Boolean storage
refuse. Per-result policies are mandatory; integer value policies require zero
tolerances. Floating tolerances must be explicit, finite and nonnegative.

`words` compares the original carrier bytes, preserving signed zeros, NaN
payloads and wide integer low bits. `values` uses the existing logical tensor
comparison. BF16 retains its uint16 carrier and decodes only for a value
comparison. A numeric tolerance never relaxes diagnostic bytes. Existing i1,
i8/i16/i32/i64, f16/BF16/f32/f64 carriers remain distinct from broader numeric
operation support.

The additional explicit policy `{"kind":"ieee"}` compares every finite,
infinite and signed-zero word exactly, while requiring NaN positions to match
without choosing their payload, sign or signaling representation. It accepts
only declared f16/BF16/f32/f64 tensors, with no tolerances or extra options.
This permits exact rounding checks for operations whose NaN result class is
defined but whose NaN payload is unspecified. It supplies no implementation
or rounding permission to compilation. Older v2 readers that do not implement
this policy must refuse its unknown kind; the original `words` and `values`
policies keep their meanings. NaN representation-sensitive source code still
requires an exact word reference and an applicable implementation contract.

The outer `t2m.effect-kernel-receipt.v1`, qualification and invocation schemas
remain unchanged. The nested v2 reference and added archive are explicit
extensions: older readers must refuse an unknown reference version. Original
v1 void and initially failing packets remain readable. A source assertion
failure still uses reference v1, retains the actual declared tensor results,
and publishes no completed output values. Reusing that library for a successful
call requires the full v2 tensor reference.

This differs from model2MLIR's numeric `golden.npy`/`extra.npz` evaluator layout.
The standalone packet has no invented FX graph or fake tensor golden; its
`extra.npz` is empty, and `compatibility.json` records the difference. The
compiler consumes the same standard tensor computation and actual public
signature through Merlin's existing compiler-only staging interface.

## Compiler and evaluator separation

`write_effect_bundle(..., references=tuple_of_arrays)` validates and freezes
the independent references, then observes one real native invocation without
passing expected values to that worker. It writes the expected archive only
after execution. The capture verifier reopens and compares the complete
observed output tuple, streams, termination and completion record.

`qualify-merlin` independently compiles the whole portable body before copying
the references into its evaluator directory. Merlin excludes `reference.json`,
`reference.npz`, invocation values, observations and individual evaluator
digests from compiler inputs. It preserves every actual result type. A separate
host observer invokes the produced library once; all tensor and diagnostic
comparisons refer to that same call, exact arguments, program and library.

`invoke-merlin` reuses that qualified library with new inputs and an adjacent
JSON/NPZ reference pair. Invalid result counts, types, shapes and policies
refuse before the library runs. Returned values and streams are independently
checked afterward; a failed comparison publishes no successful invocation
receipt. Reopening a receipt recomputes the comparisons, so updating artifact
hashes alone cannot conceal a changed result or effect.

Saved TTIR uses `shared-tensor --inputs inputs.npz --effect-reference reference.json`.
The v2 policies replace the separate `--reference`, `--rtol` and `--atol`
options for this profile.

## Retained computation and verification scope

The structured path may use upstream's supported `skip-prepass=true` only
after proving all SCF regions numeric and their state uses data-only. This
avoids the installed binary's i64 pointer-state boxing defect, without
narrowing accumulators, unrolling loops or hiding regions from conversion.
Pointer/mask uses and mixed memory recurrences retain the original prepass;
the [source-word profile](source_words.md) remains independently checked.

Native bufferization explicitly marks functional tensor inputs readonly using
the [upstream argument contract](native_execution.md). The actual post-call
input-byte check remains active. Merlin's private-state, alias and lifetime
checks also remain active.

Tests cover all three memory profiles, empty/single/tail domains, runtime loop
zero trips and branches, initial state, source failure after printing, library
reuse, all nine existing tensor carriers with exact word references, and mixed
exact/approximate per-result policies. Rehashed output, stream and completion
changes must fail capture, qualification and reused-invocation verification.
The package registry records `observable-effects-and-tensors` plus actual
per-result policies, without granting per-op execution or universal coverage.

Logical printing still explicitly abstracts physical thread replicas and
interleaving; see [the print contract](logical_prints.md). Pointer/FP8
diagnostics, external/assembly/synchronization semantics, broader low-precision
arithmetic, runtime/alias models, the maintained corpus and applicable staged
target materialization remain required work.

The saved demo at `out/mixed-results` and its actual Merlin qualification at
`out/mixed-results-merlin` pass 26/45 artifact checks. A second invocation at
`out/mixed-results-reused` changes the runtime extent from 13 to 17 and the
loop count from three to one, reuses the same library, and passes all 25
invocation artifact checks. The independent caller JSON/NPZ pair is retained
at `out/mixed-results-caller`. The explicit 20-package report at
`out/mixed-result-coverage/coverage.json` still observes 25 of 50 definitions,
with 21 present in captures of host-qualified packages. These counts describe
the selected source compositions rather than universal operation support.

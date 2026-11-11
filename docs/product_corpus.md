# Maintained product kernels, refusals and compiler failures

`product-coverage` measures the actual saved-TTIR-to-Merlin path, rather than
only an upstream pass or an explicit selection of successful packages. Every
selected case remains in the report: valid source, invalid source, conversion
refusals, failed references/compilations and cases not started within a budget.
All 50 pinned TTIR definitions remain in the denominator.

[Proved layout-independent packed maps](packed_maps.md) now add a successful
rank-two i64 case with runtime recurrence. The current copy at
`out/packed-map-product-qualified/` contains thirteen cases, eleven actual
Merlin libraries and two located refusals (asymmetric packing and source
assembly). The combined `out/packed-map-inventory-coverage/coverage.json`
retains 50 observed source definitions and 49 in native/Merlin-qualified
captures. The earlier measurements below retain their original code and scope.

The previous mapped-read corpus at `out/memory-map-product-qualified/` retains
its twelve cases, ten actual libraries and two refusals. Its report at
`out/memory-map-inventory-coverage/coverage.json` joins 67 packages and 46
standalone qualifications. See [the map memory contract](memory_maps.md).

## Reproduction

```bash
python -m t2m.examples.product_corpus --out out/product-inputs
python -m t2m product-coverage out/product-inputs/manifest.json \
  --out out/product-measurement --optimizer "$TRITON_SHARED_OPT" \
  --python "$T2M_NATIVE_PYTHON" --merlin-root "$T2M_MERLIN_ROOT" \
  --merlin-python "$T2M_MERLIN_PYTHON" --merlin-clang "$T2M_MERLIN_CLANG"
python -m t2m package-coverage --corpus out/product-measurement \
  --out out/product-observations
```

The preparation module captures real source kernels and ships one saved
pre-layout shape fixture. Independent NumPy/Python references are prepared
before conversion; the compiler never derives behavior from case names or
reference values. Inputs/outputs are ordinary generated artifacts, not tracked
weights or binaries. Block pointers remain in the corpus despite their
upstream deprecation, since pinned TTIR still defines them.

`out/ttir-product-qualified/` is a verified copy of the earlier maintained
test run: twelve cases, nine actual Merlin libraries and three located
refusals. Its standalone original-source counts are 33 observed definitions,
with 32 represented in native/Merlin-qualified captures. The cases cover shape
join/split/cat/trans/unsplat, logical program count, indexed tile gather, scalar
maps, histogram, i64 high products, ragged block-pointer GEMM, noncommutative
tuple scans, masked softmax and exact pointer round-trips. Runtime map,
histogram and high-product libraries replay empty/tail domains; other libraries
replay changed inputs. Packed maps, memory-accessing maps and source assembly
retain their actual refusal diagnostics and original typed graphs.

The softmax case exposed a missing guarded-memory allocation admission; the
[fresh tensor allocation fix](guarded_reductions.md) now preserves its original
standard allocation and qualifies through Merlin. Exact pointer round-trips
already fold through the supported upstream path; no physical address value is
invented. Pointer-to-integer observations that escape the round-trip still
need their own address-space/consumer contract.

The combined explicit report at `out/ttir-full-inventory-coverage/coverage.json`
reopens 67 earlier packages, 46 standalone qualifications and this twelve-case
corpus. It observes **50/50 source definitions**, with **49 represented in
native-qualified and Merlin-host-qualified captures**. The one definition
without a host-qualified capture is `tt.elementwise_inline_asm`.
These counts are composition observations, not universal opcode/type/attribute
support or measurements of which individual operations executed. Arbitrary
packed maps, broader mapped reads, source assembly, numeric/runtime/alias forms
and applicable targets remain completion requirements.

## Manifest and stage contract

A `t2m.product-corpus.v1` JSON object contains a `cases` list. Each case has a
unique local `id`, a relative `source` path, an `abi` object or relative JSON
path, explicit compiler `options` and optional independent `examples`:

```json
{
  "schema": "t2m.product-corpus.v1",
  "cases": [{
    "id": "kernel",
    "source": "kernel/source.ttir",
    "abi": "kernel/abi.json",
    "options": {
      "memory_profile": "guarded-buffer",
      "address_policy": "source-words"
    },
    "examples": [
      {"inputs": "kernel/inputs0.npz", "reference": "kernel/reference0.npz"},
      {"inputs": "kernel/inputs1.npz", "reference": "kernel/reference1.npz"}
    ]
  }]
}
```

ABI inputs use exactly `in0`, `in1`, ...; references include exactly `out0`,
`out1`, ... for every returned value. Comparison tolerances default to zero
and remain explicit in `rtol`/`atol`. Existing allocation bindings can be
supplied as `allocation_bindings`. The allowed compiler options match the
saved-source frontend: memory/address/effect profiles, f32 atomic-add policy,
FP8 overflow policy and logical byte order.

Effect-only, failing and mixed tensor/effect cases use
`reference_kind: "effects"`, JSON outcome references and, for mixed v2
references, an adjacent result archive supplied through each example's
`results` path. The original effect ABI, exact streams, termination and
per-result policies stay authoritative; no placeholder tensor is introduced.

| Stage | Actual evidence |
|---|---|
| `source` | Pinned native Triton parser/verifier, exact source bytes and original SSA/block inventory. No normalization or target backend. Invalid source produces no success inventory. |
| `conversion` | Actual functional/effectful standard MLIR conversion, semantic ABI and retained provenance. A located refusal remains unqualified. |
| `native_package` | Serialized MLIR compiles/executes against the first independent example; complete package integrity verifies. |
| `merlin_host` | Compiler-only staging, whole-body native library compilation and first-example reference comparison, with evaluator inputs excluded from compilation. |
| `merlin_replays` | Remaining examples invoke the same qualified library, preserving its receipt and compiled bytes. |

Native example counts and Merlin example counts stay separate. A conversion
without references is reported as converted without execution evidence.
An expected source assertion failure uses the effect outcome contract; a
compiler crash, invalid source or wrong numeric reference stays a failed case.
Diagnostics keep the conversion-reported source namespace, separate from
the native original-source graph. No source-equivalence mapping is fabricated.

The runner is serial, accepts at most 256 cases/sixteen examples per case,
checks the time budget between cases and retains existing per-tool timeouts.
Sources and archives have explicit byte/expansion budgets; paths are local and
nonsymlink. RUN comments and arbitrary commands are never executed. The exact
T2M Python source fingerprints and actual optimizer identity are recorded;
compiler-file changes during measurement abort instead of mixing versions.

`verify_product_corpus(out)` reopens case/source fingerprints, packages,
qualifications and reused-library receipts, and reconstructs aggregate status.
Changed executable/refusal/record bytes and forged all-qualified summaries
refuse. This verifies recorded custody and measured execution, not a universal
proof of source equivalence or accelerator support.

## Package coverage v2

`package-coverage` keeps its default v1 package-only report. Supplying one or
more `--corpus` selections emits `t2m.package-coverage.v2`, reopening each
corpus before joining observations. `product_cases` and `corpora` remain
distinct from `kernel_cases`, `packages` and standalone `qualifications`.
Source observations include valid refused cases; native and host counts do
not inherit that success. Each opcode retains attributed product refusals.
Native-invalid and unstarted cases remain in the case denominator without
inventing a successful source inventory. Source/normalization/target evidence
remain separate from the [full completion requirements](completion_plan.md).

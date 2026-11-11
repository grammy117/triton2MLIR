# Source effects through the Merlin target interface

`scripts/qualify_spike_effect.py` carries an existing
`t2m.effect-kernel-receipt.v1` package through ordinary Merlin staging, RISC-V
generation and actual Spike execution. Real void entries, initially failing
tensor entries and returning tensors plus diagnostics keep their signatures.
Standard `vector.print`/`cf.assert`, loops, logical grids, computation and
provenance remain executable. No placeholder tensor or reference enters MLIR.

Merlin's optional `effect_observation_bytes` requires a checked runtime
signature and complete readback. The bridge marks invocation immediately before
the actual source call, after scalar transport is ready. Source stdio is emitted
in a checked binary byte frame. NUL, Unicode and protocol-like text remain
payload. The at-most-4-MiB buffer is included in image layout.

A returning call emits its source frame/return marker, every result, original
readonly input observations and normal DONE. A source assertion emits its
source frame/assertion marker and readonly observations before the selected
assertion exit. It emits no tensor or DONE, even when its signature has results.
Setup failures, traps, timeouts, missing phases, unknown formats and capacity
exhaustion refuse. Exit code or diagnostic text alone cannot prove an assertion.
`run_process` retains raw process streams and status; numeric `run_raw` still
refuses every nonzero exit. Simulator stderr is separate from source stdio;
source stderr is empty under this implementation. Verification reopens raw
frames, runtime ABI, complete input observations and decoded streams/results.

The diagnostic runtime supports integers and exact nearest-even fixed-six
binary64 formatting, including subnormals, extrema, signed zeros and signed
NaNs. Integer scaling changes no source arithmetic or FP environment. Its
independent libc test compares 14,336 binary64 cases. Decimal Triton prints
classify NaNs from original words before widening, emitting their source sign
with standard integer/SCF computation. Finite/infinite values keep typed
extensions; hex printing retains original words. A missing typed half-extension
helper uses the pinned LLVM compiler-rt specialization with its license in
release resources; all 65,536 binary16 words pass word and typed ABI checks.

Runtime archives bind argument ordinals through a complete input manifest.
A source argument named `state` cannot select an unrelated legacy model alias.
Each target image embeds its invocation; kernel MLIR/objects can remain
identical across changed values. Existing worker/simulator budgets still apply.

The external receipt is `t2m.spike-effect-qualification.v1`, with status
`matched-expected-outcome` and scope `functional-riscv-source-effects`. Existing
v1 outcome and v2 mixed references retain their meanings. Every returning tensor
needs its explicit word/IEEE/value policy; diagnostic bytes always match exactly.
Changed calls require `--inputs` and `--reference` together, with a complete
`reference.npz` beside mixed JSON references. References stay outside compiler
staging and image-build inputs. See [package differences](package.md).

```bash
python -m t2m effect-demo --case mixed-results --out out/effect-target-source \
  --python /path/to/native-env/bin/python
python scripts/qualify_spike_effect.py --capture out/effect-target-source \
  --out out/effect-target --merlin-root /path/to/merlin \
  --python /path/to/merlin-env/bin/python
python scripts/qualify_spike_effect.py --verify --out out/effect-target \
  --merlin-root /path/to/merlin --python /path/to/merlin-env/bin/python
```

Tests cover real void/assertion/literal/integer-print demos; mixed state/loop/
print/results across all memory profiles; empty/single/tail domains and failures
after printing. One mixed object handles five runtime domains per profile.
Further cases cover decimal/hex f16/BF16/f32/f64 on a multidimensional grid and
printing plus exact returned words for all nine tensor carriers. Rehashed
decoded streams must still agree with raw target frames.

The reverified selection at `out/spike-source-effects/summary.json` pins
41 target invocations: 33 returns, eight expected assertions and all 35
returning tensor results. The source-effect/print suites pass 82 tests;
39 existing numeric/scalar/model regressions also pass. Merlin's runtime suite
passes 197 checks with two optional bitcode checks skipped for missing
`llvm-as`. Its formatter/half/release-resource checks pass separately.
The public source is `out/target-effect-mixed-source`; returning and failing
CLI invocations are `out/target-effect-mixed` and
`out/target-effect-mixed-failure`. Both preserve the same kernel-object hash.

This is finite functional CPU evidence under the existing logical print
profile. Physical replicas/order, external participants, synchronization,
physical addresses/descriptors, general model views, broader compositions,
accelerator execution and performance remain completion work. Custody and
comparisons do not prove universal source equivalence or dependency closure.

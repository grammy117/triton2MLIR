# Dynamic model packages and reusable Merlin invocation

The real two-call copy and weighted-XOR models now pass complete package writing,
native execution, Merlin host execution and actual RISC-V/Spike execution in the
tensor, buffer and guarded-buffer profiles. Their public input/result types stay
dynamic. Full arbitrary-kernel coverage remains incomplete.

```python
first = copy(x, self.weight, x.shape[0])
second = copy(first, self.weight, first.shape[0])
return second, first, x, self.weight
```

The weighted kernel computes `x ^ weight[0]`; applying it twice recovers `x`.
The unweighted variant returns three copy/input results. Both use actual
`torch.export` dimensions with a source domain of zero through 31. The saved
length 13 is an invocation example, never a compiler extent.

## Compiler contract

The existing [symbolic tensor profile](symbolic_tensor_bindings.md) proves the
original call operands, checked scalar conversion, actual kernel destinations
and complete inlined result relationships. Merlin independently reconstructs
the same facts from the original parent and executable standard bodies, without
importing the producer analyses or opening evaluator arrays.

`merlin.compile-signature.v3` explicitly selects dimensions obtained from public
input axes. For the weighted model, the parameter is argument zero and `x` is
argument one:

```json
{
  "schema": "merlin.compile-signature.v3",
  "entry": "forward",
  "model_shape_profile": "t2m.symbolic-model-shapes.v1",
  "shape_scope": "source-proved-input-axes",
  "inputs": [
    {"shape": [1], "dtype": "i32"},
    {"shape": ["d1_0"], "dtype": "i32"}
  ],
  "outputs": [
    {"shape": ["d1_0"], "dtype": "i32"},
    {"shape": ["d1_0"], "dtype": "i32"},
    {"shape": ["d1_0"], "dtype": "i32"},
    {"shape": [1], "dtype": "i32"}
  ],
  "dimension_bindings": {"d1_0": {"argument": 1, "axis": 0}}
}
```

These coordinates are function argument positions, including parameters and
buffers. No artificial scalar arguments are added. Legacy signatures retain
their existing schemas. Missing/downgraded shape profiles, false axes/results,
altered checked ranges, reordered dependent calls and incompatible actual
destinations refuse. The logical grid, tile accesses, masks, source guards,
numeric computation and effects remain executable standard operations.

Guarded model scratch is the sum of per-call source shape upper-bound costs,
independently derived by Merlin from the original checked integer intervals.
`guard_storage_scope` explicitly records `sum-of-source-shape-upper-bounds`.
No dynamic sentinel, saved sample or golden supplies a scratch size.

## Invocation and package data

The private materializer accepts the explicit
`t2m.model-invocation-inputs.v2` profile with `null` dynamic axes. It resolves
the complete positional roster from exact weight/input/buffer bindings, checks
static extents and dtypes, preserves original BF16 words and budgets materialized
arguments at 64 MiB. The existing static materializer input remains supported.

Result-reference validation resolves dimensions from the source contract and
actual invocation input shapes before opening goldens. Bundle verification
reopens NPY/safetensors headers without loading tensor values. Parent weights,
input/reference archives and complete result ordering remain byte-identical;
the existing bound receipt and artifact rosters stay intact. Evaluator files
and their individual digests remain excluded from compiler staging.

Host compilation binds both source program and signature. Reusable invocation
checks those bindings and independently reifies the actual program's result
dimensions before allocating outputs. Rehashing a false output declaration
cannot substitute for that proof. Runtime values size descriptors and supply
arguments; they do not specialize or rewrite the program.

For C runtime generation, `t2m.runtime-input-projection.v2` carries the complete
positional source argument roster, including captured parameters. Input words
are rederived on verification, using an existing native materialization
environment. Stored parameter arguments must match the original weight
container exactly; the C harness retains its normal weight-blob transport.
Reusable host calls can supply independent parameter values directly. This
distinction is explicit and does not invent a target runtime parameter-update
protocol. Temporary projection work is bounded and cleaned under project
scratch; no compiler sees runtime values or references.

## Evidence and remaining work

`tests/test_symbolic_model_package.py` executes six real model/profile
configurations through native, Merlin host and Spike. Every complete result is
checked. Four further host invocations per configuration reuse the same library
and program at lengths zero, one, seven and 31; the weighted cases also vary
the scalar parameter words. Length 32 aborts through the actual source-domain
guard without publishing outputs. Rehashed signature tests refuse before
calling the compiled library. Focused producer and independent consumer tests
add false-profile/axis/body/range/reference/parameter cases and compiler-only
staging checks.

The final focused selection is `.scratch/pytest-symbolic-package-final4/`,
with 87 passing checks. Its tensor-copy Spike case correctly refused an
in-flight runtime source formatting edit; the source-frozen recheck under
`.scratch/pytest-symbolic-package-final4-recheck/` passes that case. Together
they cover all 88 focused checks. `out/symbolic-model-packages/summary.json`
reopens six complete model/profile configurations, 21 initial results, 24
further host invocations with 84 complete results, and six actual guard aborts.
The earlier broader selection retains 141 passes, three existing surrounding-
arithmetic/slicing integration failures and one unselected optional target case
skipped in `.scratch/symbolic-package-final3.log`; it is not an all-green run.
Those three integration failures are now closed by the independently qualified
[dynamic surrounding computation](dynamic_surrounding.md).

The matching Merlin implementation is
[`2b32726086`](https://github.com/ucb-bar/merlin/commit/2b32726086be311bd2a28ed7736ee8f1b49d986d).
Its broader frontend/runtime selection passes 355, final runtime selection 71,
and layout/access selection 146 with two unavailable environment checks skipped.
These selections overlap; documentation and structure gates pass. The packages
extend the older native-only binding evidence. They are not yet folded into
the 352-package inventory; historical counts must not increase by assumption.

General derived dimension expressions, cross-axis relationship closure,
dynamic value/source views, resizing loop state, data-dependent result sizes,
and arbitrary numeric/effect/model
compositions remain open. Host/Spike execution establishes functional behavior
for the selected programs, not GPU correspondence, accelerator placement or
preserved performance.

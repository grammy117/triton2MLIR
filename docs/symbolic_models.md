# Symbolic public model interfaces

## Current boundary

Model2MLIR commit
[`f85f949`](https://github.com/grammy117/model2MLIR/commit/f85f9491917f926da7fe00540474b57dd5088622)
preserves backed symbolic dimensions instead of converting their example hints
to static types. Its shared helper uses PyTorch's literal check; ordinary
integers and literal `SymInt` expressions stay concrete. Public tensor types,
registered calls/declarations, dependent SSA and complete model results now
retain `?` where the original export is symbolic. Original source ranges remain
in the FX trace. The subsequent entry-domain expansion now emits executable
guards from actual expressions and ValueRanges. A general symbolic model
binding/invocation ABI remains unfinished. [Checked index arguments](checked_index_arguments.md)
now prove the direct guarded narrowing; symbolic tensor-dimension relationships
are the current binding boundary.

## Executable domains now implemented

The producer implementation is model2MLIR `088496f`; Merlin `4ca31cc24`
provides the matching standard scalar unsigned-cast reader.

The importer emits standard `tensor.dim`, scalar `arith.index_castui`, integer
arithmetic/comparisons and `cf.assert` before model computation. It checks the
round trip through logical i64 sizes without fixing the native index width.
Interval-sized arithmetic preserves intermediate shape products and offsets.
Shared dimensions, source ranges and invertible affine root reconstruction
retain executable equality/divisibility checks. Integer infinities are handled
as structured upstream values, not parsed strings. Advisory
`prov.shape_domain` identifies the original entry-domain intent; it grants no
permission to remove checks or infer values.

Dynamic reduced axes additionally reach ordinary Linalg when the retained
output axes and existing dtype rules permit it. The real model
`x.sum(0) + y.sum(0)` now executes shared, `2*n+1`, inverse-derived, empty/
singleton and unbounded-source configurations. Fifteen valid native executions
compare exact dyadic f32 results; fifteen invalid invocations abort through
actual compiled guards. A separate unsigned-cast execution retains high source
bits. These runs compile unchanged source MLIR for each invocation; they do
not establish one reusable compiled library or accelerator performance.

The final focused source-domain/axis/native selection passes 41 tests in
`.scratch/shape-domains-native-final5.log`. Model2MLIR's broader selection passes 137
with 26 deselected, and the final dimension/domain selection passes seventeen;
counts overlap. Merlin's independent reader/staging/layout/access selection
passes 410 with two unavailable environment checks skipped.
`out/shape-domain-native-axis/summary.json` reopens the five captures, all fifteen
exact native reduction results and all fifteen compiled guard outcomes under
the explicit input-axis contract. An earlier native-only observation remains
separately retained in `out/shape-domain-native/`.

## Runtime input-axis contract

Explicit `t2m.runtime-shapes.v2` / `merlin.runtime-shapes.v2` declarations bind
dimension names to actual public input axes without adding scalar arguments:

```json
{
  "schema": "t2m.runtime-shapes.v2",
  "inputs": [{"shape": ["rows", 3], "dtype": "i32"}],
  "outputs": [{"shape": ["rows", 3], "dtype": "i32"}],
  "dimension_bindings": {
    "rows": {"argument": 0, "axis": 0, "bounds": [0, 31]}
  }
}
```

The caller checks named axes, optional inclusive i64 bounds, shared input
dimensions, static extents and exact program/array dtypes. Scalar and axis
names cannot shadow each other. Binding records reject unknown fields; a v1
downgrade cannot admit the new field. Output allocation follows declared
dimension relationships, with no golden/reference inference. A dynamic-result
identity program executes empty/singleton and independent sizes through this
contract while retaining its original public ABI. Wrong contract dtypes refuse
before invocation. Existing v1 scalar contracts and mandatory explicit shape
contracts for dynamic input signatures remain intact.

This runtime declaration supplies invocation shapes, not a proof of the
source program's shape/effect or memory semantics. General dynamic model
bundle writing and independent Merlin compiler signatures still require the
integration below. The runtime helper accepts the Merlin prefix; current
compiler staging does not publish an input-axis signature from it. No TTIR inventory,
new bound-model package or Merlin/device execution qualification is added by
these native source-domain cases.

## Remaining binding boundary

The preceding [live-argument qualification](live_runtime_arguments.md) used
model2MLIR `5a681ea7c6e3dacbe9a1a7316a205549014f9298`, whose importer specialized
the public shape to 13. Its retained packages and execution receipts establish
that static scope. They are not new dynamic-model execution evidence after the
symbolic importer correction.

A fresh real registered Triton copy model now emits:

```mlir
func.func @forward(%x: tensor<?xi32>)
    -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>)
```

Two dependent calls use the actual original `tensor.dim` SSA and preserve all
three model references. The Triton body is the existing masked
`live_extent_copy`, translated through shared into standard tensor computation.
Binding currently refuses with:

```text
live runtime shape dimension needs an actual parent SSA/type fact
```

The new checked-index profile derives `[2, 31]` from the original dominating
round trip and standard integer comparisons. It emits an unsigned conversion
with no native index-width assumption; Merlin reconstructs the proof separately.
The public dimension remains genuinely unknown: neither its captured value 13
nor a range enclosure establishes tensor-dimension equality or a concrete size.
The remaining refusal must stay until the symbolic shape relationship is proved.
No successful bound package or Merlin execution is claimed by this probe.

On this VM, `.scratch/symbolic-model-boundary3/` retains a fresh guarded capture,
source TTIR, shared translation and the new symbolic-dimension refusal.
`.scratch/symbolic-model-boundary2/` retains the preceding guarded index refusal.
The original unguarded capture remains under
`.scratch/symbolic-model-boundary1/`.
The earlier focused tensor-profile shape integration retained the preceding
index refusal in `.scratch/symbolic-existing-shape1.log`. This remains an open integration
regression; do not turn it into a passing refusal test or weaken its checks.
Model2MLIR's relevant frontend selection passes 134 tests with 26 unrelated
cases deselected; the final twelve dimension contracts pass after source freeze.
These are frontend/capture tests, not compiled dynamic kernel/model evidence.

## Execution order

1. **Complete source domains and surrounding dynamic computation.** Entry
   guards for the forms above are implemented. Continue using the
   actual exported symbolic expressions, ranges and placeholder axes. Preserve
   equality/derived-dimension relations and emit standard checked computation
   where the source requires it. Do not parse printed range strings, read
   evaluator samples, or treat FX/provenance custody as an arithmetic proof.
   Static-only surrounding decompositions must gain actual dynamic computation;
   an unresolved ATen call cannot count as a runnable model.
2. **Bind symbolic kernel operands without losing words or dimensions.** The
   checked-index profile now proves direct unsigned narrowing from dominating
   typed source checks. Preserve kernel scalar guards and extend actual proof
   coverage where other source expressions require it.
   Establish input/result dimension relationships from actual original SSA,
   including dependent calls. Keep concrete captured scalars and live scalar
   data distinct from shape proofs. Unknown relations remain explicit refusals.
3. **Extend the package and independent Merlin signature together.** Public
   input axes can supply runtime dimensions without adding artificial scalar
   inputs. Output dimensions need executable/source-derived relationships;
   data-dependent results may require a separately declared result-allocation
   interface. Reuse the existing runtime shape machinery with an explicit
   version/profile and deterministic checks. Independently reconstruct the
   original parent contract in Merlin; evaluator archives remain excluded.
   Bind native argument materialization, reference comparison, scratch budgets
   and reusable host invocation to that same contract. Dynamic guarded scratch
   must retain real allocation/ownership/lifetime and budget checks.
4. **Execute through applicable consumers.** Reuse one complete model library
   across independent valid shapes and compare every result and readonly
   argument. Execute the same payload on Spike; distinguish unchanged kernel
   code from an invocation-specific harness. Qualify accelerator mapping and
   performance separately while retaining logical tiles, reuse and recurrence.

The first symbolic end-to-end acceptance case is the two-call copy model above,
with several valid sizes, shared source dimension SSA, all three results and
out-of-domain refusals. It is an initial interface case, not completion of
arbitrary symbolic expressions, aliases, mutation, effects or Triton kernels.
The full [completion plan](completion_plan.md) remains active.

## Package differences

The importer correction changes symbolic MLIR types to reflect the source;
entry domains add executable standard guards and advisory `prov.shape_domain`.
The scalar unsigned-cast readers retain the native `nonNeg` poison promise;
domain expansion adds no such promise. Older readers lacking the registered
operation refuse. Neither change adds a receipt schema, compiler artifact or
successful dynamic binding profile. Concrete reference arrays and model2MLIR `input_abi`/`output_abi` still
describe the saved invocation. They must not specialize compiler types or
authorize runtime result shapes. The existing static-public-model profiles
remain separately scoped; a future symbolic profile must explicitly declare
its source-domain, dimension, result-allocation and consumer requirements.

# Symbolic public model interfaces

## Current boundary

Model2MLIR commit
[`f85f949`](https://github.com/grammy117/model2MLIR/commit/f85f9491917f926da7fe00540474b57dd5088622)
preserves backed symbolic dimensions instead of converting their example hints
to static types. Its shared helper uses PyTorch's literal check; ordinary
integers and literal `SymInt` expressions stay concrete. Public tensor types,
registered calls/declarations, dependent SSA and complete model results now
retain `?` where the original export is symbolic. Original source ranges remain
in the FX trace. This does not implement executable source-domain guards or a
general dynamic model invocation ABI.

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
live index conversion needs a representable source SSA shape/integer fact
```

The public dimension is genuinely unknown; neither its captured value 13 nor
an intent annotation may supply that missing fact. The current live profile's
refusal must remain until conversion is proved or checked in executable IR.
No successful bound package or Merlin execution is claimed by this probe.

On this VM, `.scratch/symbolic-model-boundary1/` retains the original capture,
source TTIR, shared translation and separate capture/binding observations.
The focused existing tensor-profile shape integration also reproduces this
failure in `.scratch/symbolic-existing-shape1.log`. This is an open integration
regression; do not turn it into a passing refusal test or weaken its checks.
Model2MLIR's relevant frontend selection passes 134 tests with 26 unrelated
cases deselected; the final twelve dimension contracts pass after source freeze.
These are frontend/capture tests, not compiled dynamic kernel/model evidence.

## Execution order

1. **Carry original shape domains as executable source contracts.** Use the
   actual exported symbolic expressions, ranges and placeholder axes. Preserve
   equality/derived-dimension relations and emit standard checked computation
   where the source requires it. Do not parse printed range strings, read
   evaluator samples, or treat FX/provenance custody as an arithmetic proof.
   Static-only surrounding decompositions must gain actual dynamic computation;
   an unresolved ATen call cannot count as a runnable model.
2. **Bind symbolic kernel operands without losing words or dimensions.** Prove
   a narrowing index conversion from dominating typed source checks, or emit
   a representability check before the cast. Preserve kernel scalar guards.
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
it adds no receipt schema, compiler artifact or successful dynamic binding
profile. Concrete reference arrays and model2MLIR `input_abi`/`output_abi` still
describe the saved invocation. They must not specialize compiler types or
authorize runtime result shapes. The existing static-public-model profiles
remain separately scoped; a future symbolic profile must explicitly declare
its source-domain, dimension, result-allocation and consumer requirements.

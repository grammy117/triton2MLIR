# Live runtime arguments in complete model captures

The qualification below used static parent captures from model2MLIR `5a681ea`.
Model2MLIR `f85f949` now preserves actual symbolic public types; that path stops
at the dynamic whole-model interface. A separate two-call copy model now has
checked tensor relationships and native execution in three memory profiles;
package materialization and independent consumer signatures remain open. See the
[symbolic model contract and next execution steps](symbolic_models.md).
The retained static results do not establish current symbolic-model execution.

`KernelBinding.runtime_operands` maps a source runtime-scalar name to an actual
original model-call SSA operand. The explicit
`argument_binding_profile: "t2m.live-runtime-arguments.v1"` keeps these values
live through standard computation. This extends the
[captured-attribute profile](runtime_model_arguments.md); it does not introduce
a dynamic public model signature or derive compiler values from saved inputs.

## Supported value boundary

- Exact scalar SSA values retain their i1/i8/i16/i32/i64/f32/f64 type.
- Unencoded singleton tensors of any static rank, including rank zero, supply
  their one logical element through standard `tensor.extract` with zero indices.
- An `index` value may become a signed i8/i16/i32/i64 scalar only when original
  parent SSA/type semantics prove its value fits the destination width.
- Source scalar bounds remain executable. A known value outside them refuses
  before binding; an unknown value keeps the original runtime checks.

There are no implicit lossy numeric or Boolean conversions. Source-scalar names
partition into captured attributes (`runtime_arguments`) and live operands
(`runtime_operands`). A schema argument cannot be both an attribute and an SSA
operand. Every schema operand and source scalar remains accounted for.

One original readonly SSA value may supply several scalar formals. A singleton
tensor may supply both a tensor argument and its scalar element; this adds no
buffer alias, mutation or ownership permission. Existing distinct buffer
mappings and model source/value-view checks remain mandatory.

```python
binding = KernelBinding(
    "t2m_demo::live_step_state",
    translated_kernel,
    operands={"x": "x", "y": "y", "state": "initial"},
    scalar_arguments={"n": 19, "capacity": 128, "origin": 80,
                      "stride": -3, "choice": 1},
    runtime_arguments={name: name for name in
                       ("n", "capacity", "origin", "stride", "choice")},
    runtime_operands={"steps": "steps"},
)
```

The public `bind-model` JSON configuration accepts the same additive
`runtime_operands` dictionary. This example retains captured geometry while
actual scalar-tensor data controls the original kernel loop.

## Compiler facts and independent consumer checks

Integer facts come from original standard integer constants, static
`tensor.dim` results with a known axis, or singleton `tensor.from_elements`
whose element has such a fact. Inspection has a finite depth limit. Unknown
runtime data can control computation and loops; it cannot invent a tensor
extent or scratch budget. Every named kernel tensor dimension still needs a
fact established in the original parent graph.

The producer snapshots facts before inlining any kernel. Replacing an earlier
opaque call with an implementation that returns a constant does not turn that
constant into an original-parent fact for a later call. Actual dependent SSA
values still flow through the inlined computation.

Call records retain `runtime_operand_bindings`: source/schema correspondence,
original caller/kernel types, the exact standard adapter, and an optional
`constant_integer` fact. Producer bundle verification reconstructs these
records from the archived original parent. Merlin independently reopens each
original call, checks the live/captured partition, complete mappings, types,
representability and concrete tensor/result dimensions, then derives per-call
scratch. A missing/downgraded profile or changed fact refuses.

No producer import, evaluator array, saved weight value or provenance tag
supplies a consumer scalar/shape fact. These checks establish custody and
interface consistency, not source equivalence or arithmetic/effect correctness.
Executable memory, numeric, ownership and lifetime gates remain separate.

## Real models and measured evidence

`scripts/capture_live_scalar_model.py` captures two dependent registered Triton
calls around learned scalar state and a public singleton control tensor. Seven
configurations cover every scalar type above; all five original model results
match exact words through native LLVM, Merlin host libraries and actual Spike
images. Four changed-control/input invocations per configuration reuse the
same library. Floating examples use finite dyadic values, without claiming
general GPU exceptional-value correspondence. The i64 case also exercises the
public JSON/CLI binding path.

The same script's `--extent --dtype i32` model uses actual official
`torch.export` dynamic shapes. Its imported parent contains real `tensor.dim`
SSA operands, two dependent live-shape calls and one captured empty call. All
five original results pass native/Merlin-host/Spike exact-word comparison in
the tensor, buffer and guarded-buffer profiles. **The public parent type remains
specialized to 13 elements.** This does not qualify changing public model shapes.

`scripts/capture_live_step_model.py` retains a live scalar-tensor step count,
alternating logical address roots, signed strides, dependent state and original
snapshots. All seven model results pass those three execution boundaries.
Four additional host invocations include zero trips and changed input data;
two out-of-range counts abort through actual compiled guards without publishing
outputs or successful evaluations. The source operation clones incoming state;
this does not admit argument-mutating model schemas.

The final selection in `.scratch/live-model-final1.log` passes 51 tests: forty
producer contracts and eleven complete model configurations. The separate
174-case model/binding/view/bundle regression selection passes; counts overlap.
Merlin's independent staging/layout/access selection passes 446 tests with two
unavailable bubblewrap/release-wheel checks skipped. Retained target/replay
receipts reopen in `out/live-runtime-models/summary.json`: eleven images,
57 complete exact-word results, 32 further host invocations and two guard aborts.
The separate `out/live-runtime-model-inventory/coverage.json` reopens 352
packages, 278 standalone qualifications and fourteen product cases. All fifty
pinned operation names remain observed in selected native/Merlin-host-qualified
captures; this does not establish universal type/attribute/kernel coverage or
target qualification for every inventory member.

Run the focused producer/model selection with the existing environments from
the [handoff](handoff.md#reproduction-on-this-vm), additionally setting
`T2M_MODEL2MLIR_PYTHON` to model2MLIR's Python:

```bash
python -m pytest -q tests/test_model_live_arguments.py \
  tests/test_live_scalar_model.py tests/test_live_step_model.py \
  --basetemp=.scratch/pytest-live-runtime-review
```

Inspect skips when optional toolchain variables are absent. General symbolic
model ABIs, unknown-value shape facts, physical/mutating views, broader numeric/
effect/source combinations and applicable accelerator correctness/performance
remain under the full [completion plan](completion_plan.md).

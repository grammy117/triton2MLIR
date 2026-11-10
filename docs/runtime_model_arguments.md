# Binding runtime kernel arguments into captured models

`KernelBinding.runtime_arguments` maps source runtime-scalar names to the
original `prov.kernel_args` schema names. `scalar_arguments` still must match
the complete typed captured dictionary. Tensor mappings remain complete and
distinct; missing mappings and mismatched types, extents, results or bounds
refuse. No input archive, shape sample, golden or evaluator digest supplies
compiler facts.

The binder materializes captured Boolean/integer/floating attributes as typed
SSA constants. Integers must fit their source width and declared range;
Booleans remain distinct from integers. F32 runtime floats use an explicit
standard f64-to-f32 RTNE cast from the original finite captured attribute.
Dynamic kernel tensor arguments receive standard `tensor.cast` only when
each caller extent exactly matches the ABI dimension resolved from those
attributes. The same check precedes result casts back to the model's type.
Original scalar guards, source assertions, loop structure and per-operation
rounding remain executable.

This is the explicit `t2m.captured-runtime-arguments.v1` binding subprofile.
Call records retain source/schema mappings, original call ordinal/attributes,
typed adaptations and scalar evidence; bridge operations have distinct
callsite provenance. Core model2MLIR ABI, weights, inputs and references stay
byte-identical. The existing binding configuration schema has the additive
`runtime_arguments` field. Older consumers without this subprofile cannot
qualify its runtime argument/scratch declarations.

Merlin reopens each original typed parent call independently. It checks its
identity/declaration, ordinal, schema operands, scalar attributes, width/range,
all tensor/result dimensions and mapping completeness. Guarded scratch is
computed per call from these concrete result extents, including zero, and
must agree with the derived model and binding declarations within i64.
Repeated calls may use one implementation with different captured shapes.
This establishes compiler custody/shape consistency, not numeric equivalence,
memory safety, alias ownership or target execution. Those gates remain separate.

The model argument suite covers dynamic/empty shapes in all three memory
profiles, multiple calls with different extents, missing/conflicting facts,
Boolean/integer distinction and explicit float rounding. The scaled model
suite additionally captures actual model2MLIR calls: both static M/N-packed
FP4 and ragged runtime K-packed kernels qualify through native LLVM and
Merlin with learned C/bias, scale buffers, packed parameters and both outputs.
Signed PyTorch byte carriers are bitcast explicitly to Triton's unsigned
packed words; no numeric conversion changes their storage bits.
The final producer regression passes 120 cases. Merlin's 255 compiler staging
cases include typed parent-call checks, per-call and zero scratch costs,
rehashed claim refusal and evaluator read barriers; structure checks pass.
The static/dynamic model demos verify 66/80 receipt-bound artifacts. The explicit
32-package report at `out/scaled-dot-model-coverage/coverage.json` retains
30/50 definitions observed and 26 in host-qualified captures. These remain
composition observations rather than universal operation coverage.

```bash
/path/to/m2m-python scripts/capture_scaled_dot_model.py --dynamic \
  --out out/scaled-dot-model-dynamic/capture
python -m t2m bind-model out/scaled-dot-model-dynamic/capture \
  --bindings examples/scaled_dot_dynamic_binding.json \
  --out out/scaled-dot-model-dynamic/bound --python /path/to/native-python \
  --merlin-root /path/to/merlin --merlin-python /path/to/frontend-python \
  --merlin-clang /path/to/clang --merlin-host
```

First create the `scaled-dot-dynamic` source demo as documented in
[scaled contractions](scaled_dot.md). The original whole-model capture has
a static public signature; these are captured call attributes, not newly
invented public scalar inputs. Arbitrary symbolic model signatures, schema
mutation/aliases, broader numeric/effect profiles and applicable accelerator
materialization remain under the full [completion plan](completion_plan.md).

# Symbolic tensor relationships across kernel calls

`bind_model` now preserves checked dynamic tensor dimensions across dependent
Triton implementations. The real exported example in
`scripts/capture_symbolic_copy_model.py` is:

```python
first = copy(x, x.shape[0])
second = copy(first, first.shape[0])
return second, first, x
```

Its public function remains `tensor<?xi32>` to three `tensor<?xi32>` results.
The source domain admits lengths zero through 31. All three tensor, buffer and
guarded-buffer implementations bind and execute natively at independent sizes.
Model package writing and independent Merlin dynamic signatures remain open;
this establishes no new complete model package or reusable dynamic library.

## Executable shape proof

The binder keeps an immutable original parent module beside the detached module
being transformed. Constants and checked integer enclosures are frozen before
inlining. Symbolic index adapters still require the
[checked-index proof](checked_index_arguments.md); a sampled length never
supplies an integer fact or tensor extent.

`model_shape_facts.py` reifies dimensions from actual standard operations:

- `tensor.dim` identifies its actual tensor SSA value and constant axis.
- `tensor.empty` and `memref.alloc`/`alloca` retain their explicit dynamic sizes.
- Tensor/memref casts and standard bufferization tensor/buffer transport retain
  their logical shape without supplying layout, alias or ownership permission.
- `tensor.insert_slice`, Linalg tensor destinations and `scf.forall` results
  retain their actual destination shapes.
- Dynamic `scf.for` results require an invariant actual yield, proved with a
  separate body environment; resizing state cannot inherit the initial shape.
- `scf.if` results require the same relationship from both actual yields.
- Checked unsigned caller transport followed by the kernel's index conversion
  preserves the original dimension. Typed nonnegative literals and exact
  index identities with zero/one additionally retain their standard meaning.

These rules follow native LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`,
TensorOps.cpp result-shape reification and SCF.cpp `DimOfForallOp`. No source
operation is removed or replaced by a shape annotation. The kernel's logical
grid, tile accesses, tails, source guards and memory/effect computation remain
executable. An unsupported shape producer stays unknown and produces a located
refusal when its relationship is needed. Analysis limits are 64 nested queries
and 16,384 visits per environment, including checked loop-body work.

Every symbolic kernel input must match its actual parent relationship. Every
kernel result is reified from the implementation body and checked against its
declared buffer relationship. That proved result can supply a later call's
tensor dimension; it does not become an original-parent integer constant.
Reifying the final inlined body must reproduce the same complete public shape
contract. No golden, input array value, kernel name or provenance tag participates.

## Explicit profile and runtime contract

The binding report declares `model_shape_profile: t2m.symbolic-model-shapes.v1`,
per-call `symbolic_tensor_shapes`, and a source-derived `runtime_shape_contract`.
Dimension names such as `d0_0` identify function argument zero, axis zero, rather
than an evaluator input position. Public parameter arguments retain their real
positions. The existing runtime-shape v2 input-axis mechanism resolves output
allocation without adding artificial scalar arguments.

Producer verification reopens the original parent, actual implementation
functions and complete inlined body. It checks source call order, scalar
records, every tensor relationship and the public ABI/contract. Changed profiles,
sampled constants, unrelated axes, incorrect implementation result dimensions,
reordered calls and altered contracts refuse. This proof remains separate from
numeric, source-equivalence, memory and target qualification.

Guarded calls reserve the sum of their shape upper bounds using the actual
checked source scalar intervals and existing per-result guard costs. The
binding explicitly sets `guard_storage_scope` to
`sum-of-source-shape-upper-bounds`. It never multiplies an MLIR dynamic sentinel
or substitutes the saved length. Allocation, initialization, ownership,
conflict and lifetime checks remain executable. Independent consumer scratch
admission is still required with the new model interface.

This extends the binding report only. Artifact rosters, model2MLIR parent data
and receipt schemas are unchanged; no successful dynamic receipt is emitted by
the current writer. Older static consumers refuse these public dynamic types.
General derived dimensions, cross-axis equality closure, dynamic value views,
data-dependent result allocation and unsupported shape/control producers remain
completion work rather than opaque successful conversions.

## Evidence and remaining interface work

`.scratch/symbolic-model-binding2.log` retains 163 passing producer/model tests,
including existing whole-model Merlin host paths. The final 32-check selection
in `.scratch/symbolic-model-binding-final4.log` also checks the existing i32
runtime-control model through native/Merlin host/Spike. Counts overlap.
`out/symbolic-model-binding-native/summary.json` reopens the three actual binding
configurations, twelve valid native invocations, all 36 exact-word results and
three compiled source-domain aborts. Each invocation compiles the same source
MLIR separately; these runs do not establish an unchanged compiled library.
New source/native cases add no TTIR package-inventory qualification.

The complete writer probe under `.scratch/symbolic-model-package-probe1/` now
passes binding and refuses at:

```text
native model qualification requires static ranked tensor arguments
```

Next extend argument materialization and reference validation using this
source-derived contract, then independently reconstruct it in Merlin staging.
Publish an explicit input-axis compiler signature and teach the reusable host
caller and runtime generator to allocate from it. Preserve evaluator exclusion,
all results, input snapshots and scratch budgets. Run one model library across
independent shapes, then qualify the applicable Spike/accelerator interfaces.
The older surrounding-arithmetic shape integration remains open; this copy model
does not establish arbitrary dynamic model composition or full TTIR coverage.

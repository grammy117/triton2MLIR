# External descriptors over shared allocation state

`DescriptorArgument` binds an original `!tt.tensordesc` parameter to a declared
owning tensor, runtime or constant element origin, logical shape/strides and
padding/rounding flags. Multiple descriptor and ordinary pointer arguments may
share that root. Its public tensor value appears once and its updated value is
returned once. This extends the [logical field ABI](external_descriptors.md)
through the existing [allocation-state interface](allocation_state.md).

## Executable semantics and retained structure

The frontend runs the actual pinned Triton pre-layout descriptor-to-pointer
pass before constructing the allocation wrapper. It structurally verifies the
native one-to-many signature, substitutes declared root pointers and `tt.addptr`
origins, and calls the normalized source body. Upstream helper inlining exposes
those shared roots before memory analysis. Runtime i32 shape/stride fields are
extended to native i64 fields by real `arith.extsi`; flags require i1. Original
ordinary AST metadata arguments remain explicit bindings in source order.

The wrapper does not simulate descriptor effects from annotations. Original
tile loads, store masks, loop recurrence and pointer arithmetic execute through
the existing guarded source-word memory path. A whole-tile read observes one
state version before its following store. Partial overlap within one logical
program is supported; conflicting accesses across programs abort under the
existing ownership contract. Physical GPU layout and launch mapping remain
undecided, while tiling, ordering and reuse remain in standard MLIR.

```python
from t2m.allocation_bindings import DescriptorArgument, ScalarArgument

# One entry in original source-argument order, followed by any ordinary
# AST metadata and source arguments. Root/scalar names come from the public ABI.
view = DescriptorArgument("state", ("m", "n"), ("row_stride", "col_stride"),
                          offset="origin", padding_nan=False)
```

`shared-tensor --allocation-bindings` accepts the strict
`t2m.source-allocation-arguments.v2` configuration when its complete original
argument roster contains descriptor records. A descriptor record has
`kind: descriptor`, `allocation`, `shape`, `strides`, and optional `offset`,
`padding_nan`, `round_f32_to_tf32`. Integer fields accept constants or declared
scalar names; flags accept Boolean constants or i1 scalar names. Shape bounds
must be nonnegative. Unknown fields, wrong ranks/widths and incomplete rosters
refuse. Pointer-only v1 configurations remain accepted.

## Portable original-source custody

New allocation packages carry `t2m.source-allocation-capsule.v1` inside the
existing descriptive `source_argument_binding` record. Five artifacts retain:

| Artifact | Bytes retained |
|---|---|
| `allocation-original.ttir` | Exact pre-wrapper input, including line endings |
| `allocation-input.mlir` | Actual native generic input to wrapper construction |
| `allocation-arguments.json` | Complete original source-argument configuration |
| `allocation-source-stage.json` | Source-only stage flags, tool fingerprint and artifact hashes |
| `allocation-source-stderr.log` | Native source-stage diagnostics |

Normalized inputs additionally retain `allocation-normalized.ttir`,
`allocation-intake.json`, and `allocation-inventory.json`: actual pass output,
normalizer/version/pipeline identity, original typed graph and field expansion.
`source.ttir` remains the executable bound program. Its graph, the original
graph and later normalized/emitted graphs stay distinct. Provenance retains
descriptor intent, original fields, roots and origins without granting semantics.

Standalone tensor/effect and derived-model receipts bind these files. Merlin
independently validates the known compiler roster, all byte links, entry
correspondence and source-only stage projection before staging them. Evaluator
archives and reference fields do not enter that projection. This establishes
byte custody, **not source equivalence**, alias permission or target admission;
`source_closure_verified` stays false. Legacy pointer-only packages without a
capsule remain readable; descriptor-origin records require the complete capsule.
Older readers may refuse the added artifact inventory explicitly.

## Protected contracts before allocation normalization

An explicit FP8 overflow selection or source-assembly provider now selects
`t2m.source-allocation-capsule.v2`. Source-only native serialization exposes
the complete original graph before any normalization. A source-entry preflight
audits operations, numeric contracts and effects without admitting a wrapper
ABI. Existing FP8 and selected assembly recipes then emit standard executable
arithmetic. Native descriptor/pointer/dot passes consume that protected source;
the complete allocation binding and final TensorABI preflight remain mandatory.

V2 adds exactly four compiler artifacts to the existing roster:

| Artifact | Bytes retained |
|---|---|
| `allocation-generic.mlir` | Complete native generic original source, before annotations/expansion |
| `allocation-native-print.json` | Source/generic hashes, verified no-pass round trip and actual printer identity |
| `allocation-protected.ttir` | Executable arithmetic expansion before allocation normalization |
| `allocation-protection.json` | Original typed graph, selected cast/provider records and original/generic/protected byte links |

The producer and independent Merlin reader check the entire chain. With native
normalization, its input hash must name `allocation-protected.ttir`. Without
normalization, the shared source-only stage consumes that protected file
directly. Existing original/bound/generic links still apply. Evaluator fields,
missing files, profile downgrades, changed links and authority grants refuse
even when outer artifact hashes are updated. V1 remains supported; v1-only
readers explicitly refuse v2. Public tensor/scalar/model ABI and capture receipt
schemas are unchanged. Custody still supplies no source-equivalence proof.

Coverage observes the original graph in `allocation-protection.json`, including
casts removed by executable expansion, separately from the bound graph. The
source descriptor provenance likewise uses the original pre-expansion graph.
Numeric and effect semantics come from the payload, never those annotations.

## Reproducible demonstrations and evidence

```bash
python -m t2m shared-demo --kernel descriptor-state \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/descriptor-state --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/descriptor-state \
  --out out/descriptor-state-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
```

`descriptor-state-f32` and `descriptor-state-bfloat` use the same real kernel
and binding route. The BF16 host feature preserves source arithmetic rounding;
[its independent word tests](bfloat_precision.md) remain separate evidence.
The API demo reads and writes 2×16 tiles through two external views, with
runtime origins, signed strides, dimensions, iteration count and root capacity.

Three semantic libraries each replay eight additional valid invocations and
four actual guard failures. These cover exact and partial sharing, reversed
views, multi-program tails, zero iterations/domains/capacity, untouched words,
out-of-bounds roots, duplicate stores and cross-program conflicts. Nineteen
protocol checks cover configuration/type/shape rules. An unused descriptor
parameter also normalizes and executes without any descriptor operation.
The public CLI package passes native execution, Merlin qualification and seven
rehashed custody-corruption refusals.

`scripts/capture_descriptor_state_model.py` captures an actual model2MLIR
custom-op model with learned i64 weights, three descriptor calls, dense matrix
value adapters and an empty result. Native and Merlin executions match every
tuple result byte for byte; original data/weights and source-capsule files
remain byte identical through derived-model staging. This is CPU reference
and host compiler evidence; the script's GPU path is not device-qualified.
The visible `out/descriptor-state` package verifies 41 artifacts, its separate
Merlin qualification verifies 60, and `out/descriptor-state-model` verifies 96.
The explicit 52-package/37-qualification report at
`out/descriptor-state-coverage/coverage.json` retains 34/50 observed definitions
and 30 in host-qualified captures. These are composition observations, not
universal operation coverage.

## Remaining completion work

This does not close arbitrary-kernel coverage. Mixed-type/byte aliases,
physical/non-dense or mutating model views, independently strided contraction
accumulators, broader numeric/effect compositions, external synchronization,
physical descriptor interfaces and applicable target materialization remain
required. No provenance tag can replace their executable semantics or evidence.

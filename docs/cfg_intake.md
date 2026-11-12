# Original control-flow graphs through the Merlin interface

Multi-block TTIR entries now use LLVM's official CFG-to-SCF conversion before
the existing tensor/memory frontend. Original early returns, block arguments
and backedges remain in the source inventory. The emitted computation retains
structured conditions, loops, tile state and logical program IDs; it does not
introduce a GPU launch, flatten the kernel into a coarse name or unroll its
runtime recurrence.

## Source and version boundary

The installed shared optimizer advertises `lift-cf-to-scf`, but a real TTIR
probe leaves its CFG unchanged: the pass visits `func.func`. A narrow structural
adapter temporarily changes TT function/call/return interfaces into equivalent
standard interfaces, preserving types, attributes, regions and locations. The
official pass then raises the graph, and the adapter restores TT interfaces for
the existing shared conversion. There is no independent CFG compiler here.

Original operations and numeric/effect contracts are audited before any
canonicalization. Explicit cast/source implementations become executable first.
Current block-pointer and descriptor types require protected native normalization
before the older shared parser can consume them. Canonicalization removes AST
unreachable blocks before lifting; the original graph still records those
blocks and operations. The restored TT module receives native verification and
the full actual ABI preflight. A source-only audit admits no signature.

The original graph now records each operation's block and successor edges in
addition to typed SSA, attributes/properties, region membership and block
arguments. Normalized operations and emitted IDs remain a separate graph.
Source descriptions and byte custody grant no omitted behavior or independent
source-equivalence proof.

Reviewed primary source and tests:

- [LLVM ControlFlowToSCF, a47bddcc](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/lib/Conversion/ControlFlowToSCF/ControlFlowToSCF.cpp)
  provides function-scoped lifting and standard structured branches/loops.
- [LLVM CFG-to-SCF fixtures](https://github.com/llvm/llvm-project/blob/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Conversion/ControlFlowToSCF/test.mlir)
  cover block arguments, conditional edges and loop structures.
- [Meta/shared early-return fixture, 17f0c234](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonArithToLinalg/early_return.mlir)
  retains actual TTIR CFG branches. Its [function conversion](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp)
  preserves block terminators; that alone does not supply our structured tensor ABI.

These are source references. The actual optimizer binary is fingerprinted
separately; no matching source build, toolchain installation or Triton-CPU
dependency is claimed.

## Package and consumer contract

The optional `t2m.control-flow-normalization.v1` profile adds six compiler files:

| File | Meaning |
|---|---|
| `cfg-input.mlir` | Typed standard-function input to the native pass. |
| `cfg-lifted.mlir` | Actual structured native output. |
| `cfg-output.ttir` | Restored TT interfaces before subsequent conversion. |
| `cfg-source-graph.json` | Complete original typed graph and successor edges. |
| `cfg-stage.json` | Fixed pipeline, native input/output hashes and optimizer fingerprint. |
| `cfg-stderr.log` | Byte-bound native diagnostics. |

The fixed stage is canonicalize → lift-cf-to-scf → canonicalize, with threading
disabled and verification enabled. Producers and Merlin independently check
its exact roster and byte links. Standalone and enclosing-model compiler
projections retain all six files while excluding evaluator inputs/references.
Public tensor/scalar ABI and receipt schemas remain unchanged; provenance v2
already separates original and normalized graphs. Older readers need the new
custody support to retain this evidence. The [package contract](package.md)
and generated compatibility metadata record the difference explicitly.

The numeric shared-bundle reader also admits these six known compiler files
before checking the exact declared source roster and every byte link. Unknown,
missing, undeclared and rehashed inconsistent files still refuse. This matches
the existing writer and effect/model readers without changing a receipt schema
or artifact profile; earlier numeric readers rejected such valid CFG packets.

The early-return probe also exposed a pointer-state prepass issue: a result-free
outer condition caused the existing numeric-loop proof to reject an otherwise
pure integer recurrence. Such a condition carries no boxed state. Stateful
nested controls still receive their complete numeric/transitive-use checks;
all actual pointer conversion, bounds, ownership and initialization checks stay
active. This reuses shared's supported `skip-prepass` option.

## Evidence and remaining scope

Real i32/i64 early-return kernels retain masked tails, original output state,
logical grid selection and runtime recurrence through native, Merlin host and
actual Spike in all three memory profiles. Separate CFG backedge/phi kernels
retain initial state on zero iterations and exact i64 words. A genuine
model2MLIR capture keeps learned weights, dependent calls, an empty result and
its live original snapshot; all four model results match exact words through
those execution boundaries.

General CFG/value-returning entry bindings, remaining pointer/descriptor phi
compositions, switches/irreducible graphs, exceptional values/effects and
applicable accelerator qualification remain part of the full completion scope.
A successful official transformation is insufficient unless the complete
resulting payload also passes the consumer and execution gates.

The selection at `out/cfg-intake/summary.json` independently reopens nine
standalone libraries, one complete model, ten actual Spike images, thirteen
exact-word results and thirty additional host replays. The original-source
i64 early-return CLI capture also verifies. The final 54-check frontend/model/
corpus/control suite passes, as do Merlin's 297 selected source/compiler-reader
checks and structure/documentation gates. Merlin's independent reader is
[88db420c3](https://github.com/ucb-bar/merlin/commit/88db420c3435077c8901e57a34d9f43caa19bd3f).

The combined `out/cfg-intake-inventory/coverage.json` reopens 123 packages,
90 standalone qualifications and fourteen maintained product cases. All fifty
pinned definition names remain observed in selected native/Merlin-host captures.
These are composition observations; general typed/attribute/kernel and
applicable target coverage remain incomplete.

# Checked source index arguments

The independent consumer implementation is Merlin `50873d7cc9bb544116b2ae8f60cc4a8e18d7f8d4`.

`t2m.checked-index-runtime-arguments.v1` extends the existing live-argument
profile with a standard unsigned index adapter. It is selected only when an
original parent index value needs an executable assertion proof rather than
an exact constant/static-dimension fact. The public model ABI, artifact roster
and receipt schemas stay unchanged. Older profiles refuse the new adapter.

## Source proof and executable lowering

The source must establish an actual round trip, for example:

```mlir
%word = arith.index_castui %size : index to i64
%back = arith.index_castui %word : i64 to index
%same = arith.cmpi eq, %size, %back : index
cf.assert %same, "source round trip"
```

Dominating integer comparison assertions then restrict `%word`. Signed and
unsigned comparisons retain their own fixed-width word meaning; extended
integer literals retain their original signed/unsigned extension. Asserted
conjunctions supply both predicates. Neither assertion messages nor provenance
attributes supply facts. Guards in sibling branches or after the call cannot
justify a conversion inserted before it.

The checked round trip allows an inclusive unsigned enclosure to transfer back
to the source index without choosing a target index width. When the enclosure
fits the nonnegative range of the declared signed kernel integer, the adapter
emits `arith.index_castui`. A signed cast could sign-extend an index whose high
bit is set on a narrower target. Original source guards and kernel bounds stay
executable; the adapter introduces no `nonNeg` poison promise.

Duplicate `tensor.dim` queries match only the same actual tensor SSA value and
constant axis. Duplicate unsigned bridges additionally match integer width.
No eventual result of an inlined opaque call creates an original-parent fact.
The producer freezes constants and enclosures before any kernel is inlined.

Per-call records retain `adapter: checked-index-castui` and
`unsigned_integer_range: [low, high]`. Producer verification and Merlin reopen
the original parent separately and reconstruct that exact record. Modified
enclosures, invented constants, extra authority fields and profile downgrades
refuse. Merlin imports no producer or evaluator code and reads no evaluator
archive to establish these facts.

## Scope and reuse boundary

The native LLVM source at `a47bddccec30255619bb8c37fa59700e661d4e66`,
`mlir/lib/Dialect/Arith/IR/InferIntRangeInterfaceImpls.cpp` and
`mlir/test/Dialect/Arith/int-range-interface.mlir`, supplies the signed/unsigned
cast and extension contract. Its range solver is not an installed Python API
and exports no parent-call assertion witness. This reader bridges that source
boundary; Triton pointer and computation conversion still use triton-shared.
No native build or Triton CPU backend is involved.

The analysis bounds each query to 4,096 predecessor operations, 4,096 assertion
expression visits, 64 enclosing blocks, 16 constant-expression levels and
4,096-bit integers. It reasons about direct fixed-width comparisons and checked
unsigned bridges. Arithmetic-derived bounds, general CFG dominance and
loop-carried/equality closure require additional proofs. Missing proof refuses.
Range enclosures do not establish concrete tensor extents, equality of
different tensor dimensions, dynamic result allocation or scratch budgets.

## Current evidence and next boundary

The focused producer selection passes **100 tests** in
`.scratch/checked-index-final3.log`. It includes exhaustive small-word signed/unsigned
comparison checks, dominance and unrelated-root refusals, profile tampering,
the original live/captured argument regressions, and actual native execution.
A real translated i32 XOR kernel bound into a typed standard source module
retains three independent runtime controls. Four invalid controls abort through
the original compiled assertions and publish no successful outputs. Those runs
compile per invocation; they establish neither a reusable library nor a new
complete PyTorch model package.

Merlin's independent reader/staging/layout/access selection passes **425 tests**
with two unavailable environment checks skipped. Five new compiler-lineage
cases admit the explicit profile, exclude evaluator reads and reject rehashed
profile/range changes. Those protocol fixtures grant no numeric qualification.
The separate existing i32 runtime-control PyTorch model also passes through
native/Merlin host/Spike with its original live profile and all five results;
four changed-control invocations reuse its host library. Its regression log is
`.scratch/checked-index-v1-model.log`; no new kernel coverage is inferred.

The actual `capture_live_scalar_model.py --extent --dtype i32` export supplies
the checked `[2, 31]` entry dimension to both original calls. Torch export reuses
the original dimension SSA; no opaque-result shape fact is inferred.
A fresh two-call copy probe in `.scratch/symbolic-model-boundary3/` now passes
index-conversion planning and stops at:

```text
live runtime shape dimension needs an actual parent SSA/type fact
```

The next work is symbolic tensor-dimension relationships, result/scratch
allocation and independent Merlin signatures under an explicit model profile.
The existing dynamic whole-model integration remains open. No TTIR inventory
or target qualification is added by these source-proof cases.

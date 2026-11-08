# R1: tensor-memory recovery

**2026-11-07.** This milestone advances memory semantics while retaining the same abstraction
boundary as model2MLIR: tensor values, linalg payloads and structured logical iteration. It does
not introduce GPU memory/launch dialects or a Triton runtime requirement for the emitted artifact.

## Implemented

- Signed affine comparisons and conjunctions become symbolic active lane intervals. Proofs cover
  the whole logical grid without enumerating programs or tensor elements.
- Boundary loads use only safe active slices, then pad with the source's exact f32 `other`.
  Negative nominal bases and entirely inactive programs cannot trigger invalid extraction.
- Omitted `other` stays undefined in analysis. Definedness propagates through the supported
  pointwise operations, and every store must prove its active lanes defined. Only then may an
  arbitrary inactive-lane fill be chosen. A shifted load whose undefined values reach the output
  is refused.
- Output stores insert only active contributions. Write-only results still require full output
  coverage; partial readwrite results start from the incoming tensor and preserve unwritten values.
- Positive static strides and storage offsets are inverted into logical coordinates. Misalignment,
  invalid view extents and accesses outside the logical view are refused.
- Readwrite reads must stay in the program's own output tile and precede its sole store. This
  establishes independence; it prevents replacing cross-program communication with a snapshot.
- Identical nonvolatile reads share a tensor SSA value, with all original load IDs retained.
  Different source cache policies do not force different hardware behavior for stable inputs.
- Integer expression folding/CSE reduces address scaffolding. Floating-point operations retain
  source grouping; no reassociation is introduced.

The 1,000-element/BLOCK=256 demo has four logical programs and a ragged tail. Its body retains
tile loads/padding, a recognizable `linalg.generic`, and cropped parallel tensor insertion.
Repeated reads and live tiles remain dataflow; physical caches, registers and SRAM are undecided.

```bash
python -m t2m demo --size 1000 --out out/r1_demo
python -m t2m verify-bundle out/r1_demo
python -m t2m verify-mlir out/r1_demo/model.mlir --python /path/to/native-env/bin/python --lower-to-llvm
```

## Validation and practical limits

Local validation: **81 tests passed**, including seven opt-in native verification/legalization
checks. The 1,000-element add and multiply-add bundles pass numerical/integrity checks; an offline
wheel build and packaged ragged-tail demo pass. Original R0 bundle hashes remain verifiable.

The suite covers low/high masks, nonzero fills, signed shifts beyond an entire tile, zero active
elements, extra inactive programs, undefined-lane nonobservability, positive-stride views,
partial/empty readwrite updates, cache-policy abstraction and shared-read provenance. Negative
cases include unsafe lower bounds, nonrectangular masks, exposed undefined values, cross-program
readwrite dependencies, stale metadata and source integer overflow. Explicit signed-zero TTIR fills
are checked bitwise; NaN constants are refused until payload-preserving literal handling is available.
The stock Triton AST adapter normalized the Python `other=-0.0` probe to positive zero, so the
signed-zero test uses saved TTIR to qualify this converter's actual input boundary independently.

Native checks use existing torch-mlir **20260913.875** bindings from the model2MLIR environment,
with unregistered dialects disabled. Representative tail, shifted-fill, partial-update, scalar-
constant, reuse and strided cases verify and legalize using standard passes:

```text
scf.forall -> scf.for
canonicalize / CSE / one-shot bufferization with function boundaries
linalg -> loops; affine/SCF -> control flow
memref / func / arith / cf -> LLVM; reconcile casts
```

The verifier refuses residual non-LLVM computation or unrealized casts. These passes run in an
isolated verification process; exported MLIR remains tensor/linalg/SCF. Legalization proves neither
native execution nor linking of runtime helpers such as `memrefCopy`. No existing environment was
modified and no compiler was built.

R1 is still open for rank-two/transposed views, runtime scalar arguments, arbitrary predicates,
multiple-store memory versions, general source regions and a typed rule registry. The current
nonaliasing/stable-input preconditions remain explicit. Native host execution and Merlin staged
execution remain separate, unqualified gates.

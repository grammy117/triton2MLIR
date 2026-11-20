# Reading partially initialized source tensors

The portable tensor path can now read a completed source row, strided column or
plane before the rest of a fresh output is written. It also retains immediate
read-after-write and a second source loop that folds over the completed region.
Actual loop order, loaded-value reuse, word arithmetic, masks and observations
remain standard executable tensor/Linalg/SCF data flow. No initialization bitmap
or whole-operation replacement is emitted.

`t2m/examples/partial_memory_kernels.py` contains real Triton sources. One pattern
loads a bias outside an inner write loop, then rereads that same completed region:

```python
for row in range(rows):
    reused = tl.load(bias + row)
    for col in range(cols):
        offset = row * cols + col
        tl.store(out + offset, tl.load(x + offset) ^ reused)
    checksum = tl.full((), 0, tl.int64)
    for col in range(cols):
        checksum ^= tl.load(out + row * cols + col)
    tl.store(first + row, checksum)
```

Column traversal uses the original nonunit element stride; the 3D variants
observe complete planes under different traversal orders. Neither requires
physical thread mapping or a dense substitute for the source algorithm.

## Proof and scope

`loop_write_coverage.scalar_stores` visits actual unconditional scalar stores
under signed, zero-based, unit-step rectangular loops. `partial_initialization`
splits the original index polynomial into an origin and independent induction
digits with their actual trip counts and element strides. The resulting image
describes precisely what those iterations wrote. Counts may be zero, strides
may repeat indices, and values fixed in an enclosing loop remain local symbols.
This is definedness evidence; it grants no disjointness or overwrite permission.

Every admitted scalar read needs a witness in one actual image: each coordinate
is nonnegative and less than its trip count, and the reconstructed address
equals the original read address. A bounded symbolic search handles polynomial
strides and corner/induction coordinates without enumerating elements. Ordinary
scalar writes additionally establish point images at their actual program point.
Source arithmetic and casts retain their independent original-width checks.

The read's own actual loop successors and necessary mask/branch comparisons may
prove coordinate bounds or positive trip counts. The existing fixed-width
predicate reader supplies only necessary implications: true conjunctions and
false disjunctions, literal negation and signed/unsigned word comparisons.
Unknown controls supply no fact. These use facts never escape the selected read
or authorize an earlier overflowing definition. A zero-trip write image contains
no elements; an inactive read still returns its original defined fill.

Partial facts intersect at branch joins and expire when their varying source
loop exits. Whole-result initialization still requires the separate complete
overwrite proof. Future/unwritten regions, one-arm/masked writes, missing
nonempty premises and parallel conflicts retain their ordinary refusals. The
partial-image budget is 128 facts per buffer; exhaustion supplies no additional
optional fact and cannot disable an existing whole-write proof. Membership search
has a separate 128-state budget. Existing polynomial budgets remain unchanged.

The implementation reuses the existing shared pointer/index adapter, polynomial
reader, loop parser and fixed-width predicate constraints. The upstream review
pins Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and historical
Microsoft `050b93ae9c737111f1ca425d793419b32427a971`: `PtrAnalysis.cpp`'s
offset/stride multiplication and loop rewriting, plus
`test/Conversion/StructuredToMemref/scalar_store_nested_loop.mlir`.
Meta file bytes were checked against the pinned tree's actual Git blob IDs.
The existing licensed `shared_gather` expansion remains in use. Upstream memref
stores alone do not prove definedness of our functional fresh-tensor results;
the image/witness reader is the added analysis, without copied upstream blocks.

## Complete PyTorch model and package

`scripts/capture_progressive_model.py` captures two dependent real Triton calls:

```python
rows, cols = x.shape
first, a, b = transform(x, self.row_bias[:rows].clone(), rows, cols, 0)
second, c, d = transform(first, self.col_bias[:cols].clone(), rows, cols, 1)
return second, first, a, b, c, d, x, self.row_bias, self.col_bias
```

Each call retains its original write loop followed by a reread/XOR recurrence
and final-value observation. Runtime parameter slices, both saved parameters,
all nine results and the actual model shape domain remain. Standalone kernels
admit zero axes independently of the model's exported rows 2..7/columns 2..9.

Public tensor/scalar types, result modes, receipts, artifact rosters, mutation
and effect profiles are unchanged. `source_gathers.initialization` records the
read's source-derived witness kind, axis count and local scope. Provenance grants
no initialization, active bounds, ownership or physical-alias permission.
Guarded and buffer continuations keep their own admission requirements.

General vector-lane initialization, conditional/masked write partitions,
while-state relations, resizing/permuted ownership, irregular parallel
dependencies, shared/async effects and applicable accelerator execution and
performance remain incomplete. The historical operation-name inventory is
unchanged; these source/control/memory compositions do not prove arbitrary
kernel coverage.

# Numeric TTIR builtins inside scalar regions

Scalar maps, reductions and scans may retain the already supported pure numeric
TTIR builtins alongside their local tensor intermediates. Admission uses exact
source operation names and fields; the existing preflight, native verifier and
numeric contracts still determine whether each concrete form is valid.

Real source examples now include a map callback that computes a word's population
count through a two-bin histogram, and a callback that builds small matrices,
contracts them with the actual source accumulator, then reduces the result:

```python
a = tl.full((16, 16), left, tl.float32)
b = tl.full((16, 16), right, tl.float32)
c = tl.full((16, 16), initial, tl.float32)
product = tl.dot(a, b, c, input_precision="ieee")
return tl.sum(tl.sum(product, 0), 0)
```

These are captured Python/Triton operations. Kernel names, example values and
goldens supply no production conversion behavior. The resulting named tile
contraction and original accumulator remain executable inside the surrounding
parallel map; the frontend does not substitute a scalar closed-form answer.

## Reuse and source contracts

`value_regions.tensor_computation` admits the known fields of dot/scaled dot,
histogram, local gather, checked casts/bitcasts, precise math, clamps, unsigned
high products and logical launch queries. Existing scalar preparation and the
shared shape/arithmetic pipeline retain their original ordering and numerical
policies. This change supplies no second numeric lowerer.

Pure external calls require `external_math.contract`, including unused calls:
library name, empty path, typed prototype, symbol and actual purity must match
the established standard-math mapping. An annotation naming a pure family does
not establish any of those facts. Unknown fields or effects refuse before the
detached value-region expansion is published.

The implementation references Meta/shared revision
`17f0c234ce474bee334af36e351c62eca7852836`:

- `include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp`,
  `MatmulConverter`, `GetProgramIDConverter` and `GetNumProgramsConverter`.
- `test/Conversion/TritonArithToLinalg/dot.mlir`, including a loaded nonzero C.
- The supported/unsupported external-elementwise conversion fixtures.
- `test/Conversion/StructuredToMemref/get_num_programs.mlir`; that fixture is
  marked XFAIL upstream and supplies a representation example, not a passing
  execution oracle.

The archived Microsoft `050b93a` monolithic converter registers the same
arithmetic patterns. The pinned Triton v3.7.1 definitions establish Pure for
the numeric/domain operations and conditional effects for externals. See the
[shared reuse record](triton_shared.md), [dot precision](dot_precision.md),
[scaled-dot contract](scaled_dot.md), [histogram expansion](histogram.md) and
[external math contract](external_math.md) for their existing boundaries.

## Logical queries inside control and mapped bodies

The tensor adapter binds program IDs/counts to the actual logical grid. Queries
left inside a lifted scalar body previously bypassed that adapter, producing a
located refusal even though the values are uniform within a logical program.

`region_launch_queries.py` validates every query before moving any node. Only
the exact total, operand-free i32 query with an axis in 0..2 may move to the
entry of its own enclosing structured program function. Original locations and
source identities remain attached. Surrounding branches, loops, arithmetic and
effects stay at their original positions. The complete detached region
transaction also discards query relocation if a later container refuses.

The owning function entry is still inside one logical program invocation when
the tensor adapter constructs its `scf.forall`. Program IDs bind to its induction
variables; program counts use the actual static/runtime grid expressions. No
physical core or warp mapping is chosen. Unlike shared's optional six trailing
launch arguments, this introduces no launcher ABI. Standard pure-value capture
connects the query to its nested consumers.

`uniform-region-launch-query` records and generated compatibility metadata name
this representation. Public tensor/scalar/model ABIs, artifact rosters and
receipt schemas remain unchanged.

## Qualification and remaining scope

The current selection reopens **30 real source captures, 30 Merlin host
libraries and 30 actual Spike images**. All thirty complete results match their
declared policies: **24 exact-word results and six tolerance-qualified tanh
results**. Another **120 host invocations** reuse those unchanged libraries
across empty/singleton/exact/ragged domains. The source kernels and all public
input tensors remain independent of the references used for evaluation.

`out/numeric-regions/summary.json` binds each reopened target receipt and host
replay. The combined `out/numeric-region-inventory/coverage.json` reopens
**291 packages, 240 standalone qualifications and fourteen product cases**.
All fifty pinned TTIR definition names remain observed in selected
native/Merlin-host-qualified captures; this is not universal per-opcode support.
The 32 focused region/query contracts and 187 existing runtime-grid,
external-contract and original-combiner-effect checks pass. These selections
overlap earlier coverage and must not be added as distinct supported kernels.

`tests/test_numeric_regions.py` captures the real source callbacks in
`t2m/examples/numeric_region_kernels.py`. Its families cover i32/i64 histogram,
IEEE f32 dot with runtime or zero C, mixed FP16/BF16 logical scaled dot, i64 local
gather, logical queries including scalar branches/zero-trip loops, and mapped
f32/f64 external tanh. Each family uses the existing tensor, buffer and guarded
source-word profiles with runtime n=0..31.

Integer, contraction and logical-query examples use independently computed
exact-word references. The tanh cases use explicitly tolerance-qualified
standard-math references; they supply no GPU libdevice accuracy or bit claim.
The scaled example uses the existing explicit logical-MX contract with no
scales, so arbitrary scale/packing/backend combinations remain separate work.
Tests require retained contraction/accumulator and histogram structure as well
as complete output agreement.

Contract tests retain unknown-property and unqualified-external refusals,
transactional expansion, query scope/location/use custody, untouched effects
and failure of a later invalid candidate without moving an earlier one.
Native/Merlin bufferization keeps the existing captured-destination handling
and all ownership/lifetime checks from [local tensor regions](tensor_regions.md).

Further tensor operations/dynamic intermediates, effectful regions, general
public model shapes/mutation, arbitrary numeric/source versions and accelerator
correctness/performance remain completion requirements. A numeric builtin's
presence inside one qualified map does not establish every reduction/scan,
type, attribute or kernel composition.

## Complete PyTorch models

`scripts/capture_numeric_region_model.py` captures two actual model2MLIR models
with saved parameters and nineteen-element public inputs. The f32 model chains
mapped IEEE contractions, a mixed FP16/BF16 scaled contraction and a zero-C
contraction. Its seven results include a dependent contraction, an empty call,
the original computed input and an epilogue. The i64 model chains histogram and
gather callbacks, retaining six results including the original snapshot and an
empty output. Independent CPU implementations use ordinary Python integer math
and finite dyadic PyTorch arithmetic; no lowering helper supplies their goldens.

All three memory profiles pass complete native/Merlin-host/Spike execution.
The six model configurations retain **all 39 original results as exact words**.
Parent weights, manifests, inputs, constants, first golden and full result
archives remain byte-identical. Each typed call selects its exact n/kind
specialization; the source runtime n remains connected through the established
binding profile. Public model shapes remain static and no ABI/schema changed.

`tests/test_numeric_region_model.py` reproduces those six configurations.
`out/numeric-region-models/summary.json` reopens every actual target receipt.
The latest combined `out/numeric-region-model-inventory/coverage.json` reopens
**297 packages, 240 standalone qualifications and fourteen product cases**,
retaining all fifty observed names under selected native/Merlin-host profiles.
These finite f32 references are chosen to be exact through the declared casts
and source recurrences; they establish no general GPU precision policy.

With the existing environment variables set to the pinned tools:

```bash
python -m pytest -q tests/test_numeric_regions.py tests/test_numeric_region_model.py \
  tests/test_numeric_region_contract.py tests/test_region_launch_queries.py \
  tests/test_value_region_contract.py --basetemp=.scratch/pytest-numeric-regions
```

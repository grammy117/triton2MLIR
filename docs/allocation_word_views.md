# Typed source arguments sharing one allocation

Explicit [allocation bindings](allocation_state.md) now accept distinct source
pointer types over one owning root when their scalar elements have equal,
byte-aligned widths. This includes f16/i16, BF16/i16, f32/i32 and f64/i64.
The public root retains its declared dtype and appears once in the input and
result interfaces. Runtime origins remain root element offsets.
The qualified path uses `guarded-buffer` memory with `source-words` addresses.

## Transformation and source-backed reuse

The wrapper first applies `tt.addptr` to the owning root, then emits a real
`tt.bitcast` into the original argument type before its `tt.call`. Upstream
inlining substitutes those exact operands into the original source body. The
existing [logical word-view analysis](pointer_views.md) resolves every event
to that root, retains its storage type and bitcasts loaded/stored register
values. A later read through the other source argument observes the current
shared state; no duplicate buffer or representation snapshot is introduced.

Meta triton-shared at `17f0c234ce474bee334af36e351c62eca7852836` already
implements the base/offset rule in `TritonToUnstructuredPass.cpp`; its
`bitcast_ptr_unstructured.mlir` tests chains and carried pointers, while the
invalid fixture rejects changed strides. This change connects the explicit
allocation wrapper to that existing analysis and native event conversion.
The archived Microsoft source at `050b93ae9c737111f1ca425d793419b32427a971`
remains historical context; the actual optimizer identity stays in the stage
records, independently of either surveyed source revision.

Source scalar/tensor arithmetic keeps its original widths. The wrapper does
not grant a physical address observation, alternate address space, sub-byte
padding interpretation, different-width byte alias or additional atomic policy.
Bounds, initialization, lane uniqueness, cross-program ownership and native
lifetime checks remain mandatory. Partial aliases are valid within one logical
program; the existing race guard refuses conflicting programs.

## Descriptor and model integration

The same rule applies to normalized external descriptors. One f32 descriptor
and one i32 descriptor may share an i32 owning root, with their original field
layouts, source metadata, runtime origins and strides intact. Native descriptor
normalization runs before the executable allocation wrapper; original,
normalized and bound graphs remain separate in the portable source capsule.

`scripts/capture_allocation_word_model.py` supplies a real model2MLIR capture
with learned i32 weights, dense matrix values, three calls and all tuple
results. The first call uses partial aliases within one tile, the second uses
exact aliases across multiple programs, and the last has an empty domain.
The original model ABI/data/weights remain unchanged; explicit dense-value
adapters bind its matrix values to the runtime-sized allocation root.

`equal-width-word-view-v1` records retain the original pointer type, root
storage type and origin in the descriptive binding table. Existing
`prov.logical_word_views` tags retain event/word intent in standard MLIR.
These descriptions grant no rewrite or source-equivalence permission. Existing
ABI/receipt schemas remain unchanged; generated compatibility metadata records
the newly supported relation and its limits.

## Demonstration and evidence

```bash
python -m t2m shared-demo --kernel allocation-word-state \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/allocation-word-state --python /path/to/native-env/bin/python
python -m t2m qualify-merlin out/allocation-word-state \
  --out out/allocation-word-state-merlin --merlin-root /path/to/merlin \
  --merlin-python /path/to/merlin-env/bin/python \
  --python /path/to/native-env/bin/python --merlin-clang /path/to/clang
```

Eight actual Merlin libraries cover integer and floating owning storage for all
four pairs, with for/while loops, exact NaN/signed-zero/subnormal word transport,
untouched root words, runtime partial origins, tails and empty domains.
Each library replays six additional valid cases and three actual guard failures.
A descriptor library adds four valid replays. The public CLI workflow separately
checks native execution and Merlin qualification. The focused suite passes
48 checks, including nineteen allocation and nineteen descriptor protocol cases.
Model/native/consumer execution compares every public result with independent
word references; no floating comparison hides changed NaN payloads.
The real model capture and twelve legacy allocation/model regressions pass
thirteen checks. The visible package, separate Merlin qualification and bound
model verify 38, 57 and 90 artifacts. The explicit 54-package/38-qualification
report at `out/allocation-word-coverage/coverage.json` retains 34/50 observed
definitions and 30 represented in host-qualified captures.

Bounded probes of the same static source under the `tensor` and `buffer`
structured-address profiles refuse a retained `tt.bitcast` with its source ID;
they are not successful conversions. Their relation to recovered structured
storage remains separate work from the qualified source-word path.

Different-width aliases, arbitrary physical model views, external synchronization,
broader atomic combinations and applicable target materialization remain open.
The [byte-alias implementation plan](byte_aliases.md) retains those requirements
and the upstream offset-rescaling boundary rather than treating them as covered.

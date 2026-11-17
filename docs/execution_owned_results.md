# Results discovered during execution

Merlin now has an explicit serial host mechanism,
`merlin.execution-owned-results.v1`, for tensor results whose dimensions are
computed by the executed body. The
[Merlin implementation](https://github.com/ucb-bar/merlin/commit/99c96b1066f0e3f974da753f9d6decab2b121fad)
exposes `merlin.llvmlower.owned_results.compile_owned_results`. Explicit signature
v6 now connects ordinary captures, standalone kernels and bound models to
triton2MLIR's selected compiler-only staging and Merlin host invocation.
Existing signatures through v5 retain their preallocation contracts. The separate
native evaluator, generated C/target harnesses and arbitrary producer/binding
shapes remain further work. Earlier package profiles retain their schemas;
internally computed call sizes now use the explicit
[v4 binding and v2 package](executed_model_binding.md) extension.

## The concrete boundary

The input-word model already supports:

```python
n = count.item()                  # an actual public singleton input word
first = prefix(x, n, x.shape[0])
```

Its v5 contract independently proves a size expression before allocation.
That expression does not represent general vector computation such as:

```python
selected = x[x > 0]               # size discovered from vector contents
return selected, selected, x
```

The new consumer mechanism executes standard MLIR for positive-value
compaction, obtains the actual result descriptors afterward, and copies every
result to independent caller storage. It preserves the source's actual count
and pack loops rather than invoking a separate size predictor. Duplicate
results and returned snapshots of input values retain their full contents.
This is an actual standard-MLIR execution test, not a claim that the PyTorch
indexing syntax above already imports through model2MLIR.

## Explicit staged host interface

```python
check_merlin_interface(capture, work, merlin_root=merlin_root,
                       python=merlin_python, result_mode="execution-owned")
outputs, report = check_merlin_host(work, arguments, merlin_root=merlin_root,
                                  python=merlin_python,
                                  compiler_python=native_python, clang=clang)
```

The argument archive is the complete positional invocation, including parameter
arguments. The staging selection receives no argument/reference data or argument
path. Each original source/package validation still runs before v6 selection.
Public input/result ranks, storage and static dimensions come from actual MLIR;
unknown result extents stay null until execution. Existing input axis/scalar
relationships and scalar bounds remain restrictions, with source guards intact.
They do not supply output predictions or rewrite permissions.

The boundary versions are explicit:

| Boundary | Selected profile |
|---|---|
| Compiler result signature | `merlin.compile-signature.v6` |
| Executed result ABI | `merlin.execution-owned-results.v1` |
| Saved compiler/native program | `merlin.owned-result-program.v1` |
| Interface observation | `t2m.merlin-interface-check.v3` |
| Host compilation/evaluation | `t2m.merlin-host-compilation.v2` / `t2m.merlin-host-evaluation.v2` |

The interface observation explicitly reports C runtime generation as not
performed. The separate compiler worker lowers the entire standard body without
evaluator arguments. `invoke_merlin_host` reopens fixed, checked compiler
products and the private native image without recompilation. Every descriptor
comes from the actual call, then its complete result is copied for comparison.
Rehashed false type/result/resource declarations refuse independent source
reconstruction. Older preallocated runtime readers reject v6.

## Reused upstream behavior and resource contract

The selection retains Merlin's ordinary bufferization, ownership, deallocation
and independent lifetime checks. It omits result-to-out-parameter conversion.
Upstream ownership transfers returned allocations to the caller and inserts
copies where borrowed arguments otherwise escape. The official clone lowering
then emits those copies in standard memory operations. A generated C bridge
uses upstream's first-result-pointer convention; C owns tuple descriptor layout.

Official generic MemRefToLLVM callbacks select an invocation-local allocator.
Actual allocation records bound capacities, peak charged bytes and allocation
count, including alignment/tracking overhead. Additional executable checks on
actual dimensions precede byte-product arithmetic, so a negative or overflowing
extent cannot wrap to a small allocation request. The initial resource profile
covers supported identity-layout heap allocations in a native 64-bit host ABI;
stack/nonidentity allocations and other native widths need further contracts.

Every result is checked against a live allocation before payload copying.
Checks retain static dimensions, signed strides, alignment, accessible byte
footprints and a total output-byte limit. Empty views perform no access.
Duplicate owners release once. Cleanup frees recorded owners rather than
trusting pointers supplied by descriptors. Unreturned live allocations refuse.
Private loaded images and locally bound functions retain the selected program;
source/prepared/LLVM/bridge/library and implementation bytes are checked.

Provenance and source locations survive preparation. Neither annotations nor
the descriptor transport grant numeric, memory, alias or ownership permissions.
The portable model still returns tensors; the selected host consumer owns this
lower-level allocation/descriptor ABI.

## Qualification and remaining work

The current Merlin selection passes **355** staging, source-only, native,
allocation and lifetime checks in `out/build/execution-staging-final1.log`.
The final **10** signature/staging checks separately include executing
content-derived compaction after staging and program reopening, in
`out/build/execution-staging-final2.log`. Counts overlap. Earlier failing probes
remain distinct; the first complete-model probe exposed the native fragment
composition bug, now covered by an explicit callback regression.

The producer/model selection passes **47**, with one optional feature check
skipped, in `.scratch/execution-model-final1.log`. It includes the existing
input-word packages/Spike regressions and the new three-profile selected-host
models. All **21** original model results match captured PyTorch words; **18**
additional unchanged-library host replays check **126** results with independent
capacities, counts and parameters. Twelve invalid source-domain calls abort
without outputs; three rehashed false-result signatures refuse. The selections
are separate from the historical 352-package inventory and supply no new v6
target qualification. Negative counts execute their actual source assertion in
this mode; earlier preallocation profiles retain their own early-size refusal.
`out/execution-result-models/summary.json` independently reopens the complete
initial/replay values and actual compiled abort records. The implementation is
Merlin `99c96b1066f0e3f974da753f9d6decab2b121fad`, following the native fragment
composition fix `54b5a5c22`.
The final source-frozen model selection additionally passes all twelve checks
in `.scratch/execution-result-model-final2.log`; every retained interface
consumer-source hash matches the current Merlin implementation. The summary
now points to that final selection. Earlier logs retain their original scopes.

Merlin's focused tests exercise real content-derived output sizes, empty and
duplicate results, input snapshots, mixed result ranks, exact floating words,
native scalars, source/build substitutions, corrupted descriptors and actual
allocation failures. The allocator tests separately exercise aligned/empty
storage, byte/count bounds and invalid releases. Host correctness does not
qualify Spike or accelerators, and no new Triton source operation is counted.

The fresh resizing-loop experiment remains a located lifetime refusal.
Merlin's generation checker currently separates replaced carried buffers only
for positive static trip counts with a final same-block release. A runtime or
zero-trip loop can return its initial buffer, while positive trips return a
new generation. Conditional cloning/retention and returned ownership need a
closed proof before the checker can admit that case. The new transport keeps
the refusal; it does not disable the gate.

The remaining integration requirements are:

1. Extend the selected internally computed count path to remaining model/
   kernel shapes, views, resources and source compositions. Receipt v2 now
   packages that real model through independent v4 reconstruction and v6
   staging; no saved invocation value supplies executable source computation.
2. Extend the separate native evaluator while retaining complete parent data
   and result/effect references. The new v2 qualification explicitly uses
   Merlin host and the original PyTorch oracle; it does not claim a second
   native compiler backend. Older qualification formats do not inherit it.
3. Extend the independent generation proof for zero/runtime-trip resizing
   state, conditional final owners and transfer to function results. Close
   initial/fresh/final aliases and every use/release edge; preserve negative
   lifetime tests.
4. Materialize the same semantics in generated C and applicable target harnesses with real
   resource/lifetime contracts and complete execution evidence.

These remain part of the full arbitrary-TTIR objective, alongside broader
memory/effect/numeric/source compositions and accelerator scheduling/execution.

# Volatile reads over private invocation memory

Explicit `--effect-profile volatile` retains real `tt.load volatile=true`
observations through guarded source-word memory. The default profile still
refuses them. Each active logical lane performs an actual standard LLVM
volatile load, including reads whose values are unused. The logical grid,
tensor tiles, source loops and reused values remain visible above this leaf.

This profile requires `--memory-profile guarded-buffer --address-policy
source-words`, or explicit automatic memory selection with the same effect
profile. It does not admit external participants, physical pointer identity,
source byte reinterpretation or mixed atomic composition. Selected synchronous
program barriers additionally require the explicit [v3 phase contract](volatile_program_phases.md).
Other effects require additional source/effect contracts. A source
volatile request cannot silently become a stable tensor read.
Selected [atomic/volatile composition](atomic_volatile_effects.md) additionally
requires `--effect-profile atomic-volatile` and admission v6/word stages v4;
the atomic numeric/ownership component remains separate from the volatile one.
Pinned TTIR forbids volatile map reads; native verification still rejects them.

## Why normalization needs protection

At Meta/shared `17f0c234ce474bee334af36e351c62eca7852836`,
`TritonToUnstructuredPass.cpp` creates an ordinary `tts.gather` from a source
load without carrying `isVolatile`. Arithmetic conversion can then remove an
unused gather inside its own rewrite driver, even without an explicit cleanup
pass. Pinned LLVM `a47bddccec30255619bb8c37fa59700e661d4e66` also explicitly
lowers an ordinary `memref.load` to a nonvolatile LLVM load. Merely retaining
the original attribute in provenance cannot fix either boundary.
The [Meta implementation](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp),
its `gather_no_loop.mlir` fixture, and the
[archived Microsoft conversion](https://github.com/microsoft/triton-shared/blob/050b93ae9c737111f1ca425d793419b32427a971/lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp)
record the reused pointer boundary. Neither supplies this effect/protection ABI.

The frontend reuses Meta's actual pointer-event conversion. Source-word stages
v3 additionally retain complete, private identity functions and conservative
standard calls to keep each load value live. These introduce no external
implementation or assertion about source numeric values. Computation and memory
stages exclude cleanup until guards and real memory observations are restored.
After actual volatile loads exist, the checked identities are inlined away;
neither helper functions nor calls enter exported MLIR. Earlier removal refuses.
Original and intermediate source, complete helper bodies and native stage
fingerprints remain bound as compiler evidence.

## Standard executable leaf and granularity

The exactly-once SCF leaf reads real memref descriptor metadata: aligned base,
offset and every index/stride contribution. Element width supplies the byte
scale; standard index arithmetic and an internal LLVM pointer carry that address
to `llvm.load volatile`. This is target-neutral LLVM memory computation. It
does not encode a GPU address space, cache placement, instruction or physical
launch mapping, and no address escapes as a logical source result.

`t2m/examples/volatile_kernels.py` retains two input reads, including an unused
one, followed by runtime read/update/read iterations on private state. Original
masks, tail fills, preserved input/state words and zero-trip outputs remain.
Producer and independent Merlin readers reopen actual typed source, require
every observation, check exact descriptor SSA and real volatile fields, and
reject invented overflow/alignment/order promises or escaping low-level ops.
Annotations retain source intent; they cannot replace an actual observation.

Unlike pure model2MLIR tensor payloads, this explicitly selected representation
contains localized LLVM memory leaves. The public tensor/scalar ABI and capture
receipt schemas remain unchanged. Admission v5 and source-word stages v3 carry
the effect/protection contracts; older readers reject their versions. Standard
LLVM bindings preserve the missing volatile property without enabling arbitrary
target intrinsics. Existing bounds, initialization, program-conflict and native
ownership/lifetime checks remain authoritative.

## Current qualification and next gates

The maintained native selection exercises i32/i64/f32/f64 over empty, singleton,
ragged and multi-program lengths and runtime/zero-trip loops. All four static
volatile sites survive LLVM legalization. Separate i64/f32 packages pass
reusable Merlin host libraries and actual serial RISC-V/Spike images. Both
readers reject changed flags, missing unused observations, descriptor recipes,
identity bodies, selected effects and cleanup pipelines. These are selected
private memory observations, not GPU cache, external publication or performance
equivalence.

Original allocation wrappers and complete dependent models now have the separate
qualification below. Selected [program phases](volatile_program_phases.md) now
have their own source/consumer contract. Next compose the remaining numeric/
storage/effect profiles.
Shared/external state, asynchronous publication/progress and applicable
accelerator correctness/performance remain completion work. Observing all fifty
pinned TTIR names does not establish this semantic closure.

```bash
python -m pytest -q tests/test_volatile_memory.py \
  --basetemp=.scratch/pytest-volatile-memory
```

Use the existing source/native/Merlin/serial-target environments from the README;
no Triton CPU driver or GPU backend is selected.

## Original allocation source and complete models

`scripts/capture_volatile_model.py --dtype i64|f32` supplies actual PyTorch
models with learned bias/initial parameters, two dependent calls and an empty
call:

```python
original, initial = x + self.bias, x + self.initial
first, old1, seen1 = update(original, initial, x.shape[0], 2)
second, old2, seen2 = update(original, first, x.shape[0], 3)
empty, old0, seen0 = update(x[:0], initial[:0], 0, 1)
return second, first, old1, seen1, old2, seen2, original, x, initial, empty, old0, seen0, self.bias, self.initial
```

Each source wrapper clones its input state. Original allocation capsules retain
the pre-wrapper source separately from the bound graph; all four original
volatile sites survive in each implementation. Identity anchors use distinct
checked read IDs. An enclosing callsite location may occur on several reads and
cannot identify an individual observation. Checked anchor removal requires the
corresponding actual LLVM volatile leaf, even after helper/wrapper inlining.

The complete models preserve all fourteen original results, earlier snapshots
and every parent data/archive byte. Executed-call binding v4, receipt v2 and
Merlin signature v6 retain the existing public interface and actual returned
descriptors. Localized LLVM leaves and the explicit effect/stage profiles remain
declared representation differences; no physical caller copyback is introduced.

Two complete models retain 28 initial host results, 84 results across six
unchanged-library invocations and 28 results across two actual Spike images as
exact words. Six rehashed source/effect claims refuse in both readers; two
original public-domain violations actually abort without outputs. The standalone
selection separately retains six target results across two images and twenty
source/payload/stage claims refused by both readers. These scopes overlap other
selections and do not expand the historical opcode inventory.

The final source/automatic-selection run passes 38 checks; an earlier overlapping
word/atomic/source-protection selection passes 43. Six complete-model pytest
checks pass, and four additional refusal/domain checks execute directly against
those retained libraries. Merlin's source/derived-interface selection passes
511. `out/volatile-observations/summary.json` reopens original packages, all
results, unchanged library/input bytes and actual target qualification with
current independent-reader fingerprints. It links the logs and failure evidence.

General external/shared state, mixed atomic and broader numeric/storage
composition, applicable accelerators and measured performance remain open.

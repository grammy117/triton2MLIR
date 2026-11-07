# triton2MLIR — design plan and session handoff

**Status:** planning only. No code exists yet, in this repo or elsewhere. This document records
what has been surveyed, which decisions are locked, and the one architectural question that is
still open and blocks spec work.

**Date of survey:** 2026-11-06. **Host:** a26 (shared VM). All reference paths below are on that
host unless a URL is given.

---

## 1. Goal

A tool that parses a Triton kernel's TTIR and emits standard-dialect MLIR at the same granularity
as [model2MLIR](https://github.com/ucb-bar/model2MLIR) (`m2m`), so that a hand-written Triton
kernel and a captured PyTorch model are **interchangeable inputs** to the same downstream
compiler (merlin).

The value we are trying to carry across the boundary is the *kernel author's work*: the fused
dataflow, the blocking, the algorithm (e.g. online softmax). The thing we must not carry across is
*NVIDIA*: flat-address-space pointer arithmetic, SPMD thread mapping, shared-memory staging,
warp/stage counts, tensor-core spellings.

## 2. Decisions locked (user, 2026-11-06)

| # | Decision |
|---|---|
| D1 | **Standalone repo**: `github.com/grammy117/triton2MLIR` (this repo). Not a branch of model2MLIR, not a subpackage of merlin. |
| D2 | **Primary consumer: merlin, at the same level as model2mlir.** Target the `linalg-on-tensors` + `high-level` contract merlin already ingests, so kernel IR and model IR are interchangeable. |
| D3 | **Two output levels, selectable** — mirroring m2m's `high-level` vs `linalg-on-tensors` split, with one deterministic pass between them. |
| D4 | **Scope commits through fused attention**: elementwise → tiled GEMM → reductions/softmax → a flash-attention-shaped kernel with a loop-carried accumulator. |
| D5 | **triton-cpu is a reference only, not a dependency.** (model2MLIR's `feat/triton-integration` branch uses it; we do not.) |

## 3. OPEN QUESTION — blocks the spec

D3 says "both levels". It does not say **which form is emitted directly from TTIR and which is
derived**. That choice decides the tool's ceiling.

The constraint that drives it, verified by reading merlin's bridge
(`src/merlin/triton/bridge.py:_op_tt_dot`, `addressing.py:whole_tensor_access`):

- merlin's bridge **refuses `tt.dot` whenever the grid is not `(1,1,1)`** — a multi-program
  contraction is an explicit `BridgeError`.
- its coverage proof **enumerates concretely**, bounded at `MAX_ENUMERATED_ELEMENTS = 1 << 22`.

So a real tiled GEMM is structurally out of reach on a whole-tensor-first path, and fused attention
more so: flash attention *is* a tiled loop carrying `(acc, m, l)`. Its whole-tensor form is just
"attention", which discards the algorithm that made the kernel worth capturing.

### Option A — tile-first, fold down **(recommended)**

Emit the kernel's own iteration space as the primary form: grid → `scf.forall`, reduction loops →
`scf.for` with `iter_args`, tile load/store → `tensor.extract_slice` / `tensor.insert_slice`, masks
→ clamped slice extents. Then **one deterministic pass folds the nest to whole-tensor form when the
tiling is provably an exact partition**, and otherwise refuses and stays tiled.

- Faithful by construction; the portable form becomes a *theorem about* the tiled form rather than
  an independent guess.
- The fold pass is merlin's `whole_tensor_access` check generalized from enumeration to a symbolic
  statement about slice extents — strictly more powerful, and it removes the 2²² ceiling.
- Cost: the tiled level must be defined precisely before anything else works.

### Option B — whole-first, re-tile

Keep merlin's existing approach and add a re-tiling pass.

- Reuses working, tested code; shortest path to a first result.
- Cannot reach a tiled GEMM without being rewritten, and re-tiling *invents* a blocking instead of
  recovering the author's. That inverts the goal.

### Option C — independent dual emission

Two emitters from TTIR, one per level.

- No single source of truth and no check that the two agree. This is the silent-divergence failure
  mode both merlin and triton-shared explicitly guard against; rejected unless someone argues for it.

**Recommendation: A.** Awaiting confirmation.

## 4. The hardware/software boundary (the granularity principle)

The line proposed for where to cut, and the thing to argue about if A is wrong:

> **Keep the memory-hierarchy-agnostic dataflow. Drop where values physically live.**

| Keep in the IR | Push to `prov.*` provenance |
|---|---|
| iteration space and its bounds | `num_warps`, `num_stages` |
| tile shapes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) | warp specialization, swizzling |
| dependence and accumulation structure | shared-memory / L2 residency choices |
| which values stay live across loop iterations (as `iter_args`) | pipelining depth |
| masking/boundary behavior | `tl.multiple_of`, `tl.max_contiguous`, `tl.assume` hints |
| numeric contract (accumulator type, `ieee` vs `tf32` as an attribute) | physical address space, pointer bases |

Concretely: *"this tile is reused across the K-loop"* is dataflow and belongs in the IR as an
`iter_args` value. *"this tile lives in shared memory, triple-buffered across 8 warps"* is an
allocation decision and belongs in provenance for the consumer to make. Holding that line is what
keeps the output off NVIDIA without flattening the kernel into an op soup.

A corollary worth stating because it is load-bearing: `BLOCK_*` constants are **portable
meta-parameters**, not hardware facts. merlin already treats them this way
(`docs/design/triton_frontend.md`: "`BLOCK_*` / `GROUP_*` are kept as portable meta-parameters that
Merlin may later treat as schedule candidates"). They stay in the IR as loop bounds; the consumer
is free to re-tile.

## 5. Prior art, as verified by reading the source

### 5.1 merlin already has a TTIR → linalg bridge — read it before writing anything

`/scratch/grammy/blast/merlin/src/merlin/triton/` (branch `main`, HEAD `e45205bf9`), 2,413 lines:

| file | lines | job |
|---|---|---|
| `source.py` | 244 | `@triton.jit` → TTIR. **The only module touching Triton internals** (stock wheel, `ASTSource.make_ir`; Triton is never patched). |
| `addressing.py` | 271 | pointer → tensor re-raising: affine addresses + grid normalization |
| `bridge.py` | 637 | TTIR → linalg: abstract interpretation + linalg emission |
| `spec.py` | 235 | the semantic ABI a kernel cannot state about itself |
| `diagnostics.py` | 83 | `BridgeError` + capability report |
| `cli.py`, `toolchain.py`, `__init__.py` | 436 | `merlin-compile-kernel`, Triton pin + probe |

Design doc: `docs/design/triton_frontend.md`, with ten gate-enforced invariants (INV-1..10).
INV-6 is the one that matters most here: **linalg-on-tensors is the single convergence point.**

**What it does well, and we should port rather than reinvent:**

- **Affine index form.** Every index-space value is `const + Σ cᵍ·program_id[g] + Σ kᵈ·iota[d]`,
  kept as sparse maps so "does not depend on the grid" is a property (`not pid`), not a
  comparison against zeros. Non-affine input (a product of two varying indices) is **refused**,
  not approximated.
- **Fail-closed op accounting.** A `CapabilityReport` tracks ops *seen* vs *lowered* vs
  *deliberately discarded*; a non-empty remainder aborts translation. This is the guard against
  emitting a module that quietly computes something else.
- **Backward liveness to a fixpoint.** Triton emits an overflow guard per index expression (widen
  to i64, compare, `and`), dead only *transitively*. One-level deadness leaves the widening
  looking live and forces the bridge to interpret an i64 shadow of the real computation. This bug
  was found the hard way; inherit the fixpoint.
- **Masks settled concretely, so they cannot be ignored.** A dropped mask is the dangerous bug —
  it computes the right answer whenever the block size divides the extent. merlin's regression
  test is the same unmasked kernel **accepted at n=1024 and refused at n=1000**. Keep that test.
- **Declared, never inferred, effects.** A pointer's `read`/`write`/`readwrite` is in the spec,
  because a kernel that mutates an input the caller believes is read-only is a miscompile, not a
  detail.

**Where it stops** (`docs/design/triton_frontend.md`, "What is explicitly not being built yet"):
tiled/multi-program GEMM, masked tails on the accelerator path, softmax, model kernel replacement,
`torch.library.triton_op` integration, autotuning. **Every one of those is in our D4 scope.**

Its hard refusals, each an error today — this is our work list:

| refusal | where |
|---|---|
| `tt.dot` under grid ≠ (1,1,1) | `bridge.py:_op_tt_dot` |
| `tt.dot` from a non-zero accumulator | same ("would have to become a linalg `outs` operand carrying real data") |
| access that covers a tensor but permuted (transposed tile) | `addressing.py:whole_tensor_access` |
| > 2²² elements enumerated | same |
| two different access patterns to one argument | `bridge.py:_pointer_argument` |
| more than one store to an argument | `bridge.py:_op_tt_store` |
| a kernel that returns a value | `bridge.py:_op_tt_return` |

**Why merlin's bridge cannot simply be extended in place** (and D1 sends us elsewhere): merlin's
`src/merlin/**` is under three gates — no target-name literals, **no `import re`**, no hardcoded
opcodes — and `merlin/tests/` has a fixed bucket set with no `triton/` bucket. Those are good rules
for a compiler core and a poor fit for a frontend that must grow fast and own a kernel corpus. We
should nonetheless **keep INV-6 and the fail-closed discipline as our own invariants.**

### 5.2 triton-shared (facebookincubator) — the structured-pointer abstraction

`/scratch/grammy/refs-triton2mlir/triton-shared`, HEAD `17f0c23` (2026-10-26), Triton pin
`142586f90aac9b917af1081e4697b085027e13b2`.

> **Correction to merlin's recorded rationale.** `docs/design/triton_frontend.md` rejects
> triton-shared partly on the grounds that its README states verbatim *"This repository is no
> longer maintained."* That sentence is **not in the current README**, the repo has **moved from
> `microsoft/` to `facebookincubator/`**, and it is **actively committed** (PR #45 merged
> 2026-10-26, eleven days before this survey). The *technical* objections still stand — see below —
> but the maintenance objection is stale and merlin's doc should be corrected.

**The idea worth taking: `tts.make_tptr`.** A structured pointer is
`(base, sizes, strides, offsets, shape, order)` with static/dynamic splits, where `order` present
means "pointer to a tensor" and absent means "tensor of pointers", and `shape` carries either the
parent tensor's shape or the wrap-around boundary. `isStructuredPtr()` is exactly
`all(static_shape == 0)`. That five-field descriptor is empirically sufficient for the kernel class
we care about, flash attention included — which is the evidence that a *symbolic* strided
descriptor (Option A's requirement) is a realistic target rather than a wish.

**Its three analyses, worth mirroring one-for-one:**

1. **Pointer analysis** — walks the IR building strided accesses.
2. **Use analysis** — after pointer analysis, address-computation ops are dead. Ops used *only* in
   address computation are `MetaUse`; ops used in *both* address and data computation are
   `MixedUse` and get **cloned** so the `MetaUse` half can be deleted safely. (merlin's liveness
   fixpoint solves the same problem differently; the cloning trick handles a case liveness alone
   does not — a value that is genuinely both.)
3. **Mask analysis** — masked load/store.

**Pipeline order** (`TritonToLinalgExperimentalPass.cpp:59-87`), a useful shape to copy:
`TritonToStructured` → CSE → canonicalize → `TritonToUnstructured` → `TritonArithToLinalg` →
`TensorGatherToLinalg` → `StructuredToMemref` → `UnstructuredToMemref` → `TritonPtrToMemref` →
`TritonToPtr` → reconcile casts → `RemoveDeadValues` → CSE → canonicalize → optional
`CollapseShape`. **We stop after the `*ToLinalg` stage and never run the `*ToMemref` half.**

**Why it is not adopted as a dependency** (merlin's findings 1 and 2, re-verified):

- It builds as an **out-of-tree backend inside a Triton source build** pinned to `triton-hash.txt`
  (`TRITON_PLUGIN_DIRS`, binaries under `triton/build/<cmake>/third_party/triton_shared`). No pip
  path. Adopting it means owning a pinned Triton source build.
- Its output is **post-bufferization**: `memref.reinterpret_cast`, `memref.alloc`, `memref.copy`,
  `bufferization.to_tensor`, `affine.store` wrapped around tensor-level linalg. merlin cannot
  consume that, concretely: `interface_lowering.lower_to_interface` builds its `value_map` **only**
  from `src_block.args`, so a non-block-argument operand is a `KeyError`; and
  `contract_facts._trace_to_block_arg` walks only `linalg.transpose` and
  `tensor.{cast,reshape,collapse_shape,expand_shape}` / `linalg.copy`, so a bufferized operand
  traces to nothing and residency is never inferred.
- Known gaps in its own words: `tt.reduce` → `linalg.reduce` supports **only `addf` and `maxf`**
  reduction bodies; `MatmulConverter` **fails on rank > 3**.
- Its `program_id` handling is a tell: `TritonArithToLinalgPass` **appends six `i32` function
  arguments** (3 program ids + 3 num_programs) and rewrites `tt.get_program_id` to read one. That
  is a *launcher ABI*, not a normalization — it hands the grid to the runtime unchanged. We must
  not do this; it is exactly the "bake in one parallelism choice" mistake.

### 5.3 triton-cpu — the counterexample that defines what we must not do

`/scratch/grammy/refs-triton2mlir/triton-cpu`, HEAD `ef84a29` (2026-11-04).

Its type converter maps `tensor` → **`VectorType`** (`TypeConverter.cpp:15`), memory to
`vector.transfer_read` / `masked_load` / `gather` and `transfer_write` / `masked_store` / `scatter`,
and its `TritonCPU` dialect carries `extract_memref`, `ptr_to_memref`, `load`, `store`, `dot`,
`brgemm_create` / `brgemm_execute`. Its `make_tttcir` stage then picks a dot lowering per host
feature: ukernels (oneDNN/XSMM), nanokernel, dot-product, AMX, FMA, ARM NEON BFMMLA.

This is a **well-built backend** and the wrong altitude for us by one whole level: the ISA/feature
choice is made *inside the compiler*, so the IR is committed to a host before any consumer sees it.
It is the concrete picture of what "tying the MLIR to a specific backend" looks like, and it is why
D5 excludes it. Useful as a cross-check on individual op semantics (its `OptimizeMasks` and
`ConvertUnsupportedOps` passes are worth reading when we hit mixed precision).

### 5.4 triton-ascend — the proof that the grid is not a hardware fact

`/scratch/grammy/refs-triton2mlir/triton-ascend`, HEAD `8f0ade6` (2026-11-06, same day as survey).

Pipeline `ttir_to_linalg`: `triton_control_flow_opt` → **`triton_to_structure`** →
`discrete_mask_access_conversion` → `triton_to_annotation` → `triton_to_unstructure` →
`triton_to_hivm` → `triton_to_hfusion` → `triton_to_llvm` → `bubble_up_operation` →
**`triton_to_structure` again** → `triton_to_linalg(named_ops=True)` → `merge_concat_load_buffer` →
`dynamic_cv_pipeline`.

Three things to take from it:

1. **It has its own `TritonStructured` dialect** — the same structured-pointer abstraction as
   triton-shared, independently kept. Two unrelated vendors converging on `make_tptr` is the
   strongest available evidence that the descriptor is the right intermediate.
2. **`named_ops=True`.** Its TTIR→linalg pass has a flag choosing *named* linalg ops over generic
   bodies. That is precisely m2m's Tier-2 argument (`docs/OP_TAXONOMY.md`) arrived at from the
   other direction, and it supports making our `high-level` level named-op-rich.
3. **`ascend.annotation`** — an op whose only job is to attach key-value attributes to a tensor
   (`$src attr-dict : type($src)`). A clean precedent for carrying advisory facts without
   polluting the op vocabulary. Our `prov.*` attributes do the same job; worth noting that a
   *separate annotation op* survives passes that drop discardable attrs, which is relevant given
   the xDSL printer caveat in §6.3.

**The architectural lesson is in its docs, not its code** (`docs/en/migration_guide/architecture_difference.md`):
on Ascend the grid is a **physical core-group mapping**, capped at the AI-core count — not a logical
dimension. Its `auto-blockify` pass therefore wraps the kernel body in an `scf.for` over
`gpu.linear_block_id` with `chunk = ceildiv(logical_blocks, physical_cores)`, and clamps the
launch count to match, so a >65,535 grid works. Its on-chip budget is a hard 192 KB (Atlas 800T A2)
and its tiling vocabulary is `ncore` / `xblock` / `xblock_sub`.

Read against merlin's Radiance arm (`must_map_to_warps`, `lanes_per_warp: 16`) the conclusion is
unavoidable and it is the single most important design input here:

> **The same Triton grid must become sequential iteration on one target, warps on another, and a
> bounded fold onto physical cores on a third. Any of those three baked into the frontend breaks
> the other two.** The grid must therefore reach the consumer as an *un-decided iteration space* —
> which is exactly what Option A's `scf.forall` is, and what triton-shared's six appended launcher
> arguments are not.

Note also Ascend's order-independence caveat: `auto-blockify` visits logical blocks in chunk order,
so kernels assuming strict block-id ordering must be rewritten. If we emit `scf.forall`, we are
**asserting order-independence**; a kernel whose correctness depends on grid order must be refused,
not silently reordered. Add that to the fail-closed list.

### 5.5 model2MLIR — the output contract we must match

`/scratch/grammy/blast/model2MLIR`, branch `feat/triton-integration`, HEAD `035bb5a`.
Read `AGENTS.md`, `docs/OP_TAXONOMY.md`, `docs/guides/triton_kernel_integration.md`.

- **Two levels.** `linalg-on-tensors` (default, portable: `builtin, func, arith, math, linalg,
  tensor, scf, cf` plus `quant_ext` as the one principled exception) and `high-level` (opt-in:
  `linalg_ext.{softmax,layer_norm,rms_norm,rope,swiglu,gelu,silu}`,
  `tensor_ext.{concat,pack,unpack}`), with a deterministic `m2m.transforms.expand_to_linalg`
  lowering the second to the first. **Our D3 split should be the same shape and reuse these
  dialects.**
- **The `prov.*` attribute schema** — `prov.family` (closed vocabulary, ~19), `prov.op` (open),
  `prov.region_id`, `prov.aten`, `prov.orig_dtype`, `prov.level`. Stamped centrally at one
  chokepoint (`import_fx._forward_fx_meta` + a `module.walk()` backfill) so coverage cannot be
  forgotten when a new decomposition is added. **Copy the chokepoint discipline, not just the
  attribute names.**
- **The scalability thesis** (`OP_TAXONOMY.md`): keep the number of distinct things a downstream
  pass must match small and fixed; adding ops grows an `op → family` map, not the pass surface.
  Our equivalent is a `TTIR op → family` map.
- **The kernel registry is the integration seam.** A registered custom op
  (`m2m/ir/kernel_registry.py`, `KERNEL_REGISTRY`) is emitted as a typed black-box `func.call` to a
  private declaration, tagged `prov.kernel = "ns::name"` with `prov.kernel_operands` (schema
  argument names, in schema order) and `prov.kernel_args` (non-tensor arguments; omitted when
  none). merlin binds it by looking `prov.kernel` up in
  `src/merlin/llvmlower/kernel_table.py`.
  **triton2MLIR's natural job is to *fill in that declaration*** — the same `prov.kernel` identity,
  a real body instead of a hand-written CPU kernel. That is the end-to-end story and it needs no
  new contract.
- **The validation pair already exists.** `workloads/mlp_basic` (plain PyTorch) and
  `workloads/mlp_basic_triton` (same math, same seed and weights, with `linear→relu→linear→relu`
  fused into `triton_exp::fused_mlp2`) are byte-comparable apart from the one call. merlin runs
  both on spike rv64gcv: max abs error 3.0e-8 with the kernel (15,558 retired instructions) and
  4.5e-8 without (16,156). **This is our convergence oracle, free.** See §9.

## 6. Proposed architecture (Option A)

### 6.1 Pipeline

```
@triton.jit  ──[stock Triton wheel, ASTSource.make_ir]──▶  TTIR
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
        (1) liveness  — backward fixpoint; drop overflow-guard chains
        (2) pointers  — symbolic strided descriptors (base, sizes, strides, offsets, order)
                        over program_id and loop IVs; refuse non-affine
        (3) masks     — predicates kept exactly; become slice extents, never dropped
                        │
                        ▼
        emit  ──▶  LEVEL "kernel-tiled"   (primary; the author's iteration space)
                     scf.forall over the grid, scf.for + iter_args for reductions,
                     tensor.extract_slice / insert_slice, linalg_ext named composites
                        │
                        │  transforms.fold_grid  (deterministic; refuses when not an exact partition)
                        ▼
                   LEVEL "linalg-on-tensors"  (portable; what merlin ingests today)
                     whole-tensor linalg, standard dialects only
```

Both levels are checked by one `coverage.py`: every TTIR op is *seen*, and must be *lowered* or
*explicitly discarded*; a non-empty remainder is an error, never a warning.

### 6.2 TTIR construct → treatment

Proposed, not implemented. The `kernel-tiled` column is the primary emission; the
`linalg-on-tensors` column is what `fold_grid` produces when the fold succeeds.

| TTIR construct | `kernel-tiled` | `linalg-on-tensors` |
|---|---|---|
| `tt.get_program_id` / `tt.get_num_programs` | the `scf.forall` induction variable / bound | folded away |
| `tt.make_range`, `splat`, `expand_dims`, `broadcast`, `addptr` | consumed by pointer analysis; never emitted | same |
| `tt.load` (structured) | `tensor.extract_slice` from a tensor block argument | whole tensor block argument |
| `tt.store` (structured) | `tensor.insert_slice` / `scf.forall.in_parallel` | destination-passing result |
| masked `tt.load`/`store` | clamped slice extents; `other` → `tensor.pad` or `linalg.fill` on the pad region | clamp folds away when the mask is exactly the tail |
| `tt.load`/`store` unstructured | `tensor_ext.gather` / `scatter`, or an indexed `linalg.generic` | same |
| `tt.dot` | `linalg.matmul` / `batch_matmul` / `quantized_matmul` into an `iter_args` accumulator | `linalg.matmul` on whole tensors (grid=1 only) |
| `tt.dot` `input_precision` (`ieee`/`tf32`/`tf32x3`) | `prov.numeric_precision` attribute — **never** a dialect or op choice | same |
| `tt.reduce` | `linalg.reduce`, region body translated (not just add/max) | same |
| `tt.scan` | `linalg_ext` scan (m2m's Tier-2 `scan` candidate) | `linalg.generic` + `scf` |
| softmax shape recognized | `linalg_ext.softmax` | expanded by `expand_to_linalg` |
| `tt.trans`, `split`, `join`, `cat`, `reshape` | `linalg.transpose`, `tensor.*`, `tensor_ext.concat` | same |
| `tt.atomic_rmw` / `atomic_cas` | **refuse** initially. Later: admissible only when provably an order-independent reduction over a partition, then `tensor_ext.scatter` with a declared combine | same |
| `tt.extern_elementwise` / libdevice | `math.*` when an exact standard op exists; otherwise **refuse, naming the symbol**. Never approximate a device intrinsic | same |
| `tt.make_tensor_descriptor` / TMA descriptor loads | the same structured descriptor (this is what triton-shared's `order`/`shape` fields are for) | same |
| explicit on-chip staging / `alloca` | an SSA value in the loop body; the allocation decision is the consumer's | same |
| `num_warps`, `num_stages`, `tl.multiple_of`, `max_contiguous`, `tl.assume`, `tl.debug_barrier` | `prov.triton_hint` provenance only. A "hint" that changes semantics is refused, not recorded | same |
| `tt.print`, `tt.assert` | dropped with a recorded note / `cf.assert` | same |
| kernel whose correctness depends on grid **order** | **refuse** — `scf.forall` asserts order-independence (see §5.4) | n/a |

### 6.3 Known hazards to design around

- **xDSL drops discardable attributes on *named* ops in printed text** (verified and load-bearing
  per `OP_TAXONOMY.md`): they exist in memory but not in the text. So **named ops must be matched
  by op type, and only `linalg.generic` by `prov.*`**. If we need an advisory fact to survive on a
  named op in text, triton-ascend's separate `annotation` op is the precedent (§5.4).
- **`m2m`'s section splitter can emit use-before-def SSA** (merlin's
  `frontends/linalg_mlir.py` docstring: `sections/smolvla.model.mlir` references an undefined
  `%2034`). Do not build on sections.
- **xDSL's linalg parser rejects the parenthesized multi-result form** `} -> (tensor<…>, tensor<…>)`
  that MLIR prints; merlin repairs it textually (`strip_paren_results`). Emit the unparenthesized
  form.
- **merlin's staged path only materializes `{matmul, elementwise}`** (`compile_core.py:34`,
  `STAGED_MATERIALIZABLE`), and `lower_to_interface` **rebuilds the function body from scratch**,
  silently dropping unaccounted payload — merlin's finding 2 calls it a false-PASS generator. A
  `kernel-tiled` module with a masked store or an elementwise epilogue will *not* descend the
  staged path today, and could appear to succeed. **Our acceptance criteria must not be "merlin
  compiled it".**

### 6.4 Repo layout (proposed, mirrors `m2m`)

```
t2m/
  api.py          convert(kernel, spec, *, level=...) -> ConversionResult   # mirrors m2m.convert
  cli.py          t2m lower-kernel <kernel.py>:<fn> --spec spec.toml --level ...
  spec.py         the semantic ABI: shapes, strides, effects, grid, constexprs, assumptions
  frontend/       @triton.jit -> TTIR. THE ONLY MODULE TOUCHING TRITON INTERNALS
  analysis/       liveness.py, pointers.py (symbolic descriptors), masks.py
  ir/             emit_tiled.py, emit_whole.py, ttir_ops.py (op -> family map)
  transforms/     fold_grid.py (tiled -> whole-tensor), expand_to_linalg.py (reuse m2m's)
  coverage.py     op accounting, capability report, differential oracle
kernels/          one dir per kernel: kernel.py + spec.toml + expected .mlir (both levels)
docs/             design.md (this, evolved), plus per-milestone evidence
```

Open sub-question: **depend on `m2m` for `linalg_ext` / `tensor_ext` / `prov.*`, or vendor them?**
Recommendation: depend on `m2m` (one source of truth for the contract; the alternative is two
drifting copies of a shared vocabulary). Revisit if the dependency proves painful.

Invariants to adopt as our own, from merlin: **linalg-on-tensors is the convergence point** (never
lower toward a target dialect from here); **fail closed with a located diagnostic**; **account for
every op**; **one module owns Triton internals**; **no target names anywhere**.

## 7. Milestones

| M | content | why this order |
|---|---|---|
| **M0** | spec/ABI model, TTIR acquisition off a stock wheel, op-accounting skeleton, `vector_add` at both levels | smallest end-to-end skeleton; proves the two-level split before anything depends on it |
| **M1** | symbolic pointer analysis + mask analysis; `vector_add` with a **ragged tail** (the n=1000 test), 2D strided loads | masks and non-dividing extents are where silent wrongness lives; do them before scale |
| **M2** | **tiled GEMM**: `scf.forall` over the grid, `scf.for` K-loop with an `iter_args` accumulator, `tt.dot` → `linalg.matmul` | **first capability merlin's bridge does not have** (it refuses multi-program `tt.dot`). The real proof of Option A |
| **M3** | `tt.reduce` / `tt.scan` with translated region bodies; fused softmax with a masked tail | reductions are a prerequisite for attention; also where triton-shared stops (add/max only) |
| **M4** | **fused attention**: nested loops, online-softmax rescaling, loop-carried `(acc, m, l)` | the D4 target; stresses `iter_args` and the named-composite story |
| **M5** | `fold_grid` (tiled → whole-tensor) + the **model2MLIR convergence test** on the MLP pair + fill in a `KERNEL_REGISTRY` declaration end-to-end | the integration payoff; see §9 |

M2 is the decision point for Option A: if an `iter_args` accumulator over `extract_slice` tiles does
**not** fold cleanly to whole-tensor form for a plain GEMM, the fold pass is harder than assumed and
the plan needs revisiting before M3.

## 8. Fail-closed discipline (non-negotiable)

Inherited from both merlin and triton-shared, which independently arrived at it:

- An unsupported TTIR op, an unproven pointer pattern, or a dropped mask is a **located error**,
  never a best-effort lowering.
- Every op is accounted for: *seen* = *lowered* + *deliberately discarded*. A non-empty remainder
  aborts. "Safe to drop" is a recorded decision, not a gap.
- A scalar used in an address or mask with no compile-time value is an error naming the scalar and
  the spec field that would fix it.
- Effects are **declared**, never inferred from the body.
- No silent numeric substitution: a device intrinsic without an exact standard-dialect equivalent
  is refused by name.

## 9. Validation — three oracles

1. **Numerical.** The kernel's own reference: `TRITON_INTERPRET=1` (or the `@triton_op`'s eager
   PyTorch fallback) vs. the emitted MLIR executed through merlin's LLVM path. Must hold at
   extents that do **and do not** divide the block size.
2. **Level agreement.** `fold_grid(kernel-tiled)` must be numerically identical to the directly
   emitted `linalg-on-tensors` form wherever both exist. This is the check Option C cannot have.
3. **Convergence with model2MLIR — the strong one.** `workloads/mlp_basic` and
   `workloads/mlp_basic_triton` are the same MLP with the same seed and weights, one captured
   through FX and one through a Triton kernel. Lowering the Triton half with triton2MLIR should
   produce IR that converges with the FX half's `fc1→relu→fc2→relu` region. merlin already has the
   analogous byte-identical-convergence test for its hand-authored case
   (`docs/design/triton_frontend.md`: "Triton vs hand-authored linalg: byte-identical interface,
   target, runtime and command buffer"). **Two frontends, one contract, same math — this is the
   thesis of the whole exercise and the pair to demonstrate it on already exists.**

Per §6.3, "merlin compiled it" is **not** an acceptance criterion on its own: `lower_to_interface`
rebuilds the body and can drop unaccounted payload while still reporting success.

## 10. Risks / unresolved

| risk | note |
|---|---|
| **Fold pass harder than assumed** | symbolic exact-partition proof over `extract_slice` extents, replacing merlin's enumeration. If it fails at M2, revisit §3. |
| **`iter_args` accumulators vs. merlin's staged path** | merlin's interface layer materializes `{matmul, elementwise}` only; a loop-carried accumulator may have no staged form. Likely needs a merlin-side `interface` addition — and by merlin's own finding-4 criterion (does an equivalent *non-Triton* program hit the same wall?) that would be a merlin abstraction gap, not ours. |
| **Online softmax as a named op** | flash attention's rescaling is not `linalg_ext.softmax`. Either a new named op or an honest `scf` body. Decide at M4, not before. |
| **Non-affine addressing** | gather/scatter kernels fall outside the descriptor. triton-shared's `make_gather_scatter_tptr` + `TensorGatherToLinalg` is the reference if we need it. |
| **Dynamic shapes** | m2m emits `-1` dims and no consumer handles a dynamic kernel. Out of scope; state the refusal explicitly. |
| **Triton API churn** | confine every Triton internals import to `frontend/` (merlin's rule, and it has already paid off for them). Pin and probe like `merlin/triton/toolchain.py`. |
| **`m2m` dependency** | §6.4 open sub-question. |

## 11. Reference paths on this host

| what | where |
|---|---|
| merlin (bridge + design doc + consumer) | `/scratch/grammy/blast/merlin` @ `main` `e45205bf9` — `src/merlin/triton/`, `docs/design/triton_frontend.md`, `docs/guides/triton_kernels.md`, `src/merlin/compile_core.py`, `src/merlin/llvmlower/kernel_table.py` |
| model2MLIR (contract + the MLP pair) | `/scratch/grammy/blast/model2MLIR` @ `feat/triton-integration` `035bb5a` — `AGENTS.md`, `docs/OP_TAXONOMY.md`, `docs/guides/triton_kernel_integration.md`, `m2m/ir/kernel_registry.py`, `workloads/mlp_basic{,_triton}/` |
| triton-shared | `/scratch/grammy/refs-triton2mlir/triton-shared` @ `17f0c23` (sparse: `include lib test/Conversion python/examples`) |
| triton-cpu | `/scratch/grammy/refs-triton2mlir/triton-cpu` @ `ef84a29` (sparse: `third_party/cpu/{include,lib,backend}`) |
| triton-ascend | `/scratch/grammy/refs-triton2mlir/triton-ascend` @ `8f0ade6` (sparse: `docs/ third_party/ascend/`) |

Reference clones are shallow, blobless and sparse; re-create with
`git clone --depth 1 --filter=blob:none --sparse <url>` then `git sparse-checkout set <paths>`.

## 12. Next actions

1. **Confirm or reject Option A** (§3). Nothing below is safe to write first.
2. Settle the `m2m`-dependency sub-question (§6.4).
3. Write the M0 spec: the `TritonKernelSpec` data model and the exact `kernel-tiled` grammar
   (which ops may appear, in what nesting) — the grammar is the contract everything else is checked
   against.
4. Correct merlin's `docs/design/triton_frontend.md` on triton-shared's maintenance status (§5.2),
   keeping the technical objections intact.

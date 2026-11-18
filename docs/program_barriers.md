# Synchronous program barriers as portable memory phases

Pinned Triton 3.7.1 emits `tl.debug_barrier()` as `ttg.barrier all` even in
pre-layout TTIR. This operation orders synchronous memory observations among
the threads of one source CTA. It is distinct from a cluster barrier or an
asynchronous transfer's completion protocol.

The frontend now reifies this source ordering as standard `scf.execute_region`
phases at the original structured control positions. Each phase executes its
complete tile operations before later phases. Tensor conversion explicitly
yields the complete updated memory versions; subsequent reads consume those
versions. Standard-memory conversion retains the actual reads and writes inside
the regions. The native `no_inline` field survives serialization. It is an
optimization hint, not a substitute for the actual computation or dependencies.

The logical grid remains an unmapped `scf.forall`. Merlin still decides physical
placement and synchronization needed to implement these dependencies. No thread,
warp, shared-memory bank or target instruction is introduced by this abstraction.

## Source scope and preservation

The native-verified original graph is inventoried before transformation. A
`t2m.synchronous-program-barrier.v1` contract retains each exact address-space
bitmask, source ID and participation scope. Original and normalized graphs remain
separate; `prov.program_barriers` and source/emission correspondence retain intent.
These annotations grant no ordering or memory permission.

The admitted source has unencoded pre-layout values, one CTA per logical program,
whole-program structured participation and synchronous ordinary memory. Existing
pointer, bounds, definite-initialization, alias and independent-program ownership
checks remain active. The guarded representation additionally enforces actual
memory conflicts and complete writes at execution. A block barrier cannot grant
cross-program communication or initialize an unwritten value.

Multi-CTA participation, barriers inside scalar lane/combiner regions, other
physical operations, and barrier composition with atomic or assembly effects
still refuse. Shared/external participants, asynchronous effects and general CFG
participation require their own concrete contracts. Allocation-state wrappers
currently need the additional protected-source integration described below;
the present source-captured standalone qualification does not claim that route.

## Real source example

`t2m/examples/barrier_kernels.py` increments each program's matrix tile and reads
its transpose through the original mutable allocation:

```python
for step in range(steps):
    before = tl.load(state + pointer, row < n, other=0)
    tl.store(state + pointer, before + step.to(tl.int64) + 1, row < n)
    tl.debug_barrier()
    after = tl.load(state + transposed_row * B + transposed_column,
                    transposed_row < n, other=-3)
    tl.store(out + pointer, after, row < n)
    tl.debug_barrier()
```

The first rendezvous makes other lanes' new values visible. The second completes
reads before the next iteration overwrites state. The source defines `out=-7`
for zero trips; masked transposed reads retain their `-3` fill. Both state and
output remain explicit tensor results. The qualified source uses four-by-four
tiles, `n` in `[0,19]` and `steps` in `[0,4]`.

This example also exposed an existing dense-axis recovery gap: shared's flat
uniform row origin can remain assigned to its first source axis after transpose.
The adapter now repartitions exact dense-stride factors when that creates an
out-of-axis trailing origin, checks reconstruction of the complete flat address,
and retains the existing mask/bounds/ownership gates. It introduces no runtime
division or new layout permission.

## Source review and validation

The implementation reuses shared's ordinary pointer/arithmetic/memory passes and
standard SCF regions. Source review covers:

- [Triton's pinned BarrierOp definition](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)
  and its `create_barrier` binding in `python/src/ir.cc`.
- Meta/shared `17f0c234ce474bee334af36e351c62eca7852836` and archived Microsoft
  `050b93ae9c737111f1ca425d793419b32427a971`: the surveyed arithmetic conversion
  patterns/pass legality do not implement this barrier. Their `reducesum_scalar`
  fixtures retain parallel Linalg computation and explicit memory boundaries;
  tree searches found no dedicated barrier fixture. Their existing staged
  conversions remain authoritative for the enclosed computation.
- LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`, `SCF.cpp`'s exactly-once region
  semantics/no-inline rule and `mlir/test/Dialect/GPU/barrier-elimination.mlir`.
  That fixture retains ordinary GPU barriers across conflicting accesses.
  Erasing a raw GPU barrier without recovering complete logical dependencies is
  therefore not the transformation implemented here.

`tests/test_program_barriers.py` checks exact source fields, separate ordered
phases, source scope refusals, source-only auditing before dead-code elimination,
native and reusable Merlin execution, actual inter-program conflict failure and
selected serial target execution. References independently construct complete
integer state and transposed output, including empty/tail/zero-trip cases.
These are software/serial correctness checks; GPU binary equivalence, arbitrary
barrier/effect coverage and performance preservation remain unestablished.

The selected broad tensor/memory/attention run passes 49 checks; the final
barrier/layout selection passes 26, with one subsequent direct abort-observation
check retaining actual SIGABRT, diagnostic bytes and absence of outputs. Counts
overlap. The selected report is `out/program-barriers/summary.json`.

## Remaining allocation-source integration

`convert_allocations` has a separate pre-wrapper protected-source capsule. It
must also expand these phases before its source-version normalizer, preserve
original barrier contracts and add matching independent consumer custody checks.
Standalone source conversion and its package do not bypass that boundary.

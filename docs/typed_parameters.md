# Different-width source parameters on typed owning allocations

`convert_allocations(..., byte_order="little")` now binds ordinary pointer
and external logical descriptor parameters to differently sized typed roots.
`big` selects the other explicit logical byte order. The source-argument
configuration remains v1 for pointers and v2 for descriptors; public owning
tensor shapes/dtypes, scalar arguments and result ordering remain unchanged.

For example, one public f64 tensor can supply both original f32 and i32 source
parameters. The executable wrapper first applies each declared origin using
the owning f64 element size, then casts to the actual source pointer type.
Source pointer advances and descriptor strides subsequently use their original
source element size. An f64 origin of 3 followed by an f32 advance of 2 means
`3 * 8 + 2 * 4` bytes. It never becomes `(3 + 2) * 4`.

The wrapper retains the real source call and complete original argument roster
before native inlining. Existing [typed storage transport](typed_storage.md)
and [byte events](byte_aliases.md) then supply executable standard MLIR: only
roots needing width-changing storage receive byte planes; word events retain
their original widths/tiles, masks, whole-tile snapshots and source order.
No physical address, target pointer representation or GPU mapping is introduced.

## Reused analyses and package differences

The same actual native external-descriptor expansion binds base pointers,
shapes, element strides and padding/rounding flags. There is no second
descriptor evaluator or a replacement kernel. The source-pointer finite-root
analysis and original-byte capsule remain unchanged.

Meta `17f0c234` rejects changed-stride pointer propagation; Ascend `8f0ade66`
preserves the scalar cast boundary and converts complete element offsets.
The implementation retains those source-unit rules through the existing
logical-root byte adapter rather than copying physical ptr/int conversions.
See [the pinned implementation/test references](byte_aliases.md).

`typed-root-byte-view-v1` records the source pointer type, owning pointer type
and explicit byte order. Legacy equal-width and i8-root records retain their
existing meanings. `compatibility.json` states the extension. Original and
normalized source graphs, argument configuration and source-only stage custody
stay portable and distinct from numerical qualification. The public spec is
typed; admission binds actual internal byte storage. Merlin independently
checks that relation and derives per-public-result guard costs using its
existing typed-transport reader. Provenance grants no missing semantics.

Default bindings still require equal byte-aligned widths. Tensor pointers,
address-space changes and sub-byte root representations retain their gates.
Byte-backed indivisible atomics, physical/mutating model views, broader numeric/
effect closure and applicable target materialization remain open.

## Execution and public workflows

The measured parameter matrix covers wider/narrower i16/f16/BF16/i64/f64 roots
in both byte orders, original f32/i32 pointer and descriptor parameters, and
additional 16/64-bit source words including i32/f32 owning roots. Original i8
parameters also update disjoint bytes in public f64 words. Reused libraries retain signed descriptor
strides, partial overlapping views within one program, multi-program disjoint
state, tails, empty allocations and zero trips. Byte-bound and inter-program/
scatter conflicts remain actual compiled refusals. Real pointer and descriptor
model2MLIR captures preserve learned i64 state, every tuple result and original
source capsules through the same Merlin host compiler interface.

The 28 positive parameter cases include 80 additional runtime replays over the
main matrix. Ten descriptor cases additionally retain negative read-column
strides with independent negative write-row and positive write-column strides.
Five actual compiled guard invocations refuse a
missing word byte, an inter-program dependency or colliding scatter lanes.
Both real model captures pass API and CLI binding with all three results.
Forty-nine binding-contract checks, ten existing native alias/byte-root
regressions and eight registry checks also pass. These are finite compositions,
not universal opcode/type/attribute or accelerator coverage.

Visible pointer/descriptor demos verify 38/41 artifacts, their separate Merlin
qualifications 57/60, generic saved-source packages 25/28 and their qualifications
50/53. The complete models at `out/typed-parameter-model` and
`out/typed-descriptor-model` verify 90/96 artifacts. Their extra members are the
original source capsules, including the descriptor normalizer's separate graph.
`out/typed-parameter-coverage/coverage.json` verifies 66 packages and 45 standalone
qualifications, still observing 34/50 pinned TTIR definitions, with 30 represented
in host-qualified captures. Source/operation execution and target coverage remain
separate from this explicit selection.

```bash
python -m t2m shared-demo --kernel typed-parameter-state \
  --memory-profile guarded-buffer --address-policy source-words \
  --out out/typed-parameter-state --python /path/to/native-env/bin/python
```

`typed-descriptor-state` demonstrates original f32/i32 descriptors over BF16
owning storage. Both bundles can use the unchanged `qualify-merlin` command.
Saved-source intake accepts `shared-tensor --allocation-bindings FILE
--byte-order little|big`; `bind-model` accepts the same per-kernel configuration.

`scripts/capture_allocation_word_model.py --typed-root` and
`scripts/capture_descriptor_state_model.py --typed-views` create real i64-parent
captures for those bindings. Their CPU references use original word views and
independent source addressing; their GPU paths remain unqualified.

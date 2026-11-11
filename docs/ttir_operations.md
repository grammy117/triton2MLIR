# Pinned TTIR operation denominator

Triton **v3.7.1**, commit `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`,
defines **50 `tt` operations**. This is a source-definition count, not a claim
that 50 operations, every attribute/type combination or arbitrary kernels are
qualified. TTIR programs also contain standard arithmetic/math/control-flow
operations; those require their own composition and numeric coverage.

The inventory comes from actual `llvm-tblgen --dump-json`, selecting `Op`
records whose dialect is `Triton_Dialect`. No operation names were recovered
from printed kernel IR or README lists. The seven Triton TableGen files use the
release's LLVM pin `1f126a6dea50d185c0781743a667390037ae88bd`; twenty matching
LLVM/MLIR includes were supplied to the existing TableGen executable. No native
stack was built. `TritonOps.td` is 50,703 bytes with SHA256
`19a7c7c7b16e05ab87072fec5b381efad2ff9809ed8aee46d1173f646453b4ef`.

[Pinned definitions](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/include/triton/Dialect/Triton/IR/TritonOps.td)
retain the authoritative types, attributes, regions, effects and verifier rules.
The families below partition that inventory; their summaries describe bounded
implementation paths and open requirements.

| Family | Operations (all have `tt.` prefix) | Current path / remaining gate |
|---|---|---|
| Functions | `func`, `call`, `return` | Native helper inlining, source inventory and declared entry ABI. [ABI v6/outcome packets](effect_abi.md) now preserve void and initially failing entries through Merlin. General value-returning entries/CFGs still need explicit bindings. |
| Logical domain | `get_program_id`, `get_num_programs`, `make_range` | Unmapped logical forall, checked runtime launch bindings and source-width arithmetic. |
| Pointer memory | `addptr`, `load`, `store` | Tensor proofs or guarded source-word memory, including signed/padded/broadcast reads, strided output state, readonly input aliases and explicit writable allocation roots. Mixed-type/byte aliases, physical/mutating model views and volatile/order policies remain separate. |
| Structured descriptors | `advance`, `make_tensor_ptr`, `make_tensor_descriptor`, `descriptor_load`, `descriptor_store` | Official pre-layout normalization for measured local descriptor forms. External descriptor ABI remains open. |
| Casts | `bitcast`, `fp_to_fp` | Native arithmetic conversion where its numeric contract agrees. [Equal-width logical pointer views](pointer_views.md) additionally qualify byte-aligned ordinary loads/stores with original storage types. Other formats/rounding modes, changed-width views remain open; word RMWs and CAS use the separate typed atomic profile. |
| Scalar math | `clampf`, `precise_divf`, `precise_sqrt` | [Independent exact-rational qualification](scalar_math.md) for f16/BF16/f32/f64 typed TTIR, both clamp NaN modes and public f32 builtins through the actual Merlin interface. BF16 keeps portable types and uses an explicit native scalar-math implementation. Broader format/attribute and target closure remain open. |
| Integer high product | `mulhiui` | [Qualified i32/i64 unsigned extended-product expansion](integer_products.md), including Triton's integer Philox. Other widths and RNG policies remain separate. |
| Tile shape | `broadcast`, `cat`, `expand_dims`, `join`, `reshape`, `splat`, `split`, `trans`, `unsplat` | Native arithmetic/shape conversion plus explicit cat/version and pointer-state adapters. Layout/type compositions remain bounded. |
| Tile gather | `gather` | Standard parallel tensor generic/extract with active signed index checks. Distinct from an indexed memory access. |
| Contractions | `dot` | Measured IEEE f16/BF16/FP8 profiles retain original destinations, tiling and K loops. [F32 BF16x3/x6 and TF32/x3](dot_precision.md) reuse official ordered decompositions and explicit logical precision contracts. Broader numeric/composition and target qualification remain open. |
| Scaled contractions | `dot_scaled` | [Logical scaled contractions](scaled_dot.md) qualify FP4/FP8/half operands, K/M/N FP4 packing, optional E8M0/E4M3FN group scales, original C, runtime/ragged loops and the explicit logical-mx-f64-v1 internal precision profile. Reserved FP6 enumerants have no defined source format; broader compositions and target equivalence remain open. |
| Combiners | `reduce`, `reduce.return`, `scan`, `scan.return` | Native/reused tuple and recurrence expansions for measured combiners; arbitrary effectful/exceptional-value rules remain open. |
| Tile histogram | `histogram` | Standard indexed tensor state with exact masks; global atomics and broader element/counter types remain separate. |
| Element maps | `map_elementwise`, `map_elementwise.return` | Pack-one numeric regions retain scalar control. Packed lane grouping and memory/effectful maps remain open. |
| Atomics | `atomic_cas`, `atomic_rmw` | [Private integer profile](atomic_memory.md) retains RMW i32/i64 and CAS i16/i32/i64, original old values, collisions, scope/order, branch/loop recurrences and finite selected buffers through actual Merlin execution. The typed profile adds bitwise f32/f64 exchange, IEEE f64 addition, explicit IEEE/FTZ f32 addition and all nine integer RMWs on equal-width floating storage, retaining the source floating min/max algorithm. Integer-word CAS additionally retains i16/i32/i64 comparisons on f16/f32/f64 storage through standard atomic regions and actual Merlin/model2MLIR execution. Other floating policies, external effects and broader synchronization remain open. |
| Assertions | `assert` | [Upstream scalar/lane-wise CF assertions](source_assertions.md) retain conditions, messages, loop/branch placement and actual failures through native/Merlin execution and real model2MLIR captures in all three memory profiles. [Outcome packets](effect_abi.md) cover void and initially failing kernels. [Length-aware literals](literal_diagnostics.md) retain embedded NUL and UTF-8, including BF16/CAS compositions. Broader diagnostic and target handling remain open. |
| Logical diagnostics | `print` | [Standard scalar/string printing and structured formatting](logical_prints.md) retain numeric values, masks/fills, source regions and logical coordinates through actual Merlin execution. Physical replicas/interleaving and typed-word float hex are explicit software-profile differences. [Mixed tensor/stream references](mixed_effects.md) now retain every result and effect from one call. Physical pointers, FP8/other carriers, effectful combiners and accelerator handling remain open. |
| External effects / implementations | `extern_elementwise`, `elementwise_inline_asm` | [Pure default-library math](external_math.md) qualifies 52 typed declarations across 26 f32/f64 families with exact library/purity/prototype checks. Original source assembly and unknown/impure libraries still require actual implementations; backend observations cannot be inferred from provenance. |
| Indexed descriptor effects | `descriptor_gather`, `descriptor_scatter`, `descriptor_reduce` | [Official logical pointer expansion](indexed_descriptors.md) qualifies loaded row indices, signed bounds/padding, scatter ownership and private reducing stores through Merlin. Transfer i16/i32/f16/BF16/f32, six i32 reduction kinds, signed i64 and unsigned i32 min/max and explicit IEEE f32 addition are measured. External descriptor ABI, broader numeric compositions and applicable target materialization remain open. |
| Physical observations | `ptr_to_int`, `int_to_ptr` | Concrete address-space/identity contract required; a pure logical tensor ABI does not supply it. |

Local survey evidence is retained under `.scratch/ttir-definition-survey/`,
including every source hash, the complete TableGen JSON and the 50 selected
records. Reproduction uses the pinned includes above:

```bash
llvm-tblgen --dump-json include/triton/Dialect/Triton/IR/TritonOps.td \
  -I include > records.json
```

The [package observation registry](package_coverage.md) now joins all 50
definitions to typed source forms and receipt-bound package/native/Merlin
outcomes, retaining unmeasured definitions, unknown TTIR and other dialects.
Source dispositions, a maintained full kernel/refusal corpus and applicable
target evidence remain open. An observation in one composition is not a
universal type/attribute or source-version compatibility claim.

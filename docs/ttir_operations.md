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
| Functions | `func`, `call`, `return` | Native helper inlining, source inventory and declared entry ABI. General entry returns/CFGs still need explicit bindings. |
| Logical domain | `get_program_id`, `get_num_programs`, `make_range` | Unmapped logical forall, checked runtime launch bindings and source-width arithmetic. |
| Pointer memory | `addptr`, `load`, `store` | Tensor proofs or explicit guarded source-word memory. Aliases, non-dense storage, volatile/order policies remain separate. |
| Structured descriptors | `advance`, `make_tensor_ptr`, `make_tensor_descriptor`, `descriptor_load`, `descriptor_store` | Official pre-layout normalization for measured local descriptor forms. External descriptor ABI remains open. |
| Casts | `bitcast`, `fp_to_fp` | Native arithmetic conversion where its numeric contract agrees. All formats, rounding modes and pointer/pointee casts are not qualified. |
| Scalar math | `clampf`, `precise_divf`, `precise_sqrt` | Reused native conversions; per-dtype exceptional-value and implementation policies need complete independent coverage. |
| Integer high product | `mulhiui` | [Qualified i32/i64 unsigned extended-product expansion](integer_products.md), including Triton's integer Philox. Other widths and RNG policies remain separate. |
| Tile shape | `broadcast`, `cat`, `expand_dims`, `join`, `reshape`, `splat`, `split`, `trans`, `unsplat` | Native arithmetic/shape conversion plus explicit cat/version and pointer-state adapters. Layout/type compositions remain bounded. |
| Tile gather | `gather` | Standard parallel tensor generic/extract with active signed index checks. Distinct from an indexed memory access. |
| Contractions | `dot` | IEEE measured profiles retain original destinations, tiling and K loops. Other precision/accumulation policies require separate qualification. |
| Scaled contractions | `dot_scaled` | FP4/FP8 formats, scales, packing and accumulator policies remain open. |
| Combiners | `reduce`, `reduce.return`, `scan`, `scan.return` | Native/reused tuple and recurrence expansions for measured combiners; arbitrary effectful/exceptional-value rules remain open. |
| Tile histogram | `histogram` | Standard indexed tensor state with exact masks; global atomics and broader element/counter types remain separate. |
| Element maps | `map_elementwise`, `map_elementwise.return` | Pack-one numeric regions retain scalar control. Packed lane grouping and memory/effectful maps remain open. |
| Atomics | `atomic_cas`, `atomic_rmw` | [Private integer profile](atomic_memory.md) retains RMW i32/i64 and CAS i16/i32/i64, original old values, collisions, scope/order, branch/loop recurrences and finite selected buffers through actual Merlin execution. Floating atomics and broader synchronization remain open. |
| External effects / implementations | `extern_elementwise`, `elementwise_inline_asm`, `print`, `assert` | Qualified library/intrinsic/diagnostic contracts required. Backend-observing assembly cannot be inferred from provenance. |
| Indexed descriptor effects | `descriptor_gather`, `descriptor_scatter`, `descriptor_reduce` | Descriptor ABI, indexing and write/reduction conflict semantics remain open. |
| Physical observations | `ptr_to_int`, `int_to_ptr` | Concrete address-space/identity contract required; a pure logical tensor ABI does not supply it. |

Local survey evidence is retained under `.scratch/ttir-definition-survey/`,
including every source hash, the complete TableGen JSON and the 50 selected
records. Reproduction uses the pinned includes above:

```bash
llvm-tblgen --dump-json include/triton/Dialect/Triton/IR/TritonOps.td \
  -I include > records.json
```

This survey does not close the maintained-coverage workstream. The next registry
must join these complete source definitions to typed source dispositions,
receipt-bound package/native/Merlin outcomes and explicit refusals, while
retaining unmeasured operations and standard dialects in the report. Unknown
future operations must never inherit support from a family label.

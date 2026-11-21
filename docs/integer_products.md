# Unsigned high products and Triton Philox

`tt.mulhiui` now expands to standard `arith.mului_extended`, selecting the high
result. Scalar words remain scalar; tensor words use a parallel identity-indexed
`linalg.generic` with the exact scalar operation. Source i1/i4/i8/i16/i32/i64 bit patterns
are interpreted as unsigned, including sign-bit and all-one inputs. No physical
instruction, lane mapping, memory space or external library is selected.

The conversion adapts Ascend's `TritonMulhiuiConverter` at
`8f0ade66580245e30ad97093526a4018e4e18fc7` in
`third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp`; its source license
and attribution are retained in `t2m/integer_products.py`. Ascend's
`test_umulhi.py` demonstrates the i32 operation. Our references additionally use
Python arbitrary-width multiplication over unsigned word interpretations,
including high-bit and i64 cases outside that fixture's positive-small-i32 scope.
Meta and archived Microsoft do not supply this conversion in their surveyed
arithmetic pattern sets. LLVM's `ArithOps.td` and `ArithToLLVM/arith-to-llvm.mlir`
define/test the standard extended unsigned product.

`prov.composite_family = "unsigned_high_product"`, source word width and stable
source/emission correspondence retain intent on the standard expansion. These
annotations grant no arithmetic rewrite. The expansion retains the original
source operation and every meaningful bit in executable arithmetic. These are
all widths admitted by `TT_Int` in `TritonTypes.td` at the pinned Triton
`f797708c0626e5f9840ca5b0a98790e2c7cb09ad`. Widths outside that definition
refuse during original native source verification.

The public Python `tl.umulhi` builtin exposes only 32/64-bit words. Narrow
qualification therefore saves a new TTIR kernel with explicit truncation,
the actual narrow `tt.mulhiui`, and zero extension to an i32 public carrier.
Its memory/grid structure comes from a real captured kernel; its numeric
semantics are those of the new, independently native-verified source. This
does not assert equivalence to the original wide Python builtin or add a
public subbyte pointer ABI. Internal word arithmetic remains exact.

The shipped `philox_words` kernel uses Triton's original `tl.randint4x`, with
runtime i64 seed and counter, four i32 output tensors and a runtime logical
launch. Its original ten rounds, twenty unsigned high products, low products,
key updates and XORs reach standard MLIR. It is not replaced by a host RNG or
an opaque library call. Empty/ragged domains, negative seed/counter bit patterns
and a counter crossing 2^32 exercise all four result streams through one
unchanged Merlin library per memory profile.

The independent Python Philox4x32-10 reference is checked against three published
[Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors),
including zero, all-one and nontrivial counter/key states. Exact integer output
comparison is separate from source-structure/provenance checks. This qualifies
the measured stateless integer generator; floating distributions, stateful RNG
schemas, other round counts and accelerator execution remain separate work.

## Neutral truncation default compatibility

Native MLIR prints `arith.trunci` with an explicit
`overflowFlags = #arith.overflow<none>`. The pinned xDSL truncation schema lacks
that property and implements ordinary modular truncation. The checked adapter
represents only that empty flag set by the equivalent absent default, recording
the conversion in provenance. It runs after native computation conversion and
on the native memory continuation before xDSL verification. Original inventories
and exact native stage input/output remain retained independently.

Nonempty `nsw`/`nuw` flags impose poison semantics and refuse with the source
location; they are never erased. See LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`,
`mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`, `Arith_TruncIOp`.

The public tensor/scalar ABI and package schema are unchanged. Both actual
native stages and the functional expansion remain ordinary, inspectable MLIR.

All **35 focused cases** pass with the native and actual Merlin environments
enabled: scalar by-value and tiled i32/i64 words in all three memory profiles,
Philox's four outputs, independent domains/inputs through unchanged libraries,
the published-vector oracle and located nonempty-flag refusals. The existing
shared-tensor/buffer regression also passes all 21 cases. General scalar
irregular memory in tensor-witness profiles is a separate existing rank/proof
boundary; scalar product qualification does not imply that memory coverage.

The additional all-width selection passes **62 cases without skips**: all six
legal widths, scalar/tensor forms, three memory profiles, original invalid-width
refusals, twelve reusable Merlin libraries and twenty-four actual serial images.
Independent unbounded Python multiplication checks every pair of i4 words
(256 pairs) and i8 words (65,536 pairs) for each vector native profile and the
unchanged host libraries; wider words use boundary/sign-bit/all-one patterns.
Empty/ragged/changed runtime shapes retain every result. Serial execution
checks empty and ragged domains as exact words and retains identical computation
object hashes between those domains. This qualifies the high-product operation;
other narrow numeric/storage/effect compositions and accelerator implementations
remain full-goal work. Evidence: `.scratch/high-product-widths-final.log` and
`.scratch/high-product-regression2.log`.

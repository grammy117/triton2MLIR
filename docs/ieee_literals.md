# Exact IEEE literal words through the Merlin interface

Original scalar and dense f16/BF16/f32/f64 NaN constants now retain their sign,
payload and signaling bit. Previously scalar NaNs and noncanonical dense NaNs
were refused because parsing through a Python float can change those words.
Finite literals keep the existing path; no backend arithmetic policy is inferred.

## Representation and source accounting

The typed-token adapter follows xDSL 0.68's
`xdsl/parser/attribute_parser.py`. It captures hexadecimal scalar and dense
element bytes before `FloatAttr` converts them, and retains encoded dense bytes
directly. The upstream Apache/LLVM license is in
[licenses/xdsl-LICENSE.txt](licenses/xdsl-LICENSE.txt). The temporary approximate
scalar attribute exists only inside the closed parser/source-audit phase.

The complete original operation audit and source graph record the exact
native-compatible primitive spelling and `floating_literal_words`: original
type/shape, format, little-endian IEEE words, width and stored NaN count. After
that audit, each literal becomes an ordinary integer `arith.constant` and
`arith.bitcast` of the original scalar/tensor/vector type. Every candidate
validates before any replacement. Locations and source intent survive.

Native canonicalization can create floating constants again. Every later shared,
buffer-continuation and bound-model parser therefore restores exact words before
any executable serialization. A nonconstant NaN attribute requires its own
representation and refuses instead of silently using the approximate float.
Old source-word receipts can retain canonical dense NaNs in their byte-bound
computation input. Replay permits that exact word reification on the complete
retained expected body only. All other operations, properties, attributes,
locations and effects must still match; native byte custody is unchanged.

This reuses the ordinary upstream constant/elementwise lowering. Meta
[`TritonArithToLinalgPass.cpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp)
handles dense splats and tensor arithmetic; its
[`bitcast.mlir`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonArithToLinalg/bitcast.mlir)
fixture retains scalar bitcasts inside parallel Linalg. Neither that native pass
nor the archived Microsoft converter repairs a lossy Python token conversion.
The [reuse map](triton_shared.md) retains the separate source/binary pins.

## Package and consumer boundary

Producer verification and Merlin staging independently check source primitive,
format, type/shape, word count, encoding, exact bytes and NaN count against each
literal transformation. Missing, duplicate, malformed and rehashed inconsistent
declarations refuse. Tensor profiles receive this check before their memory
evidence early return. Derived models use the same consumer boundary. Custody
does not establish source equivalence or authorize omitted arithmetic/effects.

Allocation sources receive a complete original operation audit before any
normalization, including unused private helpers. Sources containing these
literals use existing source capsule v2: exact words expand before pointer/
descriptor normalization and wrapper inlining. The existing protection report
has an optional nonempty `literal_conversions` field; the entry's
`prov.allocation_source_contracts` also retains those original declarations.
The original, protected, bound and normalized graphs remain distinct.

Public tensor/scalar/model ABIs, receipt schemas and compiler artifact rosters
are unchanged. Older strict protection-report readers refuse the additive field.
Generated compatibility metadata records both the standard bitcast representation
and the protected-source extension. BF16 retains its existing raw-word archives
and model witnesses. There is no custom runtime dialect, literal attribute,
calling convention, GPU instruction or opaque helper in the portable payload.

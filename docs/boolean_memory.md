# Logical Boolean memory

Triton 3.7.1 implements Boolean memory using byte slots: an i1 pointer is
cast to i8 before a load/store. Loads become nonzero tests and stores
zero-extend the logical value. The source boundary is recorded at Triton
[`f797708c`](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/triton/language/semantic.py),
`_load_legacy` and `_store_legacy`; upstream `test_core.py::test_masked_load`
covers Boolean loads with masks, tails and both fill values.

The frontend recovers that checked pattern before the pinned triton-shared
pointer passes. Every byte-load observer must be a nonzero test; every stored
byte must be a zero-extension of i1. It emits ordinary logical i1 memory,
retaining addresses, source masks, nonzero fills, inactive destination state
and cache/effect attributes. Constant fills are folded locally, avoiding
changes to unrelated source-word arithmetic. Dynamic fills retain an explicit
nonzero comparison. The exported tensor dtype remains i1.

This removes the source storage convention while retaining executable
logical dataflow. Fused source locations carry the cast, memory event and
conversion IDs into emitted provenance. The trace records
`logical-boolean-byte-memory`; package compatibility metadata explains the
representation change. No GPU placement or byte-address permission follows.

General byte observations, pointer arithmetic after the cast, arbitrary byte
stores, atomics and address-space conversions are outside this normalization
and require a separately checked storage/effect profile. Native tests use
canonical logical Boolean input bytes; noncanonical raw Boolean storage is
not qualified by these tests. This profile makes no raw Boolean byte identity
claim.

`tests/test_boolean_memory.py` checks all-or-nothing refusals, source
correspondence, zero/nonzero fills, runtime empty/tail/full domains, independent
load/store masks and preserved incoming destination values in all three memory
profiles. Real Boolean literal model execution is additionally covered by
`tests/test_literal_models.py` through native and Merlin libraries.

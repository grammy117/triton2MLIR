# Explicit automatic memory representation selection

`shared-tensor --memory-profile auto` and `bind-model --memory-profile auto`
select the first admitted representation using the original TTIR and declared
ABI. The Python equivalent is `convert_shared(..., memory_profile="auto")`.
The default remains `tensor`. Result allocation and model binding modes remain
separate explicit choices.

For ordinary memory with structured addressing, selection tries at most three
conversions, sequentially:

1. Functional tensors, with statically proved source addressing and ownership.
2. Guarded logical memory, with executable bounds/initialization/conflict checks.
3. Guarded source-word memory, retaining original fixed-width integer arithmetic.

The concrete selected profile and all its existing checks remain authoritative.
The witness-buffer profile is still separately selectable; it does not broaden
admission and is not an automatic candidate. Selection uses no sample arrays,
references, kernel names, provenance labels or target performance model.

An explicit source-word address policy, logical byte order or nonordinary
effect profile restricts selection to the compatible source-word candidate.
Numeric policies, atomic semantics, source-assembly providers and source-layout
contracts are forwarded unchanged. Selection cannot infer a missing numeric
policy or supply an implementation for an unknown operation.

## Evidence and failure behavior

Each attempt uses a retained immutable source snapshot and a distinct work
directory. The original source and snapshot are checked between attempts.
`memory-selection.json` retains attempted profiles and located refusal
diagnostics. An admitted result additionally carries the versioned
`t2m.memory-selection.v1` record in its provenance trace; package verification
checks its order, disposition, source digest and actual selected profile.

Only an ordinary typed conversion refusal permits the next candidate. Unexpected
compiler failures are recorded and propagated. Exhausted candidates produce a
located refusal and no converted result. In particular, unimplemented source
assembly cannot disappear during a retry. The selection record is provenance,
not proof of source equivalence, numeric behavior, aliases, ownership, target
correctness or performance.

## Abstraction and package differences

The selected functional tensor/scalar ABI stays unchanged. Tensor candidates
retain the default model2MLIR-like granularity. Guarded candidates contain
standard memory operations and private admission state: this is an explicit
lower-level payload difference, documented in generated compatibility metadata
and the [package contract](package.md). Logical tiles and parallel dataflow
remain present; no warp, thread, physical memory placement or GPU launch ABI is
introduced. Actual consumer ownership and resource checks remain mandatory.

The focused tests exercise real pointwise native execution, genuine unsupported
assembly, bounded selection, policy forwarding, source changes and corrupted
selection metadata. Policy-unit fixtures establish selection behavior only;
they supply no TTIR numerical or target qualification.

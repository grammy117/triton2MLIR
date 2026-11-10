# Uniform conditional read extents

Shared's pinned `MaskState::parseAnd` combines a uniform scalar gate with a
rectangular mask by selecting each dimension's existing extent or zero.
See Meta `17f0c234ce474bee334af36e351c62eca7852836`,
[`MaskAnalysis.cpp`](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/lib/Analysis/MaskAnalysis.cpp)
and `mask_ld_st_scalar_dim.mlir` / `mask_loop_iter_arg_scalar.mlir`.
The tensor adapter previously recognized only the unit comparison form.

Read-only extent recovery now proves the selected true prefix independently,
retains the original select SSA as the actual runtime slice size, and uses
the true prefix solely as an upper access bound. A `ReadSubsetExtent` cannot
prove a complete output partition. Stores keep exact-prefix recovery and
cannot reconstruct the upper bound as the stored extent. Initialization,
ownership, alias and lifetime gates remain unchanged.

Sixteen tail-proof checks include exact SSA retention, static/affine prefixes,
overflow refusal and a conditional mask denied fresh-output coverage credit.
Three real TTIR-to-Merlin conditional-copy cases retain live values or explicit
inactive fills over the same libraries with changing/empty/ragged shapes in
tensor, buffer and guarded-buffer profiles. Both gate values execute through
the normal native and consumer paths; conditions are not inferred from inputs
or provenance. General conditional store partitions remain separate work.

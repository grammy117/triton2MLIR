# Runtime dimensions and logical launches (ABI v5)

`t2m.shared-tensor-abi.v5` declares named tensor dimensions tied to bounded,
nonnegative runtime integer scalars. A rank-one buffer with `shape=("n",)` emits
`tensor<?xf32>`, with an executable `tensor.dim`/`cf.assert` check against scalar
`n` for every incoming buffer. Fresh results allocate their actual runtime
extent. Static signatures retain their existing ABI versions.

`GridExtent("n", 8)` declares a logical launch extent `ceildiv(n,8)`. The emitted
unmapped forall and source program-count values use that runtime expression.
The maximum count is used only for range proofs. Zero dimensions yield zero
programs and zero-element results. No hardware launch or target mapping is chosen.

Complete-output proofs require the mask limit to equal the actual named result
dimension. Variable launches must cover that same dimension: their ceil-divisor
may be at most the source tile stride. Surplus masked programs are valid;
potential holes refuse. Ordinary per-program ownership, initialization, integer
width and source memory checks still apply. Bounds over a maximum capacity alone
cannot prove this contract.

Both tensor and mixed-memory conversion preserve native runtime allocation and
launch behavior. Native tests invoke the same serialized module with sizes
0, 1, 7, 8, 9 and 13; the payload observes the real program count. The JIT caller
uses the explicit `t2m.runtime-shapes.v1` allocation contract derived from the ABI,
checks tensor/scalar consistency and budgets output bytes before allocating.
It does not guess an output size from example data.

This first dimension profile admits dense rank-one dynamic buffers. General
multidimensional runtime strides, arbitrary shape expressions and aliasing need
separate proofs. Merlin compiler-only staging and reusable-library shape binding
are the next consumer integration gate; native JIT success alone does not close it.

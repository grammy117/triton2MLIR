# Runtime and zero-trip loop ownership in Merlin

Merlin now proves fresh returned generations for serial loops with runtime
lower/upper bounds, including zero iterations and changing result extents. The
portable tensor loop and source computation remain unchanged. This is a consumer
capability shared by ordinary lowering, execution-owned host programs and the
selected generated-C/serial-target interfaces.

The checked source fixture has this Python-equivalent behavior:

```python
state = x
for _ in range(steps):
    state = state[:max(0, len(state) - 1)].clone()
return state
```

This example describes standard-MLIR source exercised through Merlin; it adds
no claim about exporting this Python loop through FX. A second source carries
i32 and f64 buffers with independent sizes and shrink rates, retaining exact
signaling/quiet NaN, signed-zero and subnormal words.

## Executable ownership and reuse boundary

Each changed carried slot must yield its own direct body allocation. Previous
owners release exactly once under the original false-entry/true-backedge flag,
after all synchronous uses. Initial roots are actual local allocations with
closed uses. Tuple promotions retain every result coordinate and separate owner.
Runtime/zero-trip bounds are allowed with a known positive constant step.

After the loop, a true flag transfers the final generation. A false flag takes
an actual complete independent copy of the zero-trip seed. `bufferization.clone`
spelling alone supplies no freshness: the native normalizer materializes the
official ranked allocation/dimension/copy implementation in its original branch
before simplifying deallocation. No shape predictor, sampled value or provenance
tag grants ownership or replaces the loop.

The upstream liveness pass can otherwise follow a conservative alias union and
move a seed free after `return`. A typed identity view protects only seed frees
covered by the proof. The independent checker follows that view and rechecks
actual lifetimes afterward; ordinary optimization of other allocations and all
allocation-product/count/byte gates remain active. Existing tiling, arithmetic,
recurrences and source locations retain their semantics.

## Evidence and remaining scope

Native/saved programs reuse compiled products across input shapes and trip counts.
The production C driver changes trip-count input bytes without rebuilding its
model. Actual target images retain zero-trip, empty and resized results; negative
counts still fail through their source assertion before result publication.
Adversarial checks reject missing/wrong copies, false flags, early/duplicate
releases, hidden calls and escaped initial pointers.

The real three-kernel compaction package and online-attention regressions continue
through their existing interfaces. These are separate scopes: the new resizing
fixtures establish consumer ownership, while the TTIR regressions retain their
actual source/package/reference qualification. Evidence is retained in
`out/runtime-loop-ownership/summary.json`; historical opcode counts are unchanged.

The source-frozen Merlin selection passes 187 checks. Reopening its retained
products executes twelve saved-library calls without compilation and checks four
hosted-C wire observations, six successful target images with eight complete
results, and one actual source-guard failure. The two-buffer native tests also
cover independent i32/f64 extents and nonfinite storage words. The separate
twenty-check real-TTIR model/driver/attention regression passes; a fresh complete
compaction target package additionally reopens all seven results against the
final compiler sources. These selections overlap and are not an opcode count.

Lowering products, saved programs and selected frontend fingerprints additionally
pin both lifetime and promotion implementations. Public MLIR, capture/package/result
schemas and original model archives are unchanged. Branch-selected/permuted
owners, arbitrary while/nested ownership, dynamic steps, retained aliases and
other final consumers still need further proofs. Physical views, broader effects,
numeric policies and accelerators retain their independent requirements.

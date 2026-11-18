# Shared mutable views through the Merlin interface

Actual Torch-functionalized Triton calls may now retain several mutable views
of one complete base. The compiler preserves shared storage inside each call;
independent view copies and later scatters would change overlapping updates.
This extends [whole-base mutation](functional_mutation.md) while retaining
explicit functional state results and borrowed readonly public inputs.

## Representation and provenance

Model2MLIR selects `m2m.functional-kernel-storage.v1` from the actual
`auto_functionalized_v2` arguments. Readonly schema operands retain their order.
Each cloned base appears once as `@base<i>`; ordinary returns precede one updated
result per base. `prov.kernel_storage_bases` and `prov.kernel_storage_views`
retain complete root, size, stride and offset relationships. All coordinates
are actual logical i64 SSA, with executable index/size/slice checks. The initial
profile requires proved dense zero-origin bases; it does not invent hidden
storage or reinterpret a non-dense root as dense.

Frontend source snapshots reuse Torch 2.10's actual SliceViewInfo,
AsStridedViewInfo and AliasViewInfo regeneration. Exact original schemas,
owners and observed geometry account for vanished views. Introduced dimension
arithmetic must have a closed real dependency tree matching its structured
symbolic value. Changed arithmetic or offsets cannot be accounted for by an
unchanged symbolic/provenance tag. Unavailable subclass storage stays unobserved.
This is source/coordinate custody, not a custom writer's numerical proof.

Binding explicitly selects `t2m.functional-kernel-binding.v2` within binding v4.
`KernelBinding.mutation_arguments` (also accepted by `bind-model` JSON) binds
each mutable Torch argument to its original TTIR pointer ordinal. Producer and
independent Merlin readers separately check complete readwrite base/result ties,
actual coordinate scalar formals, pointer allocations and offsets. The same base
must implement every aliased argument. Original full SSA, guards, source locations,
memory evidence and native allocation/lifetime checks remain mandatory.

## Real program and granularity

`scripts/capture_mutating_view_model.py` captures two calls to a real Triton
kernel with a learned i64 bias and shared view state:

```python
left, right = x[1:-1], x[2:]       # overlapping views of x
before = x.clone()
a, b = update(left, right, bias)
middle = x.clone()
c, d = update(left, right, bias)
return before, a, b, middle, c, d, x, bias
```

Each original kernel observes right values, observes/updates left values, then
updates right values. The final right update must see earlier left writes through
the shared base. The source bias load remains outside the original loops and is
reused. No dense substitute or opaque state operation replaces these phases.

The other forms use interleaved `x[::2]` / `x[1::2]`, identical `x[1:-1]` views,
and `x[:, 1]` / `x[:, -1]` matrix columns. Vector lengths retain their actual
`[4, 31]` domain; matrix rows/columns retain `[2, 7]` / `[2, 9]`. Complete models
return nine results, including the leading explicit state port, original/middle
snapshots, both calls' observations, final state and the unchanged parameter.
Registered PyTorch CPU implementations supply independent initial references.
No Triton-CPU backend or GPU execution supplies these results.

Automatic source-only memory selection tries the tensor representation first.
Its current scalar gather case refuses with `only all-unit, zero-index
gather/scatter is qualified`. Guarded structured-address memory is admitted next;
the source-word fallback is unnecessary. Standard memory operations and private
executable admission state are an explicit granularity difference from pure
model2MLIR tensor payloads. Original loops, arithmetic, ordering and reuse remain.
Stronger tensor-gather raising may improve abstraction without changing the
source algorithm; annotations alone cannot remove its checks.

The scalar bias uses the existing count-checked runtime dense adapter from the
public rank-zero value to the kernel's one-element allocation. Source public
types, parent archives, receipt v2 and Merlin signature v6 retain their meanings.
Generated compatibility explicitly names the storage/pointer extension. Older
consumers are unqualified for it. Public physical aliases and automatic caller
copyback remain separate invocation contracts.

## Qualification and remaining scope

The four complete models qualify through Merlin host and actual serial Spike
images, retaining all 36 initial, 108 changed-invocation and 36 target results as
exact words. Twelve changed-shape/parameter invocations reuse unchanged host
libraries. The final 16-check selection passes; four additional calls to these
same libraries actually abort for invalid original public shape domains without
publishing outputs. Twenty rehashed missing/downgraded profiles,
swapped/missing pointer mappings and removed coordinate guards refuse in both
readers. `out/mutating-views/summary.json` retains the final selected evidence.

Model2MLIR's 117-check capture/kernel/trace/result/session/word/quality selection
passes, including forged geometry/ownership/arithmetic and nondense-base refusal.
Merlin's independent storage/state/correspondence/view selection passes 51.
The separate 23-check earlier whole-base/model-binding selection passes. These
counts overlap other selections and do not expand the historical opcode inventory.

Reproduction uses the existing environments documented in the README:

```bash
python -m pytest -q tests/test_mutating_view_models.py \
  --basetemp=.scratch/pytest-mutating-views
```

General base layouts/hidden storage, optional/list state, metadata-changing
buffers, broader symbolic-coordinate/effect/numeric/source-version combinations,
public physical aliases/copyback and applicable accelerator execution/performance
remain full-goal work. The finite examples establish no GPU-binary or performance
equivalence and no arbitrary-kernel completion percentage.

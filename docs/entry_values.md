# Storage-backed TTIR entry arguments and results

`convert_allocations` extends the existing executable source wrapper to numeric
value arguments and returns. It retains the real original function and private
helpers, loads typed arguments, calls the function, then stores every returned
value to its declared result port. Upstream inlines the call before memory
analysis. Standard MLIR still contains the actual computation and control;
there is no opaque source interpreter or low-precision native by-value call.

This is useful when extracting a returning helper as a standalone kernel, or
importing a TTIR entry with f16/BF16/FP8/i4 values. Ordinary Triton kernel
entries generally return void and use pointer outputs; that existing route
continues to work. A value-port adapter adds explicit caller information to
the same tensor package and public Merlin `StagedProgram` interface.

## Existing API with explicit value ports

```python
from t2m.allocation_bindings import (
    PointerArgument, ResultBinding, ValueArgument, convert_allocations,
)
from t2m.shared_tensor import TensorABI, TensorBuffer

# Original: @return_value(%x: !tt.ptr<i32>) -> i32
abi = TensorABI("bound_return", (
    TensorBuffer("x", (1,), dtype="i32"),
    TensorBuffer("result", (1,), "write", "i32"),
))
result = convert_allocations(
    "source.ttir", abi, (PointerArgument("x"),), "out/return-conversion",
    source_entry="return_value", result_bindings=(ResultBinding("result"),),
    memory_profile="guarded-buffer", address_policy="source-words",
)

# A by-value source formal uses ValueArgument("alpha") and a readable
# TensorBuffer("alpha", (1,), dtype="f16"). Other pointer/scalar formals
# retain PointerArgument/ScalarArgument in original source argument order.
```

`ValueArgument(allocation, offset=0, per_program=False)` loads the source
argument's exact scalar or static tensor type from the named readable root.
`ResultBinding(allocation, offset=0, per_program=True)` stores one source result
to a writable root. Supply results in the complete source result order.
The root dtype must match the source element type; this is value transport,
without a numeric conversion or implicit bit reinterpretation. All fourteen
admitted numeric storage types use their existing archive/descriptor carriers.

Offsets are signed i64 element constants or names of declared i32/i64 runtime
scalars. Existing active bounds and initialization checks still execute.
With `per_program=True`, a port starts at
`offset + linear_program_id * source_value_element_count`. Linear coordinates
are `x + nx * (y + ny * z)`, using the actual logical grid counts, with x
varying fastest. This is an iteration coordinate, without warp/thread/core
mapping. Dynamic logical grids, runtime origins and empty launches retain
their existing contracts. Program/tile domains must fit signed i64 addressing.

Tensor arguments load a contiguous flattened tile, then reshape in order to
the original type. Results reshape back only at the storage boundary. The
source tensor body, transpose, reuse, branches and return values remain
executable and annotated. Source tiles are static TTIR values; owning buffers
and logical grids can have declared runtime dimensions.

## Saved configuration and compatibility

The CLI's existing `--allocation-bindings` option accepts
`t2m.source-allocation-arguments.v3`:

```json
{
  "schema": "t2m.source-allocation-arguments.v3",
  "source_entry": "return_value",
  "arguments": [{"kind": "pointer", "allocation": "x"}],
  "results": [{"allocation": "result", "offset": 0, "per_program": true}]
}
```

Value inputs use `{"kind":"value","allocation":"alpha"}` with optional
`offset` and `per_program`. Pointer/scalar/constant/descriptor records retain
their existing meaning; descriptor normalization also composes with v3.
`entry_arguments_from_dict` returns `(entry, arguments, results)` for all three
configuration versions. Pass its third result as `result_bindings=`. The legacy
two-result `arguments_from_dict` refuses a nonempty result roster, so it cannot
silently drop return values. Older v1/v2 configurations remain valid.

The descriptive binding report remains
`t2m.source-allocation-bindings.v1`, with `kind=value`, actual `source_type`
fields and optional `result_bindings`. The source capsule retains its existing
artifact roster and original/bound source bytes; its argument JSON has the new
explicit v3 profile. Producer and independent Merlin intake validate the roster;
Merlin also checks actual source types, storage/effects and every result.
These checks grant no source-equivalence, numeric, alias or target permission.
The actual bound program and guards still execute. Older capsule readers
refuse the new configuration. Generated compatibility metadata states the
difference; public tensor/scalar ABI and receipt schemas are unchanged.

## Qualification and practical boundaries

`tests/test_entry_values.py` qualifies nineteen public programs with 87 public
library invocations. It checks scalar inputs and multiple results for all
fourteen numeric storage types. It covers every storage word through sixteen
bits, wider-word edge patterns, private helper calls, source branches, exact
original/result bits and changed/single/empty reuse. I4 and E4M3FNUZ use the
execution-owned result path. FP8/i4 tensor programs additionally check static
tiles, differently shaped results and actual x/y/z coordinate tags over dynamic
three-dimensional grids. Original returning-i32 and by-value-f16 witnesses also
run publicly, including saved CLI result bindings. Each numeric result uses an
independent word/formula reference, separately from native compilation.

`tests/test_entry_value_contract.py` passes 35 checks for invalid types, rosters, origins,
storage effects and result aliasing. Independent consumer checks reject altered
source/result types even when declaration fields are changed together.

Result ports require distinct allocations separate from original pointer
state. This prevents a returned value from overwriting another result or an
observable state update. Supply fresh result roots when extracting such a
helper. Value input roots must also stay separate from result ports and
mutable source pointer state. This keeps a captured value stable across the
logical grid; a later program cannot reload an earlier program's replacement.
Use readonly value roots and fresh result ports. Downstream bufferization may
recover physical reuse under its ordinary liveness/alias checks.
Scalars are exposed through one-cell or
per-program tensor ports; this does not add rank-zero tensor buffers or a
low-precision native runtime-scalar ABI.

Raw pointer and descriptor objects need their actual logical root/offset/field
contract. They cannot be serialized as invented device addresses or descriptor
bytes by a numeric value port. Internal pointer/descriptor helpers still use
existing inlining and root/descriptor binding. For a standalone value interface,
supply ordinary TTIR that exposes defined logical offsets/fields and owning
roots. Recursive/uninlineable functions, additional public entries and unknown
effects retain their existing explicit diagnostics.

Boundary loads/stores/reshapes can add storage or instructions. Inlining and
downstream optimization may remove that cost; no GPU throughput or scheduling
equivalence is claimed. No device-specific register ABI is inferred.

The source implementation/test reference is Meta/shared's function signature
conversion and [`call` fixture](https://github.com/facebookincubator/triton-shared/blob/17f0c234ce474bee334af36e351c62eca7852836/test/Conversion/TritonPtrToMemref/call.mlir)
at `17f0c234ce474bee334af36e351c62eca7852836`. That existing call/result support
is reused before the functional tensor ABI and provenance work.

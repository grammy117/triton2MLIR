"""Snapshot table captures survive mutation, recurrence and every memory route."""

import json
import os

import numpy as np
import pytest
from test_scalar_tile_reads import TOOLS

from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(
    not TOOLS, reason="existing shared/native/Merlin tools required"
)


def source(kind, dtype, axis):
    shape = "64x1" if axis == 0 else "1x64"
    scalar = "i64" if dtype == "i64" else "i32"
    output = scalar if kind == "reduce" else f"tensor<8x{scalar}>"
    properties = "axis=0:i32" + (",reverse=false" if kind == "scan" else "")
    projected = (
        f"%result=tt.splat %computed:{scalar}->tensor<8x{scalar}>"
        if kind == "reduce"
        else f"%result=arith.addi %computed,%zeros:tensor<8x{scalar}>"
    )
    return f"""module {{tt.func public @table_state(%x:!tt.ptr<{scalar}>,%table:!tt.ptr<{scalar}>,%out:!tt.ptr<{scalar}>,%n:i32,%steps:i32,%delta:{scalar}) {{
      %r=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %nv=tt.splat %n:i32->tensor<8xi32>
      %active=arith.cmpi slt,%r,%nv:tensor<8xi32>
      %zeros=arith.constant dense<0>:tensor<8x{scalar}>
      %xv=tt.splat %x:!tt.ptr<{scalar}>->tensor<8x!tt.ptr<{scalar}>>
      %xp=tt.addptr %xv,%r:tensor<8x!tt.ptr<{scalar}>>,tensor<8xi32>
      %input=tt.load %xp,%active,%zeros:tensor<8x!tt.ptr<{scalar}>>
      %t=tt.make_range {{start=0:i32,end=64:i32}}:tensor<64xi32>
      %tv=tt.splat %table:!tt.ptr<{scalar}>->tensor<64x!tt.ptr<{scalar}>>
      %tp=tt.addptr %tv,%t:tensor<64x!tt.ptr<{scalar}>>,tensor<64xi32>
      %snapshot=tt.load %tp:tensor<64x!tt.ptr<{scalar}>>
      %matrix=tt.expand_dims %snapshot {{axis={1 - axis}:i32}}:tensor<64x{scalar}>->tensor<{shape}x{scalar}>
      %change=tt.splat %delta:{scalar}->tensor<64x{scalar}>
      %updated=arith.addi %snapshot,%change:tensor<64x{scalar}>
      tt.store %tp,%updated:tensor<64x!tt.ptr<{scalar}>>
      %zero=arith.constant 0:i32
      %one=arith.constant 1:i32
      %state=scf.for %step=%zero to %steps step %one iter_args(%tile=%input)->(tensor<8x{scalar}>):i32 {{
        %computed="tt.{kind}"(%tile) <{{{properties}}}> ({{
        ^bb0(%a:{scalar},%b:{scalar}):
          %seven=arith.constant 7:{scalar}
          %eight=arith.constant 8:{scalar}
          %left=arith.andi %a,%seven:{scalar}
          %right=arith.andi %b,%seven:{scalar}
          %row=arith.muli %left,%eight:{scalar}
          %index=arith.addi %row,%right:{scalar}
          %indices=tt.splat %index:{scalar}->tensor<1x1x{scalar}>
          %selected=tt.gather %matrix[%indices] {{axis={axis}:i32}}:(tensor<{shape}x{scalar}>,tensor<1x1x{scalar}>)->tensor<1x1x{scalar}>
          %scalar=tt.unsplat %selected:tensor<1x1x{scalar}>
          "tt.{kind}.return"(%scalar):({scalar})->()
        }}):(tensor<8x{scalar}>)->{output}
        {projected}
        scf.yield %result:tensor<8x{scalar}>
      }}
      %ov=tt.splat %out:!tt.ptr<{scalar}>->tensor<8x!tt.ptr<{scalar}>>
      %op=tt.addptr %ov,%r:tensor<8x!tt.ptr<{scalar}>>,tensor<8xi32>
      tt.store %op,%state,%active:tensor<8x!tt.ptr<{scalar}>>
      tt.return
    }} }}"""


def arguments(dtype, kind, n=7, steps=2, delta=2, group="cyclic"):
    bits = 64 if dtype == "i64" else 32
    storage = np.dtype(f"int{bits}")
    base = (1 << (bits - 1)) - 8 if group == "cyclic" else -(1 << (bits - 2))
    permutation = [0, 3, 6, 1, 4, 7, 2, 5] if group == "permuted" else list(range(8))
    inverse = [permutation.index(index) for index in range(8)]

    def cell(a, b):
        return base + (
            a ^ b if group == "xor" else permutation[(inverse[a] + inverse[b]) % 8]
        )

    table = np.asarray([cell(a, b) for a in range(8) for b in range(8)], storage)
    x = np.asarray([base + index for index in (3, 1, 5, 7, 2, 4, 6, 0)][:n], storage)
    tile = [int(value) for value in x] + [0] * (8 - n)
    for _ in range(steps):
        values = [tile[0]]
        for current in tile[1:]:
            values.append(int(table[(values[-1] & 7) * 8 + (current & 7)]))
        tile = [values[-1]] * 8 if kind == "reduce" else values

    def signed(value):
        word = value & ((1 << bits) - 1)
        return word - (1 << bits) if word >> (bits - 1) else word

    updated = np.asarray([signed(int(value) + delta) for value in table], storage)
    return [
        x,
        table,
        np.asarray(n, np.int32),
        np.asarray(steps, np.int32),
        np.asarray(delta, storage),
    ], [updated, np.asarray(tile[:n], storage)]


@pytest.mark.parametrize("kind", ["reduce", "scan"])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_snapshot_table_state_reaches_reusable_merlin_and_target(
    tmp_path, kind, dtype, axis, profile
):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    path = tmp_path / "source.ttir"
    path.write_text(source(kind, dtype, axis))
    abi = TensorABI(
        "table_state",
        (
            TensorBuffer("x", ("n",), dtype=dtype),
            TensorBuffer("table", (64,), "readwrite", dtype),
            TensorBuffer("out", ("n",), "write", dtype),
        ),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 8)),
            RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
            RuntimeScalar("delta", dtype, 5, bounds=(-3, 3)),
        ),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "conversion",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    assert "tensor.extract" in result.mlir_text and "scf.for" in result.mlir_text
    assert any(
        item["kind"] == "source-scalar-tile-read"
        for item in result.trace["transformations"]
    )
    args, outputs = arguments(dtype, kind)
    policy = expected("", [{"kind": "words"}] * 2)
    captured, compiled = qualify(result, tmp_path, args, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, (n, steps, delta, group) in enumerate(
        [
            (0, 3, 2, "xor"),
            (1, 0, -3, "cyclic"),
            (8, 1, 3, "permuted"),
            (7, 3, -2, "xor"),
        ]
    ):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        args, outputs = arguments(dtype, kind, n, steps, delta, group)
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)}
        )
        np.savez(
            caller / "reference.npz",
            **{f"out{i}": value for i, value in enumerate(outputs)},
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        out = tmp_path / f"invoke-{index}"
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(out)
        assert _record(out / "source/host/model.so") == library
    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    target.qualify(captured, tmp_path / "target", **options)
    target.verify(tmp_path / "target", **options)


@pytest.mark.parametrize("kind", ["reduce", "scan"])
def test_lookup_bounds_abort_the_existing_merlin_library_inside_the_combiner(
    tmp_path, kind
):
    from test_mixed_effects import expected, qualify

    from t2m.consumer import invoke_merlin_host

    program = source(kind, "i64", 1)
    assert program.count("%delta:i64)") == 1
    assert program.count("%index=arith.addi %row,%right:i64") == 1
    program = program.replace("%delta:i64)", "%delta:i64,%bias:i64)")
    program = program.replace(
        "%index=arith.addi %row,%right:i64",
        "%unbiased=arith.addi %row,%right:i64\n%index=arith.addi %unbiased,%bias:i64",
    )
    path = tmp_path / "source.ttir"
    path.write_text(program)
    abi = TensorABI(
        "table_state",
        (
            TensorBuffer("x", ("n",), dtype="i64"),
            TensorBuffer("table", (64,), "readwrite", "i64"),
            TensorBuffer("out", ("n",), "write", "i64"),
        ),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 8)),
            RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
            RuntimeScalar("delta", "i64", 5, bounds=(-3, 3)),
            RuntimeScalar("bias", "i64", 6, bounds=(0, 64)),
        ),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "conversion",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
    )
    args, outputs = arguments("i64", kind)
    args.append(np.asarray(0, np.int64))
    _, compiled = qualify(
        result, tmp_path, args, expected("", [{"kind": "words"}] * 2), outputs
    )
    args[-1] = np.asarray(64, np.int64)
    inputs = tmp_path / "bad-inputs.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(args)})
    with pytest.raises(ValueError, match="TTIR tile gather index outside source axis"):
        invoke_merlin_host(
            compiled,
            inputs,
            tmp_path / "bad-invocation",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )

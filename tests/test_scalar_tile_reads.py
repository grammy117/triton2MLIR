"""Captured tensor tables retain immutable dataflow and actual index guards."""

import os

import numpy as np
import pytest

from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(
    os.environ.get(key)
    for key in (
        "TRITON_SHARED_OPT",
        "T2M_NATIVE_PYTHON",
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
    )
)


def source(kind="reduce", *, loop=False):
    output_type = "i64" if kind == "reduce" else "tensor<8xi64>"
    properties = "axis=0:i32" + (",reverse=false" if kind == "scan" else "")
    splat = (
        "%result=tt.splat %computed:i64->tensor<8xi64>"
        if kind == "reduce"
        else "%result=arith.addi %computed,%zeros:tensor<8xi64>"
    )
    signature = ",%steps:i32" if loop else ""
    start = (
        """%c0=arith.constant 0:i32
      %c1=arith.constant 1:i32
      %state=scf.for %step=%c0 to %steps step %c1 iter_args(%tile=%input)->(tensor<8xi64>):i32 {"""
        if loop
        else ""
    )
    end = "scf.yield %result:tensor<8xi64>\n}" if loop else ""
    return f"""module {{tt.func public @lookup(%x:!tt.ptr<i64>,%table:!tt.ptr<i64>,%out:!tt.ptr<i64>{signature}) {{
      %r=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %xv=tt.splat %x:!tt.ptr<i64>->tensor<8x!tt.ptr<i64>>
      %xp=tt.addptr %xv,%r:tensor<8x!tt.ptr<i64>>,tensor<8xi32>
      %input=tt.load %xp:tensor<8x!tt.ptr<i64>>
      %t=tt.make_range {{start=0:i32,end=64:i32}}:tensor<64xi32>
      %tv=tt.splat %table:!tt.ptr<i64>->tensor<64x!tt.ptr<i64>>
      %tp=tt.addptr %tv,%t:tensor<64x!tt.ptr<i64>>,tensor<64xi32>
      %snapshot=tt.load %tp:tensor<64x!tt.ptr<i64>>
      %zeros=arith.constant dense<0>:tensor<8xi64>
      {start}
      %computed="tt.{kind}"(%{"tile" if loop else "input"}) <{{{properties}}}> ({{
      ^bb0(%a:i64,%b:i64):
        %seven=arith.constant 7:i64
        %eight=arith.constant 8:i64
        %left=arith.andi %a,%seven:i64
        %right=arith.andi %b,%seven:i64
        %row=arith.muli %left,%eight:i64
        %index=arith.addi %row,%right:i64
        %i32=arith.trunci %index:i64 to i32
        %indices=tt.splat %i32:i32->tensor<1xi32>
        %selected=tt.gather %snapshot[%indices] {{axis=0:i32}}:(tensor<64xi64>,tensor<1xi32>)->tensor<1xi64>
        %scalar=tt.unsplat %selected:tensor<1xi64>
        "tt.{kind}.return"(%scalar):(i64)->()
      }}):(tensor<8xi64>)->{output_type}
      {splat}
      {end}
      %ov=tt.splat %out:!tt.ptr<i64>->tensor<8x!tt.ptr<i64>>
      %op=tt.addptr %ov,%r:tensor<8x!tt.ptr<i64>>,tensor<8xi32>
      tt.store %op,%{"state" if loop else "result"}:tensor<8x!tt.ptr<i64>>
      tt.return
    }} }}"""


@pytest.mark.skipif(not TOOLS, reason="existing native/shared/Merlin tools required")
@pytest.mark.parametrize("kind", ["reduce", "scan"])
@pytest.mark.parametrize("loop", [False, True])
def test_original_snapshot_table_combiner_reaches_native_and_merlin(
    tmp_path, kind, loop
):
    from test_mixed_effects import expected, qualify

    path = tmp_path / "source.ttir"
    path.write_text(source(kind, loop=loop))
    from t2m.scalar_abi import RuntimeScalar

    result = convert_shared(
        path,
        TensorABI(
            "lookup",
            (
                TensorBuffer("x", (8,), dtype="i64"),
                TensorBuffer("table", (64,), dtype="i64"),
                TensorBuffer("out", (8,), "write", "i64"),
            ),
            runtime_scalars=(RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),)
            if loop
            else (),
        ),
        tmp_path / "conversion",
        optimizer=os.environ["TRITON_SHARED_OPT"],
    )
    base = 1 << 60
    x = np.asarray([base + n for n in (3, 1, 5, 7, 2, 4, 6, 0)], np.int64)
    table = np.asarray(
        [base + (a + b) % 8 for a in range(8) for b in range(8)], np.int64
    )
    outputs = [x.copy()]
    for _ in range(2 if loop else 1):
        values = [int(outputs[0][0])]
        for current in outputs[0][1:]:
            values.append(int(table[(values[-1] & 7) * 8 + (int(current) & 7)]))
        outputs = [
            np.full(8, values[-1], np.int64)
            if kind == "reduce"
            else np.asarray(values, np.int64)
        ]
    qualify(
        result,
        tmp_path,
        [x, table, *([np.asarray(2, np.int32)] if loop else [])],
        expected("", [{"kind": "words"}]),
        outputs,
    )

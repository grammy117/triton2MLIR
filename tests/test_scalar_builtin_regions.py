"""Original nested builtins retain computation through native/Merlin/Spike."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

REQUIRED = all(
    os.environ.get(key)
    for key in (
        "TRITON_SHARED_OPT",
        "T2M_NATIVE_PYTHON",
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin tools required"
)


def source(kind, dtype):
    bits = dtype.startswith("i")
    signature = f"%x:!tt.ptr<{dtype}>,%y:!tt.ptr<{dtype}>,%out:!tt.ptr<{dtype}>,%n:i32,%steps:i32"
    prefix = f"""module {{tt.func public @nested({signature}) {{
      %pid=tt.get_program_id x:i32
      %eight=arith.constant 8:i32
      %zero=arith.constant 0:i32
      %one=arith.constant 1:i32
      %start=arith.muli %pid,%eight:i32
      %startv=tt.splat %start:i32->tensor<8xi32>
      %range=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %lane=arith.addi %startv,%range:tensor<8xi32>
      %nv=tt.splat %n:i32->tensor<8xi32>
      %mask=arith.cmpi slt,%lane,%nv:tensor<8xi32>
      %zeros=arith.constant dense<{0 if bits else "0.0"}>:tensor<8x{dtype}>
    """
    for name in ("x", "y", "out"):
        prefix += f"""%{name}v=tt.splat %{name}:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
          %{name}p=tt.addptr %{name}v,%lane:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
        """
    prefix += f"""%input=tt.load %xp,%mask,%zeros:tensor<8x!tt.ptr<{dtype}>>
      %other=tt.load %yp,%mask,%zeros:tensor<8x!tt.ptr<{dtype}>>
      %state=scf.for %step=%zero to %steps step %one iter_args(%carry=%input)->(tensor<8x{dtype}>):i32 {{
    """
    if bits:
        body = f"""%high=tt.mulhiui %a,%b:{dtype}
          %value=arith.xori %high,%a:{dtype}"""
    else:
        symbol = "__nv_fabsf" if dtype == "f32" else "__nv_fabs"
        body = (
            f"%selected=arith.maximumf %a,%b:{dtype}\n"
            if kind != "map"
            else f"%selected=arith.addf %a,%b:{dtype}\n"
        )
        body += f"""%lo=arith.constant 0.0:{dtype}
          %hi=arith.constant 8.0:{dtype}
          %unit=arith.constant 1.0:{dtype}
          %clamped=tt.clampf %selected,%lo,%hi,propagateNan=all:{dtype}
          %abs=tt.extern_elementwise %clamped {{pure=true,libname="",libpath="",symbol="{symbol}"}}:({dtype})->{dtype}
          %root=tt.precise_sqrt %{("abs" if kind == "map" else "unit")}:{dtype}
          %divided=tt.precise_divf %abs,%root:{dtype}
          %wide=tt.fp_to_fp %divided:{dtype}->f64
          %half=tt.fp_to_fp %wide,rounding=rtz:f64->f16
          %bf=tt.fp_to_fp %half,rounding=rtne:f16->bf16
          %value=tt.fp_to_fp %bf:bf16->{dtype}
        """
    if kind == "map":
        operation = f"""%next="tt.map_elementwise"(%carry,%other) <{{pack=1:i32}}> ({{
          ^bb0(%a:{dtype},%b:{dtype}):
          {body}
          "tt.map_elementwise.return"(%value):({dtype})->()
        }}):(tensor<8x{dtype}>,tensor<8x{dtype}>)->tensor<8x{dtype}>
        """
    else:
        output = dtype if kind == "reduce" else f"tensor<8x{dtype}>"
        attrs = "axis=0:i32" + (",reverse=false" if kind == "scan" else "")
        operation = f"""%computed="tt.{kind}"(%carry) <{{{attrs}}}> ({{
          ^bb0(%a:{dtype},%b:{dtype}):
          {body}
          "tt.{kind}.return"(%value):({dtype})->()
        }}):(tensor<8x{dtype}>)->{output}
        """
        operation += (
            f"%next=tt.splat %computed:{dtype}->tensor<8x{dtype}>"
            if kind == "reduce"
            else f"%next=arith.addf %computed,%zeros:tensor<8x{dtype}>"
        )
    return (
        prefix
        + operation
        + f"""
      scf.yield %next:tensor<8x{dtype}>
      }}
      tt.store %outp,%state,%mask:tensor<8x!tt.ptr<{dtype}>>
      tt.return
    }} }}"""
    )


def reference(x, y, steps, kind):
    import ml_dtypes

    integer = np.issubdtype(x.dtype, np.integer)
    bits = x.dtype.itemsize * 8
    mask = (1 << bits) - 1

    def scalar(a, b):
        if integer:
            high = ((int(a) & mask) * (int(b) & mask)) >> bits
            word = high ^ (int(a) & mask)
            return word - (1 << bits) if word >> (bits - 1) else word
        combined = x.dtype.type(a + b) if kind == "map" else np.maximum(a, b)
        clipped = np.minimum(np.maximum(combined, x.dtype.type(0)), x.dtype.type(8))
        absolute = abs(clipped)
        root = x.dtype.type(np.sqrt(absolute)) if kind == "map" else x.dtype.type(1)
        value = x.dtype.type(absolute / root)
        nearest = np.asarray(value, np.float16)
        if np.isfinite(value) and float(nearest) > float(value):
            nearest = np.asarray(int(nearest.view(np.uint16)) - 1, np.uint16).view(
                np.float16
            )
        return x.dtype.type(np.asarray(nearest, ml_dtypes.bfloat16))

    result = x.copy()
    with np.errstate(all="ignore"):
        for start in range(0, len(x), 8):
            active = min(8, len(x) - start)
            tile, other = np.zeros(8, x.dtype), np.zeros(8, y.dtype)
            tile[:active], other[:active] = (
                x[start : start + active],
                y[start : start + active],
            )
            for _ in range(steps):
                if kind == "map":
                    tile = np.asarray(
                        [scalar(a, b) for a, b in zip(tile, other, strict=True)],
                        x.dtype,
                    )
                else:
                    values = [tile[0]]
                    for current in tile[1:]:
                        values.append(scalar(values[-1], current))
                    tile = (
                        np.full(8, values[-1], x.dtype)
                        if kind == "reduce"
                        else np.asarray(values, x.dtype) + x.dtype.type(0)
                    )
            result[start : start + active] = tile[:active]
    return [result]


@pytest.mark.parametrize(
    "kind,dtype",
    [(kind, dtype) for kind in ("map", "reduce", "scan") for dtype in ("f32", "f64")]
    + [("map", "i32"), ("map", "i64")],
)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_nested_scalar_computation_reaches_reusable_merlin_and_target(
    tmp_path, kind, dtype, profile
):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    path = tmp_path / "source.ttir"
    path.write_text(source(kind, dtype))
    abi = TensorABI(
        "nested",
        tuple(
            TensorBuffer(name, ("n",), "write" if name == "out" else "read", dtype)
            for name in ("x", "y", "out")
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
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
    assert "scf.for" in result.mlir_text
    assert any(
        item["kind"]
        == (
            "ttir-unsigned-high-product"
            if dtype.startswith("i")
            else "source-scalar-builtin"
        )
        for item in result.trace["transformations"]
    )
    dtype = {"f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}[dtype]
    # The independent oracle uses NumPy storage names; source types stay MLIR types.
    numeric = np.dtype(dtype)
    integer = np.issubdtype(numeric, np.integer)

    def inputs(n, steps):
        x = np.resize(
            np.asarray(
                [0, -1, 3, -(1 << (numeric.itemsize * 8 - 1)), 17, 29]
                if integer
                else [-0.0, 0.33333, 2.13, 7.03, -3.0, np.nan, np.inf, 0.00013],
                numeric,
            ),
            n,
        )
        y = np.resize(
            np.asarray(
                [3, -1, 17, 0, 5, 2] if integer else [0.0, 0.1, -2.0, 1.03, 0.05, 0.0],
                numeric,
            ),
            n,
        )
        return [x, y, np.asarray(n, np.int32), np.asarray(steps, np.int32)], reference(
            x, y, steps, kind
        )

    args, goldens = inputs(19, 2)
    policy = expected("", [{"kind": "words" if integer else "ieee"}])
    captured, compiled = qualify(result, tmp_path, args, policy, goldens)
    library = _record(compiled / "host/model.so")
    for n, steps in [(0, 3), (1, 0), (31, 1), (13, 3)]:
        caller = tmp_path / f"caller-{n}"
        caller.mkdir()
        args, goldens = inputs(n, steps)
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)}
        )
        np.savez(caller / "reference.npz", out0=goldens[0])
        (caller / "reference.json").write_text(json.dumps(policy))
        out = tmp_path / f"invoke-{n}"
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

"""Checked scalar mappings compose before maps, reductions and scans lift."""

import os

import numpy as np
import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.external_math import SYMBOLS
from t2m.ir import text
from t2m.scalar_builtins import expand_scalar_builtins
from t2m.shared_tensor import _source_ids, parse_shared


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_existing_external_prototype_has_a_registered_scalar_expansion(symbol):
    target, dtype, arity = SYMBOLS[symbol]
    args = ",".join(f"%x{i}:{dtype}" for i in range(arity))
    values = ",".join(f"%x{i}" for i in range(arity))
    types = ",".join([str(dtype)] * arity)
    module = parse_shared(f'''module {{func.func @f({args})->{dtype} {{
      %out="tt.extern_elementwise"({values}) {{pure=true,libname="",libpath="",symbol="{symbol}"}}:({types})->{dtype} loc("ttir.op.10")
      return %out:{dtype}
    }} }}''')
    records = expand_scalar_builtins(module, _source_ids)
    module.verify()
    assert records[0]["source_ids"] == ["ttir.op.10"]
    assert records[0]["expansion"] == [target]


@pytest.mark.parametrize(
    "kind", ["cast", "clamp", "sqrt", "div", "external", "bitcast"]
)
@pytest.mark.parametrize("bad", ["property", "type"])
def test_every_scalar_candidate_validates_before_any_replacement(kind, bad):
    operation = {
        "cast": '%bad="tt.fp_to_fp"(%x) {rounding=1:i32}: (f32)->f16',
        "clamp": '%bad="tt.clampf"(%x,%x,%x) {propagateNan=65535:i32}:(f32,f32,f32)->f32',
        "sqrt": '%bad="tt.precise_sqrt"(%x):(f32)->f32',
        "div": '%bad="tt.precise_divf"(%x,%x):(f32,f32)->f32',
        "external": '%bad="tt.extern_elementwise"(%x) {pure=true,libname="",libpath="",symbol="__nv_fabsf"}:(f32)->f32',
        "bitcast": '%bad="tt.bitcast"(%x):(f32)->i32',
    }[kind]
    module = parse_shared(f"""module {{func.func @f(%x:f32)->f32 {{
      %good="tt.precise_sqrt"(%x):(f32)->f32
      {operation}
      return %good:f32
    }} }}""")
    candidate = list(next(iter(module.ops)).body.block.ops)[1]
    if bad == "property":
        candidate.attributes["unknown_policy"] = builtin.UnitAttr()
    elif kind == "cast":
        candidate.properties["rounding"] = builtin.IntegerAttr(0, builtin.i32)
    elif kind == "external":
        candidate.attributes["libname"] = builtin.StringAttr("custom")
    else:
        candidate.results[0]._type = builtin.i64
    before = text(module)
    with pytest.raises(ConversionError):
        expand_scalar_builtins(module, _source_ids)
    assert text(module) == before


@pytest.mark.parametrize("policy", [0, 65535])
def test_clamp_retains_the_exact_source_nan_policy(policy):
    module = parse_shared(f"""module {{func.func @f(%x:f32,%lo:f32,%hi:f32)->f32 {{
      %out="tt.clampf"(%x,%lo,%hi) {{propagateNan={policy}:i32}}:(f32,f32,f32)->f32
      return %out:f32
    }} }}""")
    records = expand_scalar_builtins(module, _source_ids)
    module.verify()
    assert records[0]["expansion"] == (
        ["arith.maximumf", "arith.minimumf"]
        if policy
        else ["arith.maxnumf", "arith.minnumf"]
    )


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native compiler required"
)
@pytest.mark.parametrize("width,fraction", [(32, 23), (64, 52)])
def test_scalar_numeric_bitcast_maps_preserve_original_nan_and_zero_words(
    tmp_path, width, fraction
):
    from t2m.elementwise_map import expand_elementwise_maps
    from t2m.native import execute_native

    exponent = ((1 << (width - 1)) - 1) ^ ((1 << fraction) - 1)
    sign = 1 << (width - 1)
    words = np.asarray(
        [
            0,
            sign,
            1,
            exponent,
            exponent + 1,
            exponent | (1 << (fraction - 1)),
            sign | exponent | 13,
            exponent - 1,
        ],
        dtype=f"uint{width}",
    )
    module = parse_shared(f"""module {{func.func @forward(%x:tensor<8xf{width}>)->tensor<8xi{width}> {{
      %out="tt.map_elementwise"(%x) <{{pack=1:i32}}> ({{
      ^bb0(%v:f{width}):
        %word="tt.bitcast"(%v):(f{width})->i{width} loc("ttir.op.11")
        "tt.map_elementwise.return"(%word):(i{width})->()
      }}):(tensor<8xf{width}>)->tensor<8xi{width}>
      return %out:tensor<8xi{width}>
    }} }}""")
    expand_scalar_builtins(module, _source_ids)
    expand_elementwise_maps(module, _source_ids)
    module.verify()
    actual, _ = execute_native(
        text(module),
        [words.view(f"float{width}")],
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    assert actual[0].dtype == np.dtype(f"int{width}")
    assert actual[0].tobytes() == words.tobytes()

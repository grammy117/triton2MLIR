"""Scan validation retains source execution and refuses nested effects first."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.scans import expand_scans
from t2m.shared_tensor import _source_ids, parse_shared


def scan(
    body="%sum=arith.addi %a,%b:i64", result="%sum", *, count=8, location="ttir.op.9900"
):
    return f'''%out = "tt.scan"(%x) <{{axis=0:i32,reverse=false}}> ({{
      ^bb0(%a:i64,%b:i64):
        {body}
        "tt.scan.return"({result}):(i64)->()
      }}):(tensor<{count}xi64>)->tensor<{count}xi64> loc("{location}")'''


@pytest.mark.parametrize(
    "effect",
    [
        '"tt.print"() <{prefix="effect",hex=false}>:()->()',
        'cf.assert %condition,"effect"',
        "func.call @unknown() : () -> ()",
        "%read=memref.load %p[%zero]:memref<8xi64>",
        '%unknown="arith.unknown"(%a):(i64)->i64',
        "%tensor=tensor.empty():tensor<1xi64>",
    ],
)
@pytest.mark.parametrize("count", [1, 8])
def test_nested_unused_effects_refuse_before_any_scan_is_expanded(effect, count):
    nested = f"""%condition=arith.cmpi slt,%a,%b:i64
      %sum=scf.if %condition -> (i64) {{
        {effect}
        scf.yield %a:i64
      }} else {{scf.yield %b:i64}}"""
    good = scan(location="ttir.op.1111").replace("%out", "%good")
    bad = scan(nested, count=count, location="ttir.op.2222").replace("%x", "%small")
    module = parse_shared(f"""module {{func.func private @unknown()
      func.func @f(%x:tensor<8xi64>,%p:memref<8xi64>)->tensor<{count}xi64> {{
        %zero=arith.constant 0:index
        {good}
        %small=tensor.empty():tensor<{count}xi64>
        {bad}
        return %out:tensor<{count}xi64>
      }} }}""")
    before = str(module)
    with pytest.raises(ConversionError, match="ttir.op.2222"):
        expand_scans(module, _source_ids)
    assert str(module) == before


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native compiler required"
)
def test_direct_scalar_capture_returns_are_preserved_without_a_false_ssa_mapping(
    tmp_path,
):
    from t2m.native import execute_native

    operation = scan("", "%uniform")
    module = parse_shared(f"""module {{func.func @f(%x:tensor<8xi64>,%uniform:i64)->tensor<8xi64> {{
      {operation}
      return %out:tensor<8xi64>
    }} }}""")
    expand_scans(module, _source_ids)
    module.verify()
    from t2m.ir import text

    inputs = [np.arange(8, dtype=np.int64) + 2**53 + 1, np.asarray(-17, np.int64)]
    actual, _ = execute_native(
        text(module),
        inputs,
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    expected = np.full(8, -17, np.int64)
    expected[0] = inputs[0][0]
    assert actual[0].tobytes() == expected.tobytes()

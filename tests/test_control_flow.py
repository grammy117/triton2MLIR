"""Value loops cannot hide source-buffer observations or opaque effects."""

import pytest

from t2m.control_flow import value_only_region
from t2m.shared_tensor import parse_shared


@pytest.mark.parametrize("body,arguments,expected", [
    ("%v=arith.addf %old,%old:f32", "", True),
    ("%v=memref.load %buffer[%c0]:memref<4xf32>", ",%buffer:memref<4xf32>", False),
    ('%v="arith.unmodeled"(%old):(f32)->f32', "", False),
])
def test_scalar_result_does_not_make_memory_or_unknown_operation_value_only(body, arguments, expected):
    module = parse_shared(f"""module {{ func.func @test(%bound:index{arguments}) -> f32 {{
      %c0=arith.constant 0:index
      %c1=arith.constant 1:index
      %zero=arith.constant 0.0:f32
      %result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old=%zero)->(f32) {{
        {body}
        scf.yield %v:f32
      }}
      return %result:f32
    }} }}""")
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert value_only_region(loop) is expected

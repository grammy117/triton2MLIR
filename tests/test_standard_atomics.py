"""Standard MemRef atomic region binding preserves native structural rules."""

import os

import pytest
from xdsl.utils.exceptions import VerifyException

from t2m.ir import text
from t2m.shared_tensor import parse_shared


def program(element="i32", *, current=None, result=None, indices=True, effect=False):
    return f'''builtin.module {{func.func @observe(%p: memref<?x{element}>, %i: index) -> {result or element} {{
      %old = "memref.generic_atomic_rmw"(%p{', %i' if indices else ''}) ({{
      ^bb0(%current: {current or element}):
        {'"memref.store"(%current, %p, %i) : (' + element + ', memref<?x' + element + '>, index) -> ()' if effect else ''}
        "memref.atomic_yield"(%current) : ({current or element}) -> ()
      }}) : (memref<?x{element}>{', index' if indices else ''}) -> {result or element}
      func.return %old : {result or element}
    }}}}'''


@pytest.mark.parametrize("element", ["i32", "i64", "f32"])
def test_standard_region_round_trip_and_actual_native_verification(tmp_path, element):
    module = parse_shared(program(element))
    module.verify()
    source = text(module)
    parse_shared(source).verify()
    if optimizer := os.environ.get("TRITON_SHARED_OPT"):
        from t2m.upstream import lower_with_shared
        path = tmp_path / "source.mlir"
        path.write_text(source)
        assert lower_with_shared(path, tmp_path / "native", optimizer=optimizer, stage="source")["stage_completed"]


@pytest.mark.parametrize("options", [{"current": "i64"}, {"result": "i64"}, {"indices": False}, {"effect": True}])
def test_atomic_region_rejects_wrong_elements_indices_and_effects(options):
    with pytest.raises(VerifyException):
        parse_shared(program(**options)).verify()


def test_atomic_yield_requires_the_atomic_region():
    with pytest.raises(VerifyException, match="generic atomic parent"):
        parse_shared('''builtin.module { func.func @bad(%v: i32) {
          "memref.atomic_yield"(%v) : (i32) -> ()
        }}''').verify()

"""Native and target consumers select the same typed implementation boundaries."""

import pytest

from t2m.consumer_features import COMPILER_FEATURES, select_features
from t2m.shared_tensor import parse_shared


@pytest.mark.parametrize("source,additional", [
    ('module attributes {prov.dtype="bf16",prov.atomic="word"} {func.func @f(%x:i64)->i64 {return %x:i64}}', set()),
    ('module {func.func @f(%x:bf16)->f32 {%v=arith.extf %x:bf16 to f32 return %v:f32}}', {"preserve_bfloat_transport"}),
    ('module {func.func @f(%x:bf16)->bf16 {%v=arith.addf %x,%x:bf16 return %v:bf16}}', {"preserve_bfloat_transport", "emulate_bfloat_elementwise"}),
    ('module {func.func @f(%x:f8E4M3FN)->f32 {%v=arith.extf %x:f8E4M3FN to f32 return %v:f32}}', {"lower_fp8_widening"}),
    ('module {func.func @f(%x:f32)->f8E5M2 {%v=arith.truncf %x:f32 to f8E5M2 return %v:f8E5M2}}', {"lower_fp8_narrowing"}),
    ('module {func.func @f(%p:i1) {cf.assert %p,"plain" return}}', set()),
    ('module {func.func @f(%p:i1) {cf.assert %p,"A\\00B" return}}', {"lower_length_aware_diagnostics"}),
    ('module {func.func @f(%x:i64) {"vector.print"(%x):(i64)->() return}}', {"lower_scalar_diagnostics"}),
])
def test_capabilities_follow_actual_types_operations_and_literals(source, additional):
    module = parse_shared(source)
    module.verify()
    assert select_features(module) == COMPILER_FEATURES | additional


@pytest.mark.parametrize("dtype", ["i16", "i32", "i64", "f16", "f32", "f64"])
def test_atomic_storage_type_selects_capability_without_granting_body_permission(dtype):
    module = parse_shared(f'''module {{func.func @f(%p:memref<?x{dtype}>,%i:index)->{dtype} {{
      %r="memref.generic_atomic_rmw"(%p,%i) ({{
      ^bb0(%old:{dtype}): "memref.atomic_yield"(%old):({dtype})->()
      }}):(memref<?x{dtype}>,index)->{dtype}
      return %r:{dtype}
    }}}}''')
    module.verify()
    feature = "lower_atomic_word_views" if dtype.startswith("f") else "lower_integer_compare_exchange"
    assert select_features(module) == COMPILER_FEATURES | {feature}


def test_byte_word_region_uses_existing_integer_region_capability():
    pytest.importorskip("merlin.llvmlower.atomic_integer_regions")
    module = parse_shared('''module {func.func @f(%value:i32)->i32 {
      %raw=memref.alloc():memref<32xi8>
      %zero=arith.constant 0:index
      %eight=arith.constant 8:index
      %words=memref.view %raw[%zero][%eight]:memref<32xi8> to memref<?xi32>
      memref.store %value,%words[%zero]:memref<?xi32>
      %r="memref.generic_atomic_rmw"(%words,%zero) ({
      ^bb0(%old:i32):
        %next=arith.xori %old,%value:i32
        "memref.atomic_yield"(%next):(i32)->()
      }):(memref<?xi32>,index)->i32
      memref.dealloc %raw:memref<32xi8>
      return %r:i32
    }}''')
    module.verify()
    assert select_features(module) == COMPILER_FEATURES | {"lower_integer_atomic_regions"}

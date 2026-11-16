"""Coarse numeric admission keeps original property and external effect gates."""

import pytest
from test_value_region_contract import source

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.shared_tensor import _source_ids, parse_shared
from t2m.value_regions import expand_value_regions


@pytest.mark.parametrize("family", ["dot", "histogram", "gather", "logical"])
def test_unreviewed_coarse_properties_do_not_publish_a_prepared_container(family):
    bodies = {
        "dot": '''%numeric_tile=arith.constant dense<1.0>:tensor<16x16xf32>
          %coarse="tt.dot"(%numeric_tile,%numeric_tile,%numeric_tile)
            <{inputPrecision=2:i32,maxNumImpreciseAcc=0:i32}>
            {unreviewed_layout,prov.family="pure"}:
            (tensor<16x16xf32>,tensor<16x16xf32>,tensor<16x16xf32>)->tensor<16x16xf32>''',
        "histogram": '''%numeric_tile=arith.constant dense<1>:tensor<4xi32>
          %coarse="tt.histogram"(%numeric_tile) {unreviewed_layout,prov.family="pure"}:
            (tensor<4xi32>)->tensor<2xi32>''',
        "gather": '''%numeric_tile=arith.constant dense<1>:tensor<4xi32>
          %index=arith.constant dense<0>:tensor<4xi32>
          %coarse="tt.gather"(%numeric_tile,%index) <{axis=0:i32}>
            {unreviewed_layout,prov.family="pure"}:(tensor<4xi32>,tensor<4xi32>)->tensor<4xi32>''',
        "logical": '''%coarse="tt.get_program_id"() <{axis=0:i32}>
            {unreviewed_layout,prov.family="pure"}:()->i32''',
    }
    module = parse_shared(source(bodies[family]))
    before = text(module)
    with pytest.raises(ConversionError, match="pure scalar|launch query"):
        expand_value_regions(module, _source_ids)
    assert text(module) == before


@pytest.mark.parametrize("policy", ["effectful", "library", "path", "symbol", "prototype"])
def test_unused_tensor_externals_require_the_actual_library_type_and_purity_contract(policy):
    dtype = "f64" if policy == "prototype" else "f32"
    library = "custom" if policy == "library" else "libdevice"
    path = "unqualified-library" if policy == "path" else ""
    symbol = "__nv_unqualified" if policy == "symbol" else "__nv_tanhf"
    pure = "false" if policy == "effectful" else "true"
    operation = f'''%numeric_tile=arith.constant dense<1.0>:tensor<4x{dtype}>
      %coarse="tt.extern_elementwise"(%numeric_tile)
        {{pure={pure},libname="{library}",libpath="{path}",symbol="{symbol}",prov.family="pure"}}:
        (tensor<4x{dtype}>)->tensor<4x{dtype}>'''
    module = parse_shared(source(operation))
    before = text(module)
    with pytest.raises(ConversionError, match="external"):
        expand_value_regions(module, _source_ids)
    assert text(module) == before

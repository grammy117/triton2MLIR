"""Pointer-state prepass selection must never hide address or buffer state."""

import pytest

from t2m.shared_tensor import parse_shared, _source_ids
from t2m.structured_state import data_control_report


def loop_module(width=64, *, body="", after=""):
    return parse_shared(f'''builtin.module {{ func.func @data(%ptr: !tt.ptr<i{width}>, %value: tensor<4xi{width}>) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%state = %value) -> (tensor<4xi{width}>) {{
        {body}
        %next = arith.addi %state, %value : tensor<4xi{width}>
        scf.yield %next : tensor<4xi{width}>
      }} loc("ttir.op.loop")
      {after}
      func.return
    }} }}''')


@pytest.mark.parametrize("width",[1,8,16,32,64])
def test_numeric_tensor_recurrence_uses_the_supported_upstream_option(width):
    module = loop_module(width,body=f'''"tt.print"(%state) : (tensor<4xi{width}>) -> ()''',
        after=f'''"tt.store"(%ptr, %result) : (!tt.ptr<i{width}>, tensor<4xi{width}>) -> ()''')
    report = data_control_report(module,_source_ids)
    assert report["upstream_option"] == "skip-prepass=true"
    assert report["source_ids"] == ["ttir.op.loop"]
    assert "scf.for" in str(module) and "arith.addi" in str(module)


@pytest.mark.parametrize("body",[
    '"tt.load"(%ptr) : (!tt.ptr<i64>) -> i64',
    '"tt.store"(%ptr, %value) : (!tt.ptr<i64>, tensor<4xi64>) -> ()',
    '"tt.extern_elementwise"(%state) : (tensor<4xi64>) -> ()',
    '"func.call"(%state) <{callee = @unknown}> : (tensor<4xi64>) -> ()',
])
def test_buffer_accesses_or_unknown_effects_retain_the_original_prepass(body):
    assert data_control_report(loop_module(body=body),_source_ids) is None


@pytest.mark.parametrize("after",[
    '"tt.addptr"(%ptr,%result) : (!tt.ptr<i64>,tensor<4xi64>) -> tensor<4x!tt.ptr<i64>>',
    '''%low = "arith.trunci"(%result) : (tensor<4xi64>) -> tensor<4xi32>
       "tt.addptr"(%ptr,%low) : (!tt.ptr<i64>,tensor<4xi32>) -> tensor<4x!tt.ptr<i64>>''',
    '''%zero = arith.constant dense<0> : tensor<4xi64>
       %mask = "arith.cmpi"(%result,%zero) <{predicate=0:i64}> : (tensor<4xi64>,tensor<4xi64>) -> tensor<4xi1>
       "tt.store"(%ptr,%value,%mask) : (!tt.ptr<i64>,tensor<4xi64>,tensor<4xi1>) -> ()''',
    '"tt.int_to_ptr"(%result) : (tensor<4xi64>) -> tensor<4x!tt.ptr<i64>>',
])
def test_transitive_address_or_mask_uses_retain_the_original_prepass(after):
    assert data_control_report(loop_module(after=after),_source_ids) is None


def test_nested_numeric_branch_results_retain_the_complete_recurrence():
    module = loop_module(body='''%yes = arith.constant true
        %branch = scf.if %yes -> (tensor<4xi64>) {
          "tt.print"(%state) : (tensor<4xi64>) -> ()
          scf.yield %state : tensor<4xi64>
        } else { scf.yield %value : tensor<4xi64> }''',
        after='''%sum = arith.addi %result, %value : tensor<4xi64>
          "tt.store"(%ptr,%sum) : (!tt.ptr<i64>,tensor<4xi64>) -> ()''')
    assert data_control_report(module,_source_ids)

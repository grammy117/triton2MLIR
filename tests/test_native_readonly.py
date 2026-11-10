"""Functional tensor versions cannot overwrite the native caller's input."""

import json
import os

import numpy as np
import pytest

from t2m.native import observe_native


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"),reason="existing native environment required")
@pytest.mark.parametrize("steps",[0,1,3])
def test_native_tensor_accumulator_keeps_the_borrowed_input_readonly(tmp_path,steps):
    source = '''module {func.func @recurrence(%x:tensor<2xi64>,%steps:i32)->tensor<2xi64> {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %i0 = arith.constant 0 : index
      %one = arith.constant 1 : i64
      %result = scf.for %iv = %c0 to %steps step %c1 iter_args(%state = %x) -> (tensor<2xi64>) : i32 {
        %old = tensor.extract %state[%i0] : tensor<2xi64>
        %next = arith.addi %old,%one : i64
        %updated = tensor.insert %next into %state[%i0] : tensor<2xi64>
        scf.yield %updated : tensor<2xi64>
      }
      return %result : tensor<2xi64>
    }}'''
    values = np.asarray([2**53+1,-19],np.int64)
    original = values.copy()
    observe_native(source,[values,np.asarray(steps,np.int32)],tmp_path / "native",python=os.environ["T2M_NATIVE_PYTHON"])
    report = json.loads((tmp_path / "native/execution.json").read_text())
    assert report["invocations"] == 1 and report["functional_inputs_preserved"] is True
    assert {"name":"readonly-functional-tensor-inputs","arguments":1,
        "upstream_contract":"bufferization.writable=false"} in report["compiler_adapters"]
    with np.load(tmp_path / "native/outputs.npz",allow_pickle=False) as output:
        np.testing.assert_array_equal(output["out0"],original+np.asarray([steps,0],np.int64))
    np.testing.assert_array_equal(values,original)

"""Actual standard assertions must print their literal before native abort."""

import os

import numpy as np
import pytest
from xdsl.dialects import builtin, cf, func
from xdsl.ir import Block, Region

from t2m.ir import text
from t2m.native import execute_native


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
@pytest.mark.parametrize("message", ["literal diagnostic %s %d", 'quoted "value"\nsecond line', "source diagnostic: λ"])
def test_native_fail_stop_output_survives_piped_stdout(tmp_path,message):
    tensor = builtin.TensorType(builtin.f32,[1])
    body = Block(arg_types=[tensor,builtin.i1])
    body.add_ops([cf.AssertOp(body.args[1],message),func.ReturnOp(body.args[0])])
    module = builtin.ModuleOp([func.FuncOp("diagnostic",([tensor,builtin.i1],[tensor]),Region(body))])
    x = np.asarray([3.5],np.float32)
    result,_ = execute_native(text(module),[x,np.asarray(True)],tmp_path / "good",python=os.environ["T2M_NATIVE_PYTHON"])
    np.testing.assert_array_equal(result[0],x)
    with pytest.raises(ValueError) as error:
        execute_native(text(module),[x,np.asarray(False)],tmp_path / "bad",python=os.environ["T2M_NATIVE_PYTHON"])
    assert "exit -6" in str(error.value) and message in str(error.value)

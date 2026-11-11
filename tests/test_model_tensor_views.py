"""Dense model views preserve logical order, exact words and scalar/empty ranks."""

import math
import os

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from t2m.ir import text
from t2m.model_tensor_views import compatible, reshape
from t2m.native import execute_native
from t2m.shared_tensor import DTYPES

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


@pytest.mark.parametrize("source,target",[
    ((2,3),(6,)),((6,),(2,3)),((2,3),(3,2)),((1,),()),((),(1,)),
    ((0,3),(0,)),((0,),(2,0,3)),((2,0,3),(0,6)),
])
@pytest.mark.parametrize("dtype",["i64","f32","bf16"])
def test_static_tensor_views_preserve_words_through_native_execution(tmp_path,source,target,dtype):
    left,right = (builtin.TensorType(DTYPES[dtype],shape) for shape in (source,target))
    block = Block(arg_types=[left])
    def append(op,role):
        block.add_op(op)
        return op.results[0]
    result = reshape(block.args[0],right,append,role="view")
    block.add_op(func.ReturnOp(result))
    module = builtin.ModuleOp([func.FuncOp("forward",([left],[right]),Region(block))])
    module.verify()
    assert compatible(left,right) and result.type == right
    if not NATIVE:
        return
    count = math.prod(source)
    words,carrier = ({"i64":([2**53+1,-2**53-1,0,2**63-1],np.int64),
                      "f32":([0x7F812345,0xFFC05678,0,0x80000000,0x3F800000],np.uint32),
                      "bf16":([0x7F81,0xFFC2,0,0x8000,0x3F80],np.uint16)})[dtype]
    values = np.resize(np.asarray(words,carrier),count)
    if dtype == "f32": values = values.view(np.float32)
    values = values.reshape(source)
    before = values.tobytes()
    (actual,),_ = execute_native(text(module),[values],tmp_path / "native",python=NATIVE)
    assert actual.shape == target and actual.dtype == values.dtype and actual.tobytes() == before
    assert values.tobytes() == before


@pytest.mark.parametrize("target",[
    builtin.TensorType(builtin.f32,[5]),builtin.TensorType(builtin.f64,[6]),
    builtin.TensorType(builtin.f32,[-1,6]),
    builtin.TensorType(builtin.f32,[6],builtin.StringAttr("opaque-layout")),
])
def test_tensor_views_refuse_count_type_dynamic_or_encoding_changes_before_edits(target):
    source = builtin.TensorType(builtin.f32,[2,3])
    block = Block(arg_types=[source])
    with pytest.raises(ValueError,match="static equal element counts"):
        reshape(block.args[0],target,lambda op,role:block.add_op(op),role="view")
    assert not list(block.ops)

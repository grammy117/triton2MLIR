"""Explicit dense row-major value reshapes at a captured kernel call boundary.

LLVM a47bddcc TensorOps.cpp and tensor/ops.mlir define reassociation, including
rank-zero unit values. This bridge changes neither dtype nor element order.
Actual source strides, physical aliases and model mutations require other facts.
"""

import math

from xdsl.dialects import builtin, tensor

POLICY = "dense-row-major-v1"
PROFILE = "t2m.captured-dense-tensor-views.v1"
BINDING_SCHEMA = "t2m.model-kernel-binding.v2"


def validate_profile(trace):
    calls = trace.get("calls")
    if not isinstance(calls,list) or any(not isinstance(call,dict) for call in calls):
        raise ValueError("model binding requires complete call records")
    schema = trace.get("schema")
    if schema == "t2m.model-kernel-binding.v1":
        if "tensor_view_binding_profile" in trace or any("tensor_views" in call for call in calls):
            raise ValueError("dense model tensor views require binding schema v2")
        return False
    if (schema != BINDING_SCHEMA or trace.get("tensor_view_binding_profile") != PROFILE
            or not any(call.get("tensor_views") == POLICY for call in calls)
            or any(call.get("tensor_views") not in (None,POLICY) for call in calls)):
        raise ValueError("unknown or incomplete model tensor view binding profile")
    return True


def compatible(left, right):
    return (isinstance(left,builtin.TensorType) and isinstance(right,builtin.TensorType)
            and isinstance(left.encoding,builtin.NoneAttr) and isinstance(right.encoding,builtin.NoneAttr)
            and all(n >= 0 for n in (*left.get_shape(),*right.get_shape()))
            and left.element_type == right.element_type
            and math.prod(left.get_shape()) == math.prod(right.get_shape()))


def reshape(value, target, append, *, role):
    """Expand/collapse static logical values after their complete type check."""
    if not compatible(value.type,target):
        raise ValueError("dense model tensor views require static equal element counts and exact element type")
    if value.type == target:
        return value
    count = math.prod(target.get_shape())
    flat = builtin.TensorType(target.element_type,[count])

    def group(rank):
        return builtin.ArrayAttr([builtin.ArrayAttr(builtin.IntegerAttr(i,builtin.i64) for i in range(rank))] if rank else [])

    source_rank, target_rank = len(value.type.get_shape()),len(target.get_shape())
    if source_rank > 1:
        value = append(tensor.CollapseShapeOp.build(operands=[value],result_types=[flat],
            properties={"reassociation":group(source_rank)}),role+"-collapse")
    elif source_rank == 0:
        # xDSL's expand_shape verifier rejects upstream's valid empty
        # reassociation for rank zero. Keep an equivalent standard scalar edge.
        value = append(tensor.ExtractOp.build(operands=[value,[]],result_types=[target.element_type]),role+"-scalar-extract")
        value = append(tensor.FromElementsOp(value),role+"-scalar-pack")
    if target_rank > 1:
        value = append(tensor.ExpandShapeOp(value,[],group(target_rank),target.get_shape(),target),role+"-expand")
    elif target_rank == 0:
        value = append(tensor.CollapseShapeOp.build(operands=[value],result_types=[target],
            properties={"reassociation":group(0)}),role+"-collapse")
    return value

"""Model view relationships are derived from SSA, never annotation permission."""

from copy import deepcopy

import pytest

from t2m.model_source_views import bind, recover
from t2m.model_tensor_views import validate_profile
from t2m.shared_tensor import parse_shared


def example():
    module = parse_shared('''module {func.func @f(%root:tensor<6x16xi64>,%other:tensor<6x16xi64>)->tensor<8x6xi64> {
      %empty=tensor.empty():tensor<16x6xi64>
      %t=linalg.transpose ins(%root:tensor<6x16xi64>) outs(%empty:tensor<16x6xi64>) permutation=[1,0]
      %view="tensor.extract_slice"(%t) <{static_offsets=array<i64:1,0>,static_sizes=array<i64:8,6>,
        static_strides=array<i64:2,1>,operandSegmentSizes=array<i32:1,0,0,0>}>:(tensor<16x6xi64>)->tensor<8x6xi64>
      return %view:tensor<8x6xi64>}}''')
    module.verify()
    function = next(iter(module.ops))
    root, other = function.body.block.args
    view = function.body.block.last_op.operands[0]
    config = {"view": {"root": "state", "source_arg": 1, "shape": ["rows", "cols"], "strides": ["s0", "s1"], "offset": "origin"}}
    values = {"rows": 8, "cols": 6, "s0": 2, "s1": 16, "origin": 1}
    provenance = {"source_argument_binding": {"bindings": [{"kind": "pointer", "allocation": "state", "offset": 0},
        {"kind": "pointer", "allocation": "state", "offset": "origin"}]}}
    return module, root, other, view, config, values, provenance


def test_transposed_sliced_coordinates_and_canonical_root_are_recovered():
    module, root, _, view, config, values, provenance = example()
    layout = recover(view, root)
    assert (layout["shape"], layout["strides"], layout["offset"]) == ([8, 6], [2, 16], 1)
    assert [step["op"] for step in layout["steps"]] == ["linalg.transpose", "tensor.extract_slice"]
    assert recover(root, root) == {"shape": [6, 16], "strides": [16, 1], "offset": 0, "steps": []}
    assert bind(config, {"state": "owner"}, {"owner": root, "view": view}, values, provenance)["view"]["layout"] == layout


@pytest.mark.parametrize("change", ("offset", "stride", "shape", "root", "copy", "source-arg", "source-pointer", "extra", "missing", "scalar-type", "transpose-body"))
def test_bad_relationships_and_forged_source_bindings_refuse(change):
    module, root, other, view, config, values, provenance = example()
    if change == "offset": values["origin"] = 0
    elif change == "stride": values["s0"] = 1
    elif change == "shape": values["rows"] = 7
    elif change == "root": root = other
    elif change == "copy":
        transpose = view.owner.operands[0].owner
        transpose.operands = (other, transpose.operands[1])
    elif change == "source-arg": config["view"]["source_arg"] = 0
    elif change == "source-pointer": provenance["source_argument_binding"]["bindings"][1]["offset"] = 0
    elif change == "extra": config["view"]["grants_semantics"] = True
    elif change == "missing": config["view"].pop("strides")
    elif change == "scalar-type": values["origin"] = True
    elif change == "transpose-body":
        body = view.owner.operands[0].owner.regions[0].block
        body.last_op.operands = (body.args[-1],)
    before = str(module)
    with pytest.raises(ValueError): bind(config, {"state": "owner"}, {"owner": root, "view": view}, values, provenance)
    assert str(module) == before


@pytest.mark.parametrize("change", ("downgrade", "missing", "unknown", "absent-call"))
def test_source_view_profile_cannot_be_downgraded_or_claimed_without_calls(change):
    trace = {"schema": "t2m.model-kernel-binding.v3", "source_tensor_view_binding_profile": "t2m.captured-source-tensor-views.v1",
             "calls": [{"source_tensor_views": {"view": {}}}]}
    assert validate_profile(trace) is False
    broken = deepcopy(trace)
    if change == "downgrade": broken["schema"] = "t2m.model-kernel-binding.v1"
    elif change == "missing": broken.pop("source_tensor_view_binding_profile")
    elif change == "unknown": broken["source_tensor_view_binding_profile"] = "unknown"
    else: broken["calls"] = [{}]
    with pytest.raises(ValueError): validate_profile(broken)

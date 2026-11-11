"""Binary coordinate facts and guarded parallel-row admission are structural."""

from copy import deepcopy
import hashlib

import pytest
from xdsl.dialects import arith, builtin, func

from t2m.control_flow import complete_parallel_rows
from t2m.elementwise_map import expand_elementwise_maps
from t2m.map_layout import KEY, options, validate
from t2m.shared_tensor import opcode, parse_shared


def layout():
    # Protocol fixture, not a claimed Triton source capture or native pass run.
    return {"id": "fixture", "pack": 2, "shape": [8],
        "forward": {"bases": [["register", [[4]]], ["lane", [[1], [2]]], ["warp", []], ["block", []]], "out_dims": [["dim0", 8]]},
        "inverse": {"bases": [["dim0", [[0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]]]],
                    "out_dims": [["register", 2], ["lane", 4], ["warp", 1], ["block", 1]]}}


@pytest.mark.parametrize("damage", [None, "inverse", "duplicate", "range", "pack", "shape"])
def test_coordinate_bijection_is_checked_without_element_enumeration(damage):
    row = layout()
    if damage == "inverse": row["inverse"]["bases"][0][1][2][0] = 0
    elif damage == "duplicate": row["forward"]["bases"].append(deepcopy(row["forward"]["bases"][0]))
    elif damage == "range": row["forward"]["bases"][0][1][0][0] = 8
    elif damage == "pack": row["pack"] = 4
    elif damage == "shape": row["shape"] = [16]
    if damage is None:
        assert validate(row) == {"register": 2, "lane": 4, "warp": 1, "block": 1}
    else:
        with pytest.raises(ValueError): validate(row)


@pytest.mark.parametrize("change", [{}, {"num_warps": True}, {"num_ctas": 3}, {"target": ""}, {"threads_per_warp": 1 << 20}])
def test_layout_specialization_requires_explicit_bounded_source_options(change):
    value = {"target": "source:architecture", "num_warps": 4, "threads_per_warp": 32, "num_ctas": 1}
    if not change: value.pop("target")
    else: value.update(change)
    with pytest.raises(ValueError, match="explicit target"):
        options(value)


@pytest.mark.parametrize("damage", [None, "overlap", "duplicate", "domain", "effects"])
def test_guarded_parallel_rows_require_complete_disjoint_value_updates(damage):
    module = parse_shared('''module {func.func @rows(%x:tensor<8xi64>)->tensor<8xi64> {
      %v="tt.map_elementwise"(%x) <{pack=2:i32}> ({
      ^bb0(%a0:i64,%a1:i64):
        %p=arith.addi %a0,%a1:i64
        %q=arith.subi %a1,%a0:i64
        "tt.map_elementwise.return"(%p,%q):(i64,i64)->()
      }):(tensor<8xi64>)->tensor<8xi64>
      func.return %v:tensor<8xi64>
    }}''')
    op = next(op for op in module.walk() if opcode(op) == "tt.map_elementwise")
    op.attributes[KEY] = builtin.StringAttr("fixture")
    expand_elementwise_maps(module, lambda location: [], layouts={"fixture": layout()})
    module.verify()
    loop = next(op for op in module.walk() if op.name == "scf.forall")
    body, end = loop.body.block, loop.body.block.last_op
    if damage == "overlap":
        zero = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        body.insert_op_before(zero, end)
        update = next(iter(end.body.block.ops))
        update.operands = (*update.operands[:2], zero.result)
    elif damage == "duplicate":
        end.body.block.add_op(next(iter(end.body.block.ops)).clone())
    elif damage == "domain":
        loop.properties["staticUpperBound"] = builtin.DenseArrayBase.from_list(builtin.i64, [5])
    elif damage == "effects":
        body.insert_op_before(func.CallOp("external_effect", [], []), end)
    assert complete_parallel_rows(loop) is (damage is None)


@pytest.mark.parametrize("damage", [None, "source", "converted", "identity", "inverse", "unbound", "groups"])
def test_source_capsule_checks_the_complete_grouping_witness(damage):
    from t2m.source_capsule import validate as validate_capsule
    row = layout()
    witness = {"schema": "t2m.source-map-layout.v1", "triton_version": "3.7.1",
        "pipeline": ["convert-triton-to-tritongpu"], "device_codegen": "not-invoked",
        "dependency_source_closure": "not-established", "source_ir": "protocol source", "ttgir": "protocol layout",
        "source_options": {"target": "source:architecture", "num_warps": 1, "threads_per_warp": 4, "num_ctas": 1},
        "native_library": {"path": "/explicit/protocol/fixture", "bytes": 1, "sha256": "0" * 64}, "maps": [row]}
    witness.update(source_sha256=hashlib.sha256(witness["source_ir"].encode()).hexdigest(),
                   ttgir_sha256=hashlib.sha256(witness["ttgir"].encode()).hexdigest())
    proof = {"method": "source-linear-layout-v1", "source_map": "fixture", "layout": deepcopy(row),
             "logical_groups": 4, "group_compute": "one-original-scalar-region-per-logical-group"}
    trace = {"source_layout_specialization": witness,
             "transformations": [{"kind": "ttir-elementwise-map", "packing_proof": proof}]}
    if damage == "source": witness["source_ir"] += "changed"
    elif damage == "converted": witness["ttgir"] += "changed"
    elif damage == "identity": witness.pop("native_library")
    elif damage == "inverse": witness["maps"][0]["inverse"]["bases"][0][1][2][0] = 0
    elif damage == "unbound": trace.pop("source_layout_specialization")
    elif damage == "groups": proof["logical_groups"] = 8
    if damage is None:
        validate_capsule(trace, {}, b"protocol source")
    else:
        with pytest.raises((ValueError, KeyError)):
            validate_capsule(trace, {}, b"protocol source")

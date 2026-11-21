"""Uniform values require actual dependencies, independently of invariance."""

import pytest
from test_source_branch_offsets import source as branch_source
from test_source_pointer_recurrences import source
from xdsl.dialects import builtin

from t2m.grid_uniformity import is_grid_uniform
from t2m.shared_tensor import opcode


def test_actual_numeric_loop_tuple_and_headers_can_be_uniform_despite_unrelated_memory():
    module = source()
    loop = next(op for op in module.walk() if op.name == "scf.for")
    inputs = frozenset(module.ops.first.regions[0].block.args[1:])
    assert is_grid_uniform(loop.results[2], inputs, opcode)
    assert not is_grid_uniform(loop.results[2], frozenset(), opcode)


@pytest.mark.parametrize("operation,expected", [("tt.get_program_id", False), ("tt.get_num_programs", True), ("unknown.integer", False)])
def test_actual_selected_control_distinguishes_program_id_from_uniform_grid_extent(operation, expected):
    module, _abi = branch_source()
    branch = next(op for op in module.walk() if op.name == "scf.if")
    inputs = frozenset(module.ops.first.regions[0].block.args[2:])
    assert is_grid_uniform(branch.results[0], inputs, opcode)
    chosen = builtin.UnregisteredOp.with_name(operation).create(
        result_types=[builtin.i32], properties={"axis": builtin.IntegerAttr(0, builtin.i32)})
    branch.parent.insert_op_before(chosen, branch)
    condition = branch.cond.owner
    condition.operands = [chosen.results[0], condition.operands[1]]
    branch.attributes["prov.uniform"] = builtin.StringAttr("claimed")
    assert is_grid_uniform(branch.results[0], inputs, opcode) is expected

"""Real symbolic call dimensions come from original SSA and executable bodies."""

import json
import os
import signal
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import func

from t2m.model_binding import KernelBinding, bind_model
from t2m.model_live_arguments import validate_model
from t2m.model_shape_facts import PROFILE, ShapeFacts
from t2m.shared_tensor import parse_shared


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    if not os.environ.get("T2M_MODEL2MLIR_PYTHON"):
        pytest.skip("existing model2MLIR environment required")
    root = tmp_path_factory.mktemp("symbolic-copy-model")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_symbolic_copy_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(root / "parent")],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    return (root / "parent/model.mlir").read_text()


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def bound(tmp_path_factory, parent, request):
    if not os.environ.get("TRITON_SHARED_OPT"):
        pytest.skip("existing shared optimizer required")
    from test_live_scalar_model import implementation

    root = tmp_path_factory.mktemp("symbolic-copy-" + request.param)
    kernel = implementation(root, "i32", extent=True, profile=request.param)
    result = bind_model(parent, [KernelBinding("t2m_demo::symbolic_copy", kernel, runtime_operands={"n": "n"})])
    assert result.trace["model_shape_profile"] == PROFILE
    (root / "bound.mlir").write_text(result.mlir_text)
    (root / "binding.json").write_text(json.dumps(result.trace, indent=2) + "\n")
    (root / "kernel.mlir").write_text(kernel.mlir_text)
    (root / "kernel-provenance.json").write_text(json.dumps(kernel.trace, indent=2) + "\n")
    return result, kernel


def validate(result, kernel, parent):
    function = next(op for op in kernel.module.ops if isinstance(op, func.FuncOp))
    validate_model(result.trace, parse_shared(parent),
        implementations={kernel.trace["mlir_sha256"]: function}, bound_module=result.module)


def test_actual_dynamic_model_binds_both_calls_and_every_result_without_specialization(parent, bound):
    result, kernel = bound
    validate(result, kernel, parent)
    contract = result.trace["runtime_shape_contract"]
    assert contract == {"schema": "t2m.runtime-shapes.v2", "inputs": [{"shape": ["d0_0"], "dtype": "i32"}],
        "outputs": [{"shape": ["d0_0"], "dtype": "i32"}] * 3,
        "dimension_bindings": {"d0_0": {"argument": 0, "axis": 0}}}
    assert len(result.trace["calls"]) == 2
    for call in result.trace["calls"]:
        assert call["symbolic_tensor_shapes"] == {"x": ["d0_0"], "out": ["d0_0"]}
        assert call["runtime_operand_bindings"]["n"]["unsigned_integer_range"] == [0, 31]
        assert "constant_integer" not in call["runtime_operand_bindings"]["n"]
    entry = next(op for op in result.module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    second, first, incoming = entry.body.block.last_op.operands
    assert incoming is entry.body.block.args[0] and first is not second and first is not incoming
    first_site, second_site = (call["id"] for call in result.trace["calls"])
    assert first.owner.attributes["prov.kernel_callsite"].data == first_site
    assert second.owner.attributes["prov.kernel_callsite"].data == second_site
    assert any(first in op.operands for op in entry.walk()
               if getattr(op.attributes.get("prov.kernel_callsite"), "data", None) == second_site)
    if "guard_storage_bytes" in result.trace:
        assert result.trace["guard_storage_scope"] == "sum-of-source-shape-upper-bounds"


@pytest.mark.parametrize("damage", ["profile", "contract", "dimension", "constant", "order", "scratch-scope"])
def test_rehashed_shapes_cannot_substitute_for_original_ssa_and_kernel_result_proofs(parent, bound, damage):
    result, kernel = bound
    changed = deepcopy(result.trace)
    if damage == "profile": changed.pop("model_shape_profile")
    elif damage == "contract": changed["runtime_shape_contract"]["outputs"][0]["shape"] = [13]
    elif damage == "dimension": changed["calls"][1]["symbolic_tensor_shapes"]["x"] = [13]
    elif damage == "constant": changed["calls"][0]["runtime_operand_bindings"]["n"]["constant_integer"] = 13
    elif damage == "scratch-scope": changed["guard_storage_scope"] = "sampled-output-size"
    else: changed["calls"].reverse()
    from t2m.model_binding import BoundModel

    with pytest.raises(ValueError):
        validate(BoundModel(result.module, changed), kernel, parent)


def test_unrelated_public_tensor_axes_cannot_supply_the_kernel_dimension(parent, bound):
    from xdsl.dialects import builtin

    from t2m.diagnostics import ConversionError
    from t2m.ir import text

    _result, kernel = bound
    module = parse_shared(parent)
    function = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    type_ = function.body.block.args[0].type
    other = function.body.block.insert_arg(type_, len(function.body.block.args))
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [value.type for value in function.body.block.args], function.function_type.outputs)
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    call.operands = (other, call.operands[1])
    module.verify()
    with pytest.raises(ConversionError, match="matching typed/scalar specialization"):
        bind_model(text(module), [KernelBinding("t2m_demo::symbolic_copy", kernel, runtime_operands={"n": "n"})])


def test_kernel_result_dimension_is_reconstructed_from_the_actual_body(parent, bound):
    from xdsl.dialects import arith, builtin, tensor

    from t2m.diagnostics import ConversionError
    from t2m.model_arguments import plan_arguments

    _result, kernel = bound
    changed = kernel.module.clone()
    function = next(iter(changed.ops))
    terminal = function.body.block.last_op
    seven = arith.ConstantOp.from_int_and_width(7, builtin.IndexType())
    wrong = tensor.EmptyOp([seven.result], function.function_type.outputs.data[0])
    function.body.block.insert_ops_before([seven, wrong], terminal)
    terminal.operands = tuple(wrong.results)
    changed.verify()
    original = parse_shared(parent)
    entry = next(op for op in original.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    call = next(op for op in original.walk() if isinstance(op, func.CallOp))
    with pytest.raises(ConversionError, match="executable standard-MLIR proof"):
        plan_arguments(KernelBinding("t2m_demo::symbolic_copy", kernel, runtime_operands={"n": "n"}),
            function, {"x": "x"}, ["out"], dict(zip(("x", "n"), call.operands, strict=True)),
            builtin.DictionaryAttr({}), [value.type for value in call.results],
            source_point=call, shape_facts=ShapeFacts.entry(entry))


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
def test_complete_native_results_follow_independent_input_axes_and_compiled_source_guards(tmp_path, parent, bound):
    from t2m.native import execute_native, observe_native

    result, _kernel = bound
    contract = result.trace["runtime_shape_contract"]
    program = result.mlir_text
    for length in (0, 1, 7, 31):
        x = np.arange(length, dtype=np.int32) * 83 - 49
        actual, report = execute_native(program, [x], tmp_path / f"valid-{length}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract,
            compiler_support_root=os.environ.get("T2M_MERLIN_ROOT"))
        assert len(actual) == 3
        for value in actual:
            assert (value.shape, value.dtype, value.tobytes()) == (x.shape, x.dtype, x.tobytes())
        assert report["native_mlir_verified"] and report["llvm_lowering_verified"]
        assert result.mlir_text == program
    work = tmp_path / "invalid-32"
    report = observe_native(program, [np.arange(32, dtype=np.int32)], work,
        python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract,
        compiler_support_root=os.environ.get("T2M_MERLIN_ROOT"))
    assert report["returncode"] == -signal.SIGABRT
    assert b"exported shape domain:" in (work / "stdout.bin").read_bytes() + (work / "stderr.bin").read_bytes()
    assert not (work / "outputs.npz").exists() and not (work / "execution.json").exists()


def test_resizing_loop_results_do_not_inherit_the_initial_tensor_shape():
    module = parse_shared('''module {func.func @forward(%x:tensor<?xi32>)->tensor<?xi32> {
      %zero=arith.constant 0:index
      %one=arith.constant 1:index
      %other=tensor.empty(%one):tensor<?xi32>
      %result=scf.for %i=%zero to %one step %one iter_args(%state=%x)->(tensor<?xi32>) {
        scf.yield %other:tensor<?xi32>
      }
      return %result:tensor<?xi32>
    }}''')
    module.verify()
    function = next(iter(module.ops))
    facts = ShapeFacts.entry(function)
    with pytest.raises(ValueError, match="executable source relationship"):
        facts.contract(function)

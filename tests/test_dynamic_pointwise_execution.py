"""Typed ATen emitter programs execute independent broadcasts and slice bounds.

The fixture's unused output-extent scalars size the test driver's descriptors;
they supply no compiler fact or derived-model shape qualification. Real model
packages independently derive their signatures in the enclosing-model tests.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not all(os.environ.get(name) for name in
    ("T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON")), reason="existing native/model2MLIR environments required")


def emitter_program(tmp_path, shapes, *, operation="add", slicing=None, result_shape=None):
    """Isolate the source frontend's pinned xDSL environment from the consumer."""
    import json
    import subprocess

    script = '''import json,sys
from types import SimpleNamespace
import torch
from xdsl.dialects import builtin,func
from xdsl.ir import Block,Region
from xdsl.printer import Printer
from m2m.ir.decompositions import _binary_elementwise,_arith_build2
from m2m.ir.dynamic_slices import extract_slice
options=json.loads(sys.argv[1])
shape=options['result_shape']
types=[builtin.TensorType(builtin.i32,[builtin.DYNAMIC_INDEX if n is None else n for n in shape]) for shape in options['shapes']]
input_count=len(types)
types += [builtin.i64 for size in shape if size is None]
block=Block(arg_types=types)
if options['slicing'] is None:
 names={'add':('AddfOp','AddiOp'),'sub':('SubfOp','SubiOp'),'mul':('MulfOp','MuliOp')}
 result=_binary_elementwise(list(block.args[:input_count]),{'val':SimpleNamespace(shape=['n' if n is None else n for n in shape],dtype=torch.int32)},options['operation'],_arith_build2(*names[options['operation']]))
 assert result is not None
 operations,value=result.ops,result.result
else:
 target=builtin.TensorType(builtin.i32,[builtin.DYNAMIC_INDEX if n is None else n for n in shape])
 operations,value=extract_slice(block.args[0],target,options['slicing'])
block.add_ops(operations)
block.add_op(func.ReturnOp(value))
module=builtin.ModuleOp([func.FuncOp('forward',(types,[value.type]),region=Region(block))])
module.verify()
with open(sys.argv[2],'w') as stream:Printer(stream=stream,print_generic_format=True).print_op(module)
'''
    source = tmp_path / "source.mlir"
    options = {"shapes": shapes, "operation": operation, "slicing": slicing, "result_shape": result_shape}
    (tmp_path / "source-spec.json").write_text(json.dumps(options, sort_keys=True) + "\n")
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], "-c", script, json.dumps(options), str(source)],
        capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    return source.read_text()


def invocation(shapes, arrays, result_shape, expected):
    contract = {"schema": "t2m.runtime-shapes.v2", "inputs": [], "outputs": [], "dimension_bindings": {}}
    for i, shape in enumerate(shapes):
        dimensions = []
        for axis, size in enumerate(shape):
            name = f"input_{i}_{axis}"
            dimensions.append(name if size is None else size)
            if size is None:
                contract["dimension_bindings"][name] = {"argument": i, "axis": axis}
        contract["inputs"].append({"shape": dimensions, "dtype": "i32"})
    arrays = list(arrays)
    dimensions = []
    for axis, size in enumerate(result_shape):
        name = f"result_{axis}"
        dimensions.append(name if size is None else size)
        if size is None:
            arrays.append(np.asarray(expected.shape[axis], np.int64))
            contract["inputs"].append({"shape": [], "dtype": "i64", "scalar": name})
    contract["outputs"] = [{"shape": dimensions, "dtype": "i32"}]
    return arrays, contract


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
@pytest.mark.parametrize("layout", ["scalar", "row-column", "runtime-singletons"])
def test_actual_portable_binary_body_keeps_values_across_independent_shapes(tmp_path, operation, layout):
    from t2m.native import execute_native

    shapes = [[None], []] if layout == "scalar" else [[None, 1], [1, None]] if layout == "row-column" else [[None], [None]]
    result_shape = [None, None] if layout == "row-column" else [None]
    source = emitter_program(tmp_path, shapes, operation=operation, result_shape=result_shape)
    for i, (left, right) in enumerate(((0, 1), (1, 7), (7, 1), (7, 7))):
        if layout == "scalar":
            x, y = np.arange(left, dtype=np.int32), np.array(17, dtype=np.int32)
        elif layout == "row-column":
            x, y = np.arange(left, dtype=np.int32).reshape(left, 1), np.arange(right, dtype=np.int32).reshape(1, right)
        else:
            x, y = np.arange(left, dtype=np.int32), np.arange(right, dtype=np.int32)
        expected = {"add": np.add, "sub": np.subtract, "mul": np.multiply}[operation](x, y)
        arrays, contract = invocation(shapes, [x, y], result_shape, expected)
        (actual,), report = execute_native(source, arrays, tmp_path / f"run-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract,
            compiler_support_root=os.environ.get("T2M_MERLIN_ROOT"))
        assert (actual.shape, actual.dtype, actual.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
        assert report["native_mlir_verified"] and report["llvm_lowering_verified"]


@pytest.mark.parametrize("bounds", [(0, 0, 1), (1, None, 2), (-8, -1, 3), (None, 5, 1), (-100, 100, 2),
                                    (-(1 << 63), (1 << 63)-1, (1 << 63)-1)])
def test_actual_slice_body_clips_bounds_without_example_specialization(tmp_path, bounds):
    from t2m.native import execute_native

    start, end, step = bounds
    result_shape = [0] if start == end == 0 else [None]
    source = emitter_program(tmp_path, [[None]], slicing=[0, start, end, step], result_shape=result_shape)
    for i, length in enumerate((0, 1, 7, 31)):
        x = np.arange(length, dtype=np.int32) * 17 - 9
        expected = x[slice(start, end, step)].copy()
        arrays, contract = invocation([[None]], [x], result_shape, expected)
        (actual,), _report = execute_native(source, arrays, tmp_path / f"run-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract,
            compiler_support_root=os.environ.get("T2M_MERLIN_ROOT"))
        assert (actual.shape, actual.dtype, actual.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())


@pytest.mark.parametrize("lengths", [(2, 3), (0, 2)])
def test_incompatible_runtime_broadcast_aborts_before_publishing_results(tmp_path, lengths):
    from t2m.native import observe_native

    source = emitter_program(tmp_path, [[None], [None]], result_shape=[None])
    x, y = (np.arange(length, dtype=np.int32) for length in lengths)
    arrays, contract = invocation([[None], [None]], [x, y], [None], np.zeros(3, np.int32))
    work = tmp_path / "invalid"
    report = observe_native(source, arrays, work, python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract,
                            compiler_support_root=os.environ.get("T2M_MERLIN_ROOT"))
    assert report["returncode"] == -6
    assert b"dynamic broadcast: incompatible operand dimensions" in (work / "stdout.bin").read_bytes()
    assert not (work / "outputs.npz").exists()

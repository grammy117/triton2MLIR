"""Execute m2m's actual dimension-query permutation through native Merlin."""

import os
import subprocess

import numpy as np
import pytest

TOOLS = all(os.environ.get(name) for name in ("T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing independent m2m and native Merlin environments required")

SOURCE = '''import sys
from xdsl.dialects import builtin,func
from xdsl.ir import Block,Region
from xdsl.printer import Printer
from m2m.ir.permutations import lower
element=builtin.i64 if sys.argv[1]=='i64' else builtin.f32
typ=builtin.TensorType(element,[builtin.DYNAMIC_INDEX,3,builtin.DYNAMIC_INDEX])
body=Block(arg_types=[typ])
ops,result=lower(body.args[0],[2,0,1],[-1,-1,3])
body.add_ops([*ops,func.ReturnOp(result)])
entry=func.FuncOp('forward',([typ],[result.type]),Region(body))
entry.properties['arg_attrs']=builtin.ArrayAttr([builtin.DictionaryAttr({'bufferization.writable':builtin.IntegerAttr(0,builtin.i1)})])
module=builtin.ModuleOp([entry])
module.verify()
Printer(print_generic_format=True).print_op(module)
'''

SLICE_SOURCE = '''import sys
from xdsl.dialects import builtin,func
from xdsl.ir import Block,Region
from xdsl.printer import Printer
from m2m.ir.dynamic_slices import extract_slice
element=builtin.i64 if sys.argv[1]=='i64' else builtin.f32
typ=builtin.TensorType(element,[builtin.DYNAMIC_INDEX,builtin.DYNAMIC_INDEX])
body=Block(arg_types=[typ])
results=[]
for arguments,shape in (((1,2,None,3),[-1,-1]),((0,0,None,2),[-1,-1]),((0,0,0,1),[0,-1])):
 target=builtin.TensorType(element,[builtin.DYNAMIC_INDEX if n<0 else n for n in shape])
 ops,result=extract_slice(body.args[0],target,arguments)
 body.add_ops(ops)
 results.append(result)
body.add_op(func.ReturnOp(*results))
entry=func.FuncOp('forward',([typ],[value.type for value in results]),Region(body))
entry.properties['arg_attrs']=builtin.ArrayAttr([builtin.DictionaryAttr({'bufferization.writable':builtin.IntegerAttr(0,builtin.i1)})])
module=builtin.ModuleOp([entry])
module.verify()
Printer(print_generic_format=True).print_op(module)
'''


@pytest.mark.parametrize("dtype", ("i64", "f32"))
def test_actual_permutation_reuses_one_library_for_independent_empty_and_unit_extents(tmp_path, monkeypatch, dtype):
    monkeypatch.syspath_prepend(os.environ["T2M_MERLIN_ROOT"] + "/src")
    from merlin.llvmlower.owned_results import compile_owned_results

    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], "-c", SOURCE, dtype],
        capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    (tmp_path / "model.mlir").write_text(done.stdout)
    program = compile_owned_results(done.stdout, workdir=tmp_path / "native")
    for shape in ((2, 3, 5), (0, 3, 7), (1, 3, 1), (9, 3, 2)):
        if dtype == "i64":
            original = (np.arange(np.prod(shape), dtype=np.int64) + (1 << 56) + 1).reshape(shape)
        else:
            words = np.asarray([0x7f800001, 0xffc12345, 0x80000000, 1, 0x7f800000, 0xff800000, 0x3f800000], np.uint32)
            original = np.resize(words, np.prod(shape)).view(np.float32).reshape(shape)
        before = original.tobytes()
        actual, report = program.execute([original])
        expected = original.transpose(2, 0, 1)
        assert (actual[0].shape, actual[0].dtype, actual[0].tobytes()) == (
            expected.shape, expected.dtype, expected.tobytes())
        assert original.tobytes() == before
        assert report["source_calls"] == 1


@pytest.mark.parametrize("dtype", ("i64", "f32"))
def test_actual_slices_preserve_words_and_empty_boundary_shapes(tmp_path, monkeypatch, dtype):
    monkeypatch.syspath_prepend(os.environ["T2M_MERLIN_ROOT"] + "/src")
    from merlin.llvmlower.owned_results import compile_owned_results

    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], "-c", SLICE_SOURCE, dtype],
        capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    (tmp_path / "model.mlir").write_text(done.stdout)
    program = compile_owned_results(done.stdout, workdir=tmp_path / "native")
    for shape in ((2, 2), (0, 0), (0, 5), (5, 0), (1, 1), (9, 17)):
        if dtype == "i64":
            original = (np.arange(np.prod(shape), dtype=np.int64) + (1 << 56) + 1).reshape(shape)
        else:
            words = np.asarray([0x7f800001, 0xffc12345, 0x80000000, 1, 0x7f800000, 0xff800000], np.uint32)
            original = np.resize(words, np.prod(shape)).view(np.float32).reshape(shape)
        before = original.tobytes()
        actual, report = program.execute([original])
        for observed, expected in zip(actual, (original[:, 2::3], original[::2], original[:0]), strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
        assert original.tobytes() == before
        assert report["source_calls"] == 1

"""Actual m2m selection executes with exact words and original index failures."""

import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

TOOLS = all(os.environ.get(name) for name in ("T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing independent frontend/native environments required")

SOURCE = '''import sys
from xdsl.dialects import builtin,func
from xdsl.ir import Block,Region
from xdsl.printer import Printer
from m2m.ir.tensor_selection import select
element=builtin.i64 if sys.argv[1]=='i64' else builtin.f32
typ=builtin.TensorType(element,[builtin.DYNAMIC_INDEX,1,builtin.DYNAMIC_INDEX])
body=Block(arg_types=[typ])
value=body.args[0]
steps={'row':[(0,1)],'column':[(2,-1)],'scalar':[(0,1),(0,0),(0,-1)]}[sys.argv[2]]
for axis,index in steps:
 shape=[n for i,n in enumerate(value.type.get_shape()) if i!=axis]
 operations,value=select(value,builtin.TensorType(element,shape),axis,index)
 body.add_ops(operations)
body.add_op(func.ReturnOp(value))
entry=func.FuncOp('forward',([typ],[value.type]),Region(body))
entry.properties['arg_attrs']=builtin.ArrayAttr([builtin.DictionaryAttr({'bufferization.writable':builtin.IntegerAttr(0,builtin.i1)})])
module=builtin.ModuleOp([entry]);module.verify()
Printer(print_generic_format=True).print_op(module)
'''


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    sys.path.insert(0, os.environ["T2M_MERLIN_ROOT"] + "/src")
    from merlin.compiler.lower.options.owned_result_program import save
    from merlin.compiler.lower.options.owned_results import compile_owned_results

    root = tmp_path_factory.mktemp("tensor-selection-native")
    made = {}
    for dtype in ("i64", "f32"):
        for kind in ("row", "column", "scalar"):
            done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], "-c", SOURCE, dtype, kind],
                capture_output=True, text=True, timeout=30, check=False)
            assert done.returncode == 0, done.stderr
            folder = root / (dtype + "-" + kind)
            program = compile_owned_results(done.stdout, workdir=folder)
            save(program)
            made[dtype, kind] = program
    return made


@pytest.mark.parametrize("dtype", ("i64", "f32"))
@pytest.mark.parametrize("kind", ("row", "column", "scalar"))
def test_retained_unit_axes_scalar_and_empty_results_preserve_exact_words(programs, dtype, kind):
    shapes = [(3, 1, 5), (2, 1, 1), (9, 1, 17)]
    if kind == "row":
        shapes.append((2, 1, 0))
    if kind == "column":
        shapes.extend(((0, 1, 5), (1, 1, 3)))
    for shape in shapes:
        if dtype == "i64":
            value = (np.arange(np.prod(shape), dtype=np.int64) + (1 << 56) + 1).reshape(shape)
        else:
            words = np.asarray([0x7f800001, 0xffc12345, 0x80000000, 1, 0x7f800000, 0xff800000], np.uint32)
            value = np.resize(words, np.prod(shape)).view(np.float32).reshape(shape)
        if kind == "row":
            expected = value[1, :, :]
        elif kind == "column":
            expected = value[:, :, -1]
        else:
            expected = np.asarray(value[1, 0, -1])
        before = value.tobytes()
        (actual,), _ = programs[dtype, kind].execute((value,))
        assert (actual.shape, actual.dtype, actual.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
        assert value.tobytes() == before


@pytest.mark.parametrize("kind,shape", (("row", (1, 1, 5)), ("column", (3, 1, 0)), ("scalar", (3, 1, 0))))
def test_out_of_bounds_indices_abort_instead_of_clamping_or_returning_an_empty_value(programs, tmp_path, kind, shape):
    inputs, ready, outputs = tmp_path / "inputs.npz", tmp_path / "ready", tmp_path / "outputs.npz"
    np.savez(inputs, in0=np.empty(shape, np.int64))
    script = tmp_path / "invoke.py"
    script.write_text(r"""import sys,ctypes,resource
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1]+'/src')
from merlin.compiler.lower.options.owned_result_program import load
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
libc=ctypes.CDLL(None)
libc.setbuf.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
libc.setbuf(ctypes.c_void_p.in_dll(libc,'stdout'),None)
program=load(Path(sys.argv[2])/'owned-result-program.json')
with np.load(sys.argv[3],allow_pickle=False) as data: value=data['in0'].copy()
Path(sys.argv[4]).write_text('invoking')
values,_=program.execute((value,))
np.savez(sys.argv[5],**{f'out{i}':v for i,v in enumerate(values)})
""")
    done = subprocess.run([sys.executable, str(script), os.environ["T2M_MERLIN_ROOT"],
        str(programs["i64", kind].source.parent), str(inputs), str(ready), str(outputs)],
        capture_output=True, text=True, timeout=20, check=False)
    assert ready.read_text() == "invoking"
    assert done.returncode == -signal.SIGABRT and "selection index" in done.stdout + done.stderr
    assert not outputs.exists()
    (tmp_path / "outcome.json").write_text(json.dumps({"returncode": done.returncode,
        "stdout": done.stdout, "stderr": done.stderr}, indent=2) + "\n")

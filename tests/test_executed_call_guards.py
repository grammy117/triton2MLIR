"""Actual compiled failures of invocation-checked call adapters.

These standard-MLIR harnesses test adapters, not TTIR source equivalence.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.ir import Block, Region

from t2m.executed_model_arguments import arguments, cast_index, cast_tensor
from t2m.ir import text
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

pytestmark = pytest.mark.skipif(
    not os.environ.get("T2M_MERLIN_ROOT")
    or not os.environ.get("MERLIN_COMPILER_PYTHON"),
    reason="existing Merlin native compiler required",
)


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    sys.path.insert(0, str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"))
    from merlin.llvmlower.owned_result_program import save
    from merlin.llvmlower.owned_results import compile_owned_results

    root = tmp_path_factory.mktemp("executed-call-guards")
    vector = builtin.TensorType(builtin.i32, [builtin.DYNAMIC_INDEX])
    made = {}
    for kind in ("dimensions", "index"):
        block = Block(arg_types=[vector, builtin.i64])

        def append(op, _role, *, block=block):
            block.add_op(op)
            return op.results[0] if op.results else None

        x, n = block.args
        if kind == "dimensions":
            spec = TensorABI(
                "adapter",
                (
                    TensorBuffer("x", ("n",), dtype="i32"),
                    TensorBuffer("out", ("n",), "write", "i32"),
                ),
                runtime_scalars=(RuntimeScalar("n", "i64", 1, bounds=(0, 31)),),
            )
            binding = SimpleNamespace(
                implementation=SimpleNamespace(spec=spec),
                runtime_arguments={},
                runtime_operands={"n": "n"},
            )
            interface = func.FuncOp.external(
                "interface",
                [buffer.type for buffer in spec.inputs],
                [buffer.type for buffer in spec.outputs],
            )
            values, _ = arguments(
                binding,
                interface,
                {"x": "x"},
                {"x": x, "n": n},
                builtin.DictionaryAttr({}),
                append,
            )
            result = cast_tensor(
                append, values[0], builtin.TensorType(builtin.i32, [4])
            )
        else:
            zero = append(
                arith.ConstantOp.from_int_and_width(0, builtin.IndexType()), "axis"
            )
            extent = append(tensor.DimOp(x, zero), "extent")
            cast_index(append, extent, builtin.i8)
            result = x
        block.add_op(func.ReturnOp(result))
        entry = func.FuncOp.from_region(
            "forward", [vector, builtin.i64], [result.type], Region(block)
        )
        program = compile_owned_results(
            text(builtin.ModuleOp([entry])), workdir=root / kind, timeout=30
        )
        save(program)
        made[kind] = program
    return made


def test_valid_actual_dimensions_and_lossless_index_transport(programs):
    from merlin.llvmlower.abi import ScalarArg

    x = np.arange(4, dtype=np.int32)
    (out,), observation = programs["dimensions"].execute((x, ScalarArg(4, "i64")))
    np.testing.assert_array_equal(out, x)
    assert observation["source_calls"] == 1
    for size in (0, 1, 127):
        x = np.arange(size, dtype=np.int32)
        (out,), _ = programs["index"].execute((x, ScalarArg(0, "i64")))
        np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize(
    "kind,size,n,message",
    (
        ("dimensions", 4, 3, "tensor dimension differs from scalar"),
        ("dimensions", 3, 3, "tensor narrowing changes a static dimension"),
        ("dimensions", 4, -1, "runtime scalar outside declared bounds"),
        ("dimensions", 4, 32, "runtime scalar outside declared bounds"),
        ("index", 128, 0, "integer/index conversion is not representable"),
        ("index", 256, 0, "integer/index conversion is not representable"),
    ),
)
def test_invalid_actual_contract_aborts_before_publishing_results(
    programs, tmp_path, kind, size, n, message
):
    script = tmp_path / "invoke.py"
    script.write_text("""import ctypes,resource,sys
from pathlib import Path
import numpy as np
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_program import load
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
libc=ctypes.CDLL(None)
libc.setbuf.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
libc.setbuf(ctypes.c_void_p.in_dll(libc,"stdout"),None)
program=load(Path(sys.argv[1])/"owned-result-program.json")
Path(sys.argv[4]).write_text("invoking")
outputs,_=program.execute((np.arange(int(sys.argv[2]),dtype=np.int32),ScalarArg(int(sys.argv[3]),"i64")))
np.savez(sys.argv[5],out0=outputs[0])
""")
    started, outputs = tmp_path / "started", tmp_path / "outputs.npz"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            str(programs[kind].source.parent),
            str(size),
            str(n),
            str(started),
            str(outputs),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert started.read_text() == "invoking"
    assert done.returncode == -signal.SIGABRT, done.stdout + done.stderr
    assert message in done.stdout + done.stderr
    assert not outputs.exists()

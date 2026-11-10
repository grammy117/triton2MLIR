"""FP8 reader legality and exact encodings, independently of numeric lowering."""

import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from t2m.ir import text
from t2m.shared_tensor import parse_shared
from t2m.standard_floats import FP8_TYPE_NAMES, NUMPY_NAMES, _FP8_BY_NAME


def constant(dtype, value, size):
    return (
        f"module {{func.func @f()->tensor<{size}x{dtype}> {{"
        f"%v=arith.constant {value}:tensor<{size}x{dtype}> "
        f"return %v:tensor<{size}x{dtype}>}}}}"
    )


def data(module):
    module.verify()
    return next(
        op for op in module.walk() if op.name == "arith.constant"
    ).value.data.data


@pytest.mark.parametrize("dtype", FP8_TYPE_NAMES)
def test_all_words_and_constants_roundtrip(dtype):
    pytest.importorskip("ml_dtypes")
    type_ = _FP8_BY_NAME[dtype]
    raw = bytes(range(256))
    assert type_.pack(type_.unpack(raw, 256)) == raw
    for value in (
        'dense<"0x' + raw.hex() + '">',
        "dense<[" + ",".join(f"0x{word:02X}" for word in raw) + "]>",
    ):
        module = parse_shared(constant(dtype, value, 256))
        assert data(module) == raw
        assert data(parse_shared(text(module))) == raw
    # Scalar FloatAttr must retain the raw word too; a Python float cannot
    # carry the original NaN payload, so round-tripping through it is lossy.
    source = (
        "module {func.func @f() {"
        + "\n".join(f"%v{word}=arith.constant 0x{word:02X}:{dtype}" for word in raw)
        + "\nreturn}}"
    )
    module = parse_shared(source)
    module.verify()
    again = parse_shared(text(module))
    again.verify()
    assert (
        b"".join(
            type_.pack([op.value.value.data])
            for op in again.walk()
            if op.name == "arith.constant"
        )
        == raw
    )


@pytest.mark.parametrize("dtype", FP8_TYPE_NAMES)
def test_decimal_constants_are_rounded_to_the_declared_format(dtype):
    ml = pytest.importorskip("ml_dtypes")
    from xdsl.dialects.builtin import FloatAttr

    values = [0.0, -0.0, 1.0, -1.0, 1.0625, 1.125, 1.1875, 2.0**-16]
    expected = np.asarray(values, dtype=getattr(ml, NUMPY_NAMES[dtype])).tobytes()
    module = parse_shared(
        constant(dtype, "dense<[" + ",".join(map(str, values)) + "]>", len(values))
    )
    assert data(module) == expected
    type_ = _FP8_BY_NAME[dtype]
    scalar = FloatAttr(1.1, type_)
    assert scalar.value.data == float(
        np.asarray(1.1, dtype=getattr(ml, NUMPY_NAMES[dtype]))
    )


def test_registration_preserves_float_legality_and_is_idempotent():
    from t2m.standard_floats import register_fp8_types, register_fp8_float_constraints
    from xdsl.dialects import builtin
    from xdsl.irdl import ConstraintContext
    from xdsl.irdl.constraints import BaseAttr
    from xdsl.utils.exceptions import VerifyException

    for _ in range(3):
        register_fp8_types()
        register_fp8_float_constraints()
    for dtype in FP8_TYPE_NAMES:
        module = parse_shared(
            f"module {{func.func @f(%x:{dtype})->f32 {{"
            f"%y=arith.extf %x:{dtype} to f32 return %y:f32}}}}"
        )
        module.verify()
    with pytest.raises(VerifyException):
        builtin.AnyFloatConstr.verify(builtin.i8, ConstraintContext())
    # A deliberately restricted f16/f32 set must not widen with AnyFloat.
    with pytest.raises(VerifyException):
        (BaseAttr(builtin.Float16Type) | BaseAttr(builtin.Float32Type)).verify(
            _FP8_BY_NAME[FP8_TYPE_NAMES[0]], ConstraintContext()
        )


@pytest.mark.parametrize("first", ["t2m", "merlin"])
def test_merlin_and_frontend_share_types_in_either_import_order(first):
    root = os.environ.get("T2M_MERLIN_ROOT")
    if not root:
        pytest.skip("Merlin checkout required")
    script = f"""
import sys
sys.path.insert(0, {str(Path(root) / "src")!r})
if {first!r} == 'merlin':
    import merlin.xdsl_dialects.fp8
else:
    import t2m.standard_floats
from t2m.shared_tensor import parse_shared
from t2m.ir import text
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as merlin_text
from t2m.standard_floats import _FP8_BY_NAME
from merlin.xdsl_dialects.fp8 import _FP8_BY_NAME as consumer_types
for name, type_ in _FP8_BY_NAME.items():
    assert type_ is consumer_types[name]
    source = 'module {{func.func @f()->tensor<256x'+name+'> {{%v=arith.constant dense<"0x'+bytes(range(256)).hex()+'">:tensor<256x'+name+'> return %v:tensor<256x'+name+'>}}}}'
    module = parse_mlir_text(text(parse_shared(source)))
    module.verify()
    again = parse_shared(merlin_text(module));again.verify()
    assert next(op for op in again.walk() if op.name=='arith.constant').value.data.data == bytes(range(256))
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr[-3000:]


def test_native_parser_accepts_exact_dense_encodings():
    python = os.environ.get("T2M_NATIVE_PYTHON")
    if not python:
        pytest.skip("native MLIR bindings required")
    modules = [
        text(
            parse_shared(
                constant(dtype, 'dense<"0x' + bytes(range(256)).hex() + '">', 256)
            )
        )
        for dtype in FP8_TYPE_NAMES
    ]
    done = subprocess.run(
        [
            python,
            "-c",
            "from torch_mlir import ir;import json,sys\n"
            "results=[]\nwith ir.Context():\n for source in json.load(sys.stdin):\n"
            "  module=ir.Module.parse(source);assert module.operation.verify();results.append(str(module))\n"
            "print(json.dumps(results))\n",
        ],
        input=json.dumps(modules),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    for source in json.loads(done.stdout):
        assert data(parse_shared(source)) == bytes(range(256))


def test_type_only_intake_does_not_require_the_optional_codec():
    script = """
import builtins
original = builtins.__import__
def load(name, *args, **kwargs):
    if name == 'ml_dtypes': raise ImportError('optional codec absent')
    return original(name, *args, **kwargs)
builtins.__import__ = load
from t2m.shared_tensor import parse_shared
from t2m.standard_floats import _FP8_BY_NAME
for name, type_ in _FP8_BY_NAME.items():
    module=parse_shared('module {func.func @f(%x:tensor<4x'+name+'>)->tensor<4x'+name+'> {return %x:tensor<4x'+name+'>}}')
    module.verify()
    try: type_.pack([1.0])
    except ValueError as error: assert 'optional ml-dtypes' in str(error)
    else: raise AssertionError('codec dependency was bypassed')
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr[-3000:]

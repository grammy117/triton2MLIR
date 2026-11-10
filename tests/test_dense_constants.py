"""Dense storage words survive splat classification, printing and both readers."""

import os
from pathlib import Path
import subprocess

import pytest

from t2m.ir import text
from t2m.shared_tensor import parse_shared

FORMATS = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "f8E4M3FN": 1, "f8E5M2": 1}


def program(dtype, raw):
    return f'module {{func.func @f()->tensor<2x{dtype}> {{%v=arith.constant dense<"0x{raw.hex()}">:tensor<2x{dtype}> return %v:tensor<2x{dtype}>}}}}'


@pytest.mark.parametrize("dtype,size", FORMATS.items())
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_zero_signs_are_never_serialized_as_one_splat(dtype, size, reverse):
    if size == 1:
        pytest.importorskip("ml_dtypes")
    words = [bytes(size), (1 << (size * 8 - 1)).to_bytes(size, "little")]
    raw = b"".join(reversed(words) if reverse else words)
    module = parse_shared(program(dtype, raw))
    value = next(op.value for op in module.walk() if op.name == "arith.constant")
    assert not value.is_splat()
    restored = parse_shared(text(module))
    assert (
        next(
            op.value.data.data for op in restored.walk() if op.name == "arith.constant"
        )
        == raw
    )


@pytest.mark.parametrize("dtype,size", FORMATS.items())
def test_real_splats_and_integer_storage_keep_existing_behavior(dtype, size):
    if size == 1:
        pytest.importorskip("ml_dtypes")
    raw = (1 << (size * 8 - 1)).to_bytes(size, "little") * 2
    module = parse_shared(program(dtype, raw))
    value = next(op.value for op in module.walk() if op.name == "arith.constant")
    assert value.is_splat()
    assert (
        next(
            op.value.data.data
            for op in parse_shared(text(module)).walk()
            if op.name == "arith.constant"
        )
        == raw
    )


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native SDK required"
)
@pytest.mark.parametrize("dtype,size", FORMATS.items())
def test_mixed_zero_words_survive_actual_native_execution(tmp_path, dtype, size):
    from t2m.native import execute_native

    if size == 1:
        pytest.importorskip("ml_dtypes")
    if dtype == "bf16" and not os.environ.get("T2M_MERLIN_ROOT"):
        pytest.skip("explicit BF16 transport support required")
    raw = bytes(size) + (1 << (size * 8 - 1)).to_bytes(size, "little")
    module = parse_shared(program(dtype, raw))
    output, _ = execute_native(
        text(module), [], tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"]
    )
    assert output[0].tobytes() == raw


@pytest.mark.skipif(
    not os.environ.get("T2M_MERLIN_ROOT"), reason="cooperating Merlin reader required"
)
@pytest.mark.parametrize("first", ["merlin", "t2m"])
@pytest.mark.parametrize(
    "python", [os.environ.get("T2M_MERLIN_PYTHON"), os.environ.get("T2M_NATIVE_PYTHON")]
)
def test_shared_hook_is_idempotent_in_both_reader_orders_and_versions(
    tmp_path, first, python
):
    if not python:
        pytest.skip("explicit frontend/native Python required")
    script = tmp_path / "check.py"
    script.write_text("""import sys,json
sys.path[:0]=sys.argv[1:3]
if sys.argv[3]=='merlin':
    import merlin.xdsl_dialects.fp8
from t2m.shared_tensor import parse_shared
from t2m.ir import text
from t2m.standard_floats import register_fp8_types
register_fp8_types()
import merlin.xdsl_dialects.fp8 as consumer
consumer.register_fp8_types()
from xdsl.dialects import builtin
before=builtin.DenseIntOrFPElementsAttr.is_splat
register_fp8_types();consumer.register_fp8_types()
assert builtin.DenseIntOrFPElementsAttr.is_splat is before
source='module {func.func @f()->tensor<2xf32> {%v=arith.constant dense<[0.0,-0.0]>:tensor<2xf32> return %v:tensor<2xf32>}}'
module=parse_shared(source)
value=next(op.value for op in module.walk() if op.name=='arith.constant')
assert not value.is_splat()
restored=parse_shared(text(module))
assert next(op.value.data.data for op in restored.walk() if op.name=='arith.constant')==value.data.data
print('VERIFIED')
""")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            str(python),
            str(script),
            str(root),
            str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"),
            first,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "VERIFIED"

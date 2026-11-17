"""Checked parent index values retain words through real translated kernels."""

import os
import signal
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import func

from t2m.diagnostics import ConversionError
from t2m.model_binding import KernelBinding, bind_model
from t2m.model_live_arguments import RANGE_PROFILE, validate_model
from t2m.shared_tensor import parse_shared
from t2m.source_index_ranges import unsigned_index_range


def source(upper=127):
    return f'''module {{
      func.func @forward(%x:tensor<13xi32>,%control:tensor<1xi64>)->tensor<13xi32> {{
        %zero=arith.constant 0:index
        %raw=tensor.extract %control[%zero]:tensor<1xi64>
        %index=arith.index_castui %raw:i64 to index
        %word=arith.index_castui %index:index to i64
        %restored=arith.index_castui %word:i64 to index
        %equal=arith.cmpi eq,%index,%restored:index
        cf.assert %equal,"source index round trip"
        %two=arith.constant 2:i4
        %lower_bound=arith.extsi %two:i4 to i64
        %upper_bound=arith.constant {upper}:i64
        %lower=arith.cmpi sge,%word,%lower_bound:i64
        %upper=arith.cmpi sle,%word,%upper_bound:i64
        %both=arith.andi %lower,%upper:i1
        cf.assert %both,"source control domain"
        %out=func.call @kernel(%x,%index) {{prov.kernel="test::checked",prov.kernel_operands=["x","control"]}}:
          (tensor<13xi32>,index)->tensor<13xi32>
        return %out:tensor<13xi32>
      }}
      func.func private @kernel(tensor<13xi32>,index)->tensor<13xi32> attributes {{prov.kernel="test::checked"}}
    }}'''


@pytest.fixture(scope="module")
def implementation(tmp_path_factory):
    if not os.environ.get("TRITON_SHARED_OPT"):
        pytest.skip("existing shared optimizer required")
    from test_live_scalar_model import implementation as translated

    return translated(tmp_path_factory.mktemp("checked-index-kernel"), "i32", profile="tensor")


def bind(implementation, original=None):
    return bind_model(source() if original is None else original, [
        KernelBinding("test::checked", implementation, runtime_operands={"control": "control"})])


def test_checked_cast_retains_the_original_public_abi_and_executable_guards(implementation):
    result = bind(implementation)
    assert result.trace["argument_binding_profile"] == RANGE_PROFILE
    record = result.trace["calls"][0]["runtime_operand_bindings"]["control"]
    assert record == {"schema_operand": "control", "caller_type": "index", "kernel_type": "i32",
                      "adapter": "checked-index-castui", "unsigned_integer_range": [2, 127]}
    assert "cf.assert" in result.mlir_text
    assert "index_castui" in result.mlir_text
    validate_model(result.trace, parse_shared(source()))


@pytest.mark.parametrize("damage", ["profile", "range", "constant", "adapter", "extra", "missing"])
def test_binding_records_cannot_replace_the_original_executable_proof(implementation, damage):
    trace = deepcopy(bind(implementation).trace)
    record = trace["calls"][0]["runtime_operand_bindings"]["control"]
    if damage == "profile": trace["argument_binding_profile"] = "t2m.live-runtime-arguments.v1"
    elif damage == "range": record["unsigned_integer_range"] = [2, 128]
    elif damage == "constant": record["constant_integer"] = 7
    elif damage == "adapter": record["adapter"] = "proved-index-cast"
    elif damage == "extra": record["shape_permission"] = True
    else: record.pop("unsigned_integer_range")
    with pytest.raises(ValueError):
        validate_model(trace, parse_shared(source()))


def test_destination_signed_width_and_dominance_remain_required(implementation):
    with pytest.raises(ConversionError, match="representable"):
        bind(implementation, source(1 << 31))
    delayed = source().replace('cf.assert %both,"source control domain"', '').replace(
        'return %out:tensor<13xi32>', 'cf.assert %both,"source control domain"\n return %out:tensor<13xi32>')
    with pytest.raises(ConversionError, match="representable"):
        bind(implementation, delayed)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
def test_compiled_cast_keeps_runtime_values_and_original_domain_aborts(tmp_path, implementation):
    from t2m.native import execute_native, observe_native

    result = bind(implementation)
    x = np.arange(13, dtype=np.int32) * 17 - 83
    for control in (2, 7, 127):
        (actual,), report = execute_native(result.mlir_text, [x, np.array([control], np.int64)],
            tmp_path / f"valid-{control}", python=os.environ["T2M_NATIVE_PYTHON"])
        np.testing.assert_array_equal(actual, np.bitwise_xor(x, control))
        assert report["native_mlir_verified"] and report["llvm_lowering_verified"]
    for control in (-1, 1, 128, 1 << 40):
        work = tmp_path / f"invalid-{control}"
        report = observe_native(result.mlir_text, [x, np.array([control], np.int64)], work,
                                python=os.environ["T2M_NATIVE_PYTHON"])
        assert report["returncode"] == -signal.SIGABRT
        assert b"source control domain" in (work / "stdout.bin").read_bytes() + (work / "stderr.bin").read_bytes()
        assert not (work / "outputs.npz").exists() and not (work / "execution.json").exists()


@pytest.mark.skipif(not os.environ.get("T2M_MODEL2MLIR_PYTHON"), reason="existing model2MLIR environment required")
def test_actual_export_guards_bound_both_calls_using_the_original_entry_dimension(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/capture_live_scalar_model.py"
    capture = tmp_path / "parent"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--extent", "--dtype", "i32"], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    parent = parse_shared((capture / "model.mlir").read_text())
    parent.verify()
    calls = [op for op in parent.walk() if isinstance(op, func.CallOp)
             and "prov.kernel" in op.attributes]
    assert len(calls) == 3
    assert unsigned_index_range(calls[0].operands[1], calls[0]) == (2, 31)
    # Torch export canonicalizes both shape operands to this original entry
    # query. This equality is actual SSA, rather than a sampled shape claim.
    assert calls[1].operands[1] is calls[0].operands[1]
    assert unsigned_index_range(calls[1].operands[1], calls[1]) == (2, 31)

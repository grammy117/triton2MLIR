"""Actual root/view model captures through execution-owned correspondence."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_source_view_model import TOOLS, bindings, check_structure, implementation
from test_spike_model import TOOLS as TARGET_TOOLS
from test_spike_model import Q

from t2m.bundle import verify_bundle
from t2m.model_bundle import write_model_bundle


def check_forged_views(package, root):
    from t2m.bundle import _record

    trace_path, receipt_path = (
        package / "kernel-bindings.json",
        package / "capture_receipt.json",
    )
    original_trace, original_receipt = (
        trace_path.read_bytes(),
        receipt_path.read_bytes(),
    )
    worker = r"""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/src')
from merlin.compiler.frontend.inputs import stage_compile_inputs
read=Path.read_bytes
def checked(path):
    if path.name in ('inputs.npz','golden.npy','goldens.npz','extra.npz','tensor_words.npz','execution.json'):
        raise AssertionError('compiler read evaluator data')
    return read(path)
Path.read_bytes=checked
try: stage_compile_inputs(sys.argv[2],sys.argv[3],result_mode='execution-owned')
except ValueError: pass
else: raise AssertionError('forged executed view accepted')
"""
    try:
        for change in (
            "profile",
            "unknown-profile",
            "layout",
            "source-pointer",
            "missing-call",
            "policy",
        ):
            trace, receipt = json.loads(original_trace), json.loads(original_receipt)
            call = trace["calls"][0]
            if change == "profile":
                trace.pop("execution_view_binding_profile")
            elif change == "unknown-profile":
                trace["execution_view_binding_profile"] = "unchecked"
            elif change == "layout":
                call["source_tensor_views"]["view"]["layout"]["offset"] = 0
            elif change == "source-pointer":
                call["source_tensor_views"]["view"]["binding"]["source_arg"] = 0
            elif change == "missing-call":
                call.pop("source_tensor_views")
            else:
                call["tensor_views"] = "physical-alias"
            trace_path.write_text(json.dumps(trace))
            receipt["artifacts"]["kernel-bindings.json"] = _record(trace_path)
            receipt_path.write_text(json.dumps(receipt))
            with pytest.raises(ValueError):
                verify_bundle(package)
            output = root / ("refused-" + change)
            process = subprocess.run(
                [
                    os.environ["T2M_MERLIN_PYTHON"],
                    "-c",
                    worker,
                    os.environ["T2M_MERLIN_ROOT"],
                    str(package),
                    str(output),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert process.returncode == 0, process.stderr.decode()
            assert not output.exists()
    finally:
        trace_path.write_bytes(original_trace)
        receipt_path.write_bytes(original_receipt)


def compaction(root):
    from t2m.examples.kernel_size_kernels import positive_count, positive_pack
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    bindings = []
    for kind, kernel in (
        ("positive_count", positive_count),
        ("compact", positive_pack),
    ):
        source = root / (kind + ".ttir")
        packing = kind == "compact"
        types = {
            "x": "*fp32",
            "out" if packing else "count": "*fp32" if packing else "*i64",
            "capacity": "i64",
            "B": "constexpr",
        }
        if packing:
            types["n"] = "i64"
        source.write_text(capture_ttir(kernel, types, {"B": 128}))
        spec = TensorABI(
            kernel.__name__,
            (
                TensorBuffer("x", (96,), dtype="f32"),
                TensorBuffer(
                    "out",
                    ("n",) if packing else (1,),
                    "write",
                    "f32" if packing else "i64",
                ),
            ),
            runtime_scalars=(
                RuntimeScalar("capacity", "i64", 2, bounds=(0, 96)),
                *((RuntimeScalar("n", "i64", 3, bounds=(0, 96)),) if packing else ()),
            ),
        )
        result = convert_shared(
            source,
            spec,
            root / (kind + "-converted"),
            optimizer=os.environ["TRITON_SHARED_OPT"],
            memory_profile="auto",
        )
        bindings.append(
            KernelBinding(
                "t2m_view_demo::" + kind,
                result,
                operands={"x": "x"},
                scalar_arguments={"capacity": 96},
                runtime_arguments={"capacity": "capacity"},
                runtime_operands={"n": "n"} if packing else {},
            )
        )
    return bindings


pytestmark = pytest.mark.skipif(
    not TOOLS, reason="existing source/model/Merlin tools required"
)


@pytest.mark.parametrize("dtype", ("i64", "f32", "bf16"))
def test_original_source_views_and_all_snapshots_reach_owned_host_and_target(
    tmp_path, dtype, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_source_view_model.py"
    )
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(capture),
            "--dtype",
            dtype,
        ],
        capture_output=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert process.returncode == 0, process.stderr.decode()
    result = implementation(tmp_path, dtype)
    package = write_model_bundle(
        capture,
        bindings(result),
        tmp_path / "package",
        work=tmp_path / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True,
        result_mode="execution-owned",
    )
    assert verify_bundle(package)["schema"] == "t2m.bound-model-receipt.v2"
    check_structure(package)
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["schema"] == "t2m.model-kernel-binding.v4"
    assert (
        trace["execution_view_binding_profile"] == "t2m.execution-checked-call-views.v1"
    )
    assert len(trace["calls"]) == 3 and all(
        call["source_tensor_views"] for call in trace["calls"]
    )
    from xdsl.dialects import func

    from t2m.model_bundle import _parent, _references
    from t2m.shared_tensor import parse_shared

    entry = next(
        op
        for op in parse_shared((capture / "model.mlir").read_text()).ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    )
    references, _ = _references(capture, _parent(capture), entry.function_type.outputs)
    with np.load(package / "consumer/host/outputs.npz", allow_pickle=False) as outputs:
        assert len(outputs.files) == len(references) == 7
        for i, wanted in enumerate(references):
            actual = outputs[f"out{i}"]
            assert (actual.shape, actual.dtype, actual.tobytes()) == (
                wanted.shape,
                wanted.dtype,
                wanted.tobytes(),
            )
    if TARGET_TOOLS:
        report = Q.qualify(
            package,
            tmp_path / "target",
            result_mode="execution-owned",
            comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["evaluation"]["result_count"] == 7
        assert (
            Q.verify(
                tmp_path / "target",
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
            )
            == report
        )
    if dtype == "i64":
        check_forged_views(package, tmp_path)


def test_non_dense_updates_compose_with_content_sized_scan_scatter_results(
    tmp_path, monkeypatch
):
    from t2m.bundle import _record
    from t2m.consumer import invoke_merlin_host

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_source_view_model.py"
    )
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(capture),
            "--dtype",
            "f32",
            "--execution-sizes",
        ],
        capture_output=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert done.returncode == 0, done.stderr.decode()
    result = implementation(tmp_path, "f32")
    selected = [*bindings(result), *compaction(tmp_path)]
    package = write_model_bundle(
        capture,
        selected,
        tmp_path / "package",
        work=tmp_path / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True,
        result_mode="execution-owned",
    )
    verify_bundle(package)
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 5 and len(trace["kernels"]) == 3
    assert sum("source_tensor_views" in call for call in trace["calls"]) == 3
    library = _record(package / "consumer/host/model.so")
    source = _record(package / "consumer/compiler/program.mlir")
    with np.load(
        tmp_path / "qualification/arguments/arguments.npz", allow_pickle=False
    ) as values:
        original = {name: values[name].copy() for name in values.files}
    counts = []
    for ordinal, fill in enumerate((-100, 100, None)):
        values = {name: value.copy() for name, value in original.items()}
        if fill is not None:
            values["in2"].fill(fill)
        weight, bias, x = (values["in" + str(i)] for i in range(3))
        initial = x + weight
        first = initial.copy()
        first.T[1::2, :] += 10
        second = first.copy()
        second[::2, 2::3] -= 3
        full = second + bias
        packed = full.reshape(-1)[full.reshape(-1) > 0]
        wanted = (
            full,
            first,
            first.T[1::2, :].copy(),
            second[::2, 2::3].copy(),
            first.copy(),
            np.empty((0, 6), np.float32),
            initial.T[1::2, :].copy(),
            packed,
            np.asarray([packed.size], np.int64),
        )
        arguments, reference = (
            tmp_path / f"arguments-{ordinal}.npz",
            tmp_path / f"references-{ordinal}.npz",
        )
        np.savez(arguments, **values)
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(wanted)})
        actual, report = invoke_merlin_host(
            package / "consumer",
            arguments,
            tmp_path / f"invoke-{ordinal}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"]
        for value, expected in zip(actual, wanted, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (
                expected.shape,
                expected.dtype,
                expected.tobytes(),
            )
        counts.append(packed.size)
        if TARGET_TOOLS and ordinal == 0:
            report = Q.qualify(
                package,
                tmp_path / "target-empty",
                result_mode="execution-owned",
                comparison="words",
                inputs=arguments,
                reference=reference,
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
            )
            assert report["evaluation"]["result_count"] == 9
    assert counts[:2] == [0, 96] and 0 < counts[2] < 96
    assert _record(package / "consumer/host/model.so") == library
    assert _record(package / "consumer/compiler/program.mlir") == source

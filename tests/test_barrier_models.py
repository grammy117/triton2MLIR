"""Complete source models retain dependent program phases and all results."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_allocation_barriers import implementation

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/native/Merlin tools required")


@pytest.fixture(scope="module", params=("tensor", "guarded-buffer"))
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("barrier-model-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_barrier_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    kernel_root = root / "kernel"
    kernel_root.mkdir()
    kernel = implementation(kernel_root, request.param)
    bindings = [KernelBinding("t2m_demo::program_exchange", kernel, operands={"state": "x"}, results=("state", "out"),
        scalar_arguments={"steps": steps, **({"n": 0} if empty else {})},
        runtime_arguments={"steps": "steps", **({"n": "n"} if empty else {})},
        runtime_operands={} if empty else {"n": "n"}) for steps, empty in ((2, False), (3, False), (4, True))]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_complete_model_retains_source_phases_snapshots_and_every_result(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / name).read_bytes() == (package / destination).read_bytes()
    bindings = json.loads((package / "kernel-bindings.json").read_text())
    assert len(bindings["calls"]) == 3 and len(bindings["kernels"]) == 1
    for kernel in bindings["kernels"].values():
        assert any(row["kind"] == "protected-allocation-program-memory-phases" for row in kernel["transformations"])
    with np.load(capture / "goldens.npz", allow_pickle=False) as references, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(references.files) == len(actual.files) == 9
        for name in references.files:
            assert (actual[name].dtype, actual[name].shape, actual[name].tobytes()) == (
                references[name].dtype, references[name].shape, references[name].tobytes())


def reference(x, bias):
    def observe(state):
        result = np.empty_like(state)
        for origin in range(0, len(state), 4):
            size = min(4, len(state) - origin)
            tile = np.full((4, 4), -3, dtype=state.dtype)
            tile[:size] = state[origin : origin + size]
            result[origin : origin + size] = tile.T[:size]
        return result

    original = x + bias
    first = original + 3
    second = first + 6
    empty = np.empty((0, 4), dtype=x.dtype)
    return [second, first, observe(first), observe(second), original, x, empty, empty.copy(), bias]


def invocation(root, n, index):
    x = np.arange(n * 4, dtype=np.int64).reshape(n, 4) + (1 << 55) + index
    bias = np.asarray([index + 2, -9, 11, -4], dtype=np.int64)
    path = root / f"inputs-{index}.npz"
    np.savez(path, in0=bias, in1=x)
    return path, reference(x, bias)


def test_changed_runtime_rows_and_parameters_reuse_the_original_library(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    for index, rows in enumerate((2, 19, 11)):
        inputs, expected = invocation(root, rows, index)
        actual, observation = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert observation["functional_inputs_preserved"] is True
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_rehashed_phase_claims_and_payload_refuse_both_readers(selected):
    from copy import deepcopy
    from io import StringIO

    from test_executed_model_packages import rehash
    from xdsl.printer import Printer

    from t2m.consumer import check_merlin_interface
    from t2m.shared_tensor import opcode, parse_shared
    from t2m.source_capsule import record

    root, _capture, package = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json",
        "kernels/0/provenance.json", "kernels/0/allocation-protection.json", "kernels/0/allocation-protected.ttir",
        "kernels/0/allocation-source-stage.json")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        damages = ["missing", "mask", "prefix", "live", "scope", "lost-store", "phase-order"]
        protection = json.loads(original[paths[3]])
        if protection["barrier_conversions"][0]["contract"]["schema"] == "t2m.synchronous-program-barrier.v3":
            damages += ["volatile-selection", "volatile-primitive", "volatile-flag", "bound-volatile-profile"]
        for damage in damages:
            report = json.loads(original[paths[3]])
            changed = {"allocation-protection.json"}
            if damage == "missing":
                report.pop("barrier_conversions")
            elif damage == "mask":
                report["barrier_conversions"][0]["contract"]["address_space_mask"] = 0
            elif damage == "prefix":
                report["barrier_conversions"][0]["phase_source_ids"] = []
            elif damage == "live":
                report["barrier_conversions"][0]["live_value_types"] = ["i64"]
            elif damage == "scope":
                report["barrier_conversions"][0]["contract"]["participants"] = "all-programs"
            elif damage == "volatile-selection":
                report["barrier_conversions"][0]["contract"].pop("volatile_effects")
            elif damage == "volatile-primitive":
                row = next(row for row in report["source_graph"]["operations"] if "volatile_read" in row)
                row["volatile_read"]["observation"] = "stable-tensor-read"
            elif damage != "bound-volatile-profile":
                module = parse_shared(original[paths[4]].decode())
                phases = [op for op in module.walk() if op.name == "scf.execute_region"]
                if damage == "lost-store":
                    store = next(op for op in phases[0].walk() if opcode(op) == "tt.store")
                    # Keep its source location on an inert op: tags cannot supply the lost effect.
                    from xdsl.dialects import arith, builtin
                    replacement = arith.ConstantOp.from_int_and_width(0, builtin.i64)
                    replacement.location = store.location
                    block = store.parent_block()
                    block.insert_op_before(replacement, store)
                    block.erase_op(store)
                elif damage == "volatile-flag":
                    from xdsl.dialects import builtin
                    read = next(op for op in module.walk() if opcode(op) == "tt.load")
                    read.properties["isVolatile"] = builtin.IntegerAttr(False, builtin.i1)
                else:
                    block = phases[0].parent_block()
                    block.detach_op(phases[1])
                    block.insert_op_before(phases[1], phases[0])
                stream = StringIO()
                Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
                paths[4].write_text(stream.getvalue())
                report["artifacts"]["protected"] = record(paths[4].read_bytes())
                stage = json.loads(original[paths[5]])
                stage["artifacts"]["source.ttir"] = record(paths[4].read_bytes())
                paths[5].write_text(json.dumps(stage))
                changed.update(("allocation-protected.ttir", "allocation-source-stage.json"))
            paths[3].write_text(json.dumps(report))
            provenance = json.loads(original[paths[2]])
            if damage == "bound-volatile-profile":
                provenance["memory"]["effect_profile"] = "ordinary"
            for name in changed:
                provenance["source_argument_binding"]["source_capsule"]["artifacts"][name] = record(
                    (package / "kernels/0" / name).read_bytes())
            paths[2].write_text(json.dumps(provenance))
            bindings = json.loads(original[paths[1]])
            key = next(iter(bindings["kernels"]))
            bindings["kernels"][key] = deepcopy(provenance)
            paths[1].write_text(json.dumps(bindings))
            rehash(package)
            with pytest.raises(ValueError, match="barrier|volatile") as producer:
                verify_bundle(package)
            with pytest.raises(ValueError, match="barrier|volatile") as consumer:
                check_merlin_interface(package, root / ("refused-" + damage), merlin_root=os.environ["T2M_MERLIN_ROOT"],
                    python=os.environ["T2M_MERLIN_PYTHON"], result_mode="execution-owned")
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, contents in original.items():
                path.write_bytes(contents)
    finally:
        for path, contents in original.items():
            path.write_bytes(contents)
    verify_bundle(package)
    (root / "phase-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_original_public_shape_domain_still_aborts_before_results(selected):
    import signal

    from t2m.consumer import observe_merlin_host

    root, _capture, package = selected
    inputs, _ = invocation(root, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()


def test_actual_serial_target_retains_all_phase_model_results(selected):
    import sys

    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package = selected
    inputs, expected = invocation(root, 19, 8)
    reference = root / "target-reference.npz"
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=inputs, reference=reference)
        assert report["evaluation"]["result_count"] == 9
    finally:
        sys.path.remove(scripts)

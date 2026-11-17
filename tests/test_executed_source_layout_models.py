"""Real packed reads compose with count/scan/scatter and owned model results."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_grouped_memory import capture as translate
from test_grouped_memory_model import TOOLS
from test_spike_model import TOOLS as TARGET_TOOLS
from test_spike_model import Q

from t2m.bundle import _record, verify_bundle
from t2m.consumer import check_merlin_interface, invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.model_source_layout import PROFILE
from t2m.model_tensor_views import POLICY

pytestmark = pytest.mark.skipif(
    not TOOLS, reason="existing source/model/Merlin tools required"
)
MAPPING = {
    "target": "source_target",
    "num_warps": "source_warps",
    "threads_per_warp": "source_threads",
    "num_ctas": "source_ctas",
}
LIMITS = {
    "schema": "merlin.owned-result-limits.v2",
    "max_bytes": 64 << 20,
    "max_allocations": 16384,
    "max_output_bytes": 64 << 20,
    "index_bits": 64,
}


def compaction(root, dtype):
    from t2m.examples.kernel_size_kernels import positive_count, positive_pack
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    root.mkdir()
    bindings = []
    pointer = "*fp32" if dtype == "f32" else "*i64"
    for kind, kernel in (
        ("positive_count", positive_count),
        ("compact", positive_pack),
    ):
        packing = kind == "compact"
        source = root / (kind + ".ttir")
        source.write_text(
            capture_ttir(
                kernel,
                {
                    "x": pointer,
                    "out" if packing else "count": pointer if packing else "*i64",
                    "capacity": "i64",
                    **({"n": "i64"} if packing else {}),
                    "B": "constexpr",
                },
                {"B": 512},
            )
        )
        abi = TensorABI(
            kernel.__name__,
            (
                TensorBuffer("x", (320,), dtype=dtype),
                TensorBuffer(
                    "out",
                    ("n",) if packing else (1,),
                    "write",
                    dtype if packing else "i64",
                ),
            ),
            runtime_scalars=(
                RuntimeScalar("capacity", "i64", 2, bounds=(0, 320)),
                *((RuntimeScalar("n", "i64", 3, bounds=(0, 320)),) if packing else ()),
            ),
        )
        result = convert_shared(
            source,
            abi,
            root / (kind + "-converted"),
            optimizer=os.environ["TRITON_SHARED_OPT"],
            memory_profile="auto",
        )
        bindings.append(
            KernelBinding(
                "t2m_grouped_demo::" + kind,
                result,
                scalar_arguments={"capacity": 320},
                runtime_arguments={"capacity": "capacity"},
                runtime_operands={"n": "n"} if packing else {},
            )
        )
    return bindings


@pytest.fixture(scope="module", params=(("i64", 2, True), ("f32", 4, False)))
def selected(tmp_path_factory, request):
    dtype, warps, while_loop = request.param
    root = tmp_path_factory.mktemp("executed-source-layout-" + dtype)
    capture = root / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_grouped_memory_model.py"
    )
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--execution-sizes",
            "--out",
            str(capture),
            "--dtype",
            dtype,
            "--warps",
            str(warps),
            *([] if while_loop else ["--for-loop"]),
        ],
        capture_output=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert done.returncode == 0, done.stderr.decode()
    kernel = root / "grouped"
    kernel.mkdir()
    result = translate(kernel, dtype, warps, while_loop, True)
    scalars = {
        "source_target": "cuda:80",
        "source_warps": warps,
        "source_threads": 32,
        "source_ctas": 1,
        "while_loop": while_loop,
        "selected": True,
    }
    names = ("n", "capacity", "iterations", "delta")
    bindings = [
        KernelBinding(
            "t2m_demo::grouped_read_state",
            result,
            tensor_views=POLICY,
            scalar_arguments={**dict(zip(names, values, strict=True)), **scalars},
            runtime_arguments={name: name for name in names},
            source_layout_arguments=MAPPING,
        )
        for values in ((300, 320, 3, 2), (319, 320, 1, -3), (0, 0, 2, 5))
    ]
    bindings.extend(compaction(root / "compaction", dtype))
    package = write_model_bundle(
        capture,
        bindings,
        root / "package",
        work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True,
        result_mode="execution-owned",
        execution_allocation_policy=LIMITS,
    )
    return root, capture, package, dtype, warps


def test_original_source_grouping_and_all_seventeen_model_results_survive(selected):
    _root, capture, package, _dtype, _warps = selected
    receipt = verify_bundle(package)
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / name).read_bytes() == (package / destination).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["schema"] == "t2m.model-kernel-binding.v4"
    assert trace["source_layout_binding_profile"] == PROFILE
    assert len(trace["calls"]) == 5 and len(trace["kernels"]) == 3
    for ordinal, call in enumerate(trace["calls"]):
        assert ("source_layout_bindings" in call) == (ordinal < 3)
        assert call["source_layout_arguments"] == (MAPPING if ordinal < 3 else {})
    signature = json.loads((package / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6"
    assert (
        signature["allocation_policy"]
        == receipt["execution_allocation_policy"]
        == LIMITS
    )
    assert signature["outputs"][15]["shape"] == [None]
    with (
        np.load(capture / "goldens.npz", allow_pickle=False) as wanted,
        np.load(package / "consumer/host/outputs.npz", allow_pickle=False) as actual,
    ):
        assert actual.files == [f"out{i}" for i in range(17)]
        for name in actual.files:
            assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                wanted[name].shape,
                wanted[name].dtype,
                wanted[name].tobytes(),
            )


def invocation_files(root, dtype, warps, full):
    from t2m.examples.grouped_memory_reference import reference

    word = np.int64 if dtype == "i64" else np.float32
    x = (np.arange(320).reshape(10, 32) + 11).astype(word)
    y = (np.arange(320).reshape(10, 32) * 2 + 17).astype(word)
    if not full:
        x, y = -x, -y
    first = reference(x, y, 300, 320, 3, 2, warps=warps, selected=True)
    second = reference(first[0], y, 319, 320, 1, -3, warps=warps, selected=True)
    empty = reference(x[:0], y[:0], 0, 0, 2, 5, warps=warps, selected=True)
    flat = second[0].reshape(-1)
    packed = flat[flat > 0]
    wanted = (*first, *second, *empty, packed, np.asarray([packed.size], np.int64))
    arguments, references = (
        root / f"arguments-{full}.npz",
        root / f"references-{full}.npz",
    )
    np.savez(arguments, in0=y, in1=x)
    np.savez(references, **{f"out{i}": value for i, value in enumerate(wanted)})
    assert packed.size == (320 if full else 0)
    return arguments, references, wanted


def test_one_library_preserves_source_groups_snapshots_and_changed_allocation_sizes(
    selected,
):
    root, _capture, package, dtype, warps = selected
    library = _record(package / "consumer/host/model.so")
    source = _record(package / "consumer/compiler/program.mlir")
    for full in (False, True):
        arguments, _references, wanted = invocation_files(root, dtype, warps, full)
        actual, report = invoke_merlin_host(
            package / "consumer",
            arguments,
            root / f"invoke-{full}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"]
        assert len(actual) == 17
        for value, expected in zip(actual, wanted, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (
                expected.shape,
                expected.dtype,
                expected.tobytes(),
            )
    assert _record(package / "consumer/host/model.so") == library
    assert _record(package / "consumer/compiler/program.mlir") == source


@pytest.mark.parametrize("full", (None, False))
def test_actual_target_preserves_source_layout_and_complete_owned_results(
    selected, tmp_path, monkeypatch, full
):
    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package, dtype, warps = selected
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    invocation = {}
    if full is not None:
        arguments, reference, _wanted = invocation_files(root, dtype, warps, full)
        invocation = {"inputs": arguments, "reference": reference}
    report = Q.qualify(
        package,
        tmp_path / "target",
        result_mode="execution-owned",
        comparison="words",
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        **invocation,
    )
    assert report["evaluation"]["result_count"] == 17
    assert (
        Q.verify(
            tmp_path / "target",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        == report
    )


@pytest.mark.parametrize(
    "damage",
    (
        "profile",
        "missing-bindings",
        "mapping",
        "attribute",
        "source-selection",
        "unspecialized",
    ),
)
def test_rehashed_source_layout_claims_require_original_call_proofs(
    selected, tmp_path, damage
):
    from test_executed_model_packages import rehash

    _root, _capture, package, _dtype, _warps = selected
    paths = [
        package / name for name in ("kernel-bindings.json", "capture_receipt.json")
    ]
    if damage == "source-selection":
        inventory = json.loads(paths[1].read_text())["kernel_artifacts"]
        digest = json.loads(paths[0].read_text())["calls"][0]["implementation_sha256"]
        paths.append(package / inventory[digest] / "provenance.json")
    original = {path: path.read_bytes() for path in paths}
    try:
        trace = json.loads(original[paths[0]])
        call = trace["calls"][0]
        if damage == "profile":
            trace.pop("source_layout_binding_profile")
        elif damage == "missing-bindings":
            call.pop("source_layout_bindings")
        elif damage == "mapping":
            call["source_layout_arguments"]["num_warps"] = "source_ctas"
        elif damage == "attribute":
            call["source_layout_bindings"]["num_warps"]["attribute"] = "8 : i64"
        elif damage == "source-selection":
            trace["kernels"][call["implementation_sha256"]][
                "source_layout_specialization"
            ]["source_options"]["num_warps"] = 8
            paths[2].write_text(
                json.dumps(trace["kernels"][call["implementation_sha256"]])
            )
        else:
            trace["calls"][3]["source_layout_bindings"] = call["source_layout_bindings"]
        paths[0].write_text(json.dumps(trace))
        rehash(package)
        with pytest.raises(ValueError) as producer:
            verify_bundle(package)
        output = tmp_path / "refused"
        with pytest.raises(ValueError) as consumer:
            check_merlin_interface(
                package,
                output,
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
                result_mode="execution-owned",
            )
        assert not (output / "compiler").exists()
        (tmp_path / "refusal.json").write_text(
            json.dumps(
                {
                    "damage": damage,
                    "producer": str(producer.value),
                    "consumer": str(consumer.value),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
    verify_bundle(package)


@pytest.mark.parametrize("damage", ("receipt-selection", "saved-limits"))
def test_rehashed_resource_changes_cannot_reuse_an_existing_qualification(
    selected, damage
):
    _root, _capture, package, _dtype, _warps = selected
    receipt_path, saved_path = (
        package / "capture_receipt.json",
        package / "consumer/host/owned-result-program.json",
    )
    original = {path: path.read_bytes() for path in (receipt_path, saved_path)}
    try:
        receipt = json.loads(original[receipt_path])
        if damage == "receipt-selection":
            receipt["execution_allocation_policy"]["max_allocations"] *= 2
        else:
            saved = json.loads(original[saved_path])
            saved["limits"]["max_allocations"] *= 2
            saved_path.write_text(json.dumps(saved))
            receipt["artifacts"]["consumer/host/owned-result-program.json"] = _record(
                saved_path
            )
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="allocation"):
            verify_bundle(package)
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
    verify_bundle(package)

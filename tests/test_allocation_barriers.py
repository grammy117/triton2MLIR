"""Protected allocation source retains real program phases through Merlin."""

import json
import os
from copy import deepcopy

import pytest
from test_program_barriers import arguments

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")


def implementation(root, profile):
    from t2m.allocation_bindings import (
        PointerArgument,
        ScalarArgument,
        convert_allocations,
    )
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.barrier_kernels import exchange_tile
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer

    path = root / "source.ttir"
    path.write_text(capture_ttir(exchange_tile,
        {"state": "*i64", "out": "*i64", "n": "i32", "steps": "i32", "B": "constexpr", "CROSS": "constexpr"},
        {"B": 4, "CROSS": False}))
    abi = TensorABI("allocation_exchange", (TensorBuffer("state", ("n", 4), "readwrite", "i64"),
        TensorBuffer("out", ("n", 4), "write", "i64")), grid=(GridExtent("n", 4),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 19)),
                         RuntimeScalar("steps", "i32", 3, bounds=(0, 4))))
    return convert_allocations(path, abi,
        (PointerArgument("state"), PointerArgument("out"), ScalarArgument("n"), ScalarArgument("steps")),
        root / "translation", source_entry="exchange_tile", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile)


@pytest.fixture(scope="module", params=("tensor", "guarded-buffer"))
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("allocation-barriers-" + request.param)
    return root, implementation(root, request.param)


def test_original_barrier_scope_survives_allocation_binding_and_merlin_execution(selected):
    from test_integer_products import replay_library

    root, result = selected
    report = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(report["barrier_conversions"]) == 2
    assert sum(op["opcode"] == "ttg.barrier" for op in report["source_graph"]["operations"]) == 2
    capsule = result.trace["source_argument_binding"]["source_capsule"]
    assert capsule["schema"] == "t2m.source-allocation-capsule.v2"
    retained = [r for r in result.trace["transformations"] if r["kind"] == "protected-allocation-program-memory-phases"]
    assert len(retained) == 1 and retained[0]["source_graph"] == "allocation-original"
    replay_library(root, result, [arguments(n, steps) for n, steps in [(7, 2), (0, 3), (1, 0), (19, 4)]])


def test_changed_phase_claims_refuse_producer_replay(selected):
    from t2m.source_protection import validate

    _, result = selected
    artifacts = result.auxiliary_artifacts
    original = artifacts["allocation-original.ttir"]
    report = json.loads(artifacts["allocation-protection.json"])
    for change in ("missing", "mask", "prefix", "live", "scope", "hidden-source"):
        broken = deepcopy(report)
        if change == "missing":
            broken.pop("barrier_conversions")
        elif change == "mask":
            broken["barrier_conversions"][0]["contract"]["address_space_mask"] = 0
        elif change == "prefix":
            broken["barrier_conversions"][0]["phase_source_ids"] = []
        elif change == "live":
            broken["barrier_conversions"][0]["live_value_types"] = ["i64"]
        elif change == "scope":
            broken["barrier_conversions"][0]["contract"]["participants"] = "all-programs"
        else:
            broken.pop("barrier_conversions")
            broken["source_graph"]["operations"] = [row for row in broken["source_graph"]["operations"]
                                                    if row["opcode"] != "ttg.barrier"]
        members = {**artifacts, "allocation-protection.json": json.dumps(broken)}
        with pytest.raises(ValueError, match="barrier"):
            validate(members, original, "exchange_tile")

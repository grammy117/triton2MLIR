"""Source implementation contracts expand before allocation wrapper inlining."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from test_descriptor_state import REQUIRED
from test_source_assembly import PROVIDER, source


@pytest.mark.skipif(
    not (REQUIRED and PROVIDER),
    reason="existing native/Merlin and optional source provider required",
)
def test_selected_scalar_assembly_is_executable_before_allocation_binding(tmp_path):
    from test_mixed_effects import qualify

    from t2m.allocation_bindings import PointerArgument, convert_allocations

    original, abi = source(tmp_path)
    result = convert_allocations(
        original,
        replace(abi, entry="allocation_assembly"),
        (PointerArgument("x"), PointerArgument("out")),
        tmp_path / "binding",
        source_entry="assembly",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        assembly_provider="ptx-integer-v1",
    )
    report = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(report["assembly_conversions"]) == 1 and not report["fp8_conversions"]
    capsule = result.trace["source_argument_binding"]["source_capsule"]
    assert (
        capsule["schema"] == "t2m.source-allocation-capsule.v2"
        and capsule["normalized"] is False
    )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "words"}]},
    }
    qualify(
        result,
        tmp_path,
        [np.asarray([-1], np.int32)],
        policy,
        [np.asarray([0], np.int32)],
    )

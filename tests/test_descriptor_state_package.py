"""Public descriptor state packages retain source custody through Merlin."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest
from test_descriptor_state import REQUIRED

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


def check_rehashed_custody_refusals(capture):
    from t2m.bundle import _record, verify_bundle
    from t2m.source_capsule import record

    names = ("capture_receipt.json", "provenance.json", "allocation-source-stage.json")
    saved = {name: (capture / name).read_bytes() for name in names}
    for change in (
        "evaluator",
        "transform",
        "source",
        "schema",
        "descriptor",
        "normalized",
        "roster",
    ):
        try:
            trace = json.loads(saved["provenance.json"])
            binding = trace["source_argument_binding"]
            capsule = binding["source_capsule"]
            if change in ("evaluator", "transform", "source"):
                name = "allocation-source-stage.json"
                stage = json.loads(saved[name])
                if change == "evaluator":
                    stage["reference_sha256"] = "0" * 64
                elif change == "transform":
                    stage["flags"].append("--canonicalize")
                else:
                    stage["artifacts"]["source.ttir"]["sha256"] = "0" * 64
                raw = json.dumps(stage).encode()
                (capture / name).write_bytes(raw)
                capsule["artifacts"][name] = record(raw)
            elif change == "schema":
                capsule["schema"] = "unknown"
            elif change == "descriptor":
                binding["descriptor_bindings"]["grants_semantics"] = True
            elif change == "normalized":
                capsule["normalized"] = False
            else:
                capsule["artifacts"]["inputs.npz"] = deepcopy(
                    capsule["artifacts"]["allocation-input.mlir"]
                )
            (capture / "provenance.json").write_text(json.dumps(trace))
            receipt = json.loads(saved["capture_receipt.json"])
            for name in names[1:]:
                receipt["artifacts"][name] = _record(capture / name)
            (capture / "capture_receipt.json").write_text(json.dumps(receipt))
            with pytest.raises(ValueError):
                verify_bundle(capture)
        finally:
            for name, raw in saved.items():
                (capture / name).write_bytes(raw)
    verify_bundle(capture)


def test_unused_descriptor_parameter_normalizes_without_descriptor_operations(tmp_path):
    from test_mixed_effects import qualify

    from t2m.allocation_bindings import (
        DescriptorArgument,
        PointerArgument,
        ScalarArgument,
        convert_allocations,
    )
    from t2m.examples.descriptor_reference import policy
    from t2m.examples.descriptor_state_kernels import unused_descriptor
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            unused_descriptor,
            {
                "unused": "tensordesc<i64[2,16]>",
                "state": "*i64",
                "n": "i32",
                "capacity": "i32",
            },
            {},
        )
    )
    abi = TensorABI(
        "unused_descriptor_state",
        (TensorBuffer("state", ("capacity",), "readwrite", "i64"),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 1, bounds=(0, 16)),
            RuntimeScalar("capacity", "i32", 2, bounds=(0, 64)),
        ),
    )
    bindings = (
        DescriptorArgument("state", (2, 16), (16, 1)),
        2,
        16,
        16,
        1,
        PointerArgument("state"),
        ScalarArgument("n"),
        ScalarArgument("capacity"),
    )
    result = convert_allocations(
        source,
        abi,
        bindings,
        tmp_path / "binding",
        source_entry="unused_descriptor",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
    )
    inventory = json.loads(result.auxiliary_artifacts["allocation-inventory.json"])
    assert not any(
        op["opcode"].startswith("tt.descriptor_") for op in inventory["operations"]
    )
    assert inventory["descriptor_argument_expansion"]["functions"]
    state = np.arange(32, dtype=np.int64) + 2**53 + 1
    expected = state.copy()
    expected[:13] += 1
    qualify(
        result,
        tmp_path,
        [state, np.asarray(13, np.int32), np.asarray(32, np.int32)],
        policy(),
        [expected],
    )


def test_descriptor_state_public_cli_qualifies_portable_package(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                "descriptor-state",
                "--out",
                str(capture),
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
            ]
        )
        == 0
    )
    verify_bundle(capture)
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)
    from t2m.package_coverage import package_coverage

    report = package_coverage([capture], qualifications=[qualified])
    assert report["ttir"]["tt.descriptor_load"]["merlin_cases"]
    assert report["ttir"]["tt.descriptor_store"]["merlin_cases"]
    assert (
        report["kernel_cases"][0]["source_graph"]
        == "pre-wrapper-original-source-operations"
    )
    assert (
        report["kernel_cases"][0]["allocation_bound_operation_counts"]["tt.call"] == 1
    )
    check_rehashed_custody_refusals(capture)
    # Exercise saved v2 source-order bindings through the generic tensor CLI,
    # independently of the specialized demonstration entry point.
    tensor_capture = tmp_path / "tensor-capture"
    assert (
        main(
            [
                "shared-tensor",
                str(capture / "allocation-original.ttir"),
                "--abi",
                str(capture / "spec.json"),
                "--allocation-bindings",
                str(capture / "allocation-arguments.json"),
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
                "--inputs",
                str(capture / "inputs.npz"),
                "--reference",
                str(capture / "reference.npz"),
                "--out",
                str(tensor_capture),
            ]
        )
        == 0
    )
    verify_bundle(tensor_capture)
    assert (tensor_capture / "allocation-original.ttir").read_bytes() == (
        capture / "allocation-original.ttir"
    ).read_bytes()

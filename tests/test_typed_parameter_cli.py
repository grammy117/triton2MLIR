"""Demo and generic source intake retain original typed parameter custody."""

import json
import os

import pytest
from test_descriptor_state import REQUIRED

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize(
    "kernel,order",
    (("typed-parameter-state", "big"), ("typed-descriptor-state", "little")),
)
def test_public_and_original_source_cli_qualify_the_same_typed_storage(
    tmp_path, kernel, order
):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture = tmp_path / "capture"
    native = os.environ["T2M_NATIVE_PYTHON"]
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                kernel,
                "--out",
                str(capture),
                "--python",
                native,
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
            ]
        )
        == 0
    )
    verify_bundle(capture)
    profile = json.loads((capture / "provenance.json").read_text())
    assert profile["typed_storage_transport"]["byte_order"] == order
    generic = tmp_path / "generic"
    assert (
        main(
            [
                "shared-tensor",
                str(capture / "allocation-original.ttir"),
                "--abi",
                str(capture / "spec.json"),
                "--allocation-bindings",
                str(capture / "allocation-arguments.json"),
                "--byte-order",
                order,
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                native,
                "--inputs",
                str(capture / "inputs.npz"),
                "--reference",
                str(capture / "reference.npz"),
                "--out",
                str(generic),
            ]
        )
        == 0
    )
    verify_bundle(generic)
    for index, package in enumerate((capture, generic)):
        qualified = tmp_path / f"merlin-{index}"
        qualify_kernel(
            package,
            qualified,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
            compiler_python=native,
            clang=os.environ["T2M_MERLIN_CLANG"],
        )
        verify_kernel_qualification(qualified)


def qualify_model_cli(root, capture, bindings, result):
    """Exercise original-source allocation configuration on the real parent model."""
    import numpy as np

    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.model_tensor_views import POLICY

    configuration = {
        "schema": "t2m.kernel-bindings.v1",
        "kernels": [
            {
                "kernel": binding.qualname,
                "ttir": str(root / "source/source.ttir"),
                "abi": result.spec.as_dict(),
                "tensor_views": POLICY,
                "allocation_bindings": json.loads(
                    result.auxiliary_artifacts["allocation-arguments.json"]
                ),
                "scalar_arguments": binding.scalar_arguments,
                "runtime_arguments": binding.runtime_arguments,
                "memory_profile": "guarded-buffer",
                "address_policy": "source-words",
                "byte_order": "little",
            }
            for binding in bindings
        ],
    }
    config = root / "bindings.json"
    config.write_text(json.dumps(configuration))
    out = root / "cli-bound"
    assert (
        main(
            [
                "bind-model",
                str(capture),
                "--bindings",
                str(config),
                "--out",
                str(out),
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
                "--merlin-root",
                os.environ["T2M_MERLIN_ROOT"],
                "--merlin-python",
                os.environ["T2M_MERLIN_PYTHON"],
                "--merlin-clang",
                os.environ["T2M_MERLIN_CLANG"],
                "--merlin-host",
            ]
        )
        == 0
    )
    verify_bundle(out)
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
    ):
        assert (capture / name).read_bytes() == (out / name).read_bytes()
    with (
        np.load(capture / "goldens.npz", allow_pickle=False) as expected,
        np.load(out / "consumer/host/outputs.npz", allow_pickle=False) as actual,
    ):
        assert actual.files == expected.files
        for name in actual.files:
            assert (
                actual[name].shape == expected[name].shape
                and actual[name].dtype == expected[name].dtype
            )
            assert actual[name].tobytes() == expected[name].tobytes()

"""One qualified library over runtime packed/ragged shapes and source loops."""

import json
import os

import numpy as np
import pytest

from t2m.examples.scaled_dot_reference import dynamic_inputs, dynamic_reference

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(key)
        for key in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


def convert(root, profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.scaled_dot_kernels import scaled_dot_dynamic
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    signature = {
        name: "*u8" if i < 4 else "*fp32" if i < 6 else "i32" if i < 11 else "constexpr"
        for i, name in enumerate(scaled_dot_dynamic.arg_names)
    }
    path = root / "source.ttir"
    path.write_text(
        capture_ttir(
            scaled_dot_dynamic,
            signature,
            {"BM": 16, "BN": 16},
            debug=True,
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "scaled_dot_dynamic",
        (
            TensorBuffer("a", ("m", "packed_k"), dtype="i8"),
            TensorBuffer("b", ("packed_k", "n"), dtype="i8"),
            TensorBuffer("sa", ("m", "groups"), dtype="i8"),
            TensorBuffer("sb", ("n", "groups"), dtype="i8"),
            TensorBuffer("initial", ("m", "n")),
            TensorBuffer("out", ("m", "n"), "write"),
        ),
        grid=(GridExtent("m", 16), GridExtent("n", 16)),
        runtime_scalars=tuple(
            RuntimeScalar(name, "i32", i + 6, bounds=(0, upper))
            for i, (name, upper) in enumerate(
                [("m", 35), ("n", 21), ("k", 65), ("packed_k", 33), ("groups", 3)]
            )
        ),
    )
    return convert_shared(
        path,
        abi,
        root / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_ragged_scaled_source_loops_reuse_library_and_reject_bad_extents(
    tmp_path, profile
):
    from test_mixed_effects import qualify
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.native import execute_native

    result = convert(tmp_path, profile)
    assert (
        "scf.for" in result.mlir_text
        and "prov.scaled_dot_contracts" in result.mlir_text
    )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    first = dynamic_inputs(17, 19, 37)
    _, qualified = qualify(result, tmp_path, first, policy, [dynamic_reference(*first)])
    library = _record(qualified / "host/model.so")
    for index, (m, n, k) in enumerate(
        [
            (0, 7, 65),
            (7, 0, 65),
            (5, 7, 0),
            (1, 1, 1),
            (3, 5, 31),
            (16, 16, 32),
            (35, 21, 65),
        ]
    ):
        arrays = dynamic_inputs(m, n, k)
        expected = dynamic_reference(*arrays)
        native, _ = execute_native(
            result.mlir_text,
            arrays,
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        np.testing.assert_array_equal(
            native[0].view(np.uint32), expected.view(np.uint32)
        )
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)}
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        np.savez(caller / "reference.npz", out0=expected)
        output = tmp_path / f"invoke-{index}"
        report = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            output,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(output)
        assert report["native_compilation"] == "reused-qualified-library"
        assert _record(output / "source/host/model.so") == library
    for index, which in enumerate(("packed", "groups")):
        arrays = dynamic_inputs(3, 5, 31)
        if which == "packed":
            arrays[0] = np.pad(arrays[0], ((0, 0), (0, 1)))
            arrays[1] = np.pad(arrays[1], ((0, 1), (0, 0)))
            arrays[8] = np.asarray(17, np.int32)
            message = "packed K extent differs"
        else:
            arrays[2] = np.pad(arrays[2], ((0, 0), (0, 1)))
            arrays[3] = np.pad(arrays[3], ((0, 0), (0, 1)))
            arrays[9] = np.asarray(2, np.int32)
            message = "scale group extent differs"
        caller = tmp_path / f"bad-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)}
        )
        failure = {
            "schema": "t2m.effect-reference.v1",
            "termination": "assertion_failure",
            "stdout": "Assertion `" + message + "` failed\n",
            "stderr": "",
        }
        (caller / "reference.json").write_text(json.dumps(failure))
        output = tmp_path / f"failed-{index}"
        report = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            output,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(output)
        assert report["result_count"] == 0
        assert not (output / "evaluation/outputs.npz").exists()
    assert _record(qualified / "host/model.so") == library

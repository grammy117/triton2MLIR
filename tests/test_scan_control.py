"""Structured associative combiners retain parallel stages and source control."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(
    os.environ.get(name)
    for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
)
pytestmark = pytest.mark.skipif(
    not (OPT and NATIVE), reason="existing shared/native tools required"
)


def capture(root, dtype="i64", reverse=False, profile="guarded-buffer"):
    from t2m.examples.scan_control_kernels import segmented_scan
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    pointer = "*fp32" if dtype == "f32" else "*" + dtype
    source.write_text(
        capture_ttir(
            segmented_scan,
            {
                "x": pointer,
                "heads": "*i32",
                "out": pointer,
                "combined_heads": "*i32",
                "n": "i32",
                "B": "constexpr",
                "REVERSE": "constexpr",
            },
            {"B": 8, "REVERSE": reverse},
        )
    )
    abi = TensorABI(
        "segmented_scan",
        (
            TensorBuffer("x", ("n",), dtype=dtype),
            TensorBuffer("heads", ("n",), dtype="i32"),
            TensorBuffer("out", ("n",), "write", dtype),
            TensorBuffer("combined_heads", ("n",), "write", "i32"),
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


def arguments(dtype="i64", n=19, rotation=0, reverse=False):
    carrier = {"i32": np.int32, "i64": np.int64, "f32": np.float32}[dtype]
    pattern = (
        [0, -1, 1, -(1 << 63), (1 << 63) - 1, (1 << 53) + 1, -7, 3]
        if dtype == "i64"
        else [0, -1, 1, -17, 29, 3, -7, 4]
    )
    values = np.resize(np.roll(np.asarray(pattern, dtype=carrier), rotation), n).copy()
    heads = np.resize(
        np.roll(np.asarray([0, 1, 0, 0, 1, 0, 0, 0], np.int32), rotation), n
    ).copy()
    output, flags = values.copy(), heads.copy()
    for start in range(0, n, 8):
        order = list(range(start, min(start + 8, n)))
        if reverse:
            # Explicit source padding is a (zero,false) suffix before reversal.
            order = list(reversed(order))
        previous = None
        flag = 0
        for index in order:
            value = values[index]
            if previous is not None and not heads[index]:
                if dtype.startswith("i"):
                    bits = 64 if dtype == "i64" else 32
                    word = (int(previous) + int(value)) & ((1 << bits) - 1)
                    value = carrier(
                        word - (1 << bits) if word & (1 << (bits - 1)) else word
                    )
                else:
                    value = carrier(previous + value)
            flag |= int(heads[index])
            output[index], flags[index] = value, flag
            previous = value
    return [values, heads, np.asarray(n, np.int32)], [output, flags]


@pytest.mark.parametrize("dtype", ["i64", "i32", "f32"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_segmented_scan_keeps_nested_branches_and_all_tuple_results(
    tmp_path, dtype, reverse
):
    from t2m.native import execute_native

    result = capture(tmp_path, dtype, reverse)
    for index, (n, rotation) in enumerate(((0, 0), (1, 0), (19, 0), (31, 2))):
        inputs, expected = arguments(dtype, n, rotation, reverse)
        actual, _ = execute_native(
            result.mlir_text,
            inputs,
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        for got, wanted in zip(actual, expected, strict=True):
            assert got.dtype == wanted.dtype and got.shape == wanted.shape
            assert got.tobytes() == wanted.tobytes()


def loop_capture(root, width=64, reverse=False, gcd=False, profile="guarded-buffer"):
    from t2m.examples.scan_control_kernels import loop_scan
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            loop_scan,
            {
                "x": f"*i{width}",
                "out": f"*i{width}",
                "n": "i32",
                "B": "constexpr",
                "REVERSE": "constexpr",
                "GCD": "constexpr",
            },
            {"B": 8, "REVERSE": reverse, "GCD": gcd},
        )
    )
    abi = TensorABI(
        "loop_scan",
        (
            TensorBuffer("x", ("n",), dtype=f"i{width}"),
            TensorBuffer("out", ("n",), "write", f"i{width}"),
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)),),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


def loop_arguments(width=64, n=19, rotation=0, reverse=False, gcd=False):
    import math

    dtype = np.dtype(f"int{width}")
    pattern = [0, 1, 3, (1 << (width - 1)) - 1, 9, 5, 0, 7]
    if gcd:
        pattern = [0, 144, 233, 2**53 + 1 if width == 64 else 2**24 + 1, 72, 54, 0, 37]
    values = np.resize(np.roll(np.asarray(pattern, dtype=dtype), rotation), n).copy()
    output, mask = values.copy(), (1 << width) - 1
    for start in range(0, n, 8):
        order = list(range(start, min(start + 8, n)))
        if reverse:
            order.reverse()
        previous = 0 if gcd else 1
        for index in order:
            previous = (
                math.gcd(previous, int(values[index]))
                if gcd
                else previous * int(values[index]) & mask
            )
            output[index] = (
                previous - (1 << width) if previous & (1 << (width - 1)) else previous
            )
    return [values, np.asarray(n, np.int32)], [output]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("gcd", [False, True])
def test_native_scan_keeps_actual_for_and_while_combiner_state(
    tmp_path, width, reverse, gcd
):
    from t2m.native import execute_native

    result = loop_capture(tmp_path, width, reverse, gcd)
    report = next(
        t
        for t in result.trace["transformations"]
        if t["kind"] == "associative-scan-network"
    )
    assert report["combiner_control_flow"]["scf.while" if gcd else "scf.for"] == 1
    assert report["combiner_source_ids"]
    for index, (n, rotation) in enumerate(((0, 0), (1, 0), (19, 0), (31, 2))):
        inputs, expected = loop_arguments(width, n, rotation, reverse, gcd)
        actual, _ = execute_native(
            result.mlir_text,
            inputs,
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        assert actual[0].tobytes() == expected[0].tobytes()


def qualify_and_replay(root, result, cases):
    import json
    from test_mixed_effects import expected as policy, qualify
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    inputs, reference = cases[0]
    contract = policy("", [{"kind": "words"} for _ in reference])
    bundle, compiled = qualify(result, root, inputs, contract, reference)
    library = _record(compiled / "host/model.so")
    for index, (inputs, reference) in enumerate(cases[1:]):
        caller = root / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(inputs)}
        )
        np.savez(
            caller / "reference.npz",
            **{f"out{i}": value for i, value in enumerate(reference)},
        )
        (caller / "reference.json").write_text(json.dumps(contract))
        replay = root / f"replay-{index}"
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            replay,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(replay)
        assert _record(replay / "source/host/model.so") == library
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = dict(
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        report = target.qualify(bundle, root / "target", **options)
        assert target.verify(root / "target", **options) == report
    return bundle, compiled


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("reverse", [False, True])
def test_merlin_segmented_tuple_scan_keeps_parallel_structure_and_runtime_shapes(
    tmp_path, profile, reverse
):
    dtype = "f32" if reverse else "i64"
    result = capture(tmp_path, dtype, reverse, profile)
    stages = [
        op for op in result.module.walk() if "prov.scan_distance" in op.attributes
    ]
    assert len(stages) == 3
    assert all(
        op.name == "linalg.generic" and "prov.scan_combiner_control" in op.attributes
        for op in stages
    )
    assert all(
        any(child.name == "scf.if" for child in stage.walk()) for stage in stages
    )
    cases = [
        arguments(dtype, n, rotation, reverse)
        for n, rotation in ((19, 0), (0, 0), (1, 0), (31, 2), (13, 1))
    ]
    qualify_and_replay(tmp_path, result, cases)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
@pytest.mark.parametrize("gcd", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_merlin_loop_scan_keeps_actual_integer_state_through_the_target(
    tmp_path, gcd, reverse
):
    result = loop_capture(tmp_path, 64, reverse, gcd)
    cases = [
        loop_arguments(64, n, rotation, reverse, gcd)
        for n, rotation in ((19, 0), (0, 0), (1, 0), (31, 2), (13, 1))
    ]
    qualify_and_replay(tmp_path, result, cases)

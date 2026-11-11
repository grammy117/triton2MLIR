"""Void, failed and returning source effects reach the actual target runtime."""

import json
import os

import numpy as np
import pytest

from scripts import qualify_spike_effect as E
from test_spike_model import TOOLS

TOOLS = TOOLS and all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source, Merlin and target tools required")


def options():
    return {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}


@pytest.mark.parametrize("case", ("empty-return", "constant-fail", "scalar-check", "literal-check", "tensor-check",
    "fail-before-store", "print-words", "print-hex", "mixed-results"))
def test_real_source_diagnostics_and_actual_results_qualify_on_target(tmp_path, case):
    from t2m.effect_demo import demo
    capture, target = tmp_path / "capture", tmp_path / "target"
    demo(capture, case=case, optimizer=os.environ["TRITON_SHARED_OPT"], python=os.environ["T2M_NATIVE_PYTHON"],
         memory_profile="guarded-buffer", address_policy="source-words")
    before = E.Q.pin(capture / "capture_receipt.json")
    report = E.qualify(capture, target, **options())
    assert E.verify(target, **options()) == report
    assert report["status"] == "matched-expected-outcome" and E.Q.pin(capture / "capture_receipt.json") == before
    expected = json.loads((capture / "reference.json").read_text())
    assert (target / "source-stdout.bin").read_bytes() == expected["stdout"].encode()
    assert report["evaluation"]["termination"] == expected["termination"]
    if expected["termination"] == "assertion_failure":
        assert not (target / "outputs.npz").exists()
        process = json.loads((target / "target-process.json").read_text())
        assert process["returncode"] == 134
    else:
        assert report["evaluation"]["result_count"] == (2 if case == "mixed-results" else 0)
    stage = json.loads((target / "compiler/compile-inputs.json").read_text())
    assert not set(stage["members"]) & {"inputs.npz", "reference.json", "reference.npz"}
    build = json.loads((target / "build.json").read_text())
    assert build["reference_values_supplied_to_build"] is False
    # Retained decoded bytes are checked against raw source frames even if
    # an attacker updates their outer receipt pin.
    originals = {name: (target / name).read_bytes() for name in ("source-stdout.bin", "qualification.json")}
    try:
        (target / "source-stdout.bin").write_bytes(originals["source-stdout.bin"] + b"extra")
        changed = json.loads(originals["qualification.json"])
        changed["artifacts"]["source-stdout.bin"] = E.Q.pin(target / "source-stdout.bin")
        E.Q.write_json(target / "qualification.json", changed)
        with pytest.raises(ValueError, match="source stream differs"):
            E.verify(target, **options())
    finally:
        for name, raw in originals.items(): (target / name).write_bytes(raw)
    E.verify(target, **options())


@pytest.mark.parametrize("profile", ("tensor", "buffer", "guarded-buffer"))
def test_same_mixed_kernel_object_preserves_results_prints_empty_domains_and_abort(tmp_path, profile):
    from test_mixed_effects import convert_update
    from t2m.effect_bundle import write_effect_bundle
    from t2m.examples.mixed_effect_reference import update_reference
    result = convert_update(tmp_path, profile)
    x, state = np.arange(13, dtype=np.int64) + 2**53 + 1, np.arange(13, dtype=np.int64) * 3 - 7
    reference, expected = update_reference(x, state, 13, 3, True)
    capture = tmp_path / "capture"
    write_effect_bundle(result, capture, [x, state, np.asarray(13, np.int32), np.asarray(3, np.int32), np.asarray(True)],
        reference, references=expected, work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    objects = []
    for index, (n, steps, allow) in enumerate(((13, 3, True), (0, 3, False), (1, 0, True), (17, 1, True), (13, 1, False))):
        x, state = np.arange(n, dtype=np.int64) + 2**53 + 1, np.arange(n, dtype=np.int64) * 3 - 7
        reference, expected = update_reference(x, state, n, steps, allow)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        inputs, ref = caller / "inputs.npz", caller / "reference.json"
        np.savez(inputs, in0=x, in1=state, in2=np.asarray(n, np.int32), in3=np.asarray(steps, np.int32), in4=np.asarray(allow))
        E.Q.write_json(ref, reference)
        if reference["termination"] == "return": np.savez(caller / "reference.npz", out0=expected[0], out1=expected[1])
        target = tmp_path / f"target-{index}"
        report = E.qualify(capture, target, inputs=inputs, reference=ref, **options())
        assert report["evaluation"]["result_count"] == (0 if n and not allow else 2)
        objects.append(json.loads((target / "build.json").read_text())["build"]["host_artifacts"]["object"]["sha256"])
    assert len(set(objects)) == 1


@pytest.mark.parametrize("dtype", ("f16", "bf16", "f32", "f64"))
@pytest.mark.parametrize("hex_", (False, True))
def test_original_float_words_and_fixed_decimal_streams_reach_target(tmp_path, monkeypatch, dtype, hex_):
    import test_source_prints as source_tests
    from t2m.effect_bundle import write_effect_bundle

    def qualify_on_target(result, root, arrays, reference):
        capture = root / "capture"
        write_effect_bundle(result, capture, arrays, reference, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
        report = E.qualify(capture, root / "target", **options())
        assert report["evaluation"]["result_count"] == 0
        assert (root / "target/source-stdout.bin").read_bytes() == reference["stdout"].encode()

    # Reuse the existing independent float-word/string oracle and public
    # source fixture, selecting the target consumer for this test invocation.
    monkeypatch.setattr(source_tests, "qualify", qualify_on_target)
    source_tests.test_float_words_fixed_decimal_and_multiaxis_logical_programs(tmp_path, "guarded-buffer", dtype, hex_)


@pytest.mark.parametrize("dtype", ("i1", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"))
def test_print_and_return_preserve_every_existing_tensor_carrier(tmp_path, monkeypatch, dtype):
    import test_mixed_effects as source_tests
    from t2m.effect_bundle import write_effect_bundle

    def qualify_on_target(result, root, inputs, reference, results):
        capture = root / "capture"
        write_effect_bundle(result, capture, inputs, reference, references=results,
                            work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
        report = E.qualify(capture, root / "target", **options())
        assert report["evaluation"]["result_count"] == 1
        assert report["evaluation"]["result_comparisons"] == [{"kind": "words"}]
        with np.load(root / "target/outputs.npz", allow_pickle=False) as saved:
            assert saved["out0"].dtype == results[0].dtype and saved["out0"].tobytes() == results[0].tobytes()

    monkeypatch.setattr(source_tests, "qualify", qualify_on_target)
    source_tests.test_every_existing_tensor_carrier_keeps_printed_and_returned_words(tmp_path, "guarded-buffer", dtype)

"""Standalone native scalars and dynamic shapes reach the ordinary target backend."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_spike_model import Q, TOOLS

TOOLS = TOOLS and all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing native, Merlin and functional target tools required")


def target(capture, out, **options):
    return Q.qualify(capture, out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], comparison="words", **options)


def test_one_dynamic_kernel_object_runs_empty_single_and_ragged_domains(tmp_path):
    from t2m.cli import main
    capture = tmp_path / "capture"
    assert main(["shared-demo", "--kernel", "dynamic-count", "--out", str(capture),
        "--python", os.environ["T2M_NATIVE_PYTHON"]]) == 0
    original = Q.pin(capture / "capture_receipt.json")
    objects, programs = [], []
    for n in (7, 0, 1, 13):
        x = np.arange(n, dtype=np.float32) * np.float32(-0.5) + np.float32(1.25)
        inputs, reference = tmp_path / f"inputs-{n}.npz", tmp_path / f"reference-{n}.npz"
        np.savez(inputs, in0=x, in1=np.asarray(n, np.int32))
        np.savez(reference, out0=x + np.float32((n + 7) // 8))
        out = tmp_path / f"target-{n}"
        report = target(capture, out, inputs=inputs, reference=reference)
        assert report["schema"] == "t2m.spike-kernel-qualification.v1"
        assert report["evaluation"]["result_count"] == 1
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
        build = json.loads((out / "build.json").read_text())["build"]
        abi = build["runtime_invocation_abi"]
        assert abi["inputs"][1]["kind"] == "scalar" and abi["outputs"][0]["shape"] == [n]
        assert abi["mlir_argument_specialization"] == "not-performed"
        assert Q.pin(capture / "capture_receipt.json") == original
        objects.append(build["host_artifacts"]["object"]["sha256"])
        programs.append(Q.pin(out / "compiler/program.mlir")["sha256"])
    assert len(set(objects)) == len(set(programs)) == 1
    # Updated outer hashes cannot authorize a runtime shape absent from the
    # original staged signature and the actual invocation scalar.
    out = tmp_path / "target-7"
    originals = {name: (out / name).read_bytes() for name in ("build.json", "image/cgen/runtime-abi.json", "qualification.json")}
    try:
        build = json.loads(originals["build.json"])
        build["build"]["runtime_invocation_abi"]["outputs"][0]["shape"] = [8]
        Q.write_json(out / "build.json", build)
        Q.write_json(out / "image/cgen/runtime-abi.json", build["build"]["runtime_invocation_abi"])
        report = json.loads(originals["qualification.json"])
        for name in ("build.json", "image/cgen/runtime-abi.json"):
            report["artifacts"][name] = Q.pin(out / name)
        Q.write_json(out / "qualification.json", report)
        with pytest.raises(ValueError, match="runtime ABI does not match"):
            Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    finally:
        for name, raw in originals.items(): (out / name).write_bytes(raw)
    archive = tmp_path / "inputs-7.npz"
    original = archive.read_bytes()
    try:
        np.savez(archive, in0=np.zeros(7, np.float32), in1=np.asarray(7, np.int32))
        with pytest.raises(ValueError, match="invocation archive differs"):
            Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    finally:
        archive.write_bytes(original)
    Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    np.savez(tmp_path / "bad-inputs.npz", in0=np.arange(14, dtype=np.float32), in1=np.asarray(14, np.int32))
    np.savez(tmp_path / "bad-reference.npz", out0=np.arange(14, dtype=np.float32) + 2)
    refused = tmp_path / "refused"
    with pytest.raises(ValueError, match="outside.*guarded bounds"):
        target(capture, refused, inputs=tmp_path / "bad-inputs.npz", reference=tmp_path / "bad-reference.npz")
    assert not (refused / "qualification.json").exists() and not (refused / "image/model.elf").exists()


def scalar_arguments(*, alternate=False):
    arrays = [np.asarray(-128 if not alternate else 127, np.int8), np.asarray(32767, np.int16),
        np.asarray(-(1 << 31), np.int32), np.asarray((1 << 60) + 17, np.int64),
        np.asarray(0x7f800001 if not alternate else 0x80000000, np.uint32).view(np.float32),
        np.asarray(0xfff0000000000001 if not alternate else 0x7ff8123456789abc, np.uint64).view(np.float64),
        np.asarray(not alternate)]
    expected = np.array([*(int(a) for a in arrays[:4]), int(arrays[4].view(np.uint32)),
        int(arrays[5].view(np.int64)), int(arrays[6])], np.int64)
    return arrays, expected


def test_all_native_scalar_widths_preserve_source_words_on_host_and_target(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.examples.scalar_word_kernels import scalar_words
    from t2m.frontend import capture_ttir
    from t2m.kernel_consumer import qualify_kernel
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    names, dtypes = ("s8", "s16", "s32", "s64", "f32", "f64", "choose"), ("i8", "i16", "i32", "i64", "f32", "f64", "i1")
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(scalar_words, {"out": "*i64", **{name: "fp32" if dt == "f32" else "fp64" if dt == "f64" else dt
        for name, dt in zip(names, dtypes, strict=True)}}, {}))
    spec = TensorABI("scalar_words", (TensorBuffer("out", (7,), "write", "i64"),),
        runtime_scalars=tuple(RuntimeScalar(name, dt, i + 1) for i, (name, dt) in enumerate(zip(names, dtypes, strict=True))))
    result = convert_shared(source, spec, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words")
    arrays, expected = scalar_arguments()
    capture = tmp_path / "capture"
    write_shared_bundle(result, capture, arrays, [expected], work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(capture, tmp_path / "host", merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(capture)
    target(capture, tmp_path / "target")
    arrays, expected = scalar_arguments(alternate=True)
    inputs, reference = tmp_path / "alternate-inputs.npz", tmp_path / "alternate-reference.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    np.savez(reference, out0=expected)
    target(capture, tmp_path / "alternate-target", inputs=inputs, reference=reference)
    first = json.loads((tmp_path / "target/build.json").read_text())["build"]
    second = json.loads((tmp_path / "alternate-target/build.json").read_text())["build"]
    assert first["host_artifacts"]["object"]["sha256"] == second["host_artifacts"]["object"]["sha256"]


@pytest.mark.parametrize("bfloat", (False, True))
def test_standalone_empty_input_roster_and_exact_bfloat_storage(tmp_path, bfloat):
    from t2m.examples.scalar_word_kernels import copy_words, constant_words
    from t2m.frontend import capture_ttir
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    kernel, dtype = (copy_words, "bf16") if bfloat else (constant_words, "f32")
    names = ("x", "out") if bfloat else ("out",)
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(kernel, {name: "*bf16" if bfloat else "*fp32" for name in names}, {}))
    abi = TensorABI(kernel.__name__, tuple(TensorBuffer(name, (8,), "write" if name == "out" else "read", dtype) for name in names))
    result = convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"])
    words = np.array([0, 0x8000, 0x3f80, 0x7f81, 0xffa1, 0x7fc1, 1, 0x7f80], np.uint16)
    arrays, expected = ([words], words) if bfloat else ([], np.arange(8, dtype=np.float32) + np.float32(1.25))
    capture = tmp_path / "capture"
    write_shared_bundle(result, capture, arrays, [expected], work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    report = target(capture, tmp_path / "target")
    assert report["evaluation"]["result_count"] == 1
    build = json.loads((tmp_path / "target/build.json").read_text())["build"]
    if bfloat:
        assert build["runtime_invocation_abi"]["inputs"][0]["dtype"] == "bf16"
    else:
        assert build["runtime_invocation_abi"]["inputs"] == []
        assert json.loads((tmp_path / "target/run.json").read_text())["readonly_inputs"]["not_required"] == "source-has-no-inputs"


def test_source_grouped_memory_retains_dynamic_multi_results_on_target(tmp_path):
    from test_grouped_memory import capture as translate, arguments
    from t2m.shared_bundle import write_shared_bundle
    result = translate(tmp_path, "i64", 4, True, True)
    arrays, expected = arguments(300, 307, 3, 2, "i64", 4, True)
    capture = tmp_path / "capture"
    write_shared_bundle(result, capture, arrays, expected, work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    first = target(capture, tmp_path / "target")
    arrays, expected = arguments(0, 0, 3, 5, "i64", 4, True)
    inputs, reference = tmp_path / "empty-inputs.npz", tmp_path / "empty-reference.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    second = target(capture, tmp_path / "empty-target", inputs=inputs, reference=reference)
    assert first["evaluation"]["result_count"] == second["evaluation"]["result_count"] == 5
    objects = [json.loads((tmp_path / name / "build.json").read_text())["build"]["host_artifacts"]["object"]["sha256"]
               for name in ("target", "empty-target")]
    assert len(set(objects)) == 1

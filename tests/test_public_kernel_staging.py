"""Real TTIR packages cross Merlin's public stage/preparation/compile boundary."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_packed_source_assembly import PROVIDER, TOOLS

from t2m.bundle import _record

pytestmark = pytest.mark.skipif(not (TOOLS and PROVIDER), reason="existing source/native/Merlin tools required")


@pytest.fixture(scope="module", params=("tensor", "source-words", "owned", "atomic", "effect"))
def compiled(tmp_path_factory, request):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import (
        load_compile_inputs,
        stage_compile_inputs,
    )

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    mode = request.param
    root = tmp_path_factory.mktemp("public-kernel-" + mode)
    capture = root / "capture"
    if mode in ("tensor", "source-words"):
        from test_packed_source_assembly import arguments
        from test_packed_source_assembly import capture as convert

        width = 32 if mode == "tensor" else 64
        result = convert(root / "source", width, 2, "tensor" if mode == "tensor" else "guarded-buffer")
        arrays, expected = arguments(width)
        write_shared_bundle(result, capture, arrays, expected, work=root / "native",
                            python=os.environ["T2M_NATIVE_PYTHON"])
    elif mode == "atomic":
        from test_atomic_memory import arguments
        from test_atomic_memory import capture as convert

        source = root / "source"
        source.mkdir()
        result = convert(source, os.environ["TRITON_SHARED_OPT"], operation="add", width=32)
        arrays, expected = arguments(13, "add", 32, collide=True)
        write_shared_bundle(result, capture, arrays, expected, work=root / "native",
                            python=os.environ["T2M_NATIVE_PYTHON"])
    elif mode == "owned":
        from test_linear_tile_scatter import capture as convert

        result = convert(root / "source", "f32", "buffer")
        data = np.arange(13, dtype=np.float32) - 11
        arrays, expected = [data, np.asarray(13, np.int32)], [data * 2 + 1]
        write_shared_bundle(result, capture, arrays, expected, work=root / "native",
                            python=os.environ["T2M_NATIVE_PYTHON"])
    else:
        from test_effect_bundle import convert, reference

        from t2m.effect_bundle import write_effect_bundle

        source = root / "source"
        source.mkdir()
        result = convert(source, "tensor", scalar=True)
        arrays = [np.asarray(3, np.int32), np.asarray(3, np.int32)]
        expected = []
        write_effect_bundle(result, capture, arrays, reference(), work=root / "native",
                            python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(capture, root / "stage", result_mode="execution-owned" if mode == "owned" else None)
    loaded = load_compile_inputs(root / "stage")
    assert loaded.source_profile == ("standalone-effect-kernel" if mode == "effect" else "standalone-tensor-kernel")
    assert len(loaded.argument_bindings) == len(arrays)
    assert loaded.signature["schema"] == ("merlin.compile-signature.v6" if mode == "owned" else
        "merlin.compile-signature.v1" if mode == "effect" else "merlin.compile-signature.v2")
    product = compile(StagedProgram(root / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=root / "compiled")
    product.verify()
    names = {str(artifact.path.relative_to(product.directory)) for artifact in product.artifacts}
    manifest = json.loads((root / "stage/compile-inputs.json").read_text())
    assert {"frontend/" + name for name in manifest["members"]} <= names
    assert not any(path.name in {"inputs.npz", "golden.npy", "extra.npz", "reference.json"}
                   for path in (product.directory / "frontend").rglob("*") if path.is_file())
    assert (product.directory / "frontend.mlir").read_text() == loaded.program
    return root, mode, arrays, expected


def invoke(root, arrays, index):
    from t2m.consumer import _host_worker

    inputs = root / f"public-inputs-{index}.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    out = root / f"public-invoke-{index}"
    out.mkdir()
    _host_worker("invoke", Path(os.environ["T2M_MERLIN_ROOT"]), root / "compiled/frontend",
        root / "compiled/host", [str(inputs.resolve()), str(out.resolve())],
        dict(os.environ, TMPDIR=str(out)), os.environ["T2M_MERLIN_PYTHON"])
    with np.load(out / "outputs.npz", allow_pickle=False) as result:
        return [result[f"out{i}"].copy() for i in range(len(result.files))]


def test_public_compiled_library_retains_all_results_and_runtime_values(compiled):
    root, mode, arrays, expected = compiled
    actual = invoke(root, arrays, 0)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert (left.shape, left.dtype, left.tobytes()) == (right.shape, right.dtype, right.tobytes())
    library = _record(root / "compiled/host/model.so")
    if mode in ("tensor", "source-words"):
        from test_packed_source_assembly import arguments

        for index, (n, steps) in enumerate(((0, 3), (31, 1)), 1):
            arrays, expected = arguments(32 if mode == "tensor" else 64, n, steps)
            for left, right in zip(invoke(root, arrays, index), expected, strict=True):
                assert (left.shape, left.dtype, left.tobytes()) == (right.shape, right.dtype, right.tobytes())
    elif mode == "atomic":
        from test_atomic_memory import arguments

        for index, n in enumerate((0, 31), 1):
            arrays, expected = arguments(n, "add", 32, collide=True)
            for left, right in zip(invoke(root, arrays, index), expected, strict=True):
                assert (left.shape, left.dtype, left.tobytes()) == (right.shape, right.dtype, right.tobytes())
    elif mode == "owned":
        assert invoke(root, [np.asarray([], np.float32), np.asarray(0, np.int32)], 1)[0].shape == (0,)
    else:
        assert invoke(root, [np.asarray(-1, np.int32), np.asarray(0, np.int32)], 1) == []
        from t2m.effect_reference import compare_outcome
        from t2m.process_outcome import observe

        inputs = root / "failing-inputs.npz"
        np.savez(inputs, in0=np.asarray(-1, np.int32), in1=np.asarray(3, np.int32))
        out = root / "failing-invoke"
        out.mkdir()
        worker = Path(__file__).resolve().parents[1] / "t2m/consumer_worker.py"
        observe([os.environ["T2M_MERLIN_PYTHON"], str(worker), "observe", os.environ["T2M_MERLIN_ROOT"],
            str(root / "compiled/frontend"), str(root / "compiled/host"), str(inputs), str(out)], out,
            environment=dict(os.environ, TMPDIR=str(out)), timeout=60)
        from test_effect_bundle import reference

        compare_outcome(out, reference(True, "effect scalar must be nonnegative"), backend="merlin-host",
            program=_record(root / "compiled/frontend/program.mlir"), arguments=_record(inputs),
            implementation={"kind": "shared-library", "artifact": _record(root / "compiled/host/model.so")}, result_count=0)
    assert _record(root / "compiled/host/model.so") == library


@pytest.mark.parametrize("damage", ("extra", "symlink", "signature", "program", "numeric-policy", "missing-source"))
def test_rehashed_compiler_projection_cannot_replace_source_abi_or_policy(compiled, tmp_path, damage):
    from merlin.compiler.frontend.inputs import load_compile_inputs

    root, _, _, _ = compiled
    stage = tmp_path / "stage"
    shutil.copytree(root / "stage", stage)
    manifest = json.loads((stage / "compile-inputs.json").read_text())
    member = None
    if damage == "extra":
        member = "source/golden.npy"
        (stage / member).write_bytes(b"evaluator values must stay outside compilation")
    elif damage == "symlink":
        (stage / "source/source.ttir").unlink()
        (stage / "source/source.ttir").symlink_to(root / "capture/source.ttir")
    elif damage == "signature":
        member = "signature.json"
        signature = json.loads((stage / member).read_text())
        signature["inputs"] = []
        (stage / member).write_text(json.dumps(signature))
    elif damage == "program":
        member = "program.mlir"
        (stage / member).write_text((stage / member).read_text() + "\n// independently resealed replacement\n")
    elif damage == "numeric-policy":
        member = "precision-policy.json"
        policy = json.loads((stage / member).read_text())
        policy["numerical_transforms"] = ["reassociate"]
        (stage / member).write_text(json.dumps(policy))
    else:
        member = "source/source.ttir"
        (stage / member).unlink()
        del manifest["members"][member]
        member = None
    if member:
        manifest["members"][member] = _record(stage / member)["sha256"]
    (stage / "compile-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_compile_inputs(stage)

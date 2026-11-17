"""Production generated driver/readback execute the complete real TTIR model."""

import hashlib
import json
import os
import resource
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_compact_model import REQUIRED, parent  # noqa: F401
from test_compact_model import selected as captured_selection

from t2m.bundle import verify_bundle

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing source/model/Merlin tools required")
selected = captured_selection


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@pytest.fixture(scope="module")
def program(selected):
    _root, _parent, _pack, package = selected
    verify_bundle(package)
    interface = json.loads((package / "consumer/interface.json").read_text())
    assert all(name in interface["consumer_source_sha256"] for name in (
        "src/merlin/llvmlower/owned_result_driver.py", "src/merlin/runtime/owned_result_readback.py",
        "merlin/runtime/c/owned_result_host_main.c"))
    root = Path(os.environ["T2M_MERLIN_ROOT"])
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "src"))
        from merlin.llvmlower.owned_result_driver import generate, verify
        from merlin.runtime.owned_result_readback import decode
    source = (package / "consumer/compiler/program.mlir").read_text()
    signature = json.loads((package / "consumer/compiler/signature.json").read_text())
    llvm = package / "consumer/host/model.ll"
    assert signature["schema"] == "merlin.compile-signature.v6"
    return root, package, source, signature, llvm, generate, verify, decode


def compile_driver(program, arrays, root):
    merlin, _package, source, signature, llvm, generate, verify, _decode = program
    generated = root / "driver"
    record = generate(source, signature, arrays, generated)
    runtime, abi = merlin / "merlin/runtime/c", merlin / "merlin/runtime/abi"
    executable = root / "model"
    done = subprocess.run(
        [os.environ["T2M_MERLIN_CLANG"], "-std=c11", "-Wall", "-Wextra", "-O2",
         "-I", str(runtime), "-I", str(abi), str(generated / "model_owned.c"), str(llvm),
         *(str(runtime / name) for name in ("owned_result_driver.c", "owned_result_readback.c",
                                          "owned_result_host_main.c")),
         *(str(abi / name) for name in ("owned_results.c", "owned_result_views.c", "mlir_runtime.c")),
         "-lm", "-o", str(executable)],
        capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, TMPDIR=str(root)),
    )
    assert done.returncode == 0, done.stderr
    verify(source, signature, arrays, generated)
    return executable, generated, record


@pytest.mark.parametrize("capacity", [2, 13, 31])
def test_one_driver_preserves_all_results_with_execution_produced_shapes(program, tmp_path, capacity):
    _merlin, package, source, signature, llvm, _generate, verify, decode = program
    arrays = [np.asarray(-3, np.int32), np.arange(capacity, dtype=np.int32)]
    executable, generated, record = compile_driver(program, arrays, tmp_path)
    pins = {path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (llvm, executable, generated / "model_owned.c")}
    cases = [(np.full(capacity, -1, np.int32), 0), (np.arange(capacity, dtype=np.int32) - 2, 0),
             (np.resize(np.array([-(1 << 31), (1 << 31) - 1, -1, 7], np.int32), capacity), 1),
             (np.arange(capacity, dtype=np.int32), 100)]
    for ordinal, (x, weight) in enumerate(cases):
        parameter = np.asarray(weight, np.int32)
        invocation = bytearray((generated / "invocation.bin").read_bytes())
        for row, value in zip(record["inputs"], (parameter, x), strict=True):
            assert row["bytes"] == value.nbytes
            invocation[row["offset"]:row["offset"] + row["bytes"]] = value.tobytes()
        path = tmp_path / f"input-{ordinal}.bin"
        path.write_bytes(invocation)
        done = subprocess.run([str(executable), str(path)], capture_output=True, timeout=10,
                              check=False, preexec_fn=no_core)
        assert done.returncode == 0, done.stderr
        outputs, observed = decode(source, signature, [parameter, x], done.stdout)
        original = np.add(x, parameter, dtype=np.int32)
        packed = original[original > 0]
        expected = (packed, packed, original, x, np.asarray([packed.size], np.int64), parameter, packed[::2])
        assert len(outputs) == len(expected) == 7
        for actual, wanted in zip(outputs, expected, strict=True):
            assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
        assert observed["functional_inputs_preserved"]
        (tmp_path / f"stdout-{ordinal}.bin").write_bytes(done.stdout)
        np.savez(tmp_path / f"reference-{ordinal}.npz", **{f"out{i}": value for i, value in enumerate(expected)})
        (tmp_path / f"observation-{ordinal}.json").write_text(json.dumps(observed, sort_keys=True) + "\n")
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in pins.items())
    verify(source, signature, arrays, generated)
    verify_bundle(package)


@pytest.mark.parametrize("capacity", [1, 32])
def test_generated_driver_retains_the_real_exported_source_assertions(program, tmp_path, capacity):
    _merlin, _package, _source, _signature, _llvm, _generate, _verify, _decode = program
    arrays = [np.asarray(0, np.int32), np.arange(capacity, dtype=np.int32)]
    executable, generated, _record = compile_driver(program, arrays, tmp_path)
    done = subprocess.run([str(executable), str(generated / "invocation.bin")],
                          capture_output=True, timeout=10, check=False, preexec_fn=no_core)
    assert done.returncode == -signal.SIGABRT
    assert b"exported shape domain:" in done.stdout + done.stderr
    assert b"OUT_BIN_BEGIN" not in done.stdout and b"DONE" not in done.stdout
    (tmp_path / "abort.json").write_text(json.dumps({
        "returncode": done.returncode, "stdout": done.stdout.decode(errors="replace"),
        "stderr": done.stderr.decode(errors="replace"),
    }, sort_keys=True) + "\n")

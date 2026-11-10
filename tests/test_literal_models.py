"""All measured tensor carriers keep exact lifted constants through real Merlin."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_bfloat_model import REQUIRED, M2M, NATIVE, MERLIN, CLANG, OPT

DTYPES = ("f16", "bf16", "f32", "f64", "i1", "i8", "i16", "i32", "i64")


@pytest.fixture(scope="module", params=DTYPES)
def captured(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("literal-" + request.param)
    script = Path(__file__).resolve().parents[1] / "scripts/capture_literal_model.py"
    result = subprocess.run([M2M, str(script), "--out", str(root / "capture"), "--dtype", request.param],
        capture_output=True, text=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert result.returncode == 0, result.stdout + result.stderr
    return root / "capture", request.param


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_source_literal_words_and_public_abi_survive_merlin(tmp_path, captured, profile):
    from t2m.bundle import verify_bundle
    from t2m.examples.literal_kernels import literal_copy
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_words import TensorWords
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source, dtype = captured
    ttir = tmp_path / "source.ttir"
    pointer = "*fp" + dtype[1:] if dtype.startswith("f") else "*" + dtype
    ttir.write_text(capture_ttir(literal_copy, {"literal": pointer, "out": pointer, "N": "constexpr", "B": "constexpr"}, {"N": 17, "B": 8}))
    abi = TensorABI("literal_copy", (TensorBuffer("literal", (17,), dtype=dtype), TensorBuffer("out", (17,), "write", dtype)), grid=(3,))
    implementation = convert_shared(ttir, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    out, work = tmp_path / "model", tmp_path / "qualification"
    write_model_bundle(source, [KernelBinding("t2m_demo::literal_copy", implementation)], out, work=work,
        python=NATIVE, merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=True, merlin_clang=CLANG)
    verify_bundle(out)
    projection = json.loads((out / "consumer/compiler/compiler-constants.json").read_text())
    assert len(projection["bindings"]) == 1
    compatibility = json.loads((out / "compatibility.json").read_text())
    assert any("m2m.compiler-constants.v1" in difference for difference in compatibility["differences"])
    with TensorWords(source) as words, np.load(source / "goldens.npz", allow_pickle=False) as legacy:
        expected = words.get("goldens.npz", "out0", legacy["out0"], dtype, [17])
        for path in (work / "native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual["out0"].dtype == expected.dtype
                assert actual["out0"].tobytes() == expected.tobytes()
                np.testing.assert_array_equal(actual["out1"], legacy["out1"])

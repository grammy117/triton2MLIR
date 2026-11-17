"""Real materializer subprocesses retain parent BF16 words for bound receipts."""

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_model_runtime_inputs import fixture


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"),
    reason="existing native materialization environment required",
)
@pytest.mark.parametrize(
    "schema", ("t2m.bound-model-receipt.v1", "t2m.bound-model-receipt.v2", "unknown")
)
def test_bound_argument_materialization_uses_exact_parent_word_declarations(
    tmp_path, schema
):
    capture, work, signature, raw = fixture(tmp_path)
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    receipt["schema"] = schema
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    np.savez(capture / "extra.npz")
    header = b"{}      "
    (capture / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header
    )
    (work / "abi.json").write_text(json.dumps(signature["inputs"]))
    script = Path(__file__).resolve().parents[1] / "t2m/model_inputs.py"
    done = subprocess.run(
        [os.environ["T2M_NATIVE_PYTHON"], str(script), str(capture), str(work)],
        capture_output=True,
        timeout=15,
        check=False,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(work)
        ),
    )
    if schema == "unknown":
        assert (
            done.returncode != 0 and b"exact parent archive declaration" in done.stderr
        )
        assert not (work / "arguments.npz").exists()
    else:
        assert done.returncode == 0, done.stderr.decode()
        with np.load(work / "arguments.npz", allow_pickle=False) as actual:
            assert (
                actual["in1"].dtype == np.uint16
                and actual["in1"].tobytes() == raw.tobytes()
            )
            assert actual["in0"].tolist() == [2**62 + 3, -7]

"""Public CLI emits checked effect packets without numeric placeholders."""

import json
import os

import pytest

from t2m.bundle import verify_bundle
from t2m.cli import main

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE),reason="existing native toolchain required")


@pytest.mark.parametrize("case", ["constant-fail","empty-return","fail-before-store","tensor-check"])
def test_effect_examples_are_real_checked_packets(tmp_path,capsys,case):
    out = tmp_path / "capture"
    assert main(["effect-demo","--case",case,"--out",str(out),"--optimizer",OPT,"--python",NATIVE]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reference_comparison"] == "passed" and summary["result_count"] == 0
    assert verify_bundle(out)["schema"] == "t2m.effect-kernel-receipt.v1"
    assert not (out / "golden.npy").exists()


def test_saved_ttir_can_use_an_explicit_effect_reference_with_no_input_archive(tmp_path,capsys):
    source,out = tmp_path / "example",tmp_path / "from-ttir"
    assert main(["effect-demo","--out",str(source),"--optimizer",OPT,"--python",NATIVE]) == 0
    capsys.readouterr()
    assert main(["shared-tensor",str(source / "source.ttir"),"--abi",str(source / "spec.json"),
        "--effect-reference",str(source / "reference.json"),"--out",str(out),"--optimizer",OPT,"--python",NATIVE]) == 0
    assert json.loads(capsys.readouterr().out)["checked_bundle"] is True
    verify_bundle(out)

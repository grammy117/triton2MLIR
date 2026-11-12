"""Numeric shared packets retain the exact existing CFG compiler source roster."""

import json
import os
import shutil

import pytest

from t2m.bundle import _record, verify_bundle
from t2m.cfg_intake import FILES
from t2m.shared_bundle import write_shared_bundle

pytestmark = pytest.mark.skipif(
    not (os.environ.get("TRITON_SHARED_OPT") and os.environ.get("T2M_NATIVE_PYTHON")),
    reason="existing shared/native tools required",
)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    from test_cfg_intake import arguments, capture

    root = tmp_path_factory.mktemp("shared-cfg")
    result = capture(root, "tensor")
    inputs, expected = arguments()
    out = root / "capture"
    write_shared_bundle(
        result,
        out,
        inputs,
        [expected],
        work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    return out


def test_numeric_packet_admits_the_complete_existing_cfg_profile(packet):
    receipt = verify_bundle(packet)
    assert FILES <= set(receipt["artifacts"])
    assert len(FILES) == 6
    provenance = json.loads((packet / "provenance.json").read_text())
    assert provenance["control_flow_normalization"]["grants_semantics"] is False


@pytest.mark.parametrize(
    "change", ["extra", "missing", "undeclared", "changed-output", "authority"]
)
def test_rehashed_cfg_packet_still_requires_exact_roster_and_source_byte_chain(
    packet, tmp_path, change
):
    out = tmp_path / "changed"
    shutil.copytree(packet, out)
    receipt = json.loads((out / "capture_receipt.json").read_text())
    if change == "extra":
        (out / "cfg-extra.mlir").write_text("extra compiler payload")
        receipt["artifacts"]["cfg-extra.mlir"] = _record(out / "cfg-extra.mlir")
    elif change == "missing":
        del receipt["artifacts"]["cfg-output.ttir"]
    elif change == "changed-output":
        path = out / "cfg-output.ttir"
        path.write_text(path.read_text() + "\n")
        receipt["artifacts"][path.name] = _record(path)
    else:
        path = out / "provenance.json"
        provenance = json.loads(path.read_text())
        if change == "undeclared":
            del provenance["control_flow_normalization"]
        else:
            provenance["control_flow_normalization"]["grants_semantics"] = True
        path.write_text(json.dumps(provenance))
        receipt["artifacts"][path.name] = _record(path)
    (out / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt|CFG|control-flow"):
        verify_bundle(out)

"""Resource requests refuse before opening model data or invoking conversion."""

import json

import pytest

from t2m.cli import main
from t2m.model_bundle import write_model_bundle


def test_api_does_not_apply_owned_resources_to_a_legacy_model(tmp_path):
    with pytest.raises(ValueError, match="explicit execution-owned"):
        write_model_bundle(
            tmp_path / "absent-capture",
            [],
            tmp_path / "output",
            work=tmp_path / "work",
            execution_allocation_policy={"max_allocations": 16384},
        )
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("owned", (False, True))
def test_cli_rejects_unselected_or_unbounded_policies_before_conversion(
    tmp_path, capsys, owned
):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"schema": "merlin.owned-result-limits.v2", "max_allocations": -1})
    )
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "bind-model",
                str(tmp_path / "absent-capture"),
                "--bindings",
                str(tmp_path / "absent-bindings"),
                "--out",
                str(tmp_path / "output"),
                "--execution-allocation-policy",
                str(policy),
                *(["--result-mode", "execution-owned"] if owned else []),
            ]
        )
    assert stopped.value.code == 2
    assert "execution allocation policy requires" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.work").exists()

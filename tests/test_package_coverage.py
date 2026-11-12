"""Complete denominators and byte-bound observations retain unmeasured scope."""

import json
import os
import shutil

import pytest

from t2m.package_coverage import operation_observations, package_coverage
from t2m.ttir_definitions import OPERATIONS, SOURCE


def test_empty_selection_retains_every_pinned_definition():
    report = package_coverage([])
    assert report["definition_count"] == len(OPERATIONS) == 50
    assert SOURCE["revision"] == "f797708c0626e5f9840ca5b0a98790e2c7cb09ad"
    assert len(report["unmeasured_definitions"]) == 50
    assert report["observed_definition_count"] == report["merlin_observed_definition_count"] == 0
    assert report["universal_opcode_type_attribute_coverage"] == "not-established"
    assert report["ttir"]["tt.dot_scaled"]["status"] == "unmeasured"


def test_unknown_operations_and_exact_typed_forms_never_inherit_support():
    def op(id_, name, dtype="tensor<8xi64>"):
        return {"id":id_, "opcode":name, "input_types":[dtype], "output_types":[dtype],
            "attributes":{}, "properties":{"predicate":"1 : i64"}}
    cases = [{"id":"declared-fixture", "merlin_host":False,
        "operations":[op("s0","tt.load"), op("s1","tt.load","tensor<16xi32>"),
                      op("s2","tt.future_operation"), op("s3","arith.xori")]}]
    report = operation_observations(cases)
    assert len(report["ttir"]) == 50
    assert len(report["ttir"]["tt.load"]["forms"]) == 2
    assert not report["ttir"]["tt.load"]["merlin_cases"]
    assert report["unknown_ttir"]["tt.future_operation"]["universal_coverage"] == "not-established"
    assert report["other_operations"]["arith.xori"]["cases"] == ["declared-fixture"]
    cases[0]["operations"].append(op("s0","tt.load"))
    with pytest.raises(ValueError, match="unique typed"):
        operation_observations(cases)


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../b"])
def test_foreign_artifact_paths_refuse_before_package_verification(tmp_path, name):
    (tmp_path / "capture_receipt.json").write_text(json.dumps({"artifacts":{name:{}}}))
    with pytest.raises(ValueError, match="local nonsymlink"):
        package_coverage([tmp_path])


NATIVE = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
MERLIN = all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    if not NATIVE:
        pytest.skip("existing native toolchain required")
    from t2m.cli import shared_demo
    root = tmp_path_factory.mktemp("coverage-package")
    shared_demo(root / "capture", kernel="descriptor-matmul", optimizer=os.environ["TRITON_SHARED_OPT"], python=os.environ["T2M_NATIVE_PYTHON"])
    return root


def test_actual_package_preserves_original_graph_without_promoting_native_to_merlin(captured, tmp_path):
    from t2m.cli import main
    source = captured / "capture"
    report = package_coverage([source])
    assert report["observed_definition_count"] > 0 and report["merlin_observed_definition_count"] == 0
    assert len(report["ttir"]) == 50 and report["unmeasured_definitions"]
    trace = json.loads((source / "provenance.json").read_text())
    expected = {row["opcode"] for row in trace["source_operations"] if row["opcode"].startswith("tt.")}
    assert {name for name,row in report["ttir"].items() if row["cases"]} == expected
    assert report["kernel_cases"][0]["normalized_operation_counts"]
    name, row = next((name, row) for name, row in report["ttir"].items() if row["forms"])
    original_form = row["forms"][0]
    original_op = next(op for op in trace["source_operations"] if op["opcode"] == name)
    assert original_form["attribute_evidence"] == "recorded-source-dictionaries"
    assert original_form["attributes"] == original_op["attributes"]
    assert original_form["properties"] == original_op["properties"]
    assert main(["package-coverage",str(source),"--out",str(tmp_path / "report")]) == 0
    with pytest.raises(ValueError, match="duplicate package"):
        package_coverage([source,source])


def test_changed_package_bytes_or_symlinked_parent_cannot_supply_observations(captured, tmp_path):
    source = captured / "capture"
    changed = tmp_path / "changed"
    shutil.copytree(source,changed)
    (changed / "source.ttir").write_text("changed source")
    with pytest.raises(ValueError, match="integrity mismatch"):
        package_coverage([changed])
    linked = tmp_path / "linked"
    linked.symlink_to(captured,target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        package_coverage([linked / "capture"])


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
def test_standalone_qualification_must_match_the_exact_selected_receipt(captured, tmp_path):
    from t2m.kernel_consumer import qualify_kernel
    source, target = captured / "capture", tmp_path / "merlin"
    qualify_kernel(source,target,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"],clang=os.environ["T2M_MERLIN_CLANG"],rtol=1e-5,atol=6e-6)
    report = package_coverage([source],qualifications=[target])
    assert report["merlin_observed_definition_count"] == report["observed_definition_count"]
    assert report["kernel_cases"][0]["merlin_host"]
    assert report["packages"][0]["merlin_numeric_policy"] == {"rtol":1e-5, "atol":6e-6}
    with pytest.raises(ValueError, match="no matching selected source"):
        package_coverage([],qualifications=[target])
    (target / "host/execution.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity mismatch"):
        package_coverage([source],qualifications=[target])

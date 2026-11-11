"""Real success/refusal/failure cases share one complete source denominator."""

import json
import os
import shutil

import numpy as np
import pytest

from t2m.product_corpus import qualify_product_corpus, verify_product_corpus
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing pinned/native tools required")


def prepare(root):
    root.mkdir()
    source = '''module { tt.func public @copy(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
      %v=tt.load %x:!tt.ptr<i32>
      tt.store %out,%v:!tt.ptr<i32>
      tt.return
    } }'''
    (root / "copy.ttir").write_text(source)
    (root / "physical.ttir").write_text('''module {
      tt.func public @copy(%x:!tt.ptr<i32>,%out:!tt.ptr<i64>) {
        %bits=tt.ptr_to_int %x:!tt.ptr<i32> -> i64
        tt.store %out,%bits:!tt.ptr<i64>
        tt.return
      }
    }''')
    (root / "assembly.ttir").write_text('''module {
      tt.func public @copy(%x:!tt.ptr<i32>,%out:!tt.ptr<i32>) {
        %v=tt.load %x:!tt.ptr<i32>
        %w="tt.elementwise_inline_asm"(%v) <{asm_string="add.u32 $0, $1, 1;",
          constraints="=r,r", pure=true, packed_element=1:i32}> : (i32)->i32
        tt.store %out,%w:!tt.ptr<i32>
        tt.return
      }
    }''')
    (root / "invalid.ttir").write_text('module { "tt.future_operation"() : ()->() }')
    (root / "void.ttir").write_text('module { tt.func public @empty() {tt.return} }')
    abi = TensorABI("copy", (TensorBuffer("x", (1,), dtype="i32"), TensorBuffer("out", (1,), "write", "i32")))
    for i, value in enumerate((42, -73)):
        np.savez(root / f"inputs{i}.npz", in0=np.asarray([value], np.int32))
        np.savez(root / f"reference{i}.npz", out0=np.asarray([value], np.int32))
    np.savez(root / "wrong.npz", out0=np.asarray([99], np.int32))
    np.savez(root / "empty.npz")
    (root / "outcome.json").write_text(json.dumps({"schema": "t2m.effect-reference.v1", "termination": "return",
                                                  "stdout": "", "stderr": ""}))
    base = {"source": "copy.ttir", "abi": abi.as_dict(), "options": {
        "memory_profile": "guarded-buffer", "address_policy": "source-words"}}
    physical = abi.as_dict()
    physical["buffers"][-1]["dtype"] = "i64"
    examples = [{"inputs": f"inputs{i}.npz", "reference": f"reference{i}.npz"} for i in range(2)]
    cases = [{**base, "id": "copy", "examples": examples},
        {**base, "id": "physical", "source": "physical.ttir", "abi": physical},
        {**base, "id": "assembly", "source": "assembly.ttir"},
        {**base, "id": "invalid", "source": "invalid.ttir"},
        {**base, "id": "wrong-reference", "examples": [{"inputs": "inputs0.npz", "reference": "wrong.npz"}]},
        {**base, "id": "foreign", "source": "../outside.ttir"},
        {"id": "void", "source": "void.ttir", "abi": TensorABI("empty", (), effect_only=True).as_dict(),
         "reference_kind": "effects", "examples": [{"inputs": "empty.npz", "reference": "outcome.json"}] * 2}]
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": "t2m.product-corpus.v1", "cases": cases}))
    return manifest


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("product-corpus")
    manifest = prepare(root / "inputs")
    report = qualify_product_corpus(manifest, root / "measured", optimizer=OPT, python=NATIVE,
        merlin_root=os.environ.get("T2M_MERLIN_ROOT"), merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
        clang=os.environ.get("T2M_MERLIN_CLANG"))
    return root, report


def test_every_case_and_definition_remains_in_the_actual_product_report(measured):
    root, report = measured
    assert report["selected_cases"] == report["measured_cases"] == 7 and report["complete_selection_measured"]
    assert len(report["ttir"]) == report["definition_count"] == 50
    assert not report["all_selected_cases_merlin_host_qualified"]
    cases = {row["id"]: row for row in report["cases"]}
    assert cases["copy"]["stages"]["native_package"] == cases["void"]["stages"]["native_package"] == "passed"
    assert cases["physical"]["status"] == cases["assembly"]["status"] == "refused"
    assert cases["invalid"]["stages"]["source"] == "failed" and not cases["invalid"]["operations"]
    assert cases["wrong-reference"]["failure"]["stage"] == "native_package"
    assert cases["foreign"]["failure"]["stage"] == "case_inputs"
    assert report["ttir"]["tt.ptr_to_int"]["attributed_refusals"] == ["physical"]
    assert not report["ttir"]["tt.ptr_to_int"]["stage_cases"]["native_package"]
    assert report["universal_opcode_type_attribute_coverage"] == "not-established"
    assert verify_product_corpus(root / "measured") == report


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
def test_actual_merlin_libraries_and_effect_replays_have_separate_evidence(measured):
    _, report = measured
    cases = {row["id"]: row for row in report["cases"]}
    for name in ("copy", "void"):
        assert cases[name]["status"] == "merlin-host-qualified"
        assert cases[name]["native_reference_examples"] == 1 and cases[name]["merlin_reference_examples"] == 2
    assert report["stage_definition_counts"]["merlin_host"] > 0
    assert not report["ttir"]["tt.elementwise_inline_asm"]["merlin_cases"]


@pytest.mark.parametrize("member", ("copy/package/model.mlir", "physical/failure.log", "copy/case.json"))
def test_changed_executable_refusal_or_case_record_cannot_supply_evidence(measured, tmp_path, member):
    root, _ = measured
    target = tmp_path / "changed"
    shutil.copytree(root / "measured", target)
    path = target / member
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_product_corpus(target)


def test_limit_keeps_unstarted_cases_and_definitions_in_the_report(measured, tmp_path):
    from t2m.cli import main

    root, _ = measured
    assert main(["product-coverage", str(root / "inputs/manifest.json"), "--out", str(tmp_path / "partial"),
                 "--optimizer", OPT, "--python", NATIVE, "--limit", "1"]) == 0
    report = json.loads((tmp_path / "partial/coverage.json").read_text())
    assert report["selected_cases"] == 7 and report["measured_cases"] == 1
    assert not report["complete_selection_measured"] and len(report["cases"]) == 7
    assert len(report["unmeasured_definitions"]) > 0
    assert report["cases"][1]["status"] == "not-started"
    verify_product_corpus(tmp_path / "partial")


def test_aggregate_success_cannot_override_measured_refusals(measured, tmp_path):
    root, _ = measured
    target = tmp_path / "changed"
    shutil.copytree(root / "measured", target)
    path = target / "coverage.json"
    report = json.loads(path.read_text())
    report["all_selected_cases_merlin_host_qualified"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="aggregate status"):
        verify_product_corpus(target)


def test_registry_v2_joins_verified_refusals_without_promoting_them_to_execution(measured, tmp_path):
    from t2m.cli import main
    from t2m.package_coverage import package_coverage

    root, _ = measured
    report = package_coverage([], corpora=[root / "measured"])
    assert report["schema"] == "t2m.package-coverage.v2" and len(report["product_cases"]) == 7
    assert report["observed_definition_count"] > report["native_observed_definition_count"]
    assert report["ttir"]["tt.ptr_to_int"]["attributed_product_refusals"] == ["corpus.0:physical"]
    assert not report["ttir"]["tt.ptr_to_int"]["merlin_cases"]
    assert main(["package-coverage", "--corpus", str(root / "measured"), "--out", str(tmp_path / "registry")]) == 0
    with pytest.raises(ValueError, match="duplicate product corpus"):
        package_coverage([], corpora=[root / "measured"] * 2)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin toolchain required")
def test_maintained_missing_family_corpus_qualifies_actual_packages_and_keeps_gaps(tmp_path):
    from t2m.examples.product_corpus import prepare

    manifest = prepare(tmp_path / "inputs")
    report = qualify_product_corpus(manifest, tmp_path / "report", optimizer=OPT, python=NATIVE,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    assert report["selected_cases"] == 13 and report["complete_selection_measured"]
    cases = {row["id"]: row for row in report["cases"]}
    gaps = {"packed-map", "source-assembly"}
    assert {name for name, row in cases.items() if row["status"] == "refused"} == gaps
    assert all(row["merlin_host"] for name, row in cases.items() if name not in gaps)
    assert report["ttir"]["tt.ptr_to_int"]["merlin_cases"] == ["pointer-roundtrip"]
    assert report["ttir"]["tt.reduce"]["merlin_cases"] and report["ttir"]["tt.scan"]["merlin_cases"]
    verify_product_corpus(tmp_path / "report")

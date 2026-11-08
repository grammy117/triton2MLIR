import os
from pathlib import Path

import pytest

from t2m.coverage import chunks, qualify_corpus


def test_corpus_split_uses_only_the_mlir_test_delimiter():
    assert list(chunks("// a--b\nmodule {}\n// -----\nmodule {}\n")) == ["// a--b\nmodule {}\n", "module {}\n"]
    assert list(chunks("module {}\n// ------\nmodule {}\n")) == ["module {}\n", "module {}\n"]
    assert list(chunks("// ---- explanation\nmodule {}\n")) == ["// ---- explanation\nmodule {}\n"]


@pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing optimizer not configured")
def test_actual_corpus_report_does_not_turn_stage_success_into_product_coverage(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    fixture = Path(__file__).parent / "fixtures/triton_shared/addptr_2d_example.mlir"
    (corpus / "a-good.mlir").write_bytes(fixture.read_bytes())
    (corpus / "b-negative.mlir").write_text('module { "tt.unqualified"() : () -> () }')
    report = qualify_corpus(corpus, tmp_path / "report", budget=15)
    assert report["complete_corpus_measured"] and report["measured_cases"] == 2
    assert report["source_native_verified"] == report["stage_successes"] == 1
    assert "tt.load" in report["operation_observations"]
    assert not report["cases"][0]["functional_tensor_qualified"]
    assert "tts.load" in report["cases"][0]["remaining_project_ops"]
    assert report["numeric_equivalence"] == "not-tested"
    partial = qualify_corpus(corpus, tmp_path / "partial", limit=1)
    assert not partial["complete_corpus_measured"] and partial["measured_cases"] == 1


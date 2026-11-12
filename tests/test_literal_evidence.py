"""Original typed literal declarations survive auditing and reject mismatches."""

from copy import deepcopy

import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.float_literals import materialize, validate_reports, validate_trace
from t2m.ir import text
from t2m.shared_tensor import _preflight, _source_ids, parse_shared, parse_shared_source
from t2m.source_graph import inventory


def declared_source():
    module, words = parse_shared_source(""""builtin.module"() ({
      "tt.func"() <{sym_name="literal",function_type=()->()}> ({
        %value=arith.constant 0xFFBFFFFF:f32
        "tt.return"():()->()
      }):()->()
    }):()->()""")
    records = _preflight(module, source_entry="literal", literal_words=words)
    original = inventory(module, records, literal_words=words)
    reports = materialize(module, words, source_ids=_source_ids)
    return module, original["operations"], reports


def test_original_scalar_words_and_primitive_spelling_remain_exact():
    module, records, reports = declared_source()
    validate_reports(records, reports)
    validate_trace({"source_operations": records, "transformations": reports})
    constant = next(record for record in records if "floating_literal_words" in record)
    assert constant["properties"]["value"] == "0xFFBFFFFF : f32"
    assert constant["floating_literal_words"]["words"] == "ffffbfff"
    assert "arith.bitcast" in text(module)
    assert "0xFFC" not in text(module)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "width",
        "encoding",
        "format",
        "shape",
        "words",
        "quiet",
        "sign",
        "primitive",
        "extra",
        "source",
        "type",
        "count",
        "finite",
    ],
)
def test_source_word_declarations_refuse_changed_primitive_or_storage(change):
    _, records, reports = declared_source()
    report = reports[0]
    if change == "missing":
        reports.clear()
    elif change == "duplicate":
        reports.append(deepcopy(report))
    elif change == "width":
        report["word_bits"] = 32.0
    elif change == "encoding":
        report["encoding"] = "native-endian"
    elif change == "format":
        report["dtype"] = "f16"
    elif change == "shape":
        report["shape"] = []
    elif change == "words":
        report["words"] = report["words"].upper()
    elif change in ("quiet", "sign", "finite"):
        report["words"] = {
            "quiet": "ffffffFF",
            "sign": "ffffbf7f",
            "finite": "0000803f",
        }[change].lower()
    elif change == "primitive":
        record = next(
            record for record in records if "floating_literal_words" in record
        )
        record["properties"]["value"] = "0xFFFFFFFF : f32"
    elif change == "extra":
        report["grants_semantics"] = True
    elif change == "source":
        report["source_ids"] = ["ttir.op.9999"]
    elif change == "type":
        report["type"] = "f32 trailing"
    elif change == "count":
        report["nan_count"] = True
    with pytest.raises(ValueError, match="floating literal"):
        validate_reports(records, reports)


def test_all_candidates_validate_before_any_constant_is_replaced():
    module, words = parse_shared_source("""module {
      %first=arith.constant 0x7C01:f16
      %second=arith.constant 0x7F81:bf16
    }""")
    constants = list(module.ops)
    constants[-1].properties["value"] = builtin.FloatAttr(1.0, builtin.bf16)
    before = text(module)
    with pytest.raises(ConversionError, match="changed before"):
        materialize(module, words)
    assert text(module) == before
    assert list(module.ops) == constants


def test_unimplemented_nan_attributes_cannot_escape_as_approximate_floats():
    with pytest.raises(ConversionError, match="nonconstant NaN"):
        parse_shared("module attributes {original = 0x7FC00001:f32} {}")


@pytest.mark.parametrize("container", ["tensor<2x2xf32>", "vector<4xf32>"])
def test_shaped_literals_preserve_all_words_and_standard_result_type(container):
    module = parse_shared(f"""module {{
      %literal=arith.constant dense<"0x0100807f00000080010000000100c0ff">:{container}
    }}""")
    module.verify()
    constant, restored = list(module.ops)
    assert constant.value.data.data == bytes.fromhex("0100807f00000080010000000100c0ff")
    assert str(restored.result.type) == container
    roundtrip = parse_shared(text(module))
    assert next(iter(roundtrip.ops)).value.data == constant.value.data


@pytest.mark.parametrize(
    "change", [None, "words", "attribute", "effect", "extra-value"]
)
def test_legacy_word_stage_comparison_reifies_only_exact_literals(change):
    from t2m.word_events import print_ir
    from t2m.word_evidence import same_computation

    source = "module { %v=arith.constant dense<0x7FC00000>:tensor<4xf32> }"
    original, _ = parse_shared_source(source)
    legacy = print_ir(original)
    actual = parse_shared(legacy)
    if change == "words":
        legacy = legacy.replace("0x7fc00000", "0x7fc00001")
    elif change == "attribute":
        next(iter(original.ops)).attributes["source_hint"] = builtin.StringAttr("changed")
        legacy = print_ir(original)
    elif change == "effect":
        legacy = legacy.replace('"arith.constant"()', '"unknown.effect"()')
    elif change == "extra-value":
        original.body.block.add_op(
            builtin.UnregisteredOp.with_name("unknown.effect").create()
        )
        legacy = print_ir(original)
    if change == "effect":
        with pytest.raises(ConversionError):
            same_computation(actual, legacy)
    else:
        assert same_computation(actual, legacy) is (change is None)

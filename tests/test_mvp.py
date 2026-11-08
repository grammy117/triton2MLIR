"""Compiler tests: real TTIR, actual emitted operations, and incorrect-program refusals."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import triton
import triton.language as tl

from examples.vector_add import multiply_add, vector_add
from t2m import BufferSpec, KernelSpec, convert
from t2m.bundle import verify_bundle, write_bundle
from t2m.diagnostics import ConversionError
from t2m.execute import execute
from t2m.ir import parse, text


@triton.jit
def masked_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    y = tl.load(y_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=offsets < N)


@triton.jit
def looped_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    for _ in range(2):
        x = x + y
    tl.store(out_ptr + offsets, x)


def spec(entry="vector_add", n=32, block=8, grid=None, intent="Add two vectors"):
    return KernelSpec(entry, tuple(BufferSpec(name, (n,), effect) for name, effect in
                      (("x_ptr", "read"), ("y_ptr", "read"), ("out_ptr", "write"))),
                      (grid if grid is not None else n // block,), (("BLOCK", block),), intent)


@pytest.fixture
def converted():
    return convert(vector_add, spec())


@pytest.mark.parametrize("kernel", [vector_add, multiply_add])
@pytest.mark.parametrize("n,block", [(16, 8), (128, 32), (1024, 256)])
def test_real_source_to_executable_tiled_ir(kernel, n, block):
    result = convert(kernel, spec(kernel.__name__, n, block))
    rng = np.random.default_rng(41)
    x, y = [rng.standard_normal(n, dtype=np.float32) for _ in range(2)]
    actual, = execute(result.mlir_text, [x, y])
    golden = x + y if kernel is vector_add else x * y + x
    np.testing.assert_array_equal(actual, golden)
    names = {op.name for op in result.module.walk()}
    assert {"scf.forall", "tensor.extract_slice", "linalg.generic", "tensor.parallel_insert_slice"} <= names
    assert all(name.split(".")[0] in {"builtin", "func", "scf", "tensor", "arith", "linalg"} for name in names)


def test_saved_ttir_reproducibility_and_full_accounting(converted, tmp_path):
    path = tmp_path / "source.ttir"
    path.write_text(converted.source_ttir)
    replay = convert(path, converted.spec)
    assert replay.mlir_text == converted.mlir_text
    records = replay.trace["source_operations"]
    assert len(records) == len({r["id"] for r in records})
    assert all(r["status"] in {"lowered", "analyzed", "eliminated", "structural"} for r in records)
    assert any(r["status"] == "eliminated" and r["opcode"] == "arith.extsi" for r in records)
    assert any(r["status"] == "lowered" and r["opcode"] == "arith.addf" for r in records)
    assert converted.trace["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_survives_printing_but_is_not_execution_semantics(converted):
    module = parse(converted.mlir_text)
    assert module.attributes["prov.intent"].data == "Add two vectors"
    assert module.attributes["prov.intent_origin"].data == "caller-description"
    assert not converted.trace["intent"]["grants_semantics"]
    source_ids = {r["id"] for r in converted.trace["source_operations"]}
    for op in module.walk():
        assert "prov.emission_id" in op.attributes
        assert {s.data for s in op.attributes["prov.source_ids"]} <= source_ids
        if op.name == "arith.addf":
            assert op.attributes["prov.family"].data == "elementwise"
            assert op.attributes["prov.op"].data == "add"
        for key in list(op.attributes):
            if key.startswith("prov."):
                del op.attributes[key]
    x = np.arange(32, dtype=np.float32)
    np.testing.assert_array_equal(execute(text(module), [x, x.copy()])[0], x + x)


def test_caller_intent_does_not_rewrite_the_computation():
    result = convert(multiply_add, spec("multiply_add", intent="This is only an addition"))
    assert 'prov.op = "mul"' in result.mlir_text
    x = np.arange(32, dtype=np.float32)
    np.testing.assert_array_equal(execute(result.mlir_text, [x, x.copy()])[0], x * x + x)


@pytest.mark.parametrize("n,grid,reason", [(31, 4, "outside"), (40, 4, "partition")])
def test_unmasked_tail_and_incomplete_output_are_refused(n, grid, reason):
    with pytest.raises(ConversionError, match=reason):
        convert(vector_add, spec(n=n, grid=grid))


def test_unknown_live_operation_has_a_source_id(converted, tmp_path):
    path = tmp_path / "unsupported.ttir"
    path.write_text(converted.source_ttir.replace("arith.addf", "arith.divf"))
    with pytest.raises(ConversionError, match=r"ttir.op.\d+ \(arith.divf\)"):
        convert(path, converted.spec)


def test_masked_tail_lowers_to_tensor_padding():
    masked_spec = replace(spec("masked_add", n=31, grid=4), constexprs=(("BLOCK", 8), ("N", 31)))
    result = convert(masked_add, masked_spec)
    x = np.arange(31, dtype=np.float32)
    np.testing.assert_array_equal(execute(result.mlir_text, [x, x.copy()])[0], x + x)
    assert '"tensor.pad"' in result.mlir_text


def test_nested_source_regions_are_refused_instead_of_flattened():
    with pytest.raises(ConversionError, match="nested regions"):
        convert(looped_add, spec("looped_add"))


def test_volatile_reads_are_not_converted_to_tensor_values(converted, tmp_path):
    lines = [line.replace(" : tensor", " {isVolatile = true} : tensor", 1)
             if "tt.load" in line else line for line in converted.source_ttir.splitlines()]
    path = tmp_path / "volatile.ttir"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ConversionError, match="volatile"):
        convert(path, converted.spec)


def test_triton_version_mismatch_is_rejected_before_decoding(monkeypatch):
    from t2m import frontend
    monkeypatch.setattr(frontend, "version", lambda _: "0.0")
    with pytest.raises(ConversionError, match="expected triton==3.7.1"):
        convert(vector_add, spec())


def test_multiple_stores_refused(converted, tmp_path):
    path = tmp_path / "multiple.ttir"
    line = next(line for line in converted.source_ttir.splitlines() if "tt.store" in line)
    path.write_text(converted.source_ttir.replace(line, line + "\n" + line))
    with pytest.raises(ConversionError, match="one store"):
        convert(path, converted.spec)


def test_effect_mismatch_is_not_inferred_away():
    wrong = replace(spec(), buffers=(BufferSpec("x_ptr", (32,), "write"),
                    BufferSpec("y_ptr", (32,), "read"), BufferSpec("out_ptr", (32,), "read")))
    with pytest.raises(ConversionError, match="contradicts"):
        convert(vector_add, wrong)


def test_source_argument_order_is_checked():
    correct = spec()
    wrong = replace(correct, buffers=(correct.buffers[1], correct.buffers[0], correct.buffers[2]))
    with pytest.raises(ConversionError, match="argument order"):
        convert(vector_add, wrong)


def test_index_wraparound_is_not_lost_in_index_conversion():
    with pytest.raises(ConversionError, match="wraparound"):
        convert(vector_add, spec(n=(1 << 31) + 8))


def test_large_tensor_proof_does_not_enumerate_elements(converted):
    large = convert(vector_add, spec(n=1 << 24))
    assert len(large.mlir_text) < len(converted.mlir_text) * 2
    assert large.trace["proofs"]["output_partition"] == "proved"


def test_bundle_contains_baseline_artifacts_and_explicit_differences(converted, tmp_path):
    x = np.arange(32, dtype=np.float32)
    out = write_bundle(converted, tmp_path / "bundle", [x, x.copy()], x + x)
    receipt = verify_bundle(out)
    assert receipt["schema"] == "t2m.capture-receipt.v1"
    compatibility = json.loads((out / "compatibility.json").read_text())
    assert compatibility["same_core_filenames_and_array_conventions"]
    assert compatibility["differences"]
    assert compatibility["merlin_staged_execution"] == "not-certified"
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["mlir"]["sha256"] == hashlib.sha256((out / "model.mlir").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_bundle(converted, out, [x, x.copy()], x + x)
    (out / "golden.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="golden.npy"):
        verify_bundle(out)


def test_wrong_golden_does_not_publish_a_bundle(converted, tmp_path):
    x = np.arange(32, dtype=np.float32)
    out = tmp_path / "wrong"
    with pytest.raises(AssertionError):
        write_bundle(converted, out, [x, x.copy()], x + x + 1)
    assert not out.exists()


def test_modified_ir_requires_regenerated_provenance(converted, tmp_path):
    from xdsl.dialects.builtin import StringAttr
    converted.module.attributes["prov.intent"] = StringAttr("changed")
    x = np.arange(32, dtype=np.float32)
    with pytest.raises(ValueError, match="IR changed"):
        write_bundle(converted, tmp_path / "modified", [x, x.copy()], x + x)


def test_runtime_abi_preconditions_are_checked(converted):
    x = np.arange(32, dtype=np.float32)
    with pytest.raises(ValueError, match="independent-allocation"):
        execute(converted.mlir_text, [x, x])
    with pytest.raises(ValueError, match="f32"):
        execute(converted.mlir_text, [x.astype(np.float64), x.copy()])

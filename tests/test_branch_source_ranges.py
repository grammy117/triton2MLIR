"""Actual scalar source conditions refine only the selected branch's indices."""

import os
from dataclasses import replace

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.examples.branch_index_kernels import branch_index
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")


def capture(root, style="then", *, outside=False, after=False):
    root.mkdir(exist_ok=True)
    source = root / "source.ttir"
    source.write_text(capture_ttir(branch_index,
        {"x": "*i64", "state": "*i64", "old": "*i64", "divisor": "i32", "allow": "i32",
         "STYLE": "constexpr", "OUTSIDE": "constexpr", "AFTER": "constexpr"},
        {"STYLE": style, "OUTSIDE": outside, "AFTER": after}))
    abi = TensorABI("branch_index", (TensorBuffer("x", (16,), "read", "i64"),
        TensorBuffer("state", (1,), "readwrite", "i64"), TensorBuffer("old", (1,), "write", "i64")),
        runtime_scalars=(RuntimeScalar("divisor", "i32", 3, bounds=(0, 9)),
                         RuntimeScalar("allow", "i32", 4, bounds=(0, 1))))
    return source, abi


def arguments(divisor, allow, style="then"):
    values = np.arange(16, dtype=np.int64) + (1 << 55)
    state = np.asarray([(1 << 56) + 11], np.int64)
    active = divisor > 0 and (allow != 0 if style in ("and", "or") else True)
    expected = state + values[9 // divisor] if active else state ^ 3
    return [values, state, np.asarray(divisor, np.int32), np.asarray(allow, np.int32)], [expected, state.copy()]


@pytest.mark.parametrize("style", ("then", "else", "and", "or", "unsigned", "ne"))
def test_real_branch_conditions_admit_tensor_indexing_without_changing_entry_bounds(tmp_path, style):
    source, abi = capture(tmp_path, style)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert abi.runtime_scalars[0].bounds == (0, 9)
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    assert any(op.name == "scf.if" for op in result.module.walk())
    assert any(row["kind"] == "source-branch-ranges" for row in result.trace["transformations"])


@pytest.mark.parametrize("options", ({"outside": True}, {"after": True},
                                    {"style": "unrelated"}, {"style": "wrong-else"},
                                    {"style": "false-and"}, {"style": "true-or"}))
def test_branch_facts_cannot_validate_earlier_later_unrelated_or_sibling_divisions(tmp_path, options):
    source, abi = capture(tmp_path, **options)
    with pytest.raises(ConversionError, match="zero divisor"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def test_branch_fact_cannot_hide_overflow_in_its_earlier_compared_product(tmp_path):
    source, abi = capture(tmp_path, "product")
    abi = replace(abi, runtime_scalars=(replace(abi.runtime_scalars[0], bounds=(0, 1 << 30)), abi.runtime_scalars[1]))
    with pytest.raises(ConversionError, match="overflow"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def test_advisory_range_annotations_cannot_supply_a_missing_source_condition(tmp_path):
    from xdsl.dialects.builtin import StringAttr

    from t2m.shared_indexing import check_source_addresses
    from t2m.shared_tensor import parse_shared
    from t2m.upstream import lower_with_shared

    source, abi = capture(tmp_path, "unrelated")
    lower_with_shared(source, tmp_path / "source", optimizer=os.environ["TRITON_SHARED_OPT"], stage="source")
    module = parse_shared((tmp_path / "source/lowered.mlir").read_text())
    for op in module.walk():
        op.attributes["prov.assume_positive"] = StringAttr("divisor")
    with pytest.raises(ConversionError, match="zero divisor"):
        check_source_addresses(module, abi)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
@pytest.mark.parametrize("style", ("then", "else", "and", "or", "unsigned", "ne"))
def test_native_branches_keep_exact_state_and_snapshot_words(tmp_path, style):
    from t2m.native import execute_native

    source, abi = capture(tmp_path, style)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    for i, (divisor, allow) in enumerate(((0, 1), (3, 0), (9, 1), (1, 1))):
        values, expected = arguments(divisor, allow, style)
        actual, _ = execute_native(result.mlir_text, values, tmp_path / f"native-{i}",
                                  python=os.environ["T2M_NATIVE_PYTHON"])
        assert len(actual) == len(expected) == 2
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.fixture(scope="module", params=("and", "or"))
def compiled(tmp_path_factory, request):
    if not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing native/Merlin tools required")
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    root = tmp_path_factory.mktemp("branch-" + request.param)
    source, abi = capture(root, request.param)
    result = convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    values, expected = arguments(3, 1, request.param)
    write_shared_bundle(result, root / "capture", values, expected, work=root / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def invocation(root, divisor, allow, style, name):
    values, expected = arguments(divisor, allow, style)
    inputs, reference = root / f"inputs-{name}.npz", root / f"reference-{name}.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(values)})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, reference


def test_one_merlin_library_per_source_keeps_zero_and_independent_branch_choices(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, style = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for i, (divisor, allow) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1), (3, 1), (9, 1))):
        inputs, reference = invocation(root, divisor, allow, style, str(i))
        report = invoke_kernel(root / "merlin", inputs, reference, root / f"invoke-{i}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{i}")
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_targets_keep_both_branch_outcomes_and_all_results(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, style = compiled
    for i, divisor in enumerate((0, 3)):
        inputs, reference = invocation(root, divisor, 1, style, "target-" + str(i))
        out = root / f"target-{i}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report


def test_public_scalar_domain_is_rejected_before_execution_without_results(compiled):
    from t2m.consumer import observe_merlin_host

    root, style = compiled
    values, _ = arguments(10, 1, style)
    inputs = root / "invalid-domain.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(values)})
    out = root / "refused-domain"
    outcome = observe_merlin_host(root / "merlin", inputs, out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == 1 and not (out / "outputs.npz").exists()
    assert b"outside its guarded ABI bounds" in (out / "stderr.bin").read_bytes()


def test_serialized_native_scalar_guard_still_aborts_without_results(compiled):
    import json

    from t2m.native import execute_native

    root, style = compiled
    values, _ = arguments(10, 1, style)
    out = root / "native-invalid-domain"
    with pytest.raises(ValueError, match="native host execution failed") as failure:
        execute_native((root / "capture/model.mlir").read_text(), values, out,
                       python=os.environ["T2M_NATIVE_PYTHON"])
    message = str(failure.value)
    assert "exit -6" in message and "Triton runtime scalar outside declared bounds: divisor" in message
    assert not (out / "outputs.npz").exists()
    (root / "native-domain-abort.json").write_text(json.dumps({"error": message}) + "\n")


@pytest.mark.parametrize("style", ("and", "or"))
@pytest.mark.skipif(not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing native/Merlin tools required")
def test_full_signed_source_domain_keeps_negative_and_extreme_controls(tmp_path, style):
    from test_spike_model import TOOLS, Q

    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle

    source, abi = capture(tmp_path, style)
    abi = replace(abi, runtime_scalars=(replace(abi.runtime_scalars[0], bounds=(-(1 << 31), (1 << 31) - 1)),
                                      replace(abi.runtime_scalars[1], bounds=None)))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    values, expected = arguments(-(1 << 31), -17, style)
    write_shared_bundle(result, tmp_path / "capture", values, expected, work=tmp_path / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(tmp_path / "capture", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    library, receipt = _record(tmp_path / "merlin/host/model.so"), verify_kernel_qualification(tmp_path / "merlin")
    for i, (divisor, allow) in enumerate(((-(1 << 31), -17), (-1, 1), (0, -17), (1, 0), (1, -(1 << 31)), ((1 << 31) - 1, -17))):
        inputs, reference = invocation(tmp_path, divisor, allow, style, str(i))
        report = invoke_kernel(tmp_path / "merlin", inputs, reference, tmp_path / f"invoke-{i}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(tmp_path / f"invoke-{i}")
    assert _record(tmp_path / "merlin/host/model.so") == library and verify_kernel_qualification(tmp_path / "merlin") == receipt
    if TOOLS:
        report = Q.qualify(tmp_path / "capture", tmp_path / "target", inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(tmp_path / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report

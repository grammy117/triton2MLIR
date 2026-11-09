"""Official offset-width policy preserves source words and logical dataflow."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, *, scalar=False, cancel=False, policy="source-words"):
    from t2m.examples.source_word_kernels import modular_update, pointer_cancel
    from t2m.frontend import capture_ttir
    kernel = pointer_cancel if cancel else modular_update
    path = root / "source.ttir"
    path.write_text(capture_ttir(kernel, {"x": "*fp32", "encoded": "*i32", "out": "*fp32", "n": "i32", "m": "i32",
        "B": "constexpr", "SCALAR": "constexpr"}, {"B": 8, "SCALAR": scalar}))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("n",)), TensorBuffer("encoded", ("n",), dtype="i32"),
        TensorBuffer("out", ("n",) if cancel else ("m",), "write" if cancel else "readwrite")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
                                                                   RuntimeScalar("m", "i32", 4, bounds=(0, 127))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy=policy), abi


def inputs(n, *, cancel=False, base=1 << 30):
    x = np.arange(n, dtype=np.float32) - np.float32(5)
    key = (base + np.arange(n)[::-1]).astype(np.int32)
    m = n if cancel else 4 * n + 3
    old = np.arange(m, dtype=np.float32) + 100
    expected = x + np.float32(1) if cancel else old.copy()
    if not cancel:
        index = (key.astype(np.int64) * 4 + 1).astype(np.int32)
        expected[index] = (old[index] + x) + np.float32(1)
    return [x, key, *([] if cancel else [old]), np.asarray(n, dtype=np.int32), np.asarray(m, dtype=np.int32)], expected


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_original_word_math_and_wide_pointer_sums_execute(tmp_path, scalar, cancel):
    from t2m.native import execute_native
    result, abi = capture(tmp_path, scalar=scalar, cancel=cancel)
    arrays, expected = inputs(13, cancel=cancel)
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    assert result.trace["profile"] == "shared-source-word-buffer"
    assert result.trace["memory"]["index_overflow"].startswith("original-integer-word-arithmetic")
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    function = next(iter(result.module.ops))
    assert json.loads(function.attributes["prov.source_accesses"].data)
    module = parse_shared(result.auxiliary_artifacts["word-tensor.mlir"])
    from xdsl.dialects.builtin import TensorType, i32, i64
    for op in module.walk():
        if op.name == "builtin.unregistered" and op.op_name.data in ("tts.gather", "tts.scatter"):
            type_ = op.operands[1].type
            assert (type_.element_type if isinstance(type_, TensorType) else type_) == i64
    if not cancel:
        assert any(op.name == "arith.muli" and op.results[0].type == i32 for op in module.walk())


def test_structured_policy_retains_its_existing_source_width_gate(tmp_path):
    with pytest.raises(ConversionError, match="source address arithmetic may overflow"):
        capture(tmp_path, policy="structured")
    assert not (tmp_path / "translation/tensor-ir").exists()


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_source_word_policy_needs_its_explicit_memory_contract(tmp_path, profile):
    with pytest.raises(ValueError, match="requires the explicit guarded-buffer"):
        convert_shared(tmp_path / "source.ttir", None, tmp_path / "translation", memory_profile=profile, address_policy="source-words")
    assert not (tmp_path / "translation").exists()


def pointer_loop(root):
    from t2m.examples.source_word_kernels import pointer_runtime_sum
    from t2m.frontend import capture_ttir
    path = root / "source.ttir"
    path.write_text(capture_ttir(pointer_runtime_sum, {"x": "*fp32", "out": "*fp32", "n": "i32", "rows": "i32",
        "steps": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("pointer_runtime_sum", (TensorBuffer("x", ("rows", "n")), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 13)),
        RuntimeScalar("rows", "i32", 3, bounds=(1, 5)), RuntimeScalar("steps", "i32", 4, bounds=(0, 4))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words"), abi


def test_runtime_pointer_results_include_nonexecuted_loop_initializers(tmp_path):
    from t2m.native import execute_native
    result, abi = pointer_loop(tmp_path)
    assert result.trace["memory"]["source_loops"]
    for n, steps in ((0, 0), (1, 0), (7, 1), (9, 3), (13, 4)):
        x = np.arange((steps + 1) * n, dtype=np.float32).reshape(steps + 1, n)
        arrays = [x, *[np.asarray(value, dtype=np.int32) for value in (n, steps + 1, steps)]]
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / ("native" + str(n)), python=NATIVE, shape_abi=abi.runtime_shapes())
        np.testing.assert_array_equal(actual, x.sum(axis=0))


def test_runtime_pointer_bounds_are_checked_at_the_actual_iteration(tmp_path):
    from t2m.native import execute_native
    result, abi = pointer_loop(tmp_path)
    arrays = [np.ones((1, 3), dtype=np.float32), *[np.asarray(value, dtype=np.int32) for value in (3, 1, 2)]]
    with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())


@pytest.mark.parametrize("step,valid", [(3, True), (2, False)])
def test_nonlinear_grid_preserves_bijection_or_rejects_actual_conflicts(tmp_path, step, valid):
    from t2m.examples.source_word_kernels import permuted_tiles
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    constants = {"N": 29, "B": 8, "PROGRAMS": 4, "STEP": step}
    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(permuted_tiles, {"x": "*fp32", "out": "*fp32", **{k: "constexpr" for k in constants}}, constants))
    abi = TensorABI("permuted_tiles", (TensorBuffer("x", (29,)), TensorBuffer("out", (29,), "write")), grid=(4,))
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")
    x = np.arange(29, dtype=np.float32)
    if valid:
        (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
        np.testing.assert_array_equal(actual, x + 1)
    else:
        with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
            execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar,cancel", [(False, False), (True, False), (False, True)])
def test_one_merlin_library_retains_words_for_independent_shapes_and_encoded_values(tmp_path, scalar, cancel):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.memory_contract import WORD_FILES
    from t2m.shared_bundle import write_shared_bundle
    result, _ = capture(tmp_path, scalar=scalar, cancel=cancel)
    arrays, expected = inputs(7, cancel=cancel)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    receipt = verify_bundle(bundle)
    assert WORD_FILES <= receipt["artifacts"].keys()
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    qualified, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for ordinal, (n, base) in enumerate(((0, 0), (1, 0), (7, 1 << 30), (9, 1 << 31), (13, 3 << 30))):
        arrays, expected = inputs(n, cancel=cancel, base=base)
        arguments, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(arguments, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / ("invoke" + str(ordinal))
        report = invoke_kernel(compiled, arguments, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == qualified and _record(compiled / "host/model.so") == library
    # Rehashing an edited policy artifact cannot sever its bound stage relation.
    (bundle / "word-input.mlir").write_text((bundle / "word-input.mlir").read_text() + "\n")
    receipt["artifacts"]["word-input.mlir"] = _record(bundle / "word-input.mlir")
    (bundle / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="source-word address evidence"):
        verify_bundle(bundle)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_one_merlin_library_retains_runtime_pointer_state_and_final_pointer(tmp_path):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result, _ = pointer_loop(tmp_path)
    x = np.arange(21, dtype=np.float32).reshape(3, 7)
    arguments = [x, *[np.asarray(value, dtype=np.int32) for value in (7, 3, 2)]]
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arguments, [x.sum(axis=0)], work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for ordinal, (n, steps) in enumerate(((0, 0), (1, 0), (7, 1), (9, 3), (13, 4))):
        x = np.arange((steps + 1) * n, dtype=np.float32).reshape(steps + 1, n)
        arrays = [x, *[np.asarray(value, dtype=np.int32) for value in (n, steps + 1, steps)]]
        inputs_path, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs_path, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=x.sum(axis=0))
        out = tmp_path / ("invoke" + str(ordinal))
        report = invoke_kernel(compiled, inputs_path, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


def test_source_word_cli_preserves_evidence_without_claiming_numeric_execution(tmp_path):
    from t2m.cli import main
    result, abi = capture(tmp_path)
    spec, out = tmp_path / "abi.json", tmp_path / "ir"
    spec.write_text(json.dumps(abi.as_dict()))
    assert main(["shared-tensor", str(tmp_path / "source.ttir"), "--abi", str(spec), "--out", str(out),
                 "--optimizer", OPT, "--memory-profile", "guarded-buffer", "--address-policy", "source-words"]) == 0
    assert (out / "word-stage.json").is_file() and not (out / "capture_receipt.json").exists()
    trace = json.loads((out / "provenance.json").read_text())
    assert trace["profile"] == result.trace["profile"] and trace["native_verification"] == "not-performed"

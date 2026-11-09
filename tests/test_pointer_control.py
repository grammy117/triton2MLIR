"""Same logical base across SCF edges, with real native/consumer execution."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, opcode, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, *, loop=False, scalar=False, mixed=False):
    from t2m.examples.pointer_control_kernels import branch_pointer_select, mixed_base_branch, while_pointer_sum
    from t2m.frontend import capture_ttir
    kernel = while_pointer_sum if loop else mixed_base_branch if mixed else branch_pointer_select
    signature = {"x": "*fp32", **({"y": "*fp32"} if mixed else {}), "out": "*fp32", "n": "i32"}
    signature.update({"rows": "i32", "steps": "i32"} if loop else {"choice": "i32"})
    signature["B"] = "constexpr"
    constants = {"B": 8}
    if not loop and not mixed:
        signature["SCALAR"] = "constexpr"
        constants["SCALAR"] = scalar
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, signature, constants))
    scalars = (RuntimeScalar("n", "i32", 3 if mixed else 2, bounds=(0, 13)),)
    scalars += ((RuntimeScalar("rows", "i32", 3, bounds=(1, 5)), RuntimeScalar("steps", "i32", 4, bounds=(0, 4)))
                if loop else (RuntimeScalar("choice", "i32", 4 if mixed else 3, bounds=(0, 1)),))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("rows", "n") if loop else (2, "n")),
        *((TensorBuffer("y", (2, "n")),) if mixed else ()), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, *, loop=False, count=1):
    rows = count + 1 if loop else 2
    x = (np.arange(rows * n, dtype=np.float32) - 5).reshape(rows, n)
    values = (n, count + 1, count) if loop else (n, count)
    expected = x.sum(axis=0) if loop else x[0 if count else 1].copy()
    return [x, *[np.asarray(v, dtype=np.int32) for v in values]], expected


@pytest.mark.parametrize("loop,scalar", [(False, False), (False, True), (True, False)])
def test_pointer_control_native_preserves_regions_shapes_and_edges(tmp_path, loop, scalar):
    from t2m.native import execute_native
    result = capture(tmp_path, loop=loop, scalar=scalar)
    events = [t for t in result.trace["transformations"] if t["kind"] == "same-base-pointer-control"]
    assert events and all(t["source_ids"] and t["pointer_offset_bits"] == 64 for t in events)
    word_ir = parse_shared(result.auxiliary_artifacts["word-input.mlir"])
    controls = [op for op in word_ir.walk() if opcode(op) == ("scf.while" if loop else "scf.if")]
    assert controls and all("tt.ptr" not in str(v.type) for op in controls for v in op.results)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    for i, (n, count) in enumerate(((0, 0), (1, 0), (7, 1), (9, 3 if loop else 0), (13, 4 if loop else 1))):
        arrays, expected = arguments(n, loop=loop, count=count)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}",
                                     python=NATIVE, shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


def test_mixed_base_branch_is_a_located_refusal_before_upstream_lowering(tmp_path):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="ttir.op.*mixed-base state"):
        capture(tmp_path, mixed=True)
    assert not (tmp_path / "translation/tensor-ir").exists()


def test_actual_while_iteration_checks_memory_bounds(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path, loop=True)
    arrays, _ = arguments(9, loop=True, count=0)
    arrays[-1] = np.asarray(1, dtype=np.int32)
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE,
                       shape_abi=result.spec.runtime_shapes())


def test_a_different_backedge_base_cannot_hide_in_a_pointer_cycle(tmp_path):
    from xdsl.dialects import builtin
    from t2m.diagnostics import ConversionError
    from t2m.pointer_control import normalize_pointer_control
    from t2m.shared_tensor import _source_ids
    capture(tmp_path, loop=True)
    module = parse_shared((tmp_path / "translation/inline/lowered.mlir").read_text())
    function = next(op for op in module.walk() if opcode(op) == "tt.func")
    loop = next(op for op in module.walk() if opcode(op) == "scf.while")
    update = next(op for op in loop.regions[1].block.ops if opcode(op) == "tt.addptr")
    alternate = builtin.UnregisteredOp.with_name("tt.splat").create(
        operands=[function.regions[0].block.args[1]], result_types=[update.operands[0].type])
    alternate.location = update.location
    update.parent.insert_op_before(alternate, update)
    update.operands = [alternate.results[0], update.operands[1]]
    with pytest.raises(ConversionError, match="ttir.op.*mixed-base state"):
        normalize_pointer_control(module, _source_ids)


def test_while_before_and_after_state_need_not_have_equal_arity():
    from t2m.pointer_control import normalize_pointer_control
    from t2m.shared_tensor import _source_ids
    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{function_type = (!tt.ptr<f32>, !tt.ptr<f32>) -> (), sym_name = "different_edges"}> ({
      ^bb0(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>):
        %zero = arith.constant 0 : i32
        %no = arith.constant false
        %result = scf.while (%pointer = %x, %counter = %zero) : (!tt.ptr<f32>, i32) -> !tt.ptr<f32> {
          scf.condition(%no) %pointer : !tt.ptr<f32>
        } do { ^bb0(%pointer: !tt.ptr<f32>):
          scf.yield %pointer, %zero : !tt.ptr<f32>, i32
        }
        %v = "tt.load"(%result) <{isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<f32>) -> f32
        "tt.store"(%out, %v) : (!tt.ptr<f32>, f32) -> ()
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()''')
    module.verify()
    assert normalize_pointer_control(module, _source_ids)
    module.verify()
    loop = next(op for op in module.walk() if opcode(op) == "scf.while")
    assert [str(v.type) for v in loop.operands] == ["i64", "i32"]
    assert [str(v.type) for v in loop.results] == ["i64"]
    assert [str(v.type) for v in loop.regions[1].block.args] == ["i64"]


def word_branch(root):
    from t2m.examples.pointer_control_kernels import branch_pointer_words
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(branch_pointer_words, {"x": "*fp32", "encoded": "*i32", "out": "*fp32",
        "n": "i32", "choice": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("branch_pointer_words", (TensorBuffer("x", ("n",)), TensorBuffer("encoded", ("n",), dtype="i32"),
        TensorBuffer("out", ("n",), "write")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 13)), RuntimeScalar("choice", "i32", 4, bounds=(0, 1))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def word_arguments(n, choice, base):
    x = np.arange(n, dtype=np.float32) - 5
    encoded = (base + np.arange(n)).astype(np.int32)
    return [x, encoded, np.asarray(n, dtype=np.int32), np.asarray(choice, dtype=np.int32)], x.copy()


@pytest.mark.parametrize("choice", [0, 1])
def test_control_normalization_preserves_word_wrap_and_wide_pointer_sums(tmp_path, choice):
    from t2m.native import execute_native
    result = word_branch(tmp_path)
    arrays, expected = word_arguments(13, choice, 1 << 30)
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE,
                                 shape_abi=result.spec.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    word_ir = parse_shared(result.auxiliary_artifacts["word-input.mlir"])
    assert any(opcode(op) == "arith.muli" and str(op.results[0].type) == "tensor<8xi32>" for op in word_ir.walk())


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("kind", ["branch", "while", "words"])
def test_merlin_library_retains_pointer_control_over_independent_invocations(tmp_path, kind):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    loop = kind == "while"
    result = word_branch(tmp_path) if kind == "words" else capture(tmp_path, loop=loop)
    arrays, expected = word_arguments(7, 1, 1 << 30) if kind == "words" else arguments(7, loop=loop, count=1)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, count) in enumerate(((0, 0), (1, 0), (7, 1), (9, 3 if loop else 0), (13, 4 if loop else 1))):
        arrays, expected = word_arguments(n, count, (i % 4) << 30) if kind == "words" else arguments(n, loop=loop, count=count)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library

"""Runtime source stores preserve conditional state and definite initialization."""

import os

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN, PYTHON, CLANG = (os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")


def convert(root, profile, mode):
    from t2m.examples.tensor_kernels import runtime_conditional_store, runtime_memory_store, runtime_preinit_store, runtime_memory_accumulate
    fn = {"preinit": runtime_preinit_store, "conditional": runtime_conditional_store,
          "accumulate": runtime_memory_accumulate}.get(mode, runtime_memory_store)
    effect = "readwrite" if mode in ("preserve", "conditional", "accumulate") else "write"
    lower = 1 if mode == "fresh-positive" else -2
    scalars = [RuntimeScalar("iterations", "i32", 1, bounds=(lower, 5))]
    signature = {"x": "*fp32", "iterations": "i32", "out": "*fp32", "N": "constexpr", "B": "constexpr"}
    if mode == "conditional":
        scalars.append(RuntimeScalar("flag", "i1", 2))
        signature["flag"] = "i1"
    source = root / "source.ttir"
    source.write_text(capture_ttir(fn, signature, {"N": 13, "B": 8}))
    abi = TensorABI(fn.__name__, (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), effect)),
        grid=(2,), runtime_scalars=tuple(scalars))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("mode", ["preserve", "fresh-positive", "preinit", "conditional", "accumulate"])
def test_runtime_write_state_survives_empty_loops_and_source_branches(tmp_path, profile, mode):
    result = convert(tmp_path, profile, mode)
    loop = result.trace["memory"]["source_loops"][0]
    assert loop["memory_effects"] == "conditional-loop-carried-tensor-versions"
    assert loop["initialization"] == ("post-body-facts; positive-trip-count-proved" if mode == "fresh-positive"
                                      else "entry-facts-only; zero-trip-reachable")
    x, old = np.arange(-6, 7, dtype=np.float32) / np.float32(3.14), np.arange(13, dtype=np.float32) * 7 + 99

    def data(count, flag=True):
        values = [x]
        if mode in ("preserve", "conditional", "accumulate"):
            values.append(old)
        values.append(np.asarray(count, dtype=np.int32))
        if mode == "conditional":
            values.append(np.asarray(flag))
        expected = x + np.float32(count - 1) if count > 0 and flag else (
            np.full(13, -7, dtype=np.float32) if mode == "preinit" else old.copy())
        if mode == "accumulate":
            expected = old.copy()
            for _ in range(count):
                expected += x
        return values, expected

    capture = tmp_path / "capture"
    values, expected = data(3)
    write_shared_bundle(result, capture, values, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    if all((MERLIN, PYTHON, CLANG)):
        from t2m.invocation import invoke_kernel, verify_invocation
        from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
        compiled = tmp_path / "merlin"
        qualify_kernel(capture, compiled, merlin_root=MERLIN, python=PYTHON, compiler_python=NATIVE, clang=CLANG)
        receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
        cases = [(1, True), (5, True)]
        if mode != "fresh-positive":
            cases += [(-2, True), (0, True)]
        if mode == "conditional":
            cases += [(0, False), (5, False)]
        for i, (count, flag) in enumerate(cases):
            inputs, reference = (tmp_path / (name + str(i) + ".npz") for name in ("inputs", "reference"))
            values, expected = data(count, flag)
            np.savez(inputs, **{f"in{n}": value for n, value in enumerate(values)})
            np.savez(reference, out0=expected)
            out = tmp_path / ("invoke" + str(i))
            report = invoke_kernel(compiled, inputs, reference, out, merlin_root=MERLIN, python=PYTHON)
            verify_invocation(out)
            assert report["native_compilation"] == "reused-qualified-library"
            assert report["evaluation"]["functional_inputs_preserved"]
        assert verify_kernel_qualification(compiled) == receipt
        assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_positive_trip_count_does_not_initialize_an_unexecuted_branch(tmp_path, profile):
    from t2m.examples.tensor_kernels import runtime_conditional_store
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_conditional_store, {"x": "*fp32", "iterations": "i32", "flag": "i1",
        "out": "*fp32", "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    abi = TensorABI(runtime_conditional_store.__name__, (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", "i32", 1, bounds=(1, 5)), RuntimeScalar("flag", "i1", 2)))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_positive_loop_cannot_read_a_fresh_accumulator_before_initialization(tmp_path, profile):
    from t2m.examples.tensor_kernels import runtime_memory_accumulate
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_memory_accumulate, {"x": "*fp32", "iterations": "i32", "out": "*fp32",
        "N": "constexpr", "B": "constexpr"}, {"N": 13, "B": 8}))
    abi = TensorABI(runtime_memory_accumulate.__name__, (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", "i32", 1, bounds=(1, 5)),))
    with pytest.raises(ConversionError, match="read before.*initialized"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    assert not (tmp_path / "translation/memory-ir").exists()

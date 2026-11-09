"""Buffer selection, original effect order and selector/offset loop state."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, *, kind="access", scalar=False, lanes=False, store=False, for_loop=False, switch=False):
    from t2m.examples.selected_pointer_kernels import selected_access, selected_loop, selected_shape
    from t2m.frontend import capture_ttir
    kernel = {"access": selected_access, "loop": selected_loop, "shape": selected_shape}[kind]
    constants = {"B": 8}
    if kind == "access":
        constants.update(SCALAR=scalar, LANES=lanes, STORE_SELECTED=store)
    if kind == "loop":
        constants.update(FOR_LOOP=for_loop, LANES=lanes, SWITCH=switch)
    scalars = [RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("m", "i32", 4, bounds=(0, 37))]
    if kind == "loop":
        scalars.append(RuntimeScalar("steps", "i32", 5, bounds=(0, 5)))
    if kind != "shape":
        scalars.append(RuntimeScalar("choice", "i32", 6 if kind == "loop" else 5, bounds=(0, 3)))
    signature = {**{name: "*fp32" for name in ("x", "y", "out")}, **{s.name: "i32" for s in scalars},
                 **{name: "constexpr" for name in constants}}
    path = root / "source.ttir"
    path.write_text(capture_ttir(kernel, signature, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("n",), "readwrite" if kind == "access" else "read"),
        TensorBuffer("y", ("m",), "readwrite" if store else "read"), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", 1 if scalar else 16 if kind == "shape" else 8),), runtime_scalars=tuple(scalars))
    return convert_shared(path, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, m, choice=0, *, kind="access", lanes=False, store=False, steps=0, switch=False):
    x, y = np.arange(n, dtype=np.float32) - 5, np.arange(m, dtype=np.float32) + 100
    arrays = [x.copy(), y.copy(), np.asarray(n, dtype=np.int32), np.asarray(m, dtype=np.int32)]
    index = np.arange(n)
    if kind == "access":
        take_x = (index + choice) % 2 == 0 if lanes else np.full(n, choice == 0)
        x += 3
        if store:
            x[take_x] *= 2
            y[m - 1 - index[~take_x]] *= 2
        out = x.copy()
        out[~take_x] = y[m - 1 - index[~take_x]]
        arrays.append(np.asarray(choice, dtype=np.int32))
        return arrays, [x, *([y] if store else []), out]
    if kind == "loop":
        take_x = np.ones(n, dtype=bool)
        out = np.zeros(n, dtype=np.float32)
        for step in range(steps):
            out[take_x] += x[take_x]
            out[~take_x] += y[m - 1 - index[~take_x]]
            take_x = np.zeros(n, dtype=bool) if switch else ((index + step + choice) % 2 == 0 if lanes else np.full(n, (step + choice) % 2 != 0))
        out[take_x] += x[take_x]
        out[~take_x] += y[m - 1 - index[~take_x]]
        arrays.extend(np.asarray(v, dtype=np.int32) for v in (steps, choice))
        return arrays, [out]
    source = (index // 16) * 16 + index % 8
    out = np.empty(n, dtype=np.float32)
    take_x = index % 16 < 8
    out[take_x] = x[source[take_x]]
    out[~take_x] = y[m - 1 - source[~take_x]]
    return arrays, [out]


@pytest.mark.parametrize("scalar,lanes,store", [(False, False, False), (True, False, True), (False, True, True)])
def test_selected_access_observes_intervening_and_selected_writes(tmp_path, scalar, lanes, store):
    from t2m.native import execute_native
    result = capture(tmp_path, scalar=scalar, lanes=lanes, store=store)
    reports = [r for r in result.trace["transformations"] if r["kind"] == "selected-pointer-memory"]
    assert reports and all(r["selector"] == ("per-lane" if lanes else "uniform") for r in reports)
    for i, (n, m, choice) in enumerate(((0, 0, 1), (1, 3, 0), (7, 9, 1), (13, 15, 2), (19, 21, 3))):
        arrays, expected = arguments(n, m, choice, lanes=lanes, store=store)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                   shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("for_loop,lanes,switch", [(False, False, False), (True, False, True), (False, True, False), (True, True, False)])
def test_selected_loop_preserves_changing_bases_and_final_pointer(tmp_path, for_loop, lanes, switch):
    from t2m.native import execute_native
    result = capture(tmp_path, kind="loop", for_loop=for_loop, lanes=lanes, switch=switch)
    for i, (n, steps, choice) in enumerate(((0, 0, 0), (1, 0, 1), (7, 1, 0), (13, 3, 2), (19, 5, 3))):
        arrays, expected = arguments(n, n + 2, choice, kind="loop", lanes=lanes, steps=steps, switch=switch)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                   shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual[0], expected[0])


def test_pointer_shape_operations_keep_selectors_paired_with_offsets(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path, kind="shape")
    for i, n in enumerate((0, 1, 13, 19, 31)):
        arrays, expected = arguments(n, n + 2, kind="shape")
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                   shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual[0], expected[0])


def test_unselected_empty_buffer_is_not_read_but_selected_bounds_are_checked(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path)
    arrays, expected = arguments(9, 0, 0)
    actual, _ = execute_native(result.mlir_text, arrays, tmp_path / "valid", python=NATIVE,
                               shape_abi=result.spec.runtime_shapes())
    np.testing.assert_array_equal(actual[-1], expected[-1])
    arrays[-1] = np.asarray(1, dtype=np.int32)
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "invalid", python=NATIVE,
                       shape_abi=result.spec.runtime_shapes())


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("kind,lanes", [("access", False), ("access", True), ("loop", False), ("loop", True), ("shape", False)])
def test_one_merlin_library_retains_selected_memory_and_input_values(tmp_path, kind, lanes):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    store = kind == "access"
    result = capture(tmp_path, kind=kind, lanes=lanes, store=store)
    arrays, expected = arguments(13, 15, 1, kind=kind, lanes=lanes, store=store, steps=3)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, steps, choice) in enumerate(((0, 0, 0), (1, 0, 1), (7, 1, 0), (13, 3, 2), (19, 5, 3))):
        arrays, expected = arguments(n, n + 2, choice, kind=kind, lanes=lanes, store=store, steps=steps)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, **{f"out{j}": value for j, value in enumerate(expected)})
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python)
        verify_invocation(output)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library

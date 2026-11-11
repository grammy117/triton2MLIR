"""Real byte-backed TTIR atomics through guarded source-word packages."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest

from t2m.byte_atomic_events import view_info
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native toolchains required")


def capture(root, width=32, cas=False, scalar=False, order="little", operation="add", fresh=False):
    from t2m.examples.byte_atomic_kernels import byte_atomic_addresses, byte_atomic_rmw
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    kernel = byte_atomic_addresses if cas else byte_atomic_rmw
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, **({"CAS": True} if cas else {"OP": operation})}
    source.write_text(capture_ttir(kernel, {
        "raw": "*i8", "offsets": "*i64", "values": f"*i{width}", "old": f"*i{width}", "n": "i64",
        **{name: "constexpr" for name in constants},
    }, constants))
    abi = TensorABI("byte_atomic_addresses" if cas else "byte_atomic_rmw", (
        TensorBuffer("raw", (5 * (width // 8) + 1,), "write" if fresh else "readwrite", dtype="i8"),
        TensorBuffer("offsets", ("n",), dtype="i64"),
        TensorBuffer("values", ("n",), dtype=f"i{width}"),
        TensorBuffer("old", ("n",), "write", dtype=f"i{width}"),
    ), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i64", 4, bounds=(0, 32)),))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words",
                          effect_profile="typed-atomics", byte_order=order)


def arguments(width, cas, n, order, operation="add"):
    size = width // 8
    data = np.asarray([(i * 73 + 211) & 255 for i in range(5 * size + 1)], np.uint8)
    offsets = np.resize(np.asarray([0, size, 0, 2 * size, 4 * size, size, 3 * size, 0], np.int64), n)
    unsigned = getattr(np, f"uint{width}")
    dtype = getattr(np, f"int{width}")
    values = np.arange(n, dtype=unsigned) + 3
    expected = data.copy()
    old = np.zeros(n, unsigned)
    limit = (1 << width) - 1
    for lane, at in enumerate(offsets):
        current = int.from_bytes(expected[at:at + size].tobytes(), order)
        if cas and lane % 2 == 0:
            values[lane] = current
        old[lane] = current
        value = int(values[lane])
        def signed(x):
            return x - (1 << width) if x >= 1 << (width - 1) else x
        if cas: changed = value ^ 31 if current == value else current
        elif operation == "add": changed = (current + value) & limit
        elif operation == "and": changed = current & value
        elif operation == "or": changed = current | value
        elif operation == "xor": changed = current ^ value
        elif operation == "max": changed = max(signed(current), signed(value)) & limit
        elif operation == "min": changed = min(signed(current), signed(value)) & limit
        elif operation == "umax": changed = max(current, value)
        elif operation == "umin": changed = min(current, value)
        else: changed = value
        expected[at:at + size] = np.frombuffer(changed.to_bytes(size, order), np.uint8)
    return [data.view(np.int8), offsets, values.view(dtype), np.asarray(n, np.int64)], [expected.view(np.int8), old.view(dtype)]


@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("width,cas,scalar", [(width, cas, scalar) for width, cas in ((32, False), (64, False), (16, True), (32, True), (64, True)) for scalar in (False, True)])
def test_real_kernel_keeps_word_atomicity_and_complete_byte_state(tmp_path, width, cas, scalar, order):
    from t2m.native import execute_native

    result = capture(tmp_path, width, cas, scalar, order)
    access = next(a for a in result.trace["memory"]["atomic_payload"] if "byte_view" in a)
    assert access["byte_view"] == view_info(result.spec.outputs[1].type.element_type, order)
    # Tiled unmasked CAS requires complete source lanes; RMW retains tails.
    for i, n in enumerate((0, 8, 16) if cas and not scalar else (0, 1, 7, 13)):
        arrays, expected = arguments(width, cas, n, order)
        observed, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes(), integer_atomic_regions=True)
        for actual, reference in zip(observed, expected, strict=True):
            assert actual.tobytes() == reference.tobytes()


@pytest.mark.skipif(not all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing explicit Merlin toolchain required")
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("width,cas,scalar", [(width, cas, scalar) for width, cas in ((32, False), (64, False), (16, True), (32, True), (64, True)) for scalar in (False, True)])
def test_complete_kernel_package_stages_and_executes_through_merlin(tmp_path, width, cas, scalar, order):
    from test_integer_products import replay_library

    result = capture(tmp_path, width, cas, scalar, order)
    cases = [arguments(width, cas, n, order) for n in ((0, 8, 16) if cas and not scalar else (0, 1, 7, 13))]
    replay_library(tmp_path, result, cases)


@pytest.mark.parametrize("case", ("unaligned", "negative", "partial-word", "unmasked-cas-tail"))
def test_real_kernel_refuses_invalid_complete_word_events(tmp_path, case):
    from t2m.native import observe_native

    cas = case == "unmasked-cas-tail"
    result = capture(tmp_path, cas=cas)
    arrays, _ = arguments(32, cas, 7 if cas else 1, "little")
    if not cas:
        arrays[1][0] = {"unaligned": 1, "negative": -4, "partial-word": 20}[case]
    outcome = observe_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE,
                             shape_abi=result.spec.runtime_shapes(), integer_atomic_regions=True)
    assert outcome["returncode"] == -6
    message = (tmp_path / "native/stdout.bin").read_bytes()
    assert message == (b"Triton active memory index outside logical allocation\n" if case in ("negative", "partial-word")
                       else b"Triton byte atomic address is not word aligned\n")


@pytest.mark.parametrize("width", (32, 64))
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("operation", ("and", "or", "xor", "max", "min", "umax", "umin", "exch"))
def test_all_integer_rmw_source_kinds_keep_word_bytes_through_merlin(tmp_path, width, order, operation):
    from test_integer_products import replay_library

    if not all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing explicit Merlin toolchain required")
    result = capture(tmp_path, width=width, order=order, operation=operation)
    cases = [arguments(width, False, n, order, operation) for n in (0, 1, 7, 13)]
    replay_library(tmp_path, result, cases)


@pytest.mark.parametrize("change", ("width", "alignment", "guard-tile"))
def test_producer_byte_evidence_requires_exact_integer_fields(tmp_path, change):
    from t2m.atomic_memory import validate_evidence

    result = capture(tmp_path)
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    memory = deepcopy(result.trace["memory"])
    access = next(a for a in admission["accesses"] if "byte_view" in a.get("atomic", {}))
    actual = admission["atomic_adaptations"][0]
    if change == "width":
        access["atomic"]["word_bits"] = actual["word_bits"] = 32.0
    elif change == "alignment":
        access["atomic"]["byte_view"]["alignment_bytes"] = actual["byte_view"]["alignment_bytes"] = 4.0
    else:
        access["guard_tile"][-1] = 4.0
    memory["atomic_payload"] = admission["atomic_adaptations"]
    buffers = {b["name"]: b for b in admission["spec"]["buffers"]}
    with pytest.raises(ValueError, match="exact integers|word/byte"):
        validate_evidence(admission, memory, result.trace.get("normalized_operations", result.trace["source_operations"]), buffers)


def test_byte_word_atomic_requires_every_byte_initialized(tmp_path):
    from t2m.native import observe_native

    result = capture(tmp_path, fresh=True)
    arrays, _ = arguments(32, False, 1, "little")
    outcome = observe_native(result.mlir_text, arrays[1:], tmp_path / "native", python=NATIVE,
                             shape_abi=result.spec.runtime_shapes(), integer_atomic_regions=True)
    assert outcome["returncode"] == -6
    assert (tmp_path / "native/stdout.bin").read_bytes() == b"Triton writable-buffer read before completed initialization\n"


@pytest.mark.parametrize("reverse", (False, True))
def test_last_byte_ordinary_observation_conflicts_with_foreign_word_atomic(tmp_path, reverse):
    from t2m.examples.byte_atomic_kernels import byte_atomic_conflict
    from t2m.frontend import capture_ttir
    from t2m.native import observe_native

    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(byte_atomic_conflict, {"raw": "*i8", "out": "*i32", "REVERSE": "constexpr"}, {"REVERSE": reverse}))
    abi = TensorABI("byte_atomic_conflict", (TensorBuffer("raw", (8,), "readwrite", "i8"), TensorBuffer("out", (2,), "write", "i32")), grid=(2,))
    result = convert_shared(path, abi, tmp_path / "conversion", optimizer=OPT, memory_profile="guarded-buffer",
                            address_policy="source-words", effect_profile="typed-atomics", byte_order="big")
    outcome = observe_native(result.mlir_text, [np.arange(8, dtype=np.int8)], tmp_path / "native", python=NATIVE,
                             integer_atomic_regions=True)
    assert outcome["returncode"] == -6
    assert (tmp_path / "native/stdout.bin").read_bytes() == b"Triton memory conflicts across logical programs\n"

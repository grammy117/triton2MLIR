"""Opt-in native operation verification using an existing, isolated toolchain."""

import os

import pytest

from t2m import BufferSpec, KernelSpec, convert
from t2m.examples.vector_add import masked_add
from t2m.verify import verify_mlir

PYTHON = os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not PYTHON, reason="set T2M_NATIVE_PYTHON to an existing torch-mlir environment")


def test_masked_output_verifies_as_registered_standard_ops():
    spec = KernelSpec("masked_add", tuple(BufferSpec(name, (1000,), effect) for name, effect in
                        (("x_ptr", "read"), ("y_ptr", "read"), ("out_ptr", "write"))),
                      (4,), (("BLOCK", 256), ("N", 1000)))
    result = convert(masked_add, spec)
    report = verify_mlir(result.mlir_text, python=PYTHON, lower_to_llvm=True)
    assert report["native_mlir_verified"]
    assert not report["unregistered_dialects_allowed"]
    assert report["execution"] == "not-performed"
    assert report["llvm_lowering_verified"]


@pytest.mark.parametrize("case", ["shift", "partial", "constant", "reuse", "strided"])
def test_memory_forms_legalize_through_native_standard_passes(case):
    from dataclasses import replace
    from test_memory import shifted_add, partial_update, constant_payload, repeated_read, strided_add, spec
    if case == "shift":
        result = convert(shifted_add, spec(shifted_add, extra=(("SHIFT", 17),)))
    elif case == "partial":
        result = convert(partial_update, spec(partial_update, readwrite=True, extra=(("LOW", 3), ("HIGH", 29))))
    elif case == "constant":
        result = convert(constant_payload, spec(constant_payload))
    elif case == "reuse":
        result = convert(repeated_read, spec(repeated_read))
    else:
        declared = spec(strided_add, extra=tuple(zip(("SX", "SY", "SO", "OX", "OY", "OO"), (2, 3, 4, 3, 1, 2))))
        declared = replace(declared, buffers=(BufferSpec("x_ptr", (31,), "read", strides=(2,), storage_offset=3),
                             BufferSpec("y_ptr", (31,), "read", strides=(3,), storage_offset=1),
                             BufferSpec("out_ptr", (31,), "write", strides=(4,), storage_offset=2)))
        result = convert(strided_add, declared)
    assert verify_mlir(result.mlir_text, python=PYTHON, lower_to_llvm=True)["llvm_lowering_verified"]


def test_unknown_dialect_cannot_pass_native_verification():
    with pytest.raises(ValueError, match="strict native MLIR verification failed"):
        verify_mlir('module { "unregistered_payload.op"() : () -> () }', python=PYTHON)

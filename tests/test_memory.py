"""Masked/strided tensor recovery: exact values, boundary safety and independence."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import triton
import triton.language as tl

from t2m import BufferSpec, KernelSpec, convert
from t2m.bundle import verify_bundle, write_bundle
from t2m.diagnostics import ConversionError
from t2m.execute import execute
from t2m.ir import text


@triton.jit
def shifted_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr, SHIFT: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i - SHIFT
    x = tl.load(x_ptr + j, mask=(j >= 0) & (j < N), other=7.5)
    y = tl.load(y_ptr + i, mask=i < N, other=-2.25)
    tl.store(out_ptr + i, x + y, mask=i < N)


@triton.jit
def repeated_read(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x_ptr + i, mask=i < N, other=0.0, cache_modifier=".cg")
    b = tl.load(x_ptr + i, mask=i < N, other=0.0, cache_modifier=".ca")
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, (a + b) + y, mask=i < N)


@triton.jit
def partial_update(x_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr,
                   LOW: tl.constexpr, HIGH: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    old = tl.load(out_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, old + x, mask=(i >= LOW) & (i < HIGH) & (i < N))


@triton.jit
def strided_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr,
                SX: tl.constexpr, SY: tl.constexpr, SO: tl.constexpr,
                OX: tl.constexpr, OY: tl.constexpr, OO: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i * SX + OX, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i * SY + OY, mask=i < N, other=0.0)
    tl.store(out_ptr + i * SO + OO, x + y, mask=i < N)


@triton.jit
def undefined_other(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N)
    y = tl.load(y_ptr + i, mask=i < N)
    tl.store(out_ptr + i, x + y, mask=i < N)


@triton.jit
def observable_undefined(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i - 3
    x = tl.load(x_ptr + j, mask=(j >= 0) & (j < N))
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, x + y, mask=i < N)


@triton.jit
def invalid_lower_mask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i - 3, mask=i < N, other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, x + y, mask=i < N)


@triton.jit
def neighboring_output_read(x_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    old = tl.load(out_ptr + i + 1, mask=i + 1 < N, other=0.0)
    tl.store(out_ptr + i, old + x, mask=i < N)


@triton.jit
def read_after_store(x_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, x, mask=i < N)
    tl.load(out_ptr + i, mask=i < N, other=0.0)


@triton.jit
def constant_payload(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < N, other=-0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, (x + 1.5) * y, mask=i < N)


@triton.jit
def hole_mask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=(i != 3) & (i < N), other=0.0)
    y = tl.load(y_ptr + i, mask=i < N, other=0.0)
    tl.store(out_ptr + i, x + y, mask=i < N)


@triton.jit
def signed_zero_fill(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i - 3
    x = tl.load(x_ptr + j, mask=(j >= 0) & (j < N), other=-0.0)
    tl.store(out_ptr + i, x, mask=i < N)


def spec(kernel, n=31, block=8, *, extra=(), readwrite=False, programs=None):
    buffers = (BufferSpec("x_ptr", (n,), "read"),)
    if not readwrite:
        buffers += (BufferSpec("y_ptr", (n,), "read"),)
    buffers += (BufferSpec("out_ptr", (n,), "readwrite" if readwrite else "write"),)
    return KernelSpec(kernel.__name__, buffers, (programs or ((n + block - 1) // block),),
                      (("BLOCK", block), ("N", n), *extra), "Preserve tiled tensor dataflow")


@pytest.mark.parametrize("n,block", [(1, 8), (7, 8), (8, 8), (31, 8), (1000, 256)])
@pytest.mark.parametrize("shift", [-40, -5, 0, 3, 17, 100])
def test_boundary_masks_and_nonzero_other(n, block, shift):
    result = convert(shifted_add, spec(shifted_add, n, block, extra=(("SHIFT", shift),)))
    x, y = np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) / 8
    j = np.arange(n) - shift
    expected = np.full(n, 7.5, dtype=np.float32)
    valid = (j >= 0) & (j < n)
    expected[valid] = x[j[valid]]
    np.testing.assert_array_equal(execute(result.mlir_text, [x, y])[0], expected + y)
    assert all(op.dialect_name() in {"builtin", "func", "scf", "tensor", "arith", "linalg"}
               for op in result.module.walk())


def test_empty_extra_programs_never_extract_invalid_addresses():
    result = convert(shifted_add, spec(shifted_add, extra=(("SHIFT", 0),), programs=7))
    x = np.arange(31, dtype=np.float32)
    np.testing.assert_array_equal(execute(result.mlir_text, [x, x.copy()])[0], x + x)


def test_reuse_is_explicit_and_cache_hints_do_not_choose_hardware():
    result = convert(repeated_read, spec(repeated_read))
    reused = result.trace["transformations"][0]["reused_reads"]
    assert len(reused) == 1
    x, y = np.arange(31, dtype=np.float32), np.ones(31, dtype=np.float32)
    np.testing.assert_array_equal(execute(result.mlir_text, [x, y])[0], (x + x) + y)
    assert '"tt.load"' not in result.mlir_text
    assert "cacheModifier" not in result.mlir_text
    matching = [op for op in result.trace["emitted_operations"] if reused[0]["source_id"] in op["source_ids"]]
    assert any(reused[0]["reuses"] in op["source_ids"] for op in matching)


@pytest.mark.parametrize("low,high", [(0, 31), (3, 29), (10, 10), (20, 10), (50, 100), (-10, 8)])
def test_partial_readwrite_preserves_unwritten_values(low, high):
    result = convert(partial_update, spec(partial_update, readwrite=True, extra=(("LOW", low), ("HIGH", high))))
    x = np.arange(31, dtype=np.float32)
    old = np.arange(31, dtype=np.float32) * -2
    expected = old.copy()
    mask = (np.arange(31) >= low) & (np.arange(31) < high)
    expected[mask] += x[mask]
    np.testing.assert_array_equal(execute(result.mlir_text, [x, old])[0], expected)
    assert result.trace["output_initialization"] == "incoming-values-preserved"


def test_strided_storage_becomes_logical_tensor_dataflow():
    declared = spec(strided_add, extra=tuple(zip(("SX", "SY", "SO", "OX", "OY", "OO"), (2, 3, 4, 3, 1, 2))))
    declared = replace(declared, buffers=(BufferSpec("x_ptr", (31,), "read", strides=(2,), storage_offset=3),
                         BufferSpec("y_ptr", (31,), "read", strides=(3,), storage_offset=1),
                         BufferSpec("out_ptr", (31,), "write", strides=(4,), storage_offset=2)))
    result = convert(strided_add, declared)
    x = np.arange(65, dtype=np.float32)[3::2][:31]
    y = np.arange(94, dtype=np.float32)[1::3][:31]
    np.testing.assert_array_equal(execute(result.mlir_text, [x, y])[0], x + y)
    assert "memref" not in result.mlir_text and "!tt.ptr" not in result.mlir_text
    replay = KernelSpec.from_dict(json.loads(json.dumps(declared.as_dict())))
    assert replay.digest == declared.digest
    wrong = replace(declared, buffers=(replace(declared.buffers[0], storage_offset=4), *declared.buffers[1:]))
    with pytest.raises(ConversionError, match="aligned"):
        convert(strided_add, wrong)


def test_undefined_masked_other_requires_a_proof():
    result = convert(undefined_other, spec(undefined_other))
    x = np.arange(31, dtype=np.float32)
    np.testing.assert_array_equal(execute(result.mlir_text, [x, x.copy()])[0], x + x)
    assert any("undefined_lanes" in record for record in result.trace["source_operations"])
    with pytest.raises(ConversionError, match="store may observe an undefined"):
        convert(observable_undefined, spec(observable_undefined))
    exact = convert(undefined_other, spec(undefined_other, n=32))
    x = np.arange(32, dtype=np.float32)
    np.testing.assert_array_equal(execute(exact.mlir_text, [x, x.copy()])[0], x + x)


@pytest.mark.parametrize("kernel,reason", [(invalid_lower_mask, "outside"), (hole_mask, "nonrectangular")])
def test_unproved_masks_are_not_approximated(kernel, reason):
    with pytest.raises(ConversionError, match=reason):
        convert(kernel, spec(kernel))


@pytest.mark.parametrize("kernel,reason", [(neighboring_output_read, "independently owned"),
                                          (read_after_store, "load after output store")])
def test_memory_order_cannot_be_replaced_by_a_snapshot(kernel, reason):
    with pytest.raises(ConversionError, match=reason):
        convert(kernel, spec(kernel, readwrite=True))


def test_constants_and_provenance_removal_have_no_hidden_semantics():
    result = convert(constant_payload, spec(constant_payload))
    x, y = np.arange(31, dtype=np.float32), np.ones(31, dtype=np.float32)
    for op in result.module.walk():
        op.attributes.clear()
    np.testing.assert_array_equal(execute(text(result.module), [x, y])[0], (x + np.float32(1.5)) * y)


def test_readwrite_package_uses_an_incoming_tensor(tmp_path):
    result = convert(partial_update, spec(partial_update, readwrite=True, extra=(("LOW", 3), ("HIGH", 29))))
    x, old = np.ones(31, dtype=np.float32), np.arange(31, dtype=np.float32)
    golden = old.copy()
    golden[3:29] += x[3:29]
    out = write_bundle(result, tmp_path / "readwrite", [x, old], golden)
    assert verify_bundle(out)["materialized_abi"]["inputs"] == 2
    manifest = json.loads((out / "weights.safetensors.manifest.json").read_text())
    assert manifest["1"] == {"kind": "input", "name": "out_ptr"}


def test_contiguous_spec_hash_stays_compatible_with_r0():
    declared = spec(undefined_other, n=32)
    original = declared.as_dict()
    assert all("strides" not in b and "storage_offset" not in b for b in original["buffers"])
    digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert declared.digest == digest


def test_visible_signed_zero_fill_keeps_its_bits(tmp_path):
    declared = spec(signed_zero_fill)
    captured = convert(signed_zero_fill, declared)
    # Stock Triton's AST capture normalizes this Python -0.0 literal to +0.0.
    # Exercise our actual input boundary with an explicit negative-zero TTIR literal.
    lines = captured.source_ttir.splitlines()
    index = next(i for i, line in enumerate(lines) if "dense<" in line and "xf32>" in line)
    prefix, _, suffix = lines[index].partition("dense<")
    _, _, rest = suffix.partition(">")
    lines[index] = prefix + "dense<0x80000000>" + rest
    path = tmp_path / "signed_zero.ttir"
    path.write_text("\n".join(lines) + "\n")
    result = convert(path, declared)
    x, y = np.arange(31, dtype=np.float32), np.ones(31, dtype=np.float32)
    expected = np.concatenate((np.full(3, -0.0, dtype=np.float32), x[:-3]))
    np.testing.assert_array_equal(execute(result.mlir_text, [x, y])[0].view(np.uint32), expected.view(np.uint32))


def test_nan_payload_is_not_quietly_changed_by_literal_decoding(tmp_path):
    declared = spec(signed_zero_fill)
    result = convert(signed_zero_fill, declared)
    lines = result.source_ttir.splitlines()
    index = next(i for i, line in enumerate(lines) if "dense<" in line and "xf32>" in line)
    prefix, _, suffix = lines[index].partition("dense<")
    _, _, rest = suffix.partition(">")
    lines[index] = prefix + "dense<0x7F800001>" + rest
    path = tmp_path / "nan_payload.ttir"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ConversionError, match="payload-preserving"):
        convert(path, declared)

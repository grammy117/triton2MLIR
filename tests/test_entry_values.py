"""Real source value signatures use guarded tensor ports through public Merlin."""

import json
import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.allocation_bindings import (
    PointerArgument,
    ResultBinding,
    ScalarArgument,
    ValueArgument,
    convert_allocations,
)
from t2m.bundle import _record
from t2m.dynamic_abi import GridExtent
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer
from t2m.tensor_storage import DTYPES

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def compile_public(result, root, arrays, expected, *, owned=False):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    with np.load(root / "native/outputs.npz", allow_pickle=False) as outputs:
        for i, value in enumerate(expected):
            assert outputs[f"out{i}"].tobytes() == value.tobytes()
    stage_compile_inputs(root / "capture", root / "stage", result_mode="execution-owned" if owned else None)
    return compile(StagedProgram(root / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=root / "compiled")


def width(dtype):
    return int(dtype[1:]) if dtype in ("i1", "i4") else np.dtype(DTYPES[dtype]).itemsize * 8


def scalar_program(dtype, capacity):
    bits = width(dtype)
    if dtype.startswith("i"):
        body = f"%r=arith.xori %x,%y:{dtype}"
    else:
        body = f'''%a=arith.bitcast %x:{dtype} to i{bits}
          %b=arith.bitcast %y:{dtype} to i{bits}
          %c=arith.xori %a,%b:i{bits}
          %r=arith.bitcast %c:i{bits} to {dtype}'''
    source = f'''module {{
      tt.func private @combine(%x:{dtype},%y:{dtype})->{dtype} {{
        {body}
        tt.return %r:{dtype}
      }}
      tt.func public @values(%x:{dtype},%y:{dtype},%enabled:i1)->({dtype},{dtype}) {{
        %r=scf.if %enabled -> ({dtype}) {{
          %c=tt.call @combine(%x,%y):({dtype},{dtype})->{dtype}
          scf.yield %c:{dtype}
        }} else {{scf.yield %x:{dtype}}}
        tt.return %x,%r:{dtype},{dtype}
      }}
    }}'''
    abi = TensorABI("bound_values", (
        TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("y", ("n",), dtype=dtype),
        TensorBuffer("original", ("n",), "write", dtype), TensorBuffer("computed", ("n",), "write", dtype)),
        runtime_scalars=(RuntimeScalar("enabled", "i1", 4), RuntimeScalar("n", "i32", 5, bounds=(0, capacity)),
                         RuntimeScalar("origin", "i64", 6, bounds=(-1, 1))),
        grid=(GridExtent("n", 1),))
    return source, abi


def arguments(dtype, n, enabled, shift):
    bits = width(dtype)
    word = np.dtype(f"uint{max(8, bits)}")
    x = np.arange(n, dtype=np.uint64)
    if bits > 16:
        x *= np.uint64(0x9E3779B97F4A7C15)
        seeds = [0, 1, 1 << (bits - 1), (1 << bits) - 1,
                 0x7F800001 if bits == 32 else 0x7FF0000000000001]
        x[:min(n, len(seeds))] = seeds[:min(n, len(seeds))]
    x = (x + np.uint64(shift)).astype(word)
    y = (np.arange(n, dtype=np.uint64) * np.uint64(37) + np.uint64(7 * shift)).astype(word)
    if bits in (1, 4):
        x &= (1 << bits) - 1
        y &= (1 << bits) - 1
    carrier = np.dtype(DTYPES[dtype])
    arrays = [x.view(carrier).copy(), y.view(carrier).copy(), np.asarray(enabled), np.asarray(n, np.int32), np.asarray(0, np.int64)]
    result = np.bitwise_xor(x, y) if enabled else x.copy()
    return arrays, [x.view(carrier).copy(), result.view(carrier).copy()]


@pytest.mark.parametrize("dtype", tuple(DTYPES))
def test_scalar_value_inputs_multiple_results_keep_every_storage_word(tmp_path, dtype):
    bits = width(dtype)
    capacity = (1 << bits) if bits <= 16 else 257
    source, abi = scalar_program(dtype, capacity)
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_allocations(path, abi,
        (ValueArgument("x", "origin", per_program=True), ValueArgument("y", per_program=True), ScalarArgument("enabled")),
        tmp_path / "conversion", source_entry="values",
        result_bindings=(ResultBinding("original"), ResultBinding("computed")),
        memory_profile="guarded-buffer", address_policy="source-words")
    configuration = json.loads(result.auxiliary_artifacts["allocation-arguments.json"])
    assert configuration["schema"] == "t2m.source-allocation-arguments.v3"
    assert len(configuration["results"]) == 2
    assert not any(op.name == "tt.call" or op.name.startswith("llvm.") for op in result.module.walk())
    arrays, expected = arguments(dtype, capacity, True, 0)
    product = compile_public(result, tmp_path, arrays, expected, owned=dtype in ("i4", "f8E4M3FNUZ"))
    library = _record(tmp_path / "compiled/host/model.so")
    for i, (n, enabled, shift) in enumerate(((capacity, True, 0), (capacity, False, 7),
            (min(capacity, 13), True, 3), (1, True, 9), (0, False, 0))):
        arrays, expected = arguments(dtype, n, enabled, shift)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        assert len(actual) == 2
        for value, golden in zip(actual, expected, strict=True):
            assert value.shape == golden.shape and value.dtype == golden.dtype
            assert value.tobytes() == golden.tobytes()
        assert before == [a.tobytes() for a in arrays]
    assert _record(tmp_path / "compiled/host/model.so") == library
    if dtype == "i32":
        bad, _ = arguments(dtype, 1, True, 0)
        bad[-1] = np.asarray(1, np.int64)
        invoke(tmp_path, bad, "invalid-port-origin", error="Triton active memory index outside logical allocation")
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    product.verify()


@pytest.mark.parametrize("dtype", ("f8E5M2", "i4"))
def test_tensor_value_arguments_and_differently_shaped_results_keep_tiles(tmp_path, dtype):
    source = f'''module {{tt.func public @tiles(%x:tensor<2x4x{dtype}>)->(tensor<2x4x{dtype}>,tensor<4x2x{dtype}>,i32) {{
      %t=tt.trans %x {{order=array<i32:1,0>}}:tensor<2x4x{dtype}>->tensor<4x2x{dtype}>
      %xid=tt.get_program_id x:i32
      %yid=tt.get_program_id y:i32
      %zid=tt.get_program_id z:i32
      %ten=arith.constant 10:i32
      %hundred=arith.constant 100:i32
      %yterm=arith.muli %yid,%ten:i32
      %zterm=arith.muli %zid,%hundred:i32
      %xy=arith.addi %xid,%yterm:i32
      %coordinate=arith.addi %xy,%zterm:i32
      tt.return %x,%t,%coordinate:tensor<2x4x{dtype}>,tensor<4x2x{dtype}>,i32
    }}}}'''
    abi = TensorABI("bound_tiles", (TensorBuffer("x", (2, 2, "n", 2, 4), dtype=dtype),
        TensorBuffer("original", (2, 2, "n", 2, 4), "write", dtype), TensorBuffer("transposed", (2, 2, "n", 4, 2), "write", dtype),
        TensorBuffer("coordinate", (2, 2, "n"), "write", "i32")),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 8)),), grid=(GridExtent("n", 1), 2, 2))
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_allocations(path, abi, (ValueArgument("x", per_program=True),), tmp_path / "conversion", source_entry="tiles",
        result_bindings=(ResultBinding("original"), ResultBinding("transposed"), ResultBinding("coordinate")),
        memory_profile="guarded-buffer", address_policy="source-words")
    def values(n, shift):
        x = ((np.arange(4 * n * 8, dtype=np.uint32) + shift) % (16 if dtype == "i4" else 256)).astype(np.uint8).reshape(2, 2, n, 2, 4)
        coordinates = np.indices((2, 2, n), dtype=np.int32)
        tag = coordinates[2] + 10 * coordinates[1] + 100 * coordinates[0]
        return [x, np.asarray(n, np.int32)], [x.copy(), x.transpose(0, 1, 2, 4, 3).copy(), tag]
    arrays, expected = values(8, 0)
    product = compile_public(result, tmp_path, arrays, expected)
    for i, (n, shift) in enumerate(((8, 0), (3, 7), (1, 9), (0, 0))):
        arrays, expected = values(n, shift)
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        for value, golden in zip(actual, expected, strict=True):
            assert value.shape == golden.shape and value.tobytes() == golden.tobytes()
    product.verify()


@pytest.mark.parametrize("kind", ("returned-i32", "stored-f16"))
def test_original_native_verified_entry_counterexamples_execute_publicly(tmp_path, kind):
    from pathlib import Path

    ledger = json.loads((Path(__file__).parent / "data/v1_generic_gaps.json").read_text())
    gap = next(row for row in (*ledger["gaps"], *ledger["closed_gaps"]) if row["id"] == "public-value-entry-binding")
    witness = gap["witnesses"][0 if kind == "returned-i32" else 1]
    path = tmp_path / "source.ttir"
    path.write_text(witness["source"])
    inspect_ttir(path, tmp_path / "source-verification")
    if kind == "returned-i32":
        abi = TensorABI("bound_counterexample", (TensorBuffer("x", (1,), dtype="i32"),
            TensorBuffer("result", (1,), "write", "i32")))
        bindings, returned = (PointerArgument("x"),), (ResultBinding("result"),)
        def values(x):
            return [np.asarray([x], np.int32)], [np.asarray([x + 1], np.int32)]
        samples = (41, -7, 0)
    else:
        abi = TensorABI("bound_counterexample", (TensorBuffer("alpha", (1,), dtype="f16"),
            TensorBuffer("out", (1,), "write", "f16")))
        bindings, returned = (ValueArgument("alpha"), PointerArgument("out")), ()
        def values(x):
            return [np.asarray([x], np.float16)], [np.asarray([x * 2], np.float16)]
        samples = (3.5, -7, 0)
    result = convert_allocations(path, abi, bindings, tmp_path / "conversion", source_entry=witness["bindings"]["entry"],
        result_bindings=returned, memory_profile="guarded-buffer", address_policy="source-words")
    arrays, expected = values(samples[0])
    product = compile_public(result, tmp_path, arrays, expected)
    for i, sample in enumerate(samples):
        arrays, expected = values(sample)
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        assert len(actual) == 1 and actual[0].tobytes() == expected[0].tobytes()
    product.verify()
    if kind == "returned-i32":
        from t2m.allocation_bindings import arguments_as_dict
        from t2m.bundle import verify_bundle
        from t2m.cli import main

        (tmp_path / "abi.json").write_text(json.dumps(abi.as_dict()))
        configuration = arguments_as_dict(witness["bindings"]["entry"], bindings, result_bindings=returned)
        (tmp_path / "bindings.json").write_text(json.dumps(configuration))
        arrays, expected = values(samples[0])
        np.savez(tmp_path / "cli-inputs.npz", in0=arrays[0])
        np.savez(tmp_path / "cli-reference.npz", out0=expected[0])
        output = tmp_path / "cli-capture"
        assert main(["shared-tensor", str(path), "--abi", str(tmp_path / "abi.json"), "--allocation-bindings",
            str(tmp_path / "bindings.json"), "--memory-profile", "guarded-buffer", "--address-policy", "source-words",
            "--optimizer", os.environ["TRITON_SHARED_OPT"], "--python", os.environ["T2M_NATIVE_PYTHON"],
            "--inputs", str(tmp_path / "cli-inputs.npz"), "--reference", str(tmp_path / "cli-reference.npz"),
            "--out", str(output)]) == 0
        verify_bundle(output)


def test_descriptor_arguments_compose_with_stored_values_and_returned_tiles(tmp_path):
    from t2m.descriptor_bindings import DescriptorArgument

    source = '''module {tt.func public @descriptor_values(%desc:!tt.tensordesc<tensor<2x16xf32>>,%alpha:f32)->tensor<2x16xf32> {
      %z=arith.constant 0:i32
      %tile=tt.descriptor_load %desc[%z,%z]:!tt.tensordesc<tensor<2x16xf32>>->tensor<2x16xf32>
      %bias=tt.splat %alpha:f32->tensor<2x16xf32>
      %sum=arith.addf %tile,%bias:tensor<2x16xf32>
      tt.return %sum:tensor<2x16xf32>
    }}'''
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    abi = TensorABI("bound_descriptor_values", (TensorBuffer("x", (40,)), TensorBuffer("alpha", (1,)),
        TensorBuffer("out", (2, 16), "write")))
    result = convert_allocations(path, abi,
        (DescriptorArgument("x", (2, 16), (16, 1), offset=3), ValueArgument("alpha")),
        tmp_path / "conversion", source_entry="descriptor_values", result_bindings=(ResultBinding("out"),),
        memory_profile="guarded-buffer", address_policy="source-words")
    assert "descriptor_bindings" in result.trace["source_argument_binding"]
    def values(shift, bias):
        x = np.arange(40, dtype=np.float32) + np.float32(shift)
        alpha = np.asarray([bias], np.float32)
        return [x, alpha], [x[3:35].reshape(2, 16) + alpha[0]]
    arrays, expected = values(0, 0.5)
    product = compile_public(result, tmp_path, arrays, expected)
    for i, (shift, bias) in enumerate(((0, 0.5), (-17, -2), (31, 3.5))):
        arrays, expected = values(shift, bias)
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        assert len(actual) == 1 and actual[0].tobytes() == expected[0].tobytes()
    product.verify()

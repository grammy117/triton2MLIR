"""Real packed source functions must be independent of unknown register grouping."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, pack, dtype, profile="guarded-buffer"):
    from t2m.examples.elementwise_map_kernels import independent_packed_maps
    from t2m.frontend import capture_ttir

    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.ttir"
    pointer = {"f32": "*fp32", "i32": "*i32", "i64": "*i64"}[dtype]
    names = ("x", "y", "out_a", "out_b")
    source.write_text(capture_ttir(independent_packed_maps,
        {**{name: pointer for name in names}, "n": "i32", "iterations": "i32", "B": "constexpr", "PACK": "constexpr"},
        {"B": 32, "PACK": pack}))
    abi = TensorABI("independent_packed_maps", tuple(TensorBuffer(name, ("n",),
        "write" if name.startswith("out") else "read", dtype) for name in names), grid=(GridExtent("n", 32),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 67)), RuntimeScalar("iterations", "i32", 5, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, iterations, dtype):
    dt = np.dtype({"i32": "int32", "i64": "int64", "f32": "float32"}[dtype])
    x = np.arange(n, dtype=dt) * 3 - 17
    if dtype == "i64":
        x += np.int64(1 << 60)
    y = np.resize(np.asarray([0, -2, 4, 1, -1, 0, 2], dtype=dt), n)
    a, b = x.copy(), y.copy()
    for step in range(iterations):
        if dtype == "f32":
            divisor = np.where(b != 0, b, np.float32(1))
            quotient = np.where(b != 0, a / divisor, np.float32(-7))
        else:
            # Python integers keep signed truncation exact above 2**53.
            values = []
            for left, right in zip(a, b, strict=True):
                left, right = int(left), int(right)
                value = (-1 if (left < 0) != (right < 0) else 1) * (abs(left) // abs(right)) if right else -7
                values.append(value)
            quotient = np.asarray(values, dtype=dt)
        a, b = (quotient + dt.type(step)).astype(dt), (a + b).astype(dt)
    return [x, y, np.asarray(n, np.int32), np.asarray(iterations, np.int32)], [a, b]


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("pack,dtype,profile", [(p, t, "guarded-buffer") for p in (2, 4) for t in ("i32", "i64", "f32")]
                         + [(2, "f32", profile) for profile in ("tensor", "buffer")])
def test_packed_scalar_functions_keep_runtime_tiles_and_recurrences(tmp_path, pack, dtype, profile):
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle

    result = capture(tmp_path, pack, dtype, profile)
    reports = [t for t in result.trace["transformations"] if t["kind"] == "ttir-elementwise-map"]
    assert reports and all(t["pack"] == pack and t["packing_proof"]["grouping"] == "independent-of-register-layout" for t in reports)
    assert any(op.attributes.get("prov.map_pack") is not None for op in result.module.walk())
    arrays, expected = arguments(37, 3, dtype)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    assert any("identical slot-local functions" in difference
               for difference in json.loads((bundle / "compatibility.json").read_text())["differences"])
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt = verify_kernel_qualification(compiled)
    for index, (n, count) in enumerate(((0, 0), (1, 4), (32, 0), (67, 2))):
        arrays, expected = arguments(n, count, dtype)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, **{f"out{j}": value for j, value in enumerate(expected)})
        invocation = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{index}", merlin_root=root, python=python)
        assert invocation["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt


def test_cross_slot_additions_cannot_be_mistaken_for_independent_identical_terms(tmp_path):
    from t2m.diagnostics import ConversionError
    from t2m.examples.elementwise_map_kernels import coupled_packed_map
    from t2m.frontend import capture_ttir

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(coupled_packed_map, {"x": "*i64", "out": "*i64"}, {}))
    abi = TensorABI("coupled_packed_map", (TensorBuffer("x", (8,), dtype="i64"), TensorBuffer("out", (8,), "write", "i64")))
    with pytest.raises(ConversionError, match="logical grouping.*coupled") as exc:
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT)
    assert exc.value.opcode == "tt.map_elementwise" and exc.value.source_id


def test_packed_slot_arithmetic_attributes_are_semantic_even_with_a_claimed_proof(tmp_path):
    from t2m.diagnostics import ConversionError

    source = tmp_path / "source.ttir"
    source.write_text('''module {
      tt.func public @packed_attributes(%x:!tt.ptr<f32>,%y:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
        %lane=tt.make_range {start=0:i32,end=8:i32}:tensor<8xi32>
        %xs=tt.splat %x:!tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
        %ys=tt.splat %y:!tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
        %os=tt.splat %out:!tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
        %xp=tt.addptr %xs,%lane:tensor<8x!tt.ptr<f32>>,tensor<8xi32>
        %yp=tt.addptr %ys,%lane:tensor<8x!tt.ptr<f32>>,tensor<8xi32>
        %op=tt.addptr %os,%lane:tensor<8x!tt.ptr<f32>>,tensor<8xi32>
        %a=tt.load %xp:tensor<8x!tt.ptr<f32>>
        %b=tt.load %yp:tensor<8x!tt.ptr<f32>>
        %r="tt.map_elementwise"(%a,%b) <{pack=2:i32}> ({
        ^bb0(%a0:f32,%a1:f32,%b0:f32,%b1:f32):
          %p=arith.addf %a0,%b0:f32
          %q=arith.addf %a1,%b1 fastmath<reassoc>:f32
          tt.map_elementwise.return %p,%q:f32,f32
        }) {prov.map_grouping="proved-slot-independent-v1"} : (tensor<8xf32>,tensor<8xf32>)->tensor<8xf32>
        tt.store %op,%r:tensor<8x!tt.ptr<f32>>
        tt.return
      }
    }''')
    abi = TensorABI("packed_attributes", tuple(TensorBuffer(name, (8,), "write" if name == "out" else "read", "f32")
                                               for name in ("x", "y", "out")))
    with pytest.raises(ConversionError, match="logical grouping.*different scalar functions") as exc:
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT)
    assert exc.value.opcode == "tt.map_elementwise" and exc.value.source_id


@pytest.mark.parametrize("bits", ["0x80000000", "0x7FC00001"])
def test_packed_float_words_cannot_use_lossy_attribute_equality(tmp_path, bits):
    from t2m.diagnostics import ConversionError
    from t2m.elementwise_map import expand_elementwise_maps
    from t2m.shared_tensor import parse_shared

    # Direct scalar-region input preserves the distinct literal definitions
    # before any native constant pooling. xDSL's ordinary FloatData comparison
    # merges signed zeros and NaNs, which is unsuitable for this proof.
    first = "0x00000000" if bits == "0x80000000" else "0x7FC00000"
    module = parse_shared('''module {
      func.func @words(%x:tensor<8xf32>) -> tensor<8xf32> {
        %v="tt.map_elementwise"(%x) <{pack=2:i32}> ({
        ^bb0(%a0:f32,%a1:f32):
          %p=arith.constant ''' + first + ''':f32
          %q=arith.constant ''' + bits + ''':f32
          "tt.map_elementwise.return"(%p,%q):(f32,f32)->()
        }):(tensor<8xf32>)->tensor<8xf32>
        func.return %v:tensor<8xf32>
      }
    }''')
    with pytest.raises(ConversionError, match="logical grouping.*different scalar functions"):
        expand_elementwise_maps(module, lambda location: [])


def test_dead_cross_slot_arithmetic_is_neither_executed_nor_attributed_as_equivalent():
    from t2m.elementwise_map import expand_elementwise_maps
    from t2m.shared_tensor import _source_ids, parse_shared

    module = parse_shared('''module {
      func.func @dead(%x:tensor<8xi64>) -> tensor<8xi64> {
        %v="tt.map_elementwise"(%x) <{pack=2:i32}> ({
        ^bb0(%a0:i64,%a1:i64):
          %dead=arith.addi %a0,%a1:i64 loc("ttir.op.dead-add")
          %unused=arith.divsi %a0,%a1:i64 loc("ttir.op.dead-div")
          %p=arith.addi %a0,%a0:i64 loc("ttir.op.slot-zero")
          %q=arith.addi %a1,%a1:i64 loc("ttir.op.slot-one")
          "tt.map_elementwise.return"(%p,%q):(i64,i64)->()
        }):(tensor<8xi64>)->tensor<8xi64>
        func.return %v:tensor<8xi64>
      }
    }''')
    report, = expand_elementwise_maps(module, _source_ids)
    module.verify()
    arithmetic = [op for op in module.walk() if op.name.startswith("arith.")]
    assert len(arithmetic) == 1 and arithmetic[0].name == "arith.addi"
    assert set(_source_ids(arithmetic[0].location)) == {"ttir.op.slot-zero", "ttir.op.slot-one"}
    assert report["packing_proof"]["discarded_pure_dead_source_ids"] == ["ttir.op.dead-add", "ttir.op.dead-div"]

"""Source-derived cross-slot groups execute as portable parallel tensor dataflow."""

import itertools
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def capture(root, pack, dtype, matrix, warps, profile):
    from t2m.examples.grouped_map_kernels import grouped_map
    from t2m.frontend import capture_ttir
    tile = 128 * pack
    source = root / "source.ttir"
    names = ("x", "y", "out_mix", "out_asymmetric")
    source.write_text(capture_ttir(grouped_map, {**{n: "*fp32" if dtype == "f32" else "*i64" for n in names},
        "n": "i32", "B": "constexpr", "PACK": "constexpr", "MATRIX": "constexpr"},
        {"B": tile, "PACK": pack, "MATRIX": matrix}))
    abi = TensorABI("grouped_map", tuple(TensorBuffer(name, ("n",), "write" if name.startswith("out") else "read", dtype)
        for name in names), grid=(GridExtent("n", tile),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, tile * 2 + 19)),))
    result = convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
        source_layout={"target": "cuda:80", "num_warps": warps, "threads_per_warp": 32, "num_ctas": 1})
    return result, tile


def arguments(n, dtype, tile, row, *, control=False):
    dt = np.dtype(np.int64 if dtype == "i64" else np.float32)
    x = np.arange(n, dtype=dt) * 3 - 17
    if dtype == "i64":
        x += np.int64(1 << 60)
    y = np.resize(np.asarray([0, -2, 4, 1, -1, 0, 2], dtype=dt), n) if control else np.arange(n, dtype=dt) * -2 + 5
    expected = [np.empty(n, dtype=dt), np.empty(n, dtype=dt)]
    # Independent scalar reference follows the original physical register packs
    # and forward layout. It does not use the emitted inverse/index expressions.
    sizes = dict(row["inverse"]["out_dims"])
    pack = row["pack"]
    groups = []
    for high, lane, warp, block in itertools.product(range(sizes["register"] // pack), range(sizes["lane"]),
                                                    range(sizes["warp"]), range(sizes["block"])):
        indices = []
        for slot in range(pack):
            physical = {"register": high * pack + slot, "lane": lane, "warp": warp, "block": block}
            coordinates = np.zeros(len(row["shape"]), dtype=np.int64)
            for name, vectors in row["forward"]["bases"]:
                for bit, vector in enumerate(vectors):
                    if (physical[name] >> bit) & 1:
                        coordinates ^= np.asarray(vector, dtype=np.int64)
            indices.append(np.ravel_multi_index(tuple(coordinates), tuple(row["shape"])))
        groups.append(indices)
    assert sorted(i for group in groups for i in group) == list(range(tile))
    for start in range(0, n, tile):
        active = min(tile, n - start)
        a, b = np.zeros(tile, dtype=dt), np.zeros(tile, dtype=dt)
        a[:active], b[:active] = x[start:start + active], y[start:start + active]
        first, second = np.empty(tile, dtype=dt), np.empty(tile, dtype=dt)
        for indices in groups:
            values = a[indices]
            if control:
                left, right = int(values[0]), int(b[indices[0]])
                value = (-1 if (left < 0) != (right < 0) else 1) * (abs(left) // abs(right)) if right else int(values[1])
                for _ in range(int(b[indices[1]]) & 3):
                    value += int(values[1])
                first[indices] = [value, value + int(b[indices[1]])]
            else:
                first[indices] = ([values[0] + values[1], values[1] - values[0]] if pack == 2 else
                    [values[0] + values[3], values[1] - values[2], values[2] + values[1], values[3] - values[0]])
            second[indices] = [a[i] + (b[i] if slot % 2 == 0 else -b[i]) for slot, i in enumerate(indices)]
        expected[0][start:start + active], expected[1][start:start + active] = first[:active], second[:active]
    return [x, y, np.asarray(n, np.int32)], expected


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin tools required")
@pytest.mark.parametrize("pack,dtype,matrix,warps,profile", [(2, "i64", False, w, "guarded-buffer") for w in (2, 4)]
    + [(4, "i64", True, 4, "guarded-buffer")]
    + [(2, "f32", False, 4, p) for p in ("tensor", "buffer", "guarded-buffer")])
def test_actual_source_groups_keep_cross_slot_and_asymmetric_computation(tmp_path, pack, dtype, matrix, warps, profile):
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result, tile = capture(tmp_path, pack, dtype, matrix, warps, profile)
    witness = result.trace["source_layout_specialization"]
    assert witness["device_codegen"] == "not-invoked"
    assert witness["ttgir"] and "#ttg" not in result.mlir_text
    row, = witness["maps"]
    if not matrix:
        assert row["forward"]["bases"][0][1][0] == [warps * 32]
    assert any(op.attributes.get("prov.map_grouping") is not None for op in result.module.walk())
    arrays, expected = arguments(tile + 7, dtype, tile, row)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt = verify_kernel_qualification(compiled)
    for index, n in enumerate((0, 1, tile, tile * 2 + 19)):
        arrays, expected = arguments(n, dtype, tile, row)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": a for i, a in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": a for i, a in enumerate(expected)})
        report = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{index}", merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin tools required")
def test_source_grouped_coupled_branches_and_while_state_keep_full_execution(tmp_path):
    from t2m.examples.grouped_map_kernels import grouped_control_map
    from t2m.frontend import capture_ttir
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle
    source = tmp_path / "source.ttir"
    names = ("x", "y", "out_mix", "out_asymmetric")
    source.write_text(capture_ttir(grouped_control_map, {**{n: "*i64" for n in names}, "n": "i32", "B": "constexpr"}, {"B": 256}))
    abi = TensorABI("grouped_control_map", tuple(TensorBuffer(name, ("n",), "write" if name.startswith("out") else "read", "i64")
        for name in names), grid=(GridExtent("n", 256),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 531)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words",
        source_layout={"target": "cuda:80", "num_warps": 4, "threads_per_warp": 32, "num_ctas": 1})
    row, = result.trace["source_layout_specialization"]["maps"]
    arrays, expected = arguments(263, "i64", 256, row, control=True)
    assert any(op.name == "scf.while" for op in result.module.walk())
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native", python=NATIVE)
    qualify_kernel(tmp_path / "capture", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])

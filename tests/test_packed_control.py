"""Packed scalar branches and data-dependent loops retain their exact execution."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared


OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def capture(root, pack, dtype, while_loop, profile):
    from t2m.examples.packed_control_kernels import packed_control
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    names = ("x", "y", "counts", "out_a", "out_b")
    source.write_text(capture_ttir(packed_control, {
        **{name: "*fp32" if dtype == "f32" else "*" + dtype for name in names if name != "counts"},
        "counts": "*i32", "n": "i32", "iterations": "i32", "B": "constexpr", "PACK": "constexpr", "WHILE": "constexpr"},
        {"B": 32, "PACK": pack, "WHILE": while_loop}))
    abi = TensorABI("packed_control", tuple(TensorBuffer(name, ("n",),
        "write" if name.startswith("out") else "read", "i32" if name == "counts" else dtype) for name in names),
        grid=(GridExtent("n", 32),), runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 67)),
        RuntimeScalar("iterations", "i32", 6, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, iterations, dtype):
    dt = np.dtype({"i32": "int32", "i64": "int64", "f32": "float32"}[dtype])
    x = np.arange(n, dtype=dt) * 3 - 17
    if dtype == "i64":
        x += np.int64(1 << 60)
    y = np.resize(np.array([0, -2, 4, 1, -1, 0, 2], dtype=dt), n)
    counts = np.resize(np.array([0, 3, -4, 1, 2], dtype=np.int32), n)
    first = []
    for a, b, count in zip(x, y, counts, strict=True):
        if dtype == "f32":
            value = dt.type(a / b) if b else dt.type(-7)
        else:
            left, right = int(a), int(b)
            value = dt.type((-1 if (left < 0) != (right < 0) else 1) * (abs(left) // abs(right)) if right else -7)
        for step in range(max(0, int(count) + iterations)):
            value = dt.type(value + dt.type(b * dt.type(step)))
        first.append(value)
    return [x, y, counts, np.asarray(n, np.int32), np.asarray(iterations, np.int32)], [np.array(first, dtype=dt), x + y]


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin tools required")
@pytest.mark.parametrize("pack,dtype,while_loop,profile", [
    (pack, dtype, loop, "guarded-buffer") for pack in (2, 4) for dtype in ("i32", "i64", "f32") for loop in (False, True)
] + [(2, "f32", True, profile) for profile in ("tensor", "buffer")])
def test_packed_control_keeps_branch_execution_and_each_elements_loop_state(tmp_path, pack, dtype, while_loop, profile):
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, pack, dtype, while_loop, profile)
    reports = [r for r in result.trace["transformations"] if r["kind"] == "ttir-elementwise-map"]
    assert reports and all(r["packing_proof"]["method"] == "typed-structured-ssa-congruence-v1" for r in reports)
    assert any(op.name == ("scf.while" if while_loop else "scf.for") for op in result.module.walk())
    assert any(op.name == "scf.if" for op in result.module.walk())
    maps = [op for op in result.module.walk() if op.attributes.get("prov.map_pack") is not None]
    assert len(maps) == 1
    assert maps[0].attributes["prov.map_pack"].value.data == pack
    # Two original slot-local loops become one logical scalar loop; its runtime
    # bounds and branch are retained, rather than unrolled or eagerly selected.
    assert sum(op.name == ("scf.while" if while_loop else "scf.for") for op in maps[0].walk()) == 1
    arrays, expected = arguments(37, 2, dtype)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt = verify_kernel_qualification(compiled)
    for index, (n, count) in enumerate(((0, 0), (1, 4), (32, 0), (67, 3))):
        arrays, expected = arguments(n, count, dtype)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        invocation = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{index}", merlin_root=root, python=python)
        assert invocation["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt

"""Visible FP4 M/N-packed MX contraction and independently checked package."""

from pathlib import Path

import numpy as np

from .effect_bundle import write_effect_bundle
from .examples.scaled_dot_kernels import scaled_dot_tile
from .examples.scaled_dot_reference import reference
from .frontend import capture_ttir
from .scaled_dot import PROFILE
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def demo(
    out,
    *,
    optimizer=None,
    python=None,
    memory_profile="tensor",
    address_policy="structured",
    effect_profile="ordinary",
    f32_atomic_add=None,
):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    m, n, k = 16, 16, 64
    params = dict(
        M=m,
        N=n,
        K=k,
        A_FORMAT="e2m1",
        B_FORMAT="e2m1",
        LHS_K_PACK=False,
        RHS_K_PACK=False,
        HAS_A_SCALE=True,
        HAS_B_SCALE=True,
        GROUP=32,
    )
    signature = {
        name: "*u8" if i < 4 else "*fp32"
        for i, name in enumerate(scaled_dot_tile.arg_names[:6])
    }
    signature.update({name: "constexpr" for name in params})
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            scaled_dot_tile, signature, params, source_target=("hip", "gfx942", 64)
        )
    )
    a = np.resize(np.arange(256, dtype=np.uint8), (m // 2, k)).view(np.int8)
    b = np.resize(np.arange(255, -1, -1, dtype=np.uint8), (k, n // 2)).view(np.int8)
    sa = np.resize(np.asarray([126, 128, 127, 129], np.uint8), (m, k // 32)).view(
        np.int8
    )
    sb = np.resize(np.asarray([128, 126, 127, 125], np.uint8), (n, k // 32)).view(
        np.int8
    )
    initial = np.resize(
        np.asarray([0, -0.0, 2**24, -(2**24), 3, -5], np.float32), (m, n)
    )
    arrays = [a, b, sa, sb, initial]
    abi = TensorABI(
        "scaled_dot_tile",
        tuple(
            TensorBuffer(name, value.shape, dtype=dtype)
            for name, value, dtype in zip(
                ("a", "b", "sa", "sb", "initial"),
                arrays,
                ("i8", "i8", "i8", "i8", "f32"),
                strict=True,
            )
        )
        + (TensorBuffer("out", (m, n), "write"),),
    )
    result = convert_shared(
        source,
        abi,
        work / "translate",
        optimizer=optimizer,
        memory_profile=memory_profile,
        address_policy=address_policy,
        effect_profile=effect_profile,
        f32_atomic_add=f32_atomic_add,
    )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    write_effect_bundle(
        result,
        out,
        arrays,
        policy,
        references=[reference(a, b, sa, sb, initial, lhs_k=False, rhs_k=False)],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": abi.entry,
        "logical_tiles": [m, n, k],
        "packed_axes": ["M", "N"],
        "numeric_profile": PROFILE,
        "native_execution": "passed",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }

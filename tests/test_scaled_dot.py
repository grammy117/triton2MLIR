"""Scaled logical tiles qualify separately from any GPU's MX instruction."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from t2m.diagnostics import ConversionError
from t2m.examples.scaled_dot_reference import reference
from t2m.ir import text
from t2m.scaled_dot import FORMATS, PROFILE, expand_scaled_dots
from t2m.shared_tensor import opcode, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = (
    OPT
    and NATIVE
    and all(
        os.environ.get(name)
        for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
STORAGE = {"e2m1": "i8", "e4m3": "i8", "e5m2": "i8", "fp16": "f16", "bf16": "bf16"}


def source(
    lhs="e2m1",
    rhs="e2m1",
    packs=(True, True),
    scales=("i8", "i8"),
    batch=False,
    *,
    m=2,
    n=4,
    k=64,
):
    prefix = (2,) if batch else ()

    def typ(dtype, shape):
        return "tensor<" + "x".join(map(str, (*prefix, *shape, dtype))) + ">"

    ashape = (
        m // (2 if lhs == "e2m1" and not packs[0] else 1),
        k // (2 if lhs == "e2m1" and packs[0] else 1),
    )
    bshape = (
        k // (2 if rhs == "e2m1" and packs[1] else 1),
        n // (2 if rhs == "e2m1" and not packs[1] else 1),
    )
    types = [typ(STORAGE[lhs], ashape), typ(STORAGE[rhs], bshape), typ("f32", (m, n))]
    names = ["%a", "%b", "%c"]
    for side, scale in enumerate(scales):
        if scale:
            types.append(
                typ(scale, ((m, n)[side], k // (16 if scale == "f8E4M3FN" else 32)))
            )
            names.append(("%sa", "%sb")[side])
    enums = {value: key for key, value in FORMATS.items()}
    attrs = (
        f"a_elem_type={enums[lhs]}:i32,b_elem_type={enums[rhs]}:i32,fastMath=false,"
        f"lhs_k_pack={str(packs[0]).lower()},rhs_k_pack={str(packs[1]).lower()},"
        f"operandSegmentSizes=array<i32:1,1,1,{int(bool(scales[0]))},{int(bool(scales[1]))}>"
    )
    arguments = ",".join(
        f"{name}:{typ}" for name, typ in zip(names, types, strict=True)
    )
    return f"""module {{func.func @f({arguments})->{types[2]} {{
      %r="tt.dot_scaled"({",".join(names)}) {{{attrs}}}:({",".join(types)})->{types[2]} loc("mx-source")
      return %r:{types[2]}
    }}}}"""


def arrays(lhs, rhs, packs, scales, batch=False, *, m=2, n=4, k=64):
    prefix = (2,) if batch else ()
    data = []
    for side, fmt in enumerate((lhs, rhs)):
        shape = list((*prefix, *((m, k) if side == 0 else (k, n))))
        if fmt == "e2m1":
            shape[-1 if (side == 0) == packs[side] else -2] //= 2
            # Every nibble, including both zeros, appears in each packing axis.
            values = np.resize(np.arange(256, dtype=np.uint8), shape).view(np.int8)
        elif fmt in ("e4m3", "e5m2"):
            values = np.resize(np.arange(256, dtype=np.uint8), shape).view(np.int8)
        elif fmt == "fp16":
            values = np.resize(
                np.asarray([0, -0.0, 1, -2, 65504, 2**-24], np.float16), shape
            )
        else:
            values = np.resize(
                np.asarray([0, 0x8000, 0x3F80, 0xC000, 0x7F7F, 1], np.uint16), shape
            )
        data.append(values)
    initial = np.resize(
        np.asarray([0, -0.0, 2**24, -(2**24), 3, -5], np.float32), (*prefix, m, n)
    )
    scale_arrays = []
    for side, fmt in enumerate(scales):
        if fmt is None:
            scale_arrays.append(None)
            continue
        codes = (
            [0, 254, 127, 128, 126, 255]
            if fmt == "i8"
            else [0, 0x80, 0x38, 0x7E, 1, 0x7F]
        )
        shape = (*prefix, (m, n)[side], k // (32 if fmt == "i8" else 16))
        scale_arrays.append(
            np.resize(np.asarray(codes, np.uint8), shape).view(np.int8)
            if fmt == "i8"
            else np.resize(np.asarray(codes, np.uint8), shape)
        )
    return [
        *data,
        initial,
        *(value for value in scale_arrays if value is not None),
    ], scale_arrays


@pytest.mark.parametrize(
    "packs", [(True, True), (False, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "scales",
    [(None, None), ("i8", None), (None, "i8"), ("i8", "i8"), ("f8E4M3FN", "i8")],
)
@pytest.mark.parametrize("batch", [False, True])
def test_geometry_and_transactional_standard_expansion(packs, scales, batch):
    module = parse_shared(source(packs=packs, scales=scales, batch=batch))
    report = expand_scaled_dots(module, lambda loc: ["mx-source"])[0]
    module.verify()
    assert report["profile"] == PROFILE
    assert report["logical_shapes"] == (
        [[2, 2, 64], [2, 64, 4]] if batch else [[2, 64], [64, 4]]
    )
    assert not any(opcode(op).startswith("tt.") for op in module.walk())
    assert all(opcode(op) not in ("func.call", "gpu.launch") for op in module.walk())
    assert any(
        op.name == ("linalg.generic" if batch else "linalg.matmul")
        for op in module.walk()
    )


@pytest.mark.parametrize(
    "replacement,match",
    [
        ("a_elem_type=2", "reserved FP6"),
        ("b_elem_type=3", "reserved FP6"),
        ("a_elem_type=9", "unknown pinned"),
        ("a_elem_type=5", "disagrees"),
        ("operandSegmentSizes=array<i32:1,1,1,0,0>", "segments"),
        ("fastMath=0:i32", "typed fastMath"),
    ],
)
def test_unknown_or_malformed_contract_is_located_and_does_not_edit(replacement, match):
    key = replacement.split("=")[0]
    original = {
        "a_elem_type": "a_elem_type=4",
        "b_elem_type": "b_elem_type=4",
        "operandSegmentSizes": "operandSegmentSizes=array<i32:1,1,1,1,1>",
        "fastMath": "fastMath=false",
    }[key]
    program = source().replace(original, replacement)
    module = parse_shared(program)
    before = text(module)
    with pytest.raises(ConversionError, match=match) as error:
        expand_scaled_dots(module, lambda loc: ["mx-source"])
    assert "mx-source" in str(error.value)
    assert text(module) == before


def test_later_invalid_candidate_does_not_partially_expand_earlier_dot():
    module = parse_shared(source())
    other = parse_shared(source().replace("a_elem_type=4", "a_elem_type=2"))
    function = next(op for op in other.ops if op.name == "func.func")
    function.properties["sym_name"] = StringAttr("invalid")
    function.detach()
    module.body.block.add_op(function)
    before = text(module)
    with pytest.raises(ConversionError, match="reserved FP6"):
        expand_scaled_dots(module, lambda loc: ["mx-source"])
    assert text(module) == before


@pytest.mark.skipif(not NATIVE, reason="existing native SDK required")
@pytest.mark.parametrize(
    "dtype,carrier,word",
    [("f16", np.float16, 3), ("bf16", np.uint16, 0x4040), ("f64", np.float64, 3)],
)
def test_declared_accumulator_result_format_is_retained(tmp_path, dtype, carrier, word):
    from t2m.float_casts import expand_float_casts
    from t2m.native import execute_native

    module = parse_shared(source(scales=(None, None)).replace("xf32", "x" + dtype))
    expand_scaled_dots(module, lambda loc: ["mx-source"])
    expand_float_casts(module, lambda loc: ["mx-source"])
    data, _ = arrays("e2m1", "e2m1", (True, True), (None, None))
    data[0].fill(0x22)
    data[1].fill(0x22)
    data[2] = np.full((2, 4), word, carrier)
    output, _ = execute_native(text(module), data, tmp_path / "native", python=NATIVE)
    expected = (
        (np.full((2, 4), 67, np.float32).view(np.uint32) >> 16).astype(np.uint16)
        if dtype == "bf16"
        else np.full((2, 4), 67, carrier)
    )
    np.testing.assert_array_equal(output[0], expected)


@pytest.mark.skipif(not NATIVE, reason="existing native SDK required")
def test_all_e8m0_encodings_and_opposite_extremes(tmp_path):
    from t2m.native import execute_native

    module = parse_shared(source(m=256, n=1, k=32))
    expand_scaled_dots(module, lambda loc: ["mx-source"])
    a = np.full((256, 16), 0x22, np.int8)
    b = np.full((16, 1), 0x22, np.int8)
    initial = np.zeros((256, 1), np.float32)
    sa = np.arange(256, dtype=np.uint8).reshape(256, 1).view(np.int8)
    sb = np.full((1, 1), 127, np.int8)
    expected = reference(a, b, sa, sb, initial)
    actual, _ = execute_native(
        text(module), [a, b, initial, sa, sb], tmp_path / "all-scales", python=NATIVE
    )
    assert expected[0, 0] == np.float32(2**-122)  # E8M0 zero is a finite scale.
    np.testing.assert_array_equal(np.isnan(actual[0]), np.isnan(expected))
    np.testing.assert_array_equal(
        actual[0].view(np.uint32)[:-1], expected.view(np.uint32)[:-1]
    )
    for index, (left, right) in enumerate([(254, 0), (0, 254)]):
        sa.fill(left if left < 128 else left - 256)
        sb.fill(right if right < 128 else right - 256)
        actual, _ = execute_native(
            text(module),
            [a, b, initial, sa, sb],
            tmp_path / f"opposite{index}",
            python=NATIVE,
        )
        # Premature f16/BF16/f32 scale materialization cannot satisfy this case.
        np.testing.assert_array_equal(actual[0], np.full_like(initial, 32.0))


NATIVE_CASES = [
    ("e2m1", "e2m1", (a, b), ("i8", "i8"), batch)
    for a, b, batch in [
        (True, True, False),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ]
] + [
    (lhs, rhs, (True, True), scales, batch)
    for lhs, rhs, scales, batch in [
        ("e4m3", "e5m2", ("i8", "i8"), False),
        ("e5m2", "e4m3", ("f8E4M3FN", "i8"), True),
        ("bf16", "fp16", ("i8", "i8"), False),
        ("fp16", "bf16", (None, "i8"), True),
        ("e2m1", "e5m2", ("i8", None), False),
        ("e4m3", "e2m1", (None, None), False),
    ]
]


@pytest.mark.skipif(not NATIVE, reason="existing native SDK required")
@pytest.mark.parametrize("lhs,rhs,packs,scales,batch", NATIVE_CASES)
def test_native_scaling_packing_nonfinites_and_original_c(
    tmp_path, lhs, rhs, packs, scales, batch
):
    from t2m.native import execute_native

    module = parse_shared(source(lhs, rhs, packs, scales, batch))
    expand_scaled_dots(module, lambda loc: ["mx-source"])
    module.verify()
    inputs, scale_arrays = arrays(lhs, rhs, packs, scales, batch)
    expected = reference(
        *inputs[:2], *scale_arrays, inputs[2], lhs, rhs, *packs, scale_formats=scales
    )
    output, _ = execute_native(text(module), inputs, tmp_path / "native", python=NATIVE)
    actual = output[0]
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(
        actual.view(np.uint32)[~np.isnan(expected)],
        expected.view(np.uint32)[~np.isnan(expected)],
    )


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize(
    "profile,packs,scales,lhs,rhs",
    [
        ("tensor", (False, False), ("i8", "i8"), "e2m1", "e2m1"),
        ("buffer", (True, False), (None, "i8"), "e4m3", "e2m1"),
        ("guarded-buffer", (False, True), ("i8", None), "e2m1", "e5m2"),
        ("tensor", (True, True), ("f8E4M3FN", "f8E4M3FN"), "bf16", "fp16"),
    ],
)
def test_real_source_merlin_package(tmp_path, profile, packs, scales, lhs, rhs):
    from test_mixed_effects import qualify
    from t2m.examples.scaled_dot_kernels import scaled_dot_tile
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    m, n, k = 16, 16, 64
    params = dict(
        M=m,
        N=n,
        K=k,
        A_FORMAT=lhs,
        B_FORMAT=rhs,
        LHS_K_PACK=packs[0],
        RHS_K_PACK=packs[1],
        HAS_A_SCALE=bool(scales[0]),
        HAS_B_SCALE=bool(scales[1]),
        GROUP=16 if scales[0] == "f8E4M3FN" else 32,
    )
    triton_types = {
        "e2m1": "u8",
        "e4m3": "u8",
        "e5m2": "u8",
        "bf16": "bf16",
        "fp16": "fp16",
    }
    signature = dict(
        zip(
            scaled_dot_tile.arg_names[:6],
            [
                "*" + triton_types[lhs],
                "*" + triton_types[rhs],
                *("*fp8e4nv" if scale == "f8E4M3FN" else "*u8" for scale in scales),
                "*fp32",
                "*fp32",
            ],
            strict=True,
        )
    )
    signature.update({name: "constexpr" for name in params})
    path = tmp_path / "source.ttir"
    path.write_text(
        capture_ttir(
            scaled_dot_tile, signature, params, source_target=("hip", "gfx942", 64)
        )
    )
    # Unused scale pointers stay in the actual source ABI; they are real readonly
    # arguments rather than silently deleted/reordered pointer slots.
    inputs, scale_arrays = arrays(lhs, rhs, packs, scales, m=m, n=n, k=k)
    storage_scales = [
        value if value is not None else np.zeros((1,), np.int8)
        for value in scale_arrays
    ]
    abi_inputs = [*inputs[:2], *storage_scales, inputs[2]]
    dtypes = [STORAGE[lhs], STORAGE[rhs], *(scale or "i8" for scale in scales), "f32"]
    abi = TensorABI(
        "scaled_dot_tile",
        tuple(
            TensorBuffer(name, data.shape, dtype=dtype)
            for name, data, dtype in zip(
                ("a", "b", "sa", "sb", "initial"), abi_inputs, dtypes, strict=True
            )
        )
        + (TensorBuffer("out", (m, n), "write"),),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    expected = reference(
        *inputs[:2], *scale_arrays, inputs[2], lhs, rhs, *packs, scale_formats=scales
    )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    capture, _ = qualify(result, tmp_path, abi_inputs, policy, [expected])
    assert "prov.scaled_dot_contracts" in result.mlir_text
    assert any(
        record.get("opcode") == "tt.dot_scaled"
        for record in result.trace["source_operations"]
    )
    assert any(
        PROFILE in difference
        for difference in json.loads((capture / "compatibility.json").read_text())[
            "differences"
        ]
    )

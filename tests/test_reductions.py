"""General reductions retain real initialization, tuple order and scalar control."""

import math
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.reductions import expand_reductions
from t2m.shared_tensor import _source_ids, parse_shared

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


def operation(shape, axis, body, *, elements=("i64",), returned=None, name="out"):
    types = [
        "tensor<" + "x".join(map(str, shape)) + "x" + element + ">"
        for element in elements
    ]
    output_shape = [n for index, n in enumerate(shape) if index != axis]
    outputs = [
        "tensor<" + "x".join(map(str, output_shape)) + "x" + element + ">"
        if output_shape
        else element
        for element in elements
    ]
    count = len(elements)
    args = ",".join(
        f"%{label}{i}:{element}"
        for label in ("a", "b")
        for i, element in enumerate(elements)
    )
    values = ",".join(returned or [f"%v{i}" for i in range(count)])
    results = ",".join(f"%{name}{i}" for i in range(count))
    source = f"""{results}="tt.reduce"({",".join(f"%x{i}" for i in range(count))}) <{{axis={axis}:i32}}> ({{
      ^bb0({args}):
        {body}
        "tt.reduce.return"({values}):({",".join(elements)})->()
      }}):({",".join(types)})->({",".join(outputs)}) loc("ttir.op.1100")"""
    return source, types, outputs


def module_for(shape, axis, body, *, elements=("i64",), returned=None):
    source, types, outputs = operation(
        shape, axis, body, elements=elements, returned=returned
    )
    arguments = ",".join(f"%x{i}:{type_}" for i, type_ in enumerate(types))
    values = ",".join(f"%out{i}" for i in range(len(elements)))
    return parse_shared(f"""module {{func.func @forward({arguments})->({",".join(outputs)}) {{
      {source}
      return {values}:{",".join(outputs)}
    }} }}""")


@pytest.mark.skipif(not NATIVE, reason="existing native compiler required")
@pytest.mark.parametrize(
    "shape,axis", [((2, 4), 0), ((2, 4), 1), ((1, 8, 1), 1), ((4, 1), 1)]
)
def test_and_reduction_never_invents_one_as_a_neutral_value(tmp_path, shape, axis):
    from t2m.native import execute_native

    module = module_for(shape, axis, "%v0=arith.andi %a0,%b0:i64")
    reports = expand_reductions(module, _source_ids)
    module.verify()
    values = np.resize(
        np.asarray([0x7FFFBAFFFF00FFFE, -1, -17, 0x6FFFFBFFFFFF00FE], np.int64),
        math.prod(shape),
    ).reshape(shape)
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    expected = np.bitwise_and.reduce(values, axis=axis)
    assert (
        actual[0].shape == expected.shape and actual[0].tobytes() == expected.tobytes()
    )
    assert (
        reports[0]["initialization"]
        == "first-original-element; no-invented-neutral-value"
    )


@pytest.mark.skipif(not NATIVE, reason="existing native compiler required")
@pytest.mark.parametrize("axis", [0, 1])
def test_coupled_affine_composition_keeps_noncommutative_tuple_order(tmp_path, axis):
    from t2m.native import execute_native

    module = module_for(
        (2, 4),
        axis,
        """%v0=arith.muli %a0,%b0:i64
      %scaled=arith.muli %a1,%b0:i64
      %v1=arith.addi %scaled,%b1:i64""",
        elements=("i64", "i64"),
    )
    expand_reductions(module, _source_ids)
    module.verify()
    first = np.asarray([[3, 7, -1, 17], [19, -7, 13, 5]], np.int64)
    second = np.arange(8, dtype=np.int64).reshape(2, 4) + (1 << 62) + 1
    expected_a, expected_b = [], []
    for a, b in zip(
        np.moveaxis(first, axis, -1).reshape(-1, first.shape[axis]),
        np.moveaxis(second, axis, -1).reshape(-1, second.shape[axis]),
        strict=True,
    ):
        scale, bias = int(a[0]), int(b[0])
        for next_a, next_b in zip(a[1:], b[1:], strict=True):
            scale = scale * int(next_a) & ((1 << 64) - 1)
            bias = (bias * int(next_a) + int(next_b)) & ((1 << 64) - 1)
        expected_a.append(scale)
        expected_b.append(bias)
    expected = [
        np.asarray(values, np.uint64)
        .view(np.int64)
        .reshape(np.delete(first.shape, axis))
        for values in (expected_a, expected_b)
    ]
    actual, _ = execute_native(
        text(module), [first, second], tmp_path / "native", python=NATIVE
    )
    assert [value.tobytes() for value in actual] == [
        value.tobytes() for value in expected
    ]


@pytest.mark.skipif(not NATIVE, reason="existing native compiler required")
def test_scalar_while_gcd_combiner_retains_actual_state_and_unsigned_words(tmp_path):
    from t2m.native import execute_native

    module = module_for(
        (2, 4),
        1,
        """%zero=arith.constant 0:i64
      %left,%right=scf.while (%l=%a0,%r=%b0):(i64,i64)->(i64,i64) {
        %continue=arith.cmpi ne,%r,%zero:i64
        scf.condition(%continue) %l,%r:i64,i64
      } do {
      ^bb0(%l:i64,%r:i64):
        %next=arith.remui %l,%r:i64
        scf.yield %r,%next:i64,i64
      }""",
        returned=["%left"],
    )
    reports = expand_reductions(module, _source_ids)
    assert reports[0]["combiner_control_flow"] == {"scf.while": 1}
    module.verify()
    values = np.asarray(
        [[0, 21, 35, 49], [0xFFFFFFFFFFFFFFFE, 0x8000000000000002, 14, 42]], np.uint64
    )
    expected = np.asarray(
        [math.gcd(*(int(value) for value in row)) for row in values], np.uint64
    )
    actual, _ = execute_native(
        text(module), [values.view(np.int64)], tmp_path / "native", python=NATIVE
    )
    assert actual[0].tobytes() == expected.tobytes()


@pytest.mark.skipif(not NATIVE, reason="existing native compiler required")
@pytest.mark.parametrize(
    "dtype,word_type,word",
    [
        ("f16", "uint16", 0xFC01),
        ("bf16", "uint16", 0xFF81),
        ("f32", "uint32", 0xFF800001),
        ("f64", "uint64", 0xFFF0000000000001),
    ],
)
def test_singleton_reduction_preserves_original_signaling_nan_words(
    tmp_path, dtype, word_type, word
):
    from t2m.native import execute_native
    from t2m.tensor_storage import DTYPES

    module = module_for((4, 1), 1, f"%v0=arith.addf %a0,%b0:{dtype}", elements=(dtype,))
    expand_reductions(module, _source_ids)
    module.verify()
    words = np.asarray(
        [word, 0, 1, 1 << (np.dtype(word_type).itemsize * 8 - 1)], dtype=word_type
    ).reshape(4, 1)
    values = words if dtype == "bf16" else words.view(DTYPES[dtype])
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert actual[0].shape == (4,) and actual[0].tobytes() == words.tobytes()


@pytest.mark.parametrize("shape", [(4, 1), (2, 4)])
@pytest.mark.parametrize(
    "effect",
    [
        '"tt.print"() <{prefix="bad",hex=false}>:()->()',
        "func.call @unknown():()->()",
        "%read=memref.load %pointer[%zero]:memref<1xi64>",
        '%unknown="arith.unknown"(%a0):(i64)->i64',
    ],
)
def test_all_combiner_candidates_validate_before_any_reduction_is_replaced(
    shape, effect
):
    good, types, outputs = operation(
        shape, 1, "%v0=arith.addi %a0,%b0:i64", name="good"
    )
    bad, _, _ = operation(shape, 1, effect, returned=["%a0"])
    module = parse_shared(f"""module {{func.func private @unknown()
      func.func @forward(%x0:{types[0]},%pointer:memref<1xi64>)->{outputs[0]} {{
      %zero=arith.constant 0:index
      {good}
      {bad}
      return %out0:{outputs[0]}
    }} }}""")
    before = text(module)
    with pytest.raises(ConversionError, match="reduction combiner"):
        expand_reductions(module, _source_ids)
    assert text(module) == before


@pytest.mark.skipif(not NATIVE, reason="existing native compiler required")
@pytest.mark.parametrize(
    "dtype,bits,nan",
    [
        ("f16", 16, 0x7C01),
        ("bf16", 16, 0x7F81),
        ("f32", 32, 0x7F800001),
        ("f64", 64, 0x7FF0000000000001),
    ],
)
def test_all_nan_maxnum_has_no_synthetic_negative_infinity_seed(
    tmp_path, dtype, bits, nan
):
    from t2m.native import execute_native
    from t2m.tensor_storage import DTYPES

    module = module_for(
        (2, 4), 1, f"%v0=arith.maxnumf %a0,%b0:{dtype}", elements=(dtype,)
    )
    expand_reductions(module, _source_ids)
    module.verify()
    words = np.asarray(
        [[nan, nan + 1, nan | (1 << (bits - 1)), nan], [1 << (bits - 1)] * 4],
        dtype=f"uint{bits}",
    )
    values = words if dtype == "bf16" else words.view(DTYPES[dtype])
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    observed = actual[0].view(f"uint{bits}")
    fraction = {"f16": 10, "bf16": 7, "f32": 23, "f64": 52}[dtype]
    exponent = nan & ~((1 << fraction) - 1)
    assert int(observed[0]) & exponent == exponent
    assert int(observed[0]) & ((1 << fraction) - 1)
    assert int(observed[1]) == 1 << (bits - 1)

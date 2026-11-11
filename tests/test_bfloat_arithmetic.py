"""Native BF16 arithmetic is checked against exact binary-rational rounding."""

import os

import numpy as np
import pytest

from t2m.examples.bfloat_arithmetic_reference import arithmetic
from t2m.native import execute_native

pytestmark = pytest.mark.skipif(
    not (os.environ.get("T2M_NATIVE_PYTHON") and os.environ.get("T2M_MERLIN_ROOT")),
    reason="existing native compiler and explicit Merlin arithmetic support required",
)


def source(n, *, chain=False, multiply_chain=False):
    type_ = f"tensor<{n}xbf16>"
    expressions = (
        "%s=arith.addf %x,%y:bf16 %d=arith.subf %s,%y:bf16 %p=arith.mulf %s,%x:bf16"
        if chain
        else "%s=arith.addf %x,%y:bf16 %d=arith.subf %x,%y:bf16 %p=arith.mulf %x,%y:bf16"
    )
    if multiply_chain:
        expressions = (
            "%p=arith.mulf %x,%x:bf16 %s=arith.subf %p,%y:bf16 %d=arith.addf %s,%x:bf16"
        )
    return f"""#id=affine_map<(i)->(i)>
    module {{func.func @forward(%a:{type_},%b:{type_})->({type_},{type_},{type_}) {{
      %e0=tensor.empty():{type_}
      %e1=tensor.empty():{type_}
      %e2=tensor.empty():{type_}
      %r:3=linalg.generic {{indexing_maps=[#id,#id,#id,#id,#id],iterator_types=["parallel"]}}
        ins(%a,%b:{type_},{type_}) outs(%e0,%e1,%e2:{type_},{type_},{type_}) {{
        ^bb0(%x:bf16,%y:bf16,%o0:bf16,%o1:bf16,%o2:bf16):
          {expressions}
          linalg.yield %s,%d,%p:bf16,bf16,bf16
      }}->({type_},{type_},{type_})
      return %r#0,%r#1,%r#2:{type_},{type_},{type_}
    }}}}"""


def compare(actual, expected):
    nan = np.asarray([word is None for word in expected])
    words = np.asarray([0 if word is None else word for word in expected], np.uint16)
    assert np.all((actual[nan] & 0x7F80) == 0x7F80) and np.all((actual[nan] & 127) != 0)
    np.testing.assert_array_equal(actual[~nan], words[~nan])


def test_all_bfloat_words_against_sixteen_exact_arithmetic_partners(tmp_path):
    partners = np.asarray(
        [
            0,
            0x8000,
            1,
            0x80,
            0xD9,
            0x3B80,
            0xBB80,
            0x3F81,
            0xBF81,
            0x3F80,
            0xBF80,
            0x7F7F,
            0xFF7F,
            0x7F80,
            0xFF80,
            0x7F81,
        ],
        np.uint16,
    )
    left = np.tile(np.arange(65536, dtype=np.uint16), len(partners))
    right = np.repeat(partners, 65536)
    outputs, _ = execute_native(
        source(left.size),
        [left, right],
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    for kind, actual in zip(("add", "sub", "mul"), outputs, strict=True):
        expected = [
            arithmetic(int(a), int(b), kind) for a, b in zip(left, right, strict=True)
        ]
        compare(actual, expected)


def test_each_bfloat_source_operation_keeps_its_rounding_boundary(tmp_path):
    left = np.asarray([0x3F80, 0x3F81, 0xBF80, 0xBF81], np.uint16)
    right = np.asarray([0x4380, 0xC380, 0x4380, 0xC380], np.uint16)
    outputs, _ = execute_native(
        source(left.size, chain=True),
        [left, right],
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    sums = [arithmetic(int(a), int(b), "add") for a, b in zip(left, right, strict=True)]
    compare(outputs[0], sums)
    compare(
        outputs[1],
        [arithmetic(a, int(b), "sub") for a, b in zip(sums, right, strict=True)],
    )
    compare(
        outputs[2],
        [arithmetic(a, int(b), "mul") for a, b in zip(sums, left, strict=True)],
    )


def test_bfloat_product_rounds_before_a_following_subtraction(tmp_path):
    left = np.asarray([0x3F81, 0xBF81], np.uint16)
    right = np.asarray([0x3F82, 0x3F82], np.uint16)
    outputs, _ = execute_native(
        source(left.size, multiply_chain=True),
        [left, right],
        tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    products = [arithmetic(int(value), int(value), "mul") for value in left]
    differences = [
        arithmetic(value, int(other), "sub")
        for value, other in zip(products, right, strict=True)
    ]
    compare(outputs[0], differences)
    compare(
        outputs[1],
        [
            arithmetic(value, int(other), "add")
            for value, other in zip(differences, left, strict=True)
        ],
    )
    compare(outputs[2], products)
    assert outputs[0].tolist() == [0, 0]

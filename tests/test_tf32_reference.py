"""Boundary values for the independent E8M10 rational rounding oracle."""

import pytest

from t2m.examples.dot_precision_reference import tf32_word


@pytest.mark.parametrize("word,rtz,rna", [
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000000, 0x80000000, 0x80000000),
    (0x00000001, 0x00000000, 0x00000000),
    (0x80000001, 0x80000000, 0x80000000),
    (0x00001000, 0x00000000, 0x00002000),
    (0x80001000, 0x80000000, 0x80002000),
    (0x007FF000, 0x007FE000, 0x00800000),
    (0x3F800FFF, 0x3F800000, 0x3F800000),
    (0x3F801000, 0x3F800000, 0x3F802000),
    (0xBF801000, 0xBF800000, 0xBF802000),
    (0x7F7FFFFF, 0x7F7FE000, 0x7F800000),
    (0x7F800000, 0x7F800000, 0x7F800000),
    (0xFF800000, 0xFF800000, 0xFF800000),
    (0x7F800001, 0x7FC00000, 0x7FC00000),
    (0xFF800001, 0x7FC00000, 0x7FC00000),
])
def test_tf32_rational_boundaries(word, rtz, rna):
    assert tf32_word(word) == rtz
    assert tf32_word(word, "rna") == rna

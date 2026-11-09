import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block

from t2m.diagnostics import ConversionError
from t2m.shared_indexing import Linear
from t2m.tail_masks import TailExtent, recover_extent


def mask_case(limit=53, tile=16, *, wrong_start=False):
    block = Block(arg_types=[IndexType()])
    facts = {block.args[0]: Linear(0, (1, 0, 0))}

    def constant(n):
        op = arith.ConstantOp.from_int_and_width(n, IndexType())
        block.add_op(op)
        return op.result

    def binary(cls, a, b):
        op = cls(a, b)
        block.add_op(op)
        return op.result

    cap, end = constant(tile), constant(limit)
    start = binary(arith.MuliOp, block.args[0], cap)
    next_ = binary(arith.AddiOp, start, cap)
    clipped = binary(arith.MinSIOp, next_, end)
    nonnegative = binary(arith.MaxSIOp, clipped, start)
    extent = binary(arith.SubiOp, nonnegative, constant(1) if wrong_start else start)
    extent = binary(arith.MinSIOp, cap, extent)
    return extent, facts, Linear(0, (tile, 0, 0))


@pytest.mark.parametrize("programs,limit", [(4, 53), (6, 53), (2**25, 53), (4, 64), (4, 9)])
def test_upstream_tail_normal_form_is_proved_without_grid_enumeration(programs, limit):
    value, facts, offset = mask_case(limit)
    extent = recover_extent(value, 16, [offset], facts, (programs,))
    if limit == 64:
        assert extent == 16  # Bounds prove the mask is full in every program.
        return
    assert isinstance(extent, TailExtent)
    assert extent.fits(offset, limit)
    assert not extent.fits(offset + Linear(1), limit)
    assert not extent.fits(offset, limit - 1)
    assert extent.as_dict()["limit"] == limit


def test_changed_subtraction_is_not_accepted_as_a_tail():
    value, facts, offset = mask_case(wrong_start=True)
    with pytest.raises(ConversionError, match="clipped affine prefix"):
        recover_extent(value, 16, [offset], facts, (4,))


def test_a_mask_for_another_address_is_not_a_bounds_proof():
    value, facts, offset = mask_case()
    # A shifted mask remains a real tail, but it must be tied to the actual
    # address and limit before an access is admitted by the tensor adapter.
    extent = recover_extent(value, 16, [offset + Linear(8)], facts, (4,))
    assert extent.limit == 61
    assert not extent.fits(offset + Linear(8), 53)


def test_tail_proof_checks_intermediate_index_overflow():
    value, facts, offset = mask_case()
    with pytest.raises(ConversionError, match="overflow"):
        recover_extent(value, 16, [offset], facts, (2**60,))

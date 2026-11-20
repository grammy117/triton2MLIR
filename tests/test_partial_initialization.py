"""Independent finite-set oracle for source-derived partial initialization."""

from itertools import product

from t2m.initialization import DefiniteWrites
from t2m.partial_initialization import ReadFacts, WriteImage, store_image
from t2m.shared_indexing import Linear
from t2m.shared_tensor import TensorABI, TensorBuffer
from t2m.symbolic_layout import Polynomial


def domains(counts):
    return [(Polynomial.constant(0) if count <= 1 else Polynomial.make([((("l", i, 0, count - 1),), 1)]),
             Polynomial.constant(count)) for i, count in enumerate(counts)]


def test_every_admitted_small_strided_image_member_was_actually_written():
    proof = ReadFacts((1,))
    for counts in product(range(4), repeat=2):
        loops = domains(counts)
        for strides in product(range(-3, 4), repeat=2):
            index = Polynomial.constant(5)
            for (induction, _), stride in zip(loops, strides, strict=True):
                index += induction.scale(stride)
            image = store_image(loops, index, (1,))
            assert image is not None
            written = {5 + sum(i * stride for i, stride in zip(point, strides, strict=True))
                       for point in product(*(range(count) for count in counts))}
            for query in range(-7, 18):
                if proof.contains(image, Polynomial.constant(query)):
                    assert query in written, (counts, strides, query)


def test_single_axis_images_include_every_real_element_with_signed_and_zero_strides():
    proof = ReadFacts((1,))
    for count, stride in product(range(5), (-4, -1, 0, 1, 7)):
        loop = domains((count,))
        image = store_image(loop, loop[0][0].scale(stride) + Polynomial.constant(3), (1,))
        assert image is not None
        for i in range(count):
            assert proof.contains(image, Polynomial.constant(3 + i * stride))


def test_local_membership_uses_real_loop_and_predicate_relations_without_global_credit():
    n = Polynomial.make([((("v", "n", 0, 9),), 1)])
    j = Polynomial.make([((("l", 1, 0, 8),), 1)])
    image = WriteImage(Polynomial.constant(0), ((n, Polynomial.constant(1)),))
    proof = ReadFacts((1,))
    assert not proof.contains(image, Polynomial.constant(0))  # The write loop may be empty.
    proof.positive(n)
    assert proof.contains(image, n + Polynomial.constant(-1))
    assert not proof.contains(image, n)
    assert not proof.contains(image, j)  # Independent interval maxima supply no j<n.
    proof.nonnegative.add(n + j.scale(-1) + Polynomial.constant(-1))
    assert proof.contains(image, j)


def test_cross_products_and_dependent_unit_digits_do_not_form_rectangular_images():
    loops = domains((2, 3))
    assert store_image(loops, loops[0][0] * loops[1][0], (1,)) is None
    dependent = [(loops[0][0], loops[0][1]), (Polynomial.constant(0), loops[0][0])]
    assert store_image(dependent, loops[0][0], (1,)) is None


def test_partial_facts_intersect_at_branches_expire_at_loop_edges_and_do_not_complete_results():
    output = TensorBuffer("out", (7,), "write", "i64")
    state = DefiniteWrites(TensorABI("kernel", (output,)))
    induction = Polynomial.make([((("l", 7, 0, 4),), 1)])
    image = WriteImage(induction, ())
    before = state.snapshot()
    state.store_image(output, image)
    written = state.snapshot()
    assert not state.complete(output)
    state.merge_branches(written, before)
    assert state.images(output) == []
    state.merge_branches(written, written)
    assert state.images(output) == [image]
    state.end_loop(7)
    assert state.images(output) == [] and not state.complete(output)


def test_optional_image_budget_keeps_existing_complete_interval_coverage():
    output = TensorBuffer("out", (129,), "write", "i64")
    state = DefiniteWrites(TensorABI("kernel", (output,)))
    for index in range(129):
        state.store_image(output, WriteImage(Polynomial.constant(index), ()))
        state.store(output, (Linear(index),), (1,))
    assert len(state.images(output)) == 128
    assert state.finish()["out"]["coverage"] == "serial-source-constant-interval-union"


def test_collapsed_unit_axis_still_needs_an_actual_nonempty_write_premise():
    count = Polynomial.make([((("v", "m", 0, 3),), 1)])
    unit = Polynomial.make([((("v", "n", 0, 1),), 1)])
    induction = Polynomial.make([((("l", 4, 0, 2),), 1)])
    image = store_image(((induction, count), (Polynomial.constant(0), unit)), induction * unit, (1,))
    assert image is not None
    proof = ReadFacts((1,))
    proof.positive(count)
    query = (count + Polynomial.constant(-1)) * unit
    assert not proof.contains(image, query)
    proof.positive(unit)
    assert proof.contains(image, query)

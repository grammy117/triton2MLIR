"""Executable byte-order-neutral word atomics, before TTIR event admission."""

import os

import numpy as np
import pytest
from xdsl.dialects import arith, bufferization, builtin, func, memref, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from t2m.byte_atomic_words import encode, event, probe
from t2m.ir import text

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(
    not (NATIVE and os.environ.get("T2M_MERLIN_ROOT")),
    reason="existing native MLIR and explicit Merlin support required",
)


def module(width, operation, order, *, byte_count=None, narrow=True):
    word = builtin.IntegerType(width)
    count, size = 8, width // 8
    raw = builtin.TensorType(builtin.i8, [5 * size + 1 if byte_count is None else byte_count])
    words = builtin.TensorType(word, [count])
    inputs = [
        raw,
        builtin.TensorType(builtin.i64, [count]),
        words,
        builtin.TensorType(builtin.i1, [count]),
        *([words] if operation == "cas" else []),
    ]
    block = Block(arg_types=inputs)

    def append(op):
        block.add_op(op)
        return op

    incoming = append(
        bufferization.ToBufferOp.build(
            operands=[block.args[0]],
            result_types=[builtin.MemRefType(builtin.i8, raw.get_shape())],
            properties={"read_only": builtin.UnitAttr()},
        )
    ).memref
    storage = append(
        memref.AllocOp.get(builtin.i8, shape=raw.get_shape(), alignment=8)
    ).memref
    append(memref.CopyOp(incoming, storage))
    little = probe(word, append)

    def atomic():
        return event(
            storage,
            block.args[1],
            block.args[2],
            little,
            append,
            operation=operation,
            byte_order=order,
            mask=block.args[3],
            desired=block.args[4] if operation == "cas" else None,
        )

    first = atomic()
    # Ordinary narrow storage must remain coherent with both aliasing word events.
    if narrow:
        one = append(arith.ConstantOp.from_int_and_width(1, builtin.IndexType())).result
        byte = append(arith.ConstantOp.from_int_and_width(0x5A, builtin.i8)).result
        append(memref.StoreOp.get(byte, storage, [one]))
    second = atomic()
    empty = append(tensor.EmptyOp([], raw)).tensor
    body = Block(arg_types=[builtin.i8])
    index = IndexOp(0)
    load = memref.LoadOp.get(storage, [index.result])
    body.add_ops([index, load, YieldOp(load.res)])
    copied = GenericOp(
        [],
        [empty],
        Region(body),
        [builtin.AffineMapAttr(AffineMap.identity(1))],
        [IteratorTypeAttr.parallel()],
        [raw],
    )
    append(copied)
    append(func.ReturnOp(copied.results[0], first, second))
    result = builtin.ModuleOp(
        [func.FuncOp("byte_atomic_words", (inputs, [raw, words, words]), Region(block))]
    )
    result.verify()
    return result


def arguments(width, operation, order, *, invalid=None):
    size, limit = width // 8, (1 << width) - 1
    unsigned, signed = getattr(np, f"uint{width}"), getattr(np, f"int{width}")
    data = ((np.arange(5 * size + 1, dtype=np.uint32) * 73 + 211) & 255).astype(
        np.uint8
    )
    offsets = np.asarray(
        [0, 0, size, 2 * size, 2 * size, 4 * size, -(2**63), 2**63 - 1], np.int64
    )
    values = np.asarray(
        [1, limit, 1 << (width - 1), 3, 7, 1 << (width - 2), 0, 0], unsigned
    ).view(signed)
    mask = np.asarray([True] * 6 + [False] * 2)
    desired = (np.arange(8, dtype=unsigned) + 17).view(signed)
    if invalid == "unaligned":
        offsets[0] = 1
    if invalid == "tail":
        offsets[0] = 5 * size
    if invalid == "negative":
        offsets[0] = -size
    state = data.copy()

    def sign(n):
        return n if n < 1 << (width - 1) else n - (1 << width)

    def once():
        old = np.zeros(8, unsigned)
        for lane, at in enumerate(offsets):
            if not mask[lane] or invalid:
                continue
            word = int.from_bytes(state[at : at + size].tobytes(), order)
            old[lane] = word
            value = int(values[lane]) & limit
            if operation == "cas":
                changed = int(desired[lane]) & limit if word == value else word
            elif operation == "exch":
                changed = value
            elif operation == "add":
                changed = word + value
            elif operation == "and":
                changed = word & value
            elif operation == "or":
                changed = word | value
            elif operation == "xor":
                changed = word ^ value
            elif operation == "max":
                changed = max(sign(word), sign(value))
            elif operation == "min":
                changed = min(sign(word), sign(value))
            elif operation == "umax":
                changed = max(word, value)
            elif operation == "umin":
                changed = min(word, value)
            else:
                raise AssertionError(operation)
            state[at : at + size] = np.frombuffer(
                (changed & limit).to_bytes(size, order), np.uint8
            )
        return old.view(signed)

    if operation == "cas":
        values[0] = sign(int.from_bytes(data[:size].tobytes(), order))
        values[2] = sign(int.from_bytes(data[size : 2 * size].tobytes(), order))
    first = once()
    state[1] = 0x5A
    second = once()
    inputs = [
        data.view(np.int8),
        offsets,
        values,
        mask,
        *([desired] if operation == "cas" else []),
    ]
    return inputs, [state.view(np.int8), first, second]


@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize(
    "width,operation",
    [
        *(
            (width, op)
            for width in (32, 64)
            for op in ("add", "and", "or", "xor", "max", "min", "umax", "umin", "exch")
        ),
        *((width, "cas") for width in (16, 32, 64)),
    ],
)
def test_indivisible_words_preserve_byte_order_old_values_and_narrow_updates(
    tmp_path, width, operation, order
):
    from t2m.native import execute_native

    program = module(width, operation, order)
    inputs, expected = arguments(width, operation, order)
    outputs, _ = execute_native(
        text(program),
        inputs,
        tmp_path / "native",
        python=NATIVE,
        integer_atomic_regions=True,
    )
    for actual, reference in zip(outputs, expected, strict=True):
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        assert actual.tobytes() == reference.tobytes()


@pytest.mark.skipif(
    not os.environ.get("T2M_MERLIN_ROOT"),
    reason="explicit Merlin compiler support required",
)
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("width,operation", ((16, "cas"), (32, "add"), (64, "xor")))
def test_actual_merlin_library_keeps_byte_aliases_and_original_word_results(
    tmp_path, width, operation, order, monkeypatch
):
    import sys
    from pathlib import Path

    root = Path(os.environ["T2M_MERLIN_ROOT"])
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", NATIVE)
    if os.environ.get("T2M_MERLIN_CLANG"):
        monkeypatch.setenv("MERLIN_CLANG", os.environ["T2M_MERLIN_CLANG"])
    sys.path.insert(0, str(root / "src"))
    from merlin.llvmlower import atomic_integer_regions
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    atomic_integer_regions.ensure_registered()
    program = module(width, operation, order)
    next(iter(program.ops)).properties["sym_name"] = builtin.StringAttr("forward")
    entry = next(iter(program.ops))
    entry.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    entry.arg_attrs = builtin.ArrayAttr(
        builtin.DictionaryAttr(
            {"bufferization.writable": builtin.IntegerAttr(0, builtin.i1)}
        )
        for _ in entry.body.block.args
    )
    inputs, expected = arguments(width, operation, order)
    before = [value.tobytes() for value in inputs]
    work = tmp_path / "merlin"
    llvm = lower_to_llvm_ir(
        text(program),
        workdir=work,
        features=frozenset({atomic_integer_regions.FEATURE}),
        timeout=60,
    )
    assert llvm.count("cmpxchg") == 2 and "atomic" in llvm
    source = work / "model.ll"
    source.write_text(llvm)
    library = build_host_shared(source, work / "model.so")
    model = HostModel.load(str(library.resolve()))
    for _ in range(2):
        outputs = [np.full_like(value, 85) for value in expected]
        model([(value.ctypes.data, value.shape) for value in (*inputs, *outputs)])
        for actual, reference in zip(outputs, expected, strict=True):
            assert actual.tobytes() == reference.tobytes()
        assert before == [value.tobytes() for value in inputs]


@pytest.mark.parametrize(
    "kind,message",
    (
        ("unaligned", "Triton byte atomic address is not word aligned"),
        ("tail", "Triton active memory index outside logical allocation"),
        ("negative", "Triton active memory index outside logical allocation"),
    ),
)
def test_active_word_addresses_refuse_without_reading_partial_or_unaligned_words(
    tmp_path, kind, message
):
    import signal

    from t2m.native import observe_native

    program = module(32, "add", "big")
    inputs, _ = arguments(32, "add", "big", invalid=kind)
    work = tmp_path / "native"
    outcome = observe_native(
        text(program), inputs, work, python=NATIVE, integer_atomic_regions=True
    )
    assert outcome["returncode"] == -signal.SIGABRT
    assert (work / "stdout.bin").read_bytes() == (message + "\n").encode()


def test_general_integer_regions_require_the_explicit_native_feature(tmp_path):
    from t2m.native import execute_native

    program = module(32, "add", "little")
    inputs, _ = arguments(32, "add", "little")
    with pytest.raises(ValueError, match="closed comparison/select/yield"):
        execute_native(text(program), inputs, tmp_path / "native", python=NATIVE)


def test_empty_byte_storage_never_observes_inactive_extreme_addresses(tmp_path):
    from t2m.native import execute_native

    program = module(32, "add", "big", byte_count=0, narrow=False)
    inputs, _ = arguments(32, "add", "big")
    inputs[0] = np.empty(0, np.int8)
    inputs[3].fill(False)
    outputs, _ = execute_native(text(program), inputs, tmp_path / "native", python=NATIVE, integer_atomic_regions=True)
    assert outputs[0].size == 0
    assert all(value.tobytes() == np.zeros(8, np.int32).tobytes() for value in outputs[1:])


@pytest.mark.parametrize("width", (16, 32, 64))
@pytest.mark.parametrize("order", ("little", "big"))
def test_word_codec_covers_both_local_representations_and_complete_i16_domain(
    tmp_path, width, order
):
    from t2m.examples.allocation_word_reference import patterns
    from t2m.native import execute_native

    unsigned, signed = getattr(np, f"uint{width}"), getattr(np, f"int{width}")
    logical = list(range(1 << 16)) if width == 16 else patterns(width)
    raw, flags, expected = [], [], []
    for native_order in ("little", "big"):
        for value in logical:
            raw.append(
                int.from_bytes(int(value).to_bytes(width // 8, order), native_order)
            )
            flags.append(native_order == "little")
            expected.append(value)
    raw = np.asarray(raw, unsigned).view(signed)
    expected = np.asarray(expected, unsigned).view(signed)
    flags = np.asarray(flags, np.bool_)
    word = builtin.IntegerType(width)
    type_ = builtin.TensorType(word, [len(raw)])
    flag_type = builtin.TensorType(builtin.i1, [len(raw)])
    block = Block(arg_types=[type_, flag_type])
    empty = tensor.EmptyOp([], type_)
    body = Block(arg_types=[word, builtin.i1, word])
    converted = encode(body.args[0], body.args[1], order, body.add_op)
    body.add_op(YieldOp(converted))
    generic = GenericOp(
        list(block.args),
        [empty.tensor],
        Region(body),
        [builtin.AffineMapAttr(AffineMap.identity(1))] * 3,
        [IteratorTypeAttr.parallel()],
        [type_],
    )
    block.add_ops([empty, generic, func.ReturnOp(generic.results[0])])
    program = builtin.ModuleOp(
        [func.FuncOp("word_codec", ([type_, flag_type], [type_]), Region(block))]
    )
    program.verify()
    (actual,), _ = execute_native(
        text(program), [raw, flags], tmp_path / "native", python=NATIVE
    )
    assert actual.tobytes() == expected.tobytes()

"""Independent Python integer byte-address/word-event reference."""

import numpy as np

from .allocation_word_reference import patterns


def policy():
    return {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {
            "archive": "reference.npz",
            "comparisons": [{"kind": "words"}, {"kind": "words"}],
        },
    }


def arguments(
    width=32,
    order="little",
    *,
    n=7,
    steps=2,
    capacity=256,
    origin=9,
    stride=24,
    advance=1,
    narrow=True,
    block=4,
    invalid=False,
):
    incoming = ((np.arange(capacity, dtype=np.uint32) * 73 + 211) & 255).astype(
        np.uint8
    )
    size = width // 8
    samples = patterns(width)
    for lane in range(n):
        start = origin + lane * stride
        if 0 <= start and start + size <= capacity:
            data = int(samples[(lane + 6) % len(samples)]).to_bytes(size, order)
            incoming[start : start + size] = np.frombuffer(data, np.uint8)
    expected = incoming.copy()
    result = np.zeros((steps, n), getattr(np, f"uint{width}"))
    for first in range(0, n, block):
        for iteration in range(steps):
            snapshots = []
            for lane in range(first, min(n, first + block)):
                start = origin + lane * stride + iteration * advance * size
                if not 0 <= start or start + size > capacity:
                    if invalid:
                        continue
                    raise ValueError(
                        "independent byte event is outside its declared allocation"
                    )
                word = (
                    int.from_bytes(expected[start : start + size].tobytes(), order) ^ 3
                )
                snapshots.append((lane, start, word))
            for _, start, word in snapshots:
                expected[start : start + size] = np.frombuffer(
                    word.to_bytes(size, order), np.uint8
                )
            if narrow:
                for _, start, _ in snapshots:
                    expected[start + 1] = 0x5A
            for lane, start, _ in snapshots:
                result[iteration, lane] = int.from_bytes(
                    expected[start : start + size].tobytes(), order
                )
    values = [
        incoming.view(np.int8),
        *(
            np.asarray(v, np.int32)
            for v in (n, steps, capacity, origin, stride, advance)
        ),
    ]
    return values, [expected.view(np.int8), result.view(getattr(np, f"int{width}"))]


def fill_arguments(
    width=32, order="little", *, n=7, capacity=256, origin=9, stride=24, fill=None
):
    size = width // 8
    fill = ((1 << (width - 1)) + 1) if fill is None else fill
    padded = ((n + 3) // 4) * 4
    raw = ((np.arange(capacity, dtype=np.uint32) * 73 + 211) & 255).astype(np.uint8)
    unsigned, signed = getattr(np, f"uint{width}"), getattr(np, f"int{width}")
    expected = np.full(padded, fill, unsigned)
    for lane in range(n):
        start = origin + lane * stride
        expected[lane] = int.from_bytes(raw[start : start + size].tobytes(), order)
    values = [
        raw.view(np.int8),
        *(np.asarray(v, np.int32) for v in (n, padded, capacity, origin, stride)),
        np.asarray(fill, unsigned).view(signed),
    ]
    return values, expected.view(signed)


def selected_arguments(
    order="big",
    *,
    n=7,
    steps=2,
    capacity=256,
    origin=9,
    stride=24,
    advance=1,
    choice=True,
):
    a = ((np.arange(capacity, dtype=np.uint32) * 73 + 211) & 255).astype(np.uint8)
    b = a ^ 0xA7
    states = [a.copy(), b.copy()]
    output = np.zeros((steps, n), np.uint32)
    for first in range(0, n, 4):
        for iteration in range(steps):
            snapshots = []
            for lane in range(first, min(n, first + 4)):
                root = 0 if ((lane % 2 == 0) == choice) != bool(iteration % 2) else 1
                address = origin + lane * stride + iteration * advance * 4
                word = (
                    int.from_bytes(states[root][address : address + 4].tobytes(), order)
                    ^ 3
                )
                snapshots.append((lane, root, address, word))
            for lane, root, address, word in snapshots:
                states[root][address : address + 4] = np.frombuffer(
                    word.to_bytes(4, order), np.uint8
                )
                output[iteration, lane] = word
    values = [
        a.view(np.int8),
        b.view(np.int8),
        *(
            np.asarray(v, np.int32)
            for v in (n, steps, capacity, origin, stride, advance)
        ),
        np.asarray(choice, np.bool_),
    ]
    return values, [*(state.view(np.int8) for state in states), output.view(np.int32)]


def bound_arguments(
    order="big", *, descriptor=False, n=7, m=2, steps=2, capacity=256, read=8, write=12
):
    raw = ((np.arange(capacity, dtype=np.uint32) * 73 + 211) & 255).astype(np.uint8)
    expected = raw.copy()
    tiles = (
        [
            [lane * 4 for lane in range(first, min(n, first + 8))]
            for first in range(0, n, 8)
        ]
        if not descriptor
        else [
            [
                i * 96 + j * 4
                for i in range(row, min(m, row + 2))
                for j in range(col, min(n, col + 16))
            ]
            for row in range(0, m, 2)
            for col in range(0, n, 16)
        ]
    )
    for indices in tiles:
        for _ in range(steps):
            for toggle in (1, 2):
                snapshots = [
                    int.from_bytes(
                        expected[read + index : read + index + 4].tobytes(), order
                    )
                    ^ toggle
                    for index in indices
                ]
                for index, word in zip(indices, snapshots, strict=True):
                    expected[write + index : write + index + 4] = np.frombuffer(
                        word.to_bytes(4, order), np.uint8
                    )
    if descriptor:
        values = [
            raw.view(np.int8),
            *(np.asarray(v, np.int32) for v in (m, n, steps, capacity)),
            *(np.asarray(v, np.int64) for v in (read, write, 24, 1, 24, 1)),
        ]
    else:
        values = [
            raw.view(np.int8),
            *(np.asarray(v, np.int32) for v in (n, steps, capacity)),
            np.asarray(read, np.int64),
            np.asarray(write, np.int64),
        ]
    return values, expected.view(np.int8)

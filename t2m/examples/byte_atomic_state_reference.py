"""Independent byte/word oracle for the selected atomic recurrence demo."""

import numpy as np


def reference(a, b, n, steps, choice, *, width=32, cas=False, operation="xor", byte_order="little"):
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("state roots require equal public types and shapes")
    root_size = a.dtype.itemsize
    unsigned = np.dtype(f"u{root_size}")

    def pack(array):
        return bytearray(b"".join(int(value).to_bytes(root_size, byte_order)
                                 for value in array.reshape(-1).view(unsigned)))

    states = [pack(a), pack(b)]
    size, limit = width // 8, (1 << width) - 1
    history = np.empty((steps, n), dtype=getattr(np, f"int{width}"))
    advance = 0
    for iteration in range(steps):
        # Distinct word addresses make lane order immaterial within an event.
        for lane in range(4):
            if not cas and lane >= n:
                continue
            state = states[0 if (lane % 2 == 0) == bool(choice) else 1]
            address = (2 * lane + advance) * size
            if address < 0 or address + size > len(state):
                raise ValueError("source word outside owning state")
            current = int.from_bytes(state[address:address + size], byte_order)
            if cas:
                changed = current ^ 31 if lane % 2 == 0 else current
            else:
                value = lane + iteration + 3
                changed = current ^ value if operation == "xor" else (current + value) & limit
            state[address:address + size] = changed.to_bytes(size, byte_order)
            if lane < n:
                history[iteration, lane] = current if current < 1 << (width - 1) else current - (1 << width)
        states[0][1] = iteration + 0x5A
        states[1][1] = iteration + 0x33
        advance += 1 if iteration % 2 == 0 else -1
        choice = not choice

    def unpack(state):
        values = [int.from_bytes(state[start:start + root_size], byte_order)
                  for start in range(0, len(state), root_size)]
        return np.asarray(values, unsigned).view(a.dtype).reshape(a.shape)

    return [unpack(state) for state in states] + [history]


def cases(width=32, cas=False, operation="xor", byte_order="little", root_dtype="i8"):
    dtype = np.dtype({"i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
                      "f16": "float16", "f32": "float32", "f64": "float64"}.get(root_dtype, root_dtype))
    results = []
    for n, steps, capacity, choice in ((0, 0, 0, False), (0, 3, 97, True), (1, 1, 97, False),
                                       (3, 3, 97, True), (4, 2, 97, False)):
        def data(shift, capacity=capacity):
            bytes_ = ((np.arange(capacity * dtype.itemsize, dtype=np.uint32) * 73 + shift) & 255).astype(np.uint8)
            return bytes_.view(dtype).copy()
        a, b = data(211), data(97)
        arrays = [a, b, np.asarray(n, np.int32), np.asarray(steps, np.int32),
                  np.asarray(capacity, np.int32), np.asarray(choice, np.bool_)]
        results.append((arrays, reference(a, b, n, steps, choice, width=width, cas=cas,
                                         operation=operation, byte_order=byte_order)))
    return results

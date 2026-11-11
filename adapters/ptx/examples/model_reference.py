"""Source example and independent Python-word evaluator; never compiler lowering."""

import numpy as np


def program(width):
    return f'''{{ .reg .b{width} %temporary; .reg .pred %chosen;
      shl.b{width} %temporary, $2, $4;
      xor.b{width} $0, %temporary, $3;
      shr.s{width} $1, $3, $4;
      setp.lt.s{width} %chosen, $2, $3;
      @%chosen add.u{width} $1, $1, 1;
    }}'''


def reference(x, y, shifts, iterations):
    x, y, shifts = np.asarray(x), np.asarray(y), np.asarray(shifts)
    if x.shape != y.shape or x.shape != shifts.shape or x.dtype != y.dtype or x.dtype not in (np.dtype('int32'), np.dtype('int64')):
        raise ValueError("reference requires same-shaped signed word inputs and shift counts")
    width = x.dtype.itemsize * 8
    modulus, mask = 1 << width, (1 << width) - 1

    def signed(word):
        word &= mask
        return word - modulus if word >= modulus // 2 else word

    a, b = list(map(int, x.flat)), list(map(int, y.flat))
    counts = [min(int(c) & ((1 << 32) - 1), width) for c in shifts.flat]
    for _ in range(iterations):
        first, second = [], []
        for left, right, count in zip(a, b, counts, strict=True):
            first.append(signed(((left & mask) << count) ^ (right & mask)))
            second.append(signed((right >> count) + (left < right)))
        a, b = first, second
    return [np.asarray(v, x.dtype).reshape(x.shape) for v in (a, b)]

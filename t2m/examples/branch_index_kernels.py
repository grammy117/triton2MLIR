"""A conditional integer lookup with complete ordered state observations."""

import triton
import triton.language as tl


@triton.jit
def branch_index(x, state, old, divisor, allow, STYLE: tl.constexpr,
                 OUTSIDE: tl.constexpr, AFTER: tl.constexpr):
    denominator = divisor * 2 if STYLE == "product" else divisor
    if OUTSIDE:
        displacement = 9 // divisor
    else:
        displacement = 0
    before = tl.load(state)
    tl.store(old, before)
    if STYLE == "and" or STYLE == "false-and":
        condition = (divisor > 0) & (allow != 0)
    elif STYLE == "or":
        condition = (divisor <= 0) | (allow == 0)
    elif STYLE == "else":
        condition = divisor <= 0
    elif STYLE == "unsigned":
        condition = divisor.to(tl.uint32) > 0
    elif STYLE == "unrelated":
        condition = allow != 0
    elif STYLE == "ne":
        condition = divisor != 0
    elif STYLE == "true-or":
        condition = (divisor > 0) | (allow != 0)
    elif STYLE == "product":
        condition = denominator > 0
    else:
        condition = divisor > 0
    if STYLE == "else" or STYLE == "or" or STYLE == "wrong-else" or STYLE == "false-and":
        if condition:
            tl.store(state, before ^ 3)
        else:
            value = tl.load(x + 9 // denominator + displacement)
            tl.store(state, before + value)
    else:
        if condition:
            value = tl.load(x + 9 // denominator + displacement)
            tl.store(state, before + value)
        else:
            tl.store(state, before ^ 3)
    if AFTER:
        value = tl.load(x + 9 // divisor)
        tl.store(state, value)

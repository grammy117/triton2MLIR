"""Source snippets and independent NumPy word references for packed examples."""


def program(width, pack, mode="independent"):
    statements = []
    for slot in range(pack):
        other = (slot + 1) % pack if mode == "coupled" and slot == 0 else slot
        statements.extend((
            f"add.u{width} ${slot}, ${2 * pack + slot}, ${3 * pack + other};",
            f"xor.b{width} ${pack + slot}, ${2 * pack + slot}, ${3 * pack + slot};"))
        if mode == "asymmetric" and slot == 0:
            statements.append(f"add.u{width} $0, $0, 1;")
    if mode == "hardware":
        statements.append(f"mov.u{width} $0, %clock;")
    register = "l" if width == 64 else "r"
    constraints = ",".join(["=" + register] * (2 * pack) + [register] * (2 * pack))
    return "\n".join(statements), constraints


def reference(a, b, steps):
    # NumPy array addition retains the declared integer word width. This
    # reference evaluates logical elements, never source register grouping.
    a, b = a.copy(), b.copy()
    for _ in range(steps):
        a, b = a + b, a ^ b
    return a, b

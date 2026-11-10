"""Independent CPU reference for the shipped word-print example only."""

import itertools


def word_reference(values,n,steps,allow,*,hex_=False,block_size=8):
    lines = []
    field_width = len(str(block_size//2-1))
    for pid in range((n+block_size-1)//block_size):
        for step in range(steps):
            if step % 2:
                continue
            for operand in range(4):
                for row,col in itertools.product(range(2),range(block_size//2)):
                    offset = pid*block_size+row*(block_size//2)+col
                    value = int(values[offset]) if offset < n else -7
                    bits = [64,64,8,1][operand]
                    if operand == 1:
                        value %= 2**64
                    elif operand == 2:
                        value %= 256
                    elif operand == 3:
                        value = int(value != 0)
                    printed = f"0x{value % (1 << bits):0{max(1,bits//4)}x}" if hex_ else str(value)
                    lines.append(f"pid ({pid}, 0, 0) idx ({row}, {col:{field_width}}) words: (operand {operand}) {printed}\n")
            lines.extend([f"pid ({pid}, 0, 0) idx () control: (operand 0) {steps}\n",
                          f"pid ({pid}, 0, 0) idx () control: (operand 1) {int(allow)}\n",f"pid ({pid}, 0, 0) event\n"])
        if not allow:
            lines.append("Assertion `logical print completed` failed\n")
            break
    return {"schema":"t2m.effect-reference.v1","termination":"assertion_failure" if n > 0 and not allow else "return",
        "stdout":"".join(lines),"stderr":""}

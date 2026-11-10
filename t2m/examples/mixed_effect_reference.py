"""Independent CPU recurrence and logical stream for the shipped mixed demo."""


def update_reference(x,state,n,steps,allow,*,block_size=8):
    current = state.copy()
    lines = []
    fail = n > 0 and not allow
    width = len(str(block_size//2-1))
    for pid in range((n+block_size-1)//block_size):
        for step in range(steps):
            if step % 2:
                continue
            for lane in range(block_size):
                offset = pid*block_size+lane
                if offset < n:
                    current[offset] += x[offset]
                value = int(current[offset]) if offset < n else 0
                lines.append(f"pid ({pid}, 0, 0) idx ({lane//(block_size//2)}, {lane%(block_size//2):{width}}) state: {value}\n")
        if fail:
            lines.append("Assertion `mixed result check` failed\n")
            break
    reference = {"schema":"t2m.effect-reference.v1" if fail else "t2m.effect-reference.v2",
        "termination":"assertion_failure" if fail else "return","stdout":"".join(lines),"stderr":""}
    if not fail:
        reference["results"] = {"archive":"reference.npz","comparisons":[{"kind":"words"}]*2}
    return reference,[current,x-current]

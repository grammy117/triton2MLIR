"""Independent word-only reference for the original TTIR sentinel program."""


def update(x, state, table, sentinel, steps):
    result, snapshot = state.copy(), state.copy()
    for index in range(len(state)):
        if index % 2 == 0:
            snapshot[index] = sentinel
        for step in range(steps):
            if (step + index // 8) % 2 == 0:
                if index % 2 == 0:
                    result[index] = table[index % 8]
            elif index % 2:
                result[index] = x[index]
    return result, snapshot

"""Prepare typed input values without losing observable writable aliases.

Ordinary readonly arguments can be snapshotted before serialization: source
pointer identity and volatile observations are separately refused by intake.
An overlap between tensors involving writable storage needs an alias/state interface.
Scalar carriers are loaded by value before the callee observes pointer storage.
Validate the complete overlap graph before making any copy.
"""

import numpy as np

READONLY_SNAPSHOTS = "readonly-input-value-snapshots-v1"
MAX_ALIAS_WORK = 4096


def snapshot_readonly_aliases(spec, arrays):
    declared = spec.inputs
    if len(arrays) != len(declared):
        raise ValueError("input storage requires the complete declared argument roster")
    readonly = {value.name for value in spec.buffers if value.effect == "read"}
    by_value = {value.name for value in spec.runtime_scalars}
    planned = set()
    for i,left in enumerate(arrays):
        for j,right in enumerate(arrays[:i]):
            if not np.may_share_memory(left,right):
                continue
            if declared[i].name in by_value or declared[j].name in by_value:
                # A scalar carrier is loaded by value before invocation; it
                # cannot become an alias of the callee's pointer arguments.
                planned.update(index for index in (i,j) if declared[index].name in by_value)
                continue
            if declared[i].name in readonly and declared[j].name in readonly:
                # A bounds-only possible overlap is enough to choose copies.
                # Exact readonly alias analysis would grant no extra semantics.
                planned.update((i,j))
                continue
            try:
                overlaps = np.shares_memory(left,right,max_work=MAX_ALIAS_WORK)
            except np.exceptions.TooHardError as exc:
                raise ValueError("writable input overlap exceeds its bounded analysis") from exc
            if overlaps:
                raise ValueError("overlap involving writable source storage requires an alias/state interface: "
                                 +declared[j].name+", "+declared[i].name)
    # This is a value snapshot, not a dtype cast or a memory-view rewrite.
    return tuple(value.copy(order="C") if i in planned else value for i,value in enumerate(arrays))

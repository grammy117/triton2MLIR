"""Expose uniform logical launch queries to the existing tensor/grid adapter.

Meta/shared 17f0c234's GetProgramID/GetNumPrograms converters replace queries
with six function arguments when selected. Our functional adapter instead binds
the same values to its logical forall. Move only these total Pure TTIR queries
to their enclosing program entry, so nested numeric regions capture the values
without introducing a launcher ABI or choosing physical cores/warps.
"""

from xdsl.dialects import builtin

from .diagnostics import ConversionError


def normalize_region_launch_queries(module, source_ids):
    from .shared_tensor import opcode

    plans = []
    for operation in module.walk():
        name = opcode(operation)
        if name not in ("tt.get_program_id", "tt.get_num_programs"):
            continue
        ids = source_ids(operation.location)

        def refuse(message, *, name=name, ids=ids):
            raise ConversionError(message, opcode=name, source_id=next(iter(ids), None))

        axis = operation.properties.get("axis", operation.attributes.get("axis"))
        if (operation.operands or len(operation.results) != 1 or operation.results[0].type != builtin.i32
                or operation.regions or operation.successors
                or not isinstance(axis, builtin.IntegerAttr) or axis.type != builtin.i32
                or not 0 <= axis.value.data <= 2
                or any(key not in ("op_name", "op_name__", "axis") and not key.startswith("prov.")
                       for key in (*operation.properties, *operation.attributes))):
            refuse("logical launch query requires its exact i32 result, axis and source fields")
        function = operation.parent_op()
        while function is not None and opcode(function) not in ("tt.func", "func.func"):
            function = function.parent_op()
        if function is None or len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
            refuse("logical launch query needs one structured enclosing program entry")
        entry = function.regions[0].block
        if operation.parent is not entry:
            plans.append((operation, entry, name, axis.value.data, ids))
    reports = []
    # Check all queries before moving any node, including unused queries.
    for operation, entry, name, axis, ids in plans:
        operation.detach()
        if entry.first_op is None:
            entry.add_op(operation)
        else:
            entry.insert_op_before(operation, entry.first_op)
        reports.append({"kind": "uniform-region-launch-query", "source_ids": ids,
                        "query": name, "axis": axis,
                        "scope": "one-logical-program-invocation",
                        "placement": "enclosing-program-entry", "physical_mapping": "undecided"})
    return reports

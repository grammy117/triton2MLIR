"""Replay protected source phases without granting numeric/effect authority."""

from .diagnostics import ConversionError


def validate(graph, reports, generic, protected, *, source_entry):
    records = graph["operations"]
    required = [row for row in records if row.get("opcode") == "ttg.barrier"]
    from .program_barriers import atomic_options, expand, volatile_options
    from .shared_tensor import (
        _preflight,
        _source_ids,
        opcode,
        parse_shared,
        parse_shared_source,
    )
    from .source_graph import inventory

    module, literals = parse_shared_source(generic.decode() if isinstance(generic, bytes) else generic)
    present = any(opcode(op) == "ttg.barrier" for op in module.walk())
    if not present and not required and reports is None:
        return
    if not present or not required or not isinstance(reports, list) or not reports:
        raise ValueError("allocation source barriers require complete phase declarations")
    atomics = any(opcode(op) in ("tt.atomic_rmw", "tt.atomic_cas") for op in module.walk())
    from xdsl.dialects import builtin
    volatile = any(opcode(op) == "tt.load" and isinstance(op.properties.get("isVolatile"), builtin.IntegerAttr)
                   and op.properties["isVolatile"].value.data for op in module.walk())
    options = (atomic_options(reports[0].get("contract", {}).get("atomic_effects")) if atomics else
               volatile_options(reports[0].get("contract", {}).get("volatile_effects")) if volatile else {})
    if atomics and volatile:
        volatile_options(reports[0].get("contract", {}).get("volatile_effects"))
        if options["effect_profile"] != "typed-atomics":
            raise ValueError("combined program phases require the selected typed atomic component")
        options["effect_profile"] = "atomic-volatile"
    try:
        actual = _preflight(module, source_entry=source_entry, allow_composite_dots=True, literal_words=literals,
            atomics=atomics, typed_atomics=options.get("effect_profile") in ("typed-atomics", "atomic-volatile"),
            f32_atomic_add=options.get("f32_atomic_add"), volatile=volatile)
        if inventory(module, actual, literal_words=literals) != graph:
            raise ValueError("allocation barrier inventory differs from its actual typed source")
        expected = expand(module, _source_ids, **options)
    except ConversionError as error:
        raise ValueError("allocation barrier source is outside its participation profile") from error
    if reports != expected:
        raise ValueError("allocation source barrier declarations differ from original phases")
    retained = parse_shared(protected.decode() if isinstance(protected, bytes) else protected)
    check_phases(module, retained, reports, _source_ids, opcode)


def check_phases(expected, retained, reports, source_ids, opcode):
    """Check real phase boundaries, scopes, results and retained source payload."""
    from xdsl.dialects import builtin

    ids = {row["contract"]["source_id"] for row in reports}

    def phases(module):
        result = {}
        for op in module.walk():
            selected = set(source_ids(op.location)) & ids
            if opcode(op) == "scf.execute_region" and selected:
                if len(selected) != 1 or next(iter(selected)) in result:
                    raise ValueError("allocation barrier has duplicate or ambiguous phases")
                result[next(iter(selected))] = op
        if set(result) != ids:
            raise ValueError("allocation source barrier lost an executable phase")
        return result

    def position(op):
        path = []
        while op.parent_op() is not None:
            block, region, parent = op.parent_block(), op.parent_region(), op.parent_op()
            path.append((opcode(parent), source_ids(parent.location),
                         list(parent.regions).index(region), list(region.blocks).index(block)))
            op = parent
        return path

    original, current = phases(expected), phases(retained)
    effects = {"tt.load", "tt.store", "tt.call", "tt.assert", "cf.assert", "tt.print", "tt.atomic_rmw", "tt.atomic_cas"}
    effect_ids = {sid for op in expected.walk() if opcode(op) in effects for sid in source_ids(op.location)}
    atomic_selection = reports[0]["contract"].get("atomic_effects")
    if atomic_selection is not None:
        from .atomic_memory import contract as atomic_contract
        from .program_barriers import atomic_options

        options = atomic_options(atomic_selection)

        def atomic(op):
            return atomic_contract(op, source_ids, typed=options["effect_profile"] == "typed-atomics",
                                   f32_atomic_add=options["f32_atomic_add"])

        declared = {sid: atomic(op) for op in expected.walk() if opcode(op) in ("tt.atomic_rmw", "tt.atomic_cas")
                    for sid in source_ids(op.location)}
        for op in retained.walk():
            if opcode(op) in ("tt.atomic_rmw", "tt.atomic_cas"):
                actual = atomic(op)
                if not any(declared.get(sid) == actual for sid in source_ids(op.location)):
                    raise ValueError("allocation barrier protection changed original atomic fields")
    from .volatile_memory import contract as volatile_contract
    loads = {sid: op for op in expected.walk() if opcode(op) == "tt.load" for sid in source_ids(op.location)}
    for op in retained.walk():
        if opcode(op) != "tt.load":
            continue
        selected = set(source_ids(op.location)) & set(loads)
        if len(selected) != 1:
            raise ValueError("allocation barrier contains an unaccounted source read")
        sid = next(iter(selected))
        before = loads[sid]
        flag = before.properties.get("isVolatile")
        after_flag = op.properties.get("isVolatile")
        if isinstance(flag, builtin.IntegerAttr) and flag.value.data:
            try:
                if volatile_contract(before, sid) != volatile_contract(op, sid):
                    raise ValueError("allocation barrier protection changed original volatile fields")
            except ConversionError as error:
                raise ValueError("allocation barrier protection lost an original volatile read") from error
        elif isinstance(after_flag, builtin.IntegerAttr) and after_flag.value.data:
            raise ValueError("allocation barrier protection invented a volatile read")

    def events(phase):
        result = []
        for child in phase.walk():
            if opcode(child) in effects:
                selected = [sid for sid in source_ids(child.location) if sid in effect_ids]
                if not selected and opcode(child) != "cf.assert":
                    raise ValueError("allocation barrier contains an unaccounted source memory effect")
                if selected:
                    result.append((opcode(child), selected))
        return result

    if events(expected) != events(retained):
        raise ValueError("allocation barrier protection changed original source memory order")

    for report in reports:
        source_id = report["contract"]["source_id"]
        before, after = original[source_id], current[source_id]
        marker = after.properties.get("no_inline", after.attributes.get("no_inline"))
        if (len(after.regions) != 1 or len(after.regions[0].blocks) != 1
                or after.regions[0].block.args or not isinstance(marker, builtin.UnitAttr)
                or position(before) != position(after)
                or [str(v.type) for v in after.results] != report["live_value_types"]):
            raise ValueError("allocation barrier changed its executable phase scope or results")
        block = after.regions[0].block
        if (block.last_op is None or opcode(block.last_op) != "scf.yield"
                or [v.type for v in block.last_op.operands] != [v.type for v in after.results]):
            raise ValueError("allocation barrier phase lacks its complete typed yield")
        observed = {sid for child in after.walk() for sid in source_ids(child.location)}
        if not set(report["phase_source_ids"]) <= observed:
            raise ValueError("allocation barrier phase lost original source payload")
        if events(before) != events(after):
            raise ValueError("allocation barrier phase changed original source memory effects or order")
    for op in retained.walk():
        if opcode(op).startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl.")):
            raise ValueError("protected program barrier retains an unresolved physical operation")

"""Re-raise ordered same-address tile writes before native pointer conversion.

Meta/shared 17f0c234 and Microsoft 050b93ae UnstructuredToMemref retain each
scatter's actual predicate and inactive no-op. For fresh functional tensor
results, collect an exactly covered write group into source-ordered selects.
No incoming storage is read, no lane is invented as initialized, and later
native pointer/ownership analyses still decide admission.
"""

from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter

from .source_boolean import BooleanCoverage


@dataclass
class _Group:
    key: tuple
    type: builtin.TensorType
    properties: dict
    events: list = field(default_factory=list)
    independent_roots: set = field(default_factory=set)
    readonly_reads: list = field(default_factory=list)
    other_accesses: list = field(default_factory=list)


def expand_tile_partitions(module, abi, source_ids):
    from .shared_tensor import _array, opcode

    entries = [op for op in module.ops if opcode(op) == "tt.func" and op.properties["sym_name"].data == abi.entry]
    if len(entries) != 1 or len(entries[0].regions[0].blocks) != 1:
        return []
    entry = entries[0].regions[0].block
    roots = {arg: buffer for arg, (kind, buffer) in zip(entry.args, abi.source_bindings, strict=True) if kind == "buffer"}
    fresh = {arg for arg, buffer in roots.items() if buffer.effect == "write"}
    reports = []

    def pointer(value, depth=0):
        if depth >= 32:
            return None
        if value in roots:
            return value, ("root", value)
        op = value.owner
        if not isinstance(op, Operation):
            return None
        name = opcode(op)
        if name not in ("tt.splat", "tt.addptr", "tt.broadcast", "tt.expand_dims", "tt.trans"):
            return None
        recovered = pointer(op.operands[0], depth + 1)
        if recovered is None:
            return None
        root, base = recovered
        return root, (name, value.type, tuple(sorted(op.properties.items())), base, tuple(op.operands[1:]))

    def ordinary(op, *, load=False):
        if opcode(op) != ("tt.load" if load else "tt.store") or len(op.operands) not in ((1, 2, 3) if load else (2, 3)):
            return False
        if load:
            # Cache/eviction hints carry no value semantics; volatile does.
            volatile = op.properties.get("isVolatile")
            if volatile is not None and (not isinstance(volatile, builtin.IntegerAttr) or volatile.value.data):
                return False
            return len(op.results) == 1
        return (not op.results and isinstance(op.operands[0].type, builtin.TensorType)
                and isinstance(op.operands[1].type, builtin.TensorType))

    def pure(op):
        name = opcode(op)
        return not op.regions and (name.startswith("arith.") or name in {
            "tt.splat", "tt.addptr", "tt.make_range", "tt.broadcast", "tt.expand_dims", "tt.trans",
            "tt.reshape", "tt.cat", "tt.join", "tt.split", "tt.bitcast"})

    def flush(group):
        events = group.events
        stores = [op for op in events if opcode(op) == "tt.store"]
        if len(stores) < 2:
            return
        # A sibling group may already have reconstructed an exact observation.
        # Re-read current actual SSA, including predicates, to avoid stale
        # candidates or cached facts about erased load results.
        proof, conditions, total = BooleanCoverage(pointwise_tensors=True), [], 0
        mask_type = builtin.TensorType(builtin.i1, group.type.get_shape())
        for op in events:
            if opcode(op) == "tt.store":
                if len(op.operands) == 3 and op.operands[2].type != mask_type:
                    return
                guard = proof.condition(op.operands[2]) if len(op.operands) == 3 else 1
                total = proof.combine("or", total, guard)
                if len(op.operands) == 3:
                    collect(op.operands[2], conditions, mask_type)
            else:
                segments = _array(op, "operandSegmentSizes")
                valid = segments == (1, 0, 0) or (segments == (1, 1, 1) and op.operands[1].type == mask_type
                                                               and op.operands[2].type == group.type)
                premise = proof.condition(op.operands[1]) if len(op.operands) == 3 else 1
                if not valid or not proof.covers([total], premise=premise):
                    return
        if total in (None, 0):
            return
        # Prefer an actual dominating source predicate, preserving rectangular
        # tails for upstream MaskAnalysis rather than synthesizing an opaque OR.
        cover = next((value for value in conditions if proof.condition(value) == total), None)
        if total != 1 and cover is None:
            return
        # A coverage predicate must dominate the final store. Conditions come
        # only from preceding store/load operands and their existing ancestors.
        last = stores[-1]
        combined, location = None, None
        observed = []
        for op in events:
            if opcode(op) == "tt.store":
                value = op.operands[1]
                if combined is not None:
                    if len(op.operands) == 2:
                        combined = value
                    else:
                        selected = arith.SelectOp(op.operands[2], value, combined)
                        selected.location = builtin.FusedLoc(builtin.ArrayAttr([location, op.location]), builtin.NoneAttr())
                        op.parent.insert_op_before(selected, op)
                        combined = selected.result
                else:
                    combined = value
                location = op.location if location is None else builtin.FusedLoc(
                    builtin.ArrayAttr([location, op.location]), builtin.NoneAttr())
                if op is not last:
                    op.parent.erase_op(op)
            else:
                # Collection already proved this active read observes a
                # preceding write, with its original explicit inactive fill.
                selected = arith.SelectOp(op.operands[1], combined, op.operands[2]) if len(op.operands) == 3 else None
                replacement = combined
                if selected is not None:
                    selected.location = op.location
                    op.parent.insert_op_before(selected, op)
                    replacement = selected.result
                observed.extend(source_ids(op.location))
                if cover is op.results[0]:
                    cover = replacement
                Rewriter.replace_op(op, [], [replacement])
        last.operands = (last.operands[0], combined, *([cover] if total != 1 else []))
        segments = last.properties.get("operandSegmentSizes")
        if segments is not None:
            last.properties["operandSegmentSizes"] = builtin.DenseArrayBase.from_list(builtin.i32, [1, 1, int(total != 1)])
        last.location = builtin.FusedLoc(builtin.ArrayAttr([op.location for op in stores]), builtin.NoneAttr())
        reports.append({"kind": "source-tile-write-partition", "source_ids": list(dict.fromkeys(
            sid for op in stores for sid in source_ids(op.location))), "observed_read_ids": observed,
            "writes": len(stores), "coverage": "actual-pointwise-boolean-union-equals-retained-source-mask" if total != 1 else
                "actual-pointwise-boolean-union-is-unconditional", "order": "source-last-active-write",
            "initialization": "no-incoming-storage-read; complete-geometry-and-active-read-proofs-still-required",
            "memory": "same-typed-source-pointer-expression; fresh-functional-result-disjoint-from-other-ABI-roots",
            "independent_roots": sorted(group.independent_roots),
            "unchanged_readonly_read_ids": list(dict.fromkeys(group.readonly_reads)),
            "unchanged_other_access_ids": list(dict.fromkeys(group.other_accesses)),
            "physical_mapping": "undecided"})

    def collect(value, conditions, type_, depth=0):
        if depth >= 128 or len(conditions) >= 128 or value in conditions:
            return
        if value.type == type_:
            conditions.append(value)
        owner = value.owner
        if isinstance(owner, Operation) and opcode(owner) in ("arith.andi", "arith.ori", "arith.xori", "arith.select"):
            for operand in owner.operands:
                collect(operand, conditions, type_, depth + 1)

    blocks = [block for op in tuple(module.walk()) for region in op.regions for block in region.blocks]
    for block in blocks:
        groups, event_count = {}, 0

        def finish(groups=groups):
            for group in groups.values():
                flush(group)
            groups.clear()

        for op in tuple(block.ops):
            if pure(op):
                continue
            store, load = ordinary(op), ordinary(op, load=True)
            recovered = pointer(op.operands[0]) if store or load else None
            if recovered is None:
                finish()
                event_count = 0
                continue
            root, key = recovered
            if load and roots[root].effect == "read":
                for group in groups.values():
                    group.independent_roots.add(roots[root].name)
                    group.readonly_reads.extend(source_ids(op.location))
            elif root in fresh:
                group = groups.get(root)
                type_ = op.operands[1].type if store else op.results[0].type
                if group is not None and (group.key != key or group.type != type_
                        or store and group.properties != dict(op.properties)):
                    finish()
                    event_count = 0
                    group = None
                for other, active in groups.items():
                    if other is not root:
                        active.independent_roots.add(roots[root].name)
                if group is None and store:
                    if len(groups) >= 128:
                        finish()
                        event_count = 0
                    group = groups[root] = _Group(key, type_, dict(op.properties))
                    group.independent_roots.update(roots[other].name for other in groups if other is not root)
                if group is not None:
                    group.events.append(op)
                    if len(group.events) >= 128:
                        finish()
                        event_count = 0
            else:
                # These ordinary input-state accesses cannot observe a fresh
                # functional result allocation. Keep them and their mutual
                # ordering; do not derive independence between input roots.
                for group in groups.values():
                    group.independent_roots.add(roots[root].name)
                    group.other_accesses.extend(source_ids(op.location))
            event_count += 1
            if event_count >= 1024:
                finish()
                event_count = 0
        finish()
    return reports

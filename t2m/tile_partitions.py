"""Re-raise ordered same-address tile writes before native pointer conversion.

Meta/shared 17f0c234 and Microsoft 050b93ae UnstructuredToMemref retain each
scatter's actual predicate and inactive no-op. For fresh functional tensor
results, collect an exactly covered write group into source-ordered selects.
No incoming storage is read, no lane is invented as initialized, and later
native pointer/ownership analyses still decide admission.
"""

from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Operation
from xdsl.rewriter import Rewriter

from .tile_write_proof import analyze_writes, lexical_choices


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
    from .shared_tensor import opcode

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
        from .value_regions import tensor_computation

        name = opcode(op)
        return not op.regions and (name.startswith(("arith.", "math.")) or tensor_computation(op) or name in {
            "tt.splat", "tt.addptr", "tt.make_range", "tt.broadcast", "tt.expand_dims", "tt.trans",
            "tt.reshape", "tt.cat", "tt.join", "tt.split", "tt.bitcast"})

    def rewrite_events(events, *, erase_stores, cover=None, unconditional=False):
        stores = [op for op in events if opcode(op) == "tt.store"]
        last = stores[-1]
        combined, location, observed = None, None, []
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
                if erase_stores and op is not last:
                    op.parent.erase_op(op)
            else:
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
        if erase_stores:
            last.operands = (last.operands[0], combined, *([] if unconditional else [cover]))
            if "operandSegmentSizes" in last.properties:
                last.properties["operandSegmentSizes"] = builtin.DenseArrayBase.from_list(builtin.i32, [1, 1, int(not unconditional)])
            last.location = builtin.FusedLoc(builtin.ArrayAttr([op.location for op in stores]), builtin.NoneAttr())
        return stores, observed

    def flush(group):
        events = group.events
        stores = [op for op in events if opcode(op) == "tt.store"]
        if not stores:
            return
        # A sibling group may already have reconstructed an exact observation.
        # Re-read current actual SSA, including predicates, to avoid stale
        # candidates or cached facts about erased load results.
        proof = analyze_writes(events, group.type)
        if proof is None or proof.total in (None, 0):
            return
        # Prefer an actual dominating source predicate, preserving rectangular
        # tails for upstream MaskAnalysis rather than synthesizing an opaque OR.
        unconditional = proof.matches(None)
        covers = [value for value in proof.candidates if proof.matches(value)]
        cover = next((value for value in covers if not proof.matches(value, global_scope=True)),
                     next(iter(covers), None))
        if not unconditional and cover is None:
            return
        if len(stores) == 1 and len(events) == 1 and (len(stores[0].operands) == 2 or
                not unconditional and cover is stores[0].operands[2]):
            return
        # A coverage predicate must dominate the final store. Conditions come
        # only from preceding store/load operands and their existing ancestors.
        scoped = bool(lexical_choices(events[-1]))
        stores, observed = rewrite_events(events, erase_stores=True, cover=cover, unconditional=unconditional)
        reports.append({"kind": "source-tile-write-partition", "source_ids": list(dict.fromkeys(
            sid for op in stores for sid in source_ids(op.location))), "observed_read_ids": observed,
            "writes": len(stores), "coverage": "actual-pointwise-boolean-union-equals-retained-source-mask" if not unconditional else
                "actual-pointwise-boolean-union-is-unconditional", "order": "source-last-active-write",
            "scope": "actual-selected-scf-if-edges" if scoped else "source-block",
            "initialization": "no-incoming-storage-read; complete-geometry-and-active-read-proofs-still-required",
            "memory": "same-typed-source-pointer-expression; fresh-functional-result-disjoint-from-other-ABI-roots",
            "independent_roots": sorted(group.independent_roots),
            "unchanged_readonly_read_ids": list(dict.fromkeys(group.readonly_reads)),
            "unchanged_other_access_ids": list(dict.fromkeys(group.other_accesses)),
            "physical_mapping": "undecided"})

    def distribution_cover(group, conditional):
        prefix = analyze_writes(group.events, group.type)
        if prefix is None:
            return None
        steps = 0

        def walk(block, states, depth):
            nonlocal steps
            if depth > 32 or len(states) > 128:
                return None
            for op in block.ops if block is not None else ():
                steps += 1
                if steps > 1024:
                    return None
                if isinstance(op, scf.IfOp):
                    if any(len(region.blocks) > 1 or region.blocks and region.block.args for region in op.regions):
                        return None
                    branched = []
                    for region, selected in zip(op.regions, (True, False), strict=True):
                        arms = [(events, (*choices, (op.cond, selected))) for events, choices in states]
                        result = walk(region.block if region.blocks else None, arms, depth + 1)
                        if result is None:
                            return None
                        branched.extend(result)
                    states = branched
                    if len(states) > 128:
                        return None
                elif opcode(op) == "scf.yield" or pure(op):
                    continue
                elif ordinary(op) or ordinary(op, load=True):
                    recovered = pointer(op.operands[0])
                    if recovered is None:
                        return None
                    root, key = recovered
                    if root is target:
                        if key != group.key:
                            return None
                        states = [([*events, op], choices) for events, choices in states]
                        if any(len(events) > 128 or analyze_writes(events, group.type, choices=choices) is None
                               for events, choices in states):
                            return None
                else:
                    return None
            return states

        target = next(root for root, candidate in groups.items() if candidate is group)
        states = []
        for region, selected in zip(conditional.regions, (True, False), strict=True):
            if len(region.blocks) > 1 or region.blocks and region.block.args:
                return None
            choices = (*lexical_choices(conditional), (conditional.cond, selected))
            result = walk(region.block if region.blocks else None, [(list(group.events), choices)], 0)
            if result is None:
                return None
            states.extend(result)
        proofs = [analyze_writes(events, group.type, choices=choices) for events, choices in states]
        if any(proof is None for proof in proofs):
            return None
        for cover in (None, *prefix.candidates):
            if not prefix.matches(cover) and all(proof.matches(cover) for proof in proofs):
                return (cover,)  # None is a successful unconditional cover.
        return None

    distribution_budget = 1024
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
            if isinstance(op, scf.IfOp):
                for root, group in tuple(groups.items()):
                    stores = [event for event in group.events if opcode(event) == "tt.store"]
                    if 2 * len(stores) > distribution_budget or distribution_cover(group, op) is None:
                        continue
                    # Prefix observations were proved before considering any
                    # future edge. Reconstruct them at their original points;
                    # only actual stores move into the selected successor.
                    stores, observed = rewrite_events(group.events, erase_stores=False)
                    for region in op.regions:
                        if not region.blocks:
                            if op.results:
                                raise ValueError("verified result-bearing if has an empty successor")
                            terminator = scf.YieldOp()
                            terminator.location = op.location
                            added = Block([terminator])
                            region.add_block(added)
                            blocks.append(added)
                        copies = [store.clone() for store in stores]
                        region.block.insert_ops_before(copies, region.block.first_op)
                    for store in stores:
                        store.parent.erase_op(store)
                    distribution_budget -= 2 * len(stores)
                    groups.pop(root)
                    reports.append({"kind": "source-tile-prefix-distribution",
                        "source_ids": list(dict.fromkeys(sid for store in stores for sid in source_ids(store.location))),
                        "branch_source_ids": source_ids(op.location), "observed_read_ids": observed,
                        "root": roots[root].name, "stores": len(stores), "static_copies": 2,
                        "execution": "exactly-one-original-scalar-if-successor; no-arithmetic-speculation",
                        "coverage": "each-selected-source-path-completes-retained-prefix-cover",
                        "scope": "prefix-reads-proved-before-future-choices; branch-reads-proved-in-source-order"})
                finish()
                event_count = 0
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

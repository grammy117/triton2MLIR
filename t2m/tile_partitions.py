"""Re-raise ordered same-address tile writes before native pointer conversion.

Meta/shared 17f0c234 and Microsoft 050b93ae UnstructuredToMemref retain each
scatter's actual predicate and inactive no-op. For fresh functional tensor
results, collect an exactly covered write group into source-ordered selects.
No incoming storage is read, no lane is invented as initialized, and later
native pointer/ownership analyses still decide admission.
"""

from xdsl.dialects import arith, builtin
from xdsl.ir import Operation
from xdsl.rewriter import Rewriter

from .source_boolean import BooleanCoverage


def expand_tile_partitions(module, abi, source_ids):
    from .shared_tensor import _array, opcode

    entries = [op for op in module.ops if opcode(op) == "tt.func" and op.properties["sym_name"].data == abi.entry]
    if len(entries) != 1 or len(entries[0].regions[0].blocks) != 1:
        return []
    entry = entries[0].regions[0].block
    fresh = {arg for arg, (kind, buffer) in zip(entry.args, abi.source_bindings, strict=True)
             if kind == "buffer" and buffer.effect == "write"}
    reports = []

    def pointer(value, depth=0):
        if depth >= 32:
            return None
        if value in fresh:
            return "fresh", value
        op = value.owner
        if not isinstance(op, Operation):
            return None
        name = opcode(op)
        if name not in ("tt.splat", "tt.addptr", "tt.broadcast", "tt.expand_dims", "tt.trans"):
            return None
        base = pointer(op.operands[0], depth + 1)
        if base is None:
            return None
        return name, value.type, tuple(sorted(op.properties.items())), base, tuple(op.operands[1:])

    def ordinary(op, *, load=False):
        if opcode(op) != ("tt.load" if load else "tt.store") or len(op.operands) not in ((1, 2, 3) if load else (2, 3)):
            return False
        if not isinstance(op.operands[0].type, builtin.TensorType):
            return False
        if load:
            # Cache/eviction hints carry no value semantics; volatile does.
            volatile = op.properties.get("isVolatile")
            if volatile is not None and (not isinstance(volatile, builtin.IntegerAttr) or volatile.value.data):
                return False
            return len(op.results) == 1 and isinstance(op.results[0].type, builtin.TensorType)
        return not op.results and isinstance(op.operands[1].type, builtin.TensorType)

    def pure(op):
        name = opcode(op)
        return not op.regions and (name.startswith("arith.") or name in {
            "tt.splat", "tt.addptr", "tt.make_range", "tt.broadcast", "tt.expand_dims", "tt.trans",
            "tt.reshape", "tt.cat", "tt.join", "tt.split", "tt.bitcast"})

    def flush(events, proof, conditions):
        stores = [op for op in events if opcode(op) == "tt.store"]
        if len(stores) < 2:
            return
        guards = [proof.condition(op.operands[2]) if len(op.operands) == 3 else 1 for op in stores]
        total = 0
        for guard in guards:
            total = proof.combine("or", total, guard)
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
            "memory": "same-typed-source-pointer-expression; fresh-ABI-root; no-intervening-effect",
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
        events, key, types, properties = [], None, None, None
        proof, conditions, union = BooleanCoverage(pointwise_tensors=True), [], 0
        for op in tuple(block.ops):
            if not events:
                if ordinary(op) and len(op.operands) == 3 and (candidate := pointer(op.operands[0])) is not None:
                    proof = BooleanCoverage(pointwise_tensors=True)
                    key, types, properties = candidate, (op.operands[1].type, op.operands[2].type), dict(op.properties)
                    if types[1] != builtin.TensorType(builtin.i1, types[0].get_shape()):
                        continue
                    events = [op]
                    union = proof.condition(op.operands[2])
                    collect(op.operands[2], conditions, types[1])
                continue
            same = pointer(op.operands[0]) == key if ordinary(op) or ordinary(op, load=True) else False
            if (ordinary(op) and same and op.operands[1].type == types[0] and dict(op.properties) == properties
                    and (len(op.operands) == 2 or op.operands[2].type == types[1])):
                events.append(op)
                guard = proof.condition(op.operands[2]) if len(op.operands) == 3 else 1
                union = proof.combine("or", union, guard)
                if len(op.operands) == 3:
                    collect(op.operands[2], conditions, types[1])
            elif ordinary(op, load=True) and same and op.results[0].type == types[0]:
                segments = _array(op, "operandSegmentSizes")
                valid = segments == (1, 0, 0) or (segments == (1, 1, 1) and op.operands[1].type == types[1]
                                                               and op.operands[2].type == types[0])
                premise = proof.condition(op.operands[1]) if len(op.operands) == 3 else 1
                if valid and proof.covers([union], premise=premise):
                    events.append(op)
                else:
                    flush(events, proof, conditions)
                    events, conditions, union = [], [], 0
            elif pure(op):
                continue
            else:
                flush(events, proof, conditions)
                events, conditions, union = [], [], 0
            if len(events) >= 128:
                flush(events, proof, conditions)
                events, conditions, union = [], [], 0
        flush(events, proof, conditions)
    return reports

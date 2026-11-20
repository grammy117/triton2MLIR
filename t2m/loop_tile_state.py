"""Carry proved same-address fresh tiles through original structured regions.

Meta/shared 17f0c234 PtrAnalysis::rewriteForOp retains original loop interfaces;
its nested tensor-index fixtures exercise the corresponding native conversion.
This preceding functional analysis adds numeric tile state, not physical state.
No loop is peeled, unrolled, serialized or assigned a hardware mapping.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .tile_write_proof import analyze_writes


class _Decline(Exception):
    pass


@dataclass
class LoopTilePlan:
    group: object
    nodes: list
    cover: object
    loop_domains: dict

    def apply(self, initial, source_ids):
        from .shared_tensor import opcode

        stores = [op for op in self.group.events if opcode(op) == "tt.store"]
        reads, loops, inactive_loops, values = [], [], [], list(self.group.value_regions)

        def inventory(nodes):
            for op, kind, children in nodes:
                if kind == "store":
                    stores.append(op)
                elif kind == "load":
                    reads.append(op)
                elif kind in ("for", "if", "empty-for"):
                    if kind in ("for", "empty-for"):
                        loops.append(op)
                    if kind == "empty-for":
                        inactive_loops.append(op)
                    for child in children:
                        inventory(child)
                elif op.regions:
                    values.extend(source_ids(op.location))

        inventory(self.nodes)
        report = {"kind": "source-loop-tile-state",
            "source_ids": list(dict.fromkeys(sid for op in stores for sid in source_ids(op.location))),
            "observed_read_ids": list(dict.fromkeys(sid for op in reads for sid in source_ids(op.location))),
            "loop_source_ids": list(dict.fromkeys(sid for op in loops for sid in source_ids(op.location))),
            "inactive_loop_source_ids": list(dict.fromkeys(sid for op in inactive_loops for sid in source_ids(op.location))),
            "retained_value_region_source_ids": list(dict.fromkeys(values)),
            "recurrence": ("original-loop-bounds-and-numeric-state; additional-functional-tile-carrier"
                           if len(loops) > len(inactive_loops) else
                           "original-numeric-state; inactive-loops-retain-complete-initial-tuples"),
            "initialization": "actual-source-ordered-writes; zero-trip-keeps-entry-coverage",
            "coverage": "complete-retained-source-mask-before-final-store; independent-geometry-required",
            "memory": "same-typed-invariant-source-pointer; fresh-disjoint-functional-root",
            "physical_mapping": "undecided"}
        final = stores[0].clone()
        final.location = builtin.FusedLoc(builtin.ArrayAttr([op.location for op in stores]), builtin.NoneAttr())
        anchor, block = self.nodes[-1][0].next_op, self.nodes[-1][0].parent
        replacements = {}

        def rewrite(nodes, state):
            for op, kind, children in nodes:
                if kind == "store":
                    value = op.operands[1]
                    if len(op.operands) == 3:
                        selected = arith.SelectOp(op.operands[2], value, state)
                        selected.location = op.location
                        op.parent.insert_op_before(selected, op)
                        value = selected.result
                    state = value
                    op.parent.erase_op(op)
                elif kind == "load":
                    if len(op.operands) == 3:
                        selected = arith.SelectOp(op.operands[1], state, op.operands[2])
                        selected.location = op.location
                        Rewriter.replace_op(op, [selected], [selected.result])
                    else:
                        Rewriter.replace_op(op, [], [state])
                elif kind == "if":
                    regions = [op.detach_region(region) for region in tuple(op.regions)]
                    for region, child in zip(regions, children, strict=True):
                        if not region.blocks:
                            region.add_block(Block([scf.YieldOp()]))
                        yielded = rewrite(child, state)
                        terminator = region.block.last_op
                        terminator.operands = (*terminator.operands, yielded)
                    replacement = scf.IfOp(op.cond, [*[value.type for value in op.results], state.type], *regions)
                    replacement.attributes.update(op.attributes)
                    replacement.properties.update(op.properties)
                    replacement.location = op.location
                    replacements[op] = replacement
                    Rewriter.replace_op(op, [replacement], replacement.results[:-1])
                    state = replacement.results[-1]
                elif kind == "for":
                    body = op.body.block
                    carried = body.insert_arg(state.type, len(body.args))
                    yielded = rewrite(children[0], carried)
                    body.last_op.operands = (*body.last_op.operands, yielded)
                    replacement = scf.ForOp(op.lb, op.ub, op.step, [*op.iter_args, state], op.detach_region(op.body))
                    replacement.attributes.update(op.attributes)
                    replacement.properties.update(op.properties)
                    replacement.location = op.location
                    self.loop_domains[replacement] = self.loop_domains[op]
                    replacements[op] = replacement
                    Rewriter.replace_op(op, [replacement], replacement.results[:-1])
                    state = replacement.results[-1]
            return state

        state = rewrite(self.nodes, initial)
        final.operands = (final.operands[0], state, *([] if self.cover is None else [self.cover]))
        if "operandSegmentSizes" in final.properties:
            final.properties["operandSegmentSizes"] = builtin.DenseArrayBase.from_list(builtin.i32, [1, 1, int(self.cover is not None)])
        if anchor is None:
            block.add_op(final)
        else:
            block.insert_op_before(final, anchor)
        report["replacement_control"] = replacements.get(self.nodes[0][0], self.nodes[0][0])
        return report


def plan_loop_state(group, root, first, *, pointer, ordinary, pure, loop_domains):
    """Prove the entire rewrite before changing any executable source operation.

    A source-defined prefix supplies an SSA value only for its initialized lanes.
    Reads require actual selected-path implication. A loop can add a proved
    invariant coverage only when it executes; zero trips keep entry coverage.
    A later source branch may complete the zero-trip path. Unknown effects,
    changing addresses, unsupported control or budgets decline this optional pass.
    """
    from .shared_tensor import _array, opcode

    initial = analyze_writes(group.events, group.type)
    if initial is None or initial.total in (None, 0):
        return None
    proof, candidates = initial.booleans, [None, *initial.candidates]
    masks = builtin.TensorType(builtin.i1, group.type.get_shape())
    visited, memory, actual_loops, assumptions = 0, 0, 0, 1
    temporary = []

    def fail():
        raise _Decline

    def boolean(value):
        result = proof.condition(value) if value is not None else 1
        if result is None:
            fail()
        return result

    def combine(kind, a, b):
        result = proof.combine(kind, a, b)
        if result is None:
            fail()
        return result

    def equal(a, b, premise):
        premise = combine("and", premise, assumptions)
        return (proof.covers([a], premise=combine("and", premise, b))
                and proof.covers([b], premise=combine("and", premise, a)))

    def choose(total, premise):
        for candidate in candidates:
            if candidate is not None and candidate.owner in group.events:
                continue
            if equal(total, boolean(candidate), premise):
                return (candidate,)
        return None

    def uncast(value):
        while isinstance(value.owner, arith.BitcastOp) and value.owner.operands[0].type == value.type:
            value = value.owner.operands[0]
        return value

    def scan(operations, total, premise, depth):
        nonlocal visited, memory, actual_loops, assumptions
        if depth > 32:
            fail()
        nodes = []
        for op in operations:
            visited += 1
            if visited > 1024:
                fail()
            name, children, kind = opcode(op), [], "keep"
            if isinstance(op, scf.IfOp):
                condition = boolean(op.cond)
                alternatives = []
                for region, active in zip(op.regions, (condition, proof.negate(condition)), strict=True):
                    if len(region.blocks) > 1 or region.blocks and region.block.args:
                        fail()
                    child, coverage = scan(region.block.ops if region.blocks else (), total,
                        combine("and", premise, active), depth + 1)
                    children.append(child)
                    alternatives.append(combine("and", active, coverage))
                total, kind = combine("or", *alternatives), "if"
            elif isinstance(op, scf.ForOp):
                domain = loop_domains.get(op)
                if domain is None or len(op.body.blocks) != 1:
                    fail()
                condition = arith.CmpiOp(uncast(op.lb), uncast(op.ub), "slt")
                temporary.append(condition)
                tripped = boolean(condition.result)
                if domain.maximum_iterations == 0:
                    # This relation is valid only in the actual source scope.
                    # Preserve the original native loop and all outside reads;
                    # the tensor memory witness checks reads before omitting it.
                    implication = combine("or", proof.negate(premise), proof.negate(tripped))
                    assumptions = combine("and", assumptions, implication)
                    nodes.append((op, "empty-for", []))
                    actual_loops += 1
                    continue
                body_premise = combine("and", premise, tripped)
                child, coverage = scan(op.body.block.ops, total, body_premise, depth + 1)
                cover = choose(coverage, body_premise)
                if cover is None:
                    fail()
                # Candidates are dominating prefix SSA, so this post-body cover
                # is invariant across all later iterations, including masks
                # computed from actual changing data inside the loop.
                updated = boolean(cover[0])
                total = updated if domain.minimum_iterations > 0 else combine("or",
                    combine("and", tripped, updated), combine("and", proof.negate(tripped), total))
                children, kind = [child], "for"
                actual_loops += 1
            elif name == "scf.yield" or pure(op):
                pass
            elif ordinary(op) or ordinary(op, load=True):
                recovered = pointer(op.operands[0])
                if recovered is None:
                    fail()
                if recovered[0] is root:
                    if recovered[1] != group.key:
                        fail()
                    memory += 1
                    if memory > 128:
                        fail()
                    if name == "tt.store":
                        if op.operands[1].type != group.type or dict(op.properties) != group.properties:
                            fail()
                        active = boolean(op.operands[2]) if len(op.operands) == 3 else 1
                        if len(op.operands) == 3 and op.operands[2].type != masks:
                            fail()
                        total, kind = combine("or", total, active), "store"
                    else:
                        segments = _array(op, "operandSegmentSizes")
                        if not (segments == (1, 0, 0) or segments == (1, 1, 1)
                                and op.operands[1].type == masks and op.operands[2].type == group.type):
                            fail()
                        if op.results[0].type != group.type:
                            fail()
                        active = boolean(op.operands[1]) if len(op.operands) == 3 else 1
                        if not proof.covers([total], premise=combine("and", assumptions, combine("and", premise, active))):
                            fail()
                        kind = "load"
            else:
                fail()
            nodes.append((op, kind, children))
        return nodes, total

    nodes, total = [], initial.total
    try:
        op = first
        while op is not None:
            part, total = scan([op], total, initial.premise, 0)
            nodes.extend(part)
            cover = choose(total, initial.premise)
            if cover is not None and actual_loops and memory:
                return LoopTilePlan(group, nodes, cover[0], loop_domains)
            op = op.next_op
    except (ConversionError, _Decline):
        return None
    finally:
        for operation in temporary:
            operation.drop_all_references()
    return None

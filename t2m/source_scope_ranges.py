"""Definition-scoped source facts from actual branch and loop reachability.

LLVM a47bddcc IntegerRangeAnalysis.cpp keeps facts at program points and bounds
induction variables through the actual loop interface. Reuse that boundary:
the source-width audit validates definitions before applying narrower use facts.
The positive-product premise below is a lexical source proof, not an annotation
or an assumption about sampled sizes, GPU threads or backend integer promotion.
"""

from .diagnostics import ConversionError
from .loop_domains import enclosed_loop, require_signed_loop
from .source_predicates import branch_constraints, signed_comparison_enclosure


class SourceScopes:
    def __init__(self, bounds, opcode):
        self.bounds, self.opcode = bounds, opcode
        self.cache = {None: {}}
        self.proofs = []

    def facts(self, block):
        if block in self.cache:
            return self.cache[block]
        owner = block.parent_op()
        parent = owner.parent if owner is not None else None
        facts = dict(self.facts(parent))
        if owner is not None and owner.name == "scf.if":
            selected = block.parent is owner.regions[0]
            narrowed = set()

            def refine(value, interval):
                baseline = self.bounds(value, parent)
                low, high = facts.get(value, baseline)
                result = max(low, interval[0]), min(high, interval[1])
                if result[0] > result[1]:
                    return
                facts[value] = result
                if result != baseline:
                    narrowed.add(value)
                operation = value.owner
                kind = self.opcode(operation) if hasattr(operation, "name") else None
                if kind in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast") or (
                        kind == "arith.bitcast" and value.type == operation.operands[0].type):
                    # The original-definition audit proves this cast preserves
                    # its mathematical value before any branch fact is used.
                    refine(operation.operands[0], result)

            for value, predicate, word, width in branch_constraints(owner.operands[0], selected):
                try:
                    baseline = self.bounds(value, parent)
                    interval = signed_comparison_enclosure(baseline, predicate, word, width)
                    if interval is not None:
                        refine(value, interval)
                except ConversionError:
                    # Optional facts cannot hide an unproved earlier definition.
                    # A dependent address still fails its ordinary audit.
                    continue
            if narrowed:
                self.proofs.append((owner, len(narrowed)))
        if owner is not None and owner.name == "scf.for" and block is owner.body.block:
            try:
                require_signed_loop(owner)
                limits = [self.bounds(value, parent) for value in owner.operands[:3]]
                width = getattr(getattr(block.args[0].type, "width", None), "data", 64)
                domain = enclosed_loop(limits, width=width)
            except ConversionError:
                # No optional reachability fact. Actual IV/address uses still
                # require their ordinary source-width/control proof below.
                self.cache[block] = facts
                return facts
            if domain.lower_interval[0] >= 0:
                selected, narrowed = set(), set()

                def positive(value):
                    if value in selected:
                        return
                    low, high = self.bounds(value, parent)
                    if low < 0:
                        return
                    if high < 1:
                        raise ConversionError("source loop body has no reachable positive upper bound", opcode="scf.for")
                    facts[value] = (max(low, 1), high)
                    if low == 0:
                        narrowed.add(value)
                    selected.add(value)
                    op = value.owner
                    kind = self.opcode(op) if hasattr(op, "name") else None
                    if kind == "arith.muli" and all(self.bounds(arg, parent)[0] >= 0 for arg in op.operands):
                        for argument in op.operands:
                            positive(argument)
                    elif kind in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast") or (
                            kind == "arith.bitcast" and value.type == op.operands[0].type):
                        # The definition audit already checked that this cast
                        # preserves the source mathematical value and width.
                        positive(op.operands[0])

                positive(owner.operands[1])
                if narrowed:
                    self.proofs.append((owner, len(narrowed)))
        self.cache[block] = facts
        return facts


def definition_scope(value):
    owner = value.owner
    return owner.parent if hasattr(owner, "name") else owner

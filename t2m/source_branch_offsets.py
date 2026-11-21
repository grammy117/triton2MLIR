"""Expose actual selected pointer offsets and common integer axis factors.

Reuse the checked uniform forms and shared pointer/SCF conversion. Original
regions and operations stay in place: only their actual yield interface changes.
"""

from functools import reduce
from math import gcd

from xdsl.dialects import arith, builtin, scf
from xdsl.rewriter import Rewriter

from .runtime_pointer_recurrences import Form, uniform_form


def expose_branch_offsets(module, ranges, source_ids):
    from .shared_tensor import _pointer_type, opcode

    reports = []
    for op in reversed([node for node in module.walk() if isinstance(node, scf.IfOp)]):
        if len(op.regions) != 2 or any(len(r.blocks) != 1 or r.block.args for r in op.regions):
            continue
        blocks = [r.block for r in op.regions]
        if any(b.last_op is None or opcode(b.last_op) != "scf.yield" or len(b.last_op.operands) != len(op.results) for b in blocks):
            continue

        def pointer_form(value, owner_scope=op):
            total = Form()
            for _ in range(32):
                owner = value.owner
                if not hasattr(owner, "name") or not owner_scope.is_ancestor(owner) or opcode(owner) != "tt.addptr":
                    return value, total
                form = uniform_form(owner.operands[1], owner_scope, opcode, ranges, [256], allow_varying=True)
                if form is None:
                    return None
                total = total.add(form)
                value = owner.operands[0]
            return None

        plans = {}
        for position, result in enumerate(op.results):
            pointer = _pointer_type(result.type)
            if not pointer and (result not in ranges or not isinstance(result.type, (builtin.IntegerType, builtin.IndexType))):
                continue
            sources = [b.last_op.operands[position] for b in blocks]
            if pointer:
                forms = [pointer_form(value) for value in sources]
                if any(row is None for row in forms) or forms[0][0] is not forms[1][0]:
                    continue
                base = forms[0][0]
                if op.is_ancestor(base.owner):
                    continue
                forms = [row[1] for row in forms]
            else:
                base = None
                forms = [uniform_form(value, op, opcode, ranges, [256], allow_varying=True) for value in sources]
                if any(form is None for form in forms):
                    continue
            factor = reduce(gcd, (abs(c) for form in forms for c in (form.constant, *form.terms.values())), 0) or 1
            type_ = builtin.i64 if pointer else result.type
            width = getattr(getattr(type_, "width", None), "data", 64)
            if factor >= 1 << (width-1) or not pointer and factor == 1:
                continue
            intervals, valid = [], True
            for form in forms:
                lo = hi = form.constant // factor
                if not -(1 << (width-1)) <= lo <= hi < 1 << (width-1):
                    valid = False
                for atom, coefficient in form.terms.items():
                    amount = coefficient // factor
                    a, b = sorted(n * amount for n in ranges[atom])
                    lo, hi = lo+a, hi+b
                    if not all(-(1 << (width-1)) <= n < 1 << (width-1) for n in (amount, a, b, lo, hi)):
                        valid = False
                if not all(-(1 << (width-1)) <= n < 1 << (width-1) for n in (lo * factor, hi * factor)):
                    valid = False
                intervals.append((lo, hi))
            if valid:
                plans[position] = base, factor, forms, type_, (min(lo for lo, hi in intervals), max(hi for lo, hi in intervals))
        if not plans:
            continue

        def emit(operations, node, location=op.location):
            node.location = location
            operations.append(node)
            return node.results[0]

        def constant(operations, number, type_):
            return emit(operations, arith.ConstantOp(builtin.IntegerAttr(number, type_)))

        def wide(operations, value):
            if isinstance(value.type, builtin.IndexType):
                return emit(operations, arith.IndexCastOp(value, builtin.i64))
            return value if value.type == builtin.i64 else emit(operations, arith.ExtSIOp(value, builtin.i64))

        for arm, block in enumerate(blocks):
            edge, values = [], list(block.last_op.operands)
            for position, (base, factor, forms, type_, _interval) in plans.items():
                if base is None:
                    # Exact divisibility follows from the checked uniform form.
                    values[position] = emit(edge, arith.DivSIOp(values[position], constant(edge, factor, type_)))
                else:
                    form = forms[arm]
                    value = constant(edge, form.constant // factor, type_)
                    for atom, coefficient in form.terms.items():
                        term = wide(edge, atom)
                        if coefficient != factor:
                            term = emit(edge, arith.MuliOp(term, constant(edge, coefficient // factor, type_)))
                        value = emit(edge, arith.AddiOp(value, term))
                    values[position] = value
            block.insert_ops_before(edge, block.last_op)
            block.last_op.operands = values
        regions = [op.detach_region(r) for r in tuple(op.regions)]
        types = [plans[i][3] if i in plans else result.type for i, result in enumerate(op.results)]
        branch = scf.IfOp(op.cond, types, *regions)
        branch.location = op.location
        branch.attributes.update(op.attributes)
        after, results = [], list(branch.results)
        for position, (base, factor, _forms, type_, interval) in plans.items():
            ranges[branch.results[position]] = interval
            value = branch.results[position]
            if factor != 1:
                value = emit(after, arith.MuliOp(value, constant(after, factor, type_)))
            if base is not None:
                if isinstance(base.type, builtin.TensorType):
                    value = emit(after, builtin.UnregisteredOp.with_name("tt.splat").create(
                        operands=[value], result_types=[builtin.TensorType(builtin.i64, base.type.get_shape())]))
                value = emit(after, builtin.UnregisteredOp.with_name("tt.addptr").create(operands=[base, value], result_types=[base.type]))
            else:
                ranges[value] = ranges[op.results[position]]
            results[position] = value
        report = {"kind": "source-branch-logical-offsets", "source_ids": source_ids(op.location),
            "results": [{"position": i, "representation": "same-SSA-base-and-selected-offset" if base is not None else "exact-selected-integer-axis-factor",
                         "logical_unit_factor": factor} for i, (base, factor, _forms, _type, _bounds) in plans.items()],
            "evaluation": "original-regions-and-memory-positions-retained; selected-yields-only; no-speculation",
            "scope": "source-width-checked-actual-branch-edges; no-invariance-ownership-or-effect-grant"}
        Rewriter.replace_op(op, [branch, *after], results)
        reports.append(report)
    return reports

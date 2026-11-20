"""Recover affine pointer streams before native structured conversion.

Meta/shared 17f0c234 and historical Microsoft 050b93ae carry pointer offsets
through SCF edges in PtrAnalysis::rewriteForOp. Some installed versions cannot
recover final pointer results. For an additive stream proved to advance by the
actual loop step times a constant plus a constant per iteration, use the actual
IV and an executable iteration ordinal. Numeric state, memory events and the
original loop remain; count enclosures never replace an executed recurrence.
"""

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter


def _identity(value):
    while getattr(value.owner, "name", None) == "arith.bitcast" and value.type == value.owner.operands[0].type:
        value = value.owner.operands[0]
    return value


def _step_form(value, step, opcode, budget):
    """Match actual SSA, never equal intervals or an advisory provenance tag."""
    budget[0] -= 1
    if budget[0] < 0:
        return None
    value = _identity(value)
    if value is _identity(step):
        return 1, 0
    op = value.owner
    name = opcode(op) if hasattr(op, "name") else None
    if name == "arith.constant":
        attribute = op.properties.get("value")
        if isinstance(attribute, builtin.IntegerAttr):
            return 0, attribute.value.data
        if isinstance(attribute, builtin.DenseIntOrFPElementsAttr):
            values = attribute.get_values()
            if values and len(set(values)) == 1:
                return 0, values[0]
    if name in {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans", "tt.reshape", "arith.extsi"}:
        return _step_form(op.operands[0], step, opcode, budget)
    if name in {"arith.addi", "arith.subi", "arith.muli"}:
        a, b = (_step_form(v, step, opcode, budget) for v in op.operands)
        if a is None or b is None:
            return None
        if name == "arith.addi":
            return a[0] + b[0], a[1] + b[1]
        if name == "arith.subi":
            return a[0] - b[0], a[1] - b[1]
        if a[0] == 0:
            return b[0] * a[1], b[1] * a[1]
        if b[0] == 0:
            return a[0] * b[1], a[1] * b[1]
    return None


def _increment(value, argument, step, opcode):
    coefficient, per_iteration = 0, 0
    for _ in range(32):
        if value is argument:
            return coefficient, per_iteration
        op = value.owner
        if (not hasattr(op, "name") or opcode(op) != "tt.addptr"
                or value.type != argument.type or op.operands[0].type != argument.type):
            return None
        form = _step_form(op.operands[1], step, opcode, [256])
        if form is None:
            return None
        coefficient += form[0]
        per_iteration += form[1]
        value = op.operands[0]
    return None


def raise_pointer_recurrences(module, source_ids, loop_domains, *, scalar_ranges=None):
    from .shared_tensor import _pointer_type, opcode

    reports = []
    for loop in reversed([op for op in module.walk() if isinstance(op, scf.ForOp)]):
        domain = loop_domains.get(loop)
        if (domain is None or domain.maximum_iterations == 0
                or not isinstance(loop.lb.type, builtin.IntegerType) or loop.lb.type.width.data > 64):
            continue
        source = loop.body.block
        pointer_slots = [i for i, arg in enumerate(source.args[1:]) if _pointer_type(arg.type)]
        if not pointer_slots:
            continue
        coefficients = {i: _increment(source.last_op.operands[i], source.args[i + 1], loop.step, opcode)
                        for i in pointer_slots}
        if any(value is None for value in coefficients.values()):
            if scalar_ranges:
                from .runtime_pointer_recurrences import raise_runtime_loop
                report = raise_runtime_loop(loop, domain, source_ids, scalar_ranges)
                if report is not None:
                    reports.append(report)
            continue
        if domain.exact:
            continue
        ordinal = any(increment for _coefficient, increment in coefficients.values())
        if ordinal and domain.maximum_iterations >= 1 << 63:
            continue
        # The mathematical difference is newly emitted as i64. Require its
        # complete enclosure and each scaled difference to fit that type.
        from .loop_domains import exit_induction_bounds
        exit_low, exit_high = exit_induction_bounds(domain)
        delta_low, delta_high = exit_low - domain.lower_interval[1], exit_high - domain.lower_interval[0]
        if any(not -(1 << 63) <= value < 1 << 63 for pair in coefficients.values() for value in pair):
            continue
        spans = []
        for coefficient, increment in coefficients.values():
            low, high = sorted((delta_low * coefficient, delta_high * coefficient))
            end = domain.maximum_iterations * increment
            spans.extend((low, high, end, low + min(0, end), high + max(0, end)))
            if coefficient:
                spans.extend((delta_low, delta_high))
        if any(not -(1 << 63) <= value < 1 << 63 for value in spans):
            continue
        kept = [i for i in range(len(loop.iter_args)) if i not in pointer_slots]
        body = Block(arg_types=[source.args[0].type, *[source.args[i + 1].type for i in kept], loop.lb.type,
                               *([builtin.i64] if ordinal else [])])
        mapping = {source.args[0]: body.args[0]}
        mapping.update((source.args[i + 1], body.args[j + 1]) for j, i in enumerate(kept))

        def emit(operations, op, *, location=loop.location):
            op.location = location
            operations.append(op)
            return op.results[0]

        def pointer_at(operations, initial, current, terms, ordinal_value, *, lower=loop.lb, emit=emit):
            coefficient, increment = terms
            if coefficient == increment == 0:
                return initial
            def wide(operand):
                return operand if operand.type == builtin.i64 else emit(operations, arith.ExtSIOp(operand, builtin.i64))
            def scale(operand, amount):
                if amount == 1:
                    return operand
                factor = emit(operations, arith.ConstantOp.from_int_and_width(amount, 64))
                return emit(operations, arith.MuliOp(operand, factor))
            difference = None
            if coefficient:
                difference = scale(emit(operations, arith.SubiOp(wide(current), wide(lower))), coefficient)
            if increment:
                term = scale(ordinal_value, increment)
                difference = term if difference is None else emit(operations, arith.AddiOp(difference, term))
            if isinstance(initial.type, builtin.TensorType):
                type_ = builtin.TensorType(builtin.i64, initial.type.get_shape())
                difference = emit(operations, builtin.UnregisteredOp.with_name("tt.splat").create(
                    operands=[difference], result_types=[type_]))
            return emit(operations, builtin.UnregisteredOp.with_name("tt.addptr").create(
                operands=[initial, difference], result_types=[initial.type]))

        prefix = []
        for i in pointer_slots:
            mapping[source.args[i + 1]] = pointer_at(prefix, loop.iter_args[i], body.args[0], coefficients[i],
                                                   body.args[-1] if ordinal else None)
        body.add_ops(prefix)
        for operation in source.ops:
            if operation is not source.last_op:
                body.add_op(operation.clone(value_mapper=mapping))
        next_iv = arith.AddiOp(body.args[0], loop.step)
        next_iv.location = loop.location
        body.add_op(next_iv)
        extra_initial, extra_yield = [], []
        before = []
        if ordinal:
            extra_initial.append(emit(before, arith.ConstantOp.from_int_and_width(0, 64)))
            one = arith.ConstantOp.from_int_and_width(1, 64)
            next_ordinal = arith.AddiOp(body.args[-1], one.result)
            one.location = next_ordinal.location = loop.location
            body.add_ops([one, next_ordinal])
            extra_yield.append(next_ordinal.result)
        yielded = source.last_op.operands
        end = scf.YieldOp(*[mapping.get(yielded[i], yielded[i]) for i in kept], next_iv.result, *extra_yield)
        end.location = source.last_op.location
        body.add_op(end)
        replacement = scf.ForOp(loop.lb, loop.ub, loop.step, [*[loop.iter_args[i] for i in kept], loop.lb, *extra_initial], Region(body))
        replacement.location = loop.location
        replacement.attributes.update(loop.attributes)
        after, results = [], {}
        results.update((i, replacement.results[j]) for j, i in enumerate(kept))
        for i in pointer_slots:
            results[i] = pointer_at(after, loop.iter_args[i], replacement.results[len(kept)], coefficients[i],
                                   replacement.results[-1] if ordinal else None)
        Rewriter.replace_op(loop, [*before, replacement, *after], [results[i] for i in range(len(loop.results))])
        reports.append({"kind": "source-ordinal-pointer-recurrences" if ordinal else "source-step-coupled-pointer-recurrences", "source_ids": source_ids(loop.location),
            "streams": [{"carried_argument": i, "step_coefficient": coefficients[i][0],
                         **({"iteration_increment": coefficients[i][1]} if ordinal else {}),
                         "update_source_ids": source_ids(getattr(yielded[i].owner, "location", loop.location))} for i in pointer_slots],
            "control": ("original-lower-upper-step-SSA; actual-zero-plus-one-iteration-carrier" if ordinal else
                        "original-lower-upper-step-SSA; actual-IV-plus-step-exit-carrier"),
            "zero_trips": "original-initial-pointers-and-complete-numeric-tuple",
            "scope": "source-width-checked-additive-streams; no-ownership-effect-or-complete-write-grant"})
    return reports

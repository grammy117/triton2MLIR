"""Carry checked bounded logical offsets instead of pointer-valued SCF state.

Native shared pointer conversion remains responsible for memory and tiling.
Source constant factors keep progress in logical axis units. No arithmetic is
hoisted: expressions defined in the body are used at its actual yield edge.
"""

from dataclasses import dataclass, field
from functools import reduce
from math import gcd

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .invariant_recurrences import is_loop_invariant, recurrence_bounds


@dataclass
class Form:
    constant: int = 0
    terms: dict = field(default_factory=dict)

    def scale(self, factor):
        return Form(self.constant * factor, {v: c * factor for v, c in self.terms.items() if c * factor})

    def add(self, other):
        terms = dict(self.terms)
        for value, coefficient in other.terms.items():
            terms[value] = terms.get(value, 0) + coefficient
        return Form(self.constant + other.constant, {v: c for v, c in terms.items() if c})


def uniform_form(value, loop, opcode, ranges, budget, *, allow_varying=False):
    budget[0] -= 1
    if budget[0] < 0:
        return None
    op = value.owner
    kind = opcode(op) if hasattr(op, "name") else None
    if kind == "arith.constant":
        attribute = op.properties.get("value")
        if isinstance(attribute, builtin.IntegerAttr):
            return Form(attribute.value.data)
        if isinstance(attribute, builtin.DenseIntOrFPElementsAttr):
            values = attribute.get_values()
            if values and isinstance(values[0], int) and len(set(values)) == 1:
                return Form(values[0])
    if (kind in {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans", "tt.reshape", "arith.extsi"}
            or kind == "arith.bitcast" and value.type == op.operands[0].type):
        return uniform_form(op.operands[0], loop, opcode, ranges, budget, allow_varying=allow_varying)
    if kind in {"arith.addi", "arith.subi", "arith.muli"}:
        a, b = (uniform_form(v, loop, opcode, ranges, budget, allow_varying=allow_varying) for v in op.operands)
        if a is not None and b is not None:
            if kind == "arith.addi":
                return a.add(b)
            if kind == "arith.subi":
                return a.add(b.scale(-1))
            if not a.terms:
                return b.scale(a.constant)
            if not b.terms:
                return a.scale(b.constant)
    if (value in ranges and isinstance(value.type, (builtin.IntegerType, builtin.IndexType))
            and (allow_varying or is_loop_invariant(value, loop, opcode))):
        return Form(terms={value: 1})
    return None


def raise_runtime_loop(loop, domain, source_ids, ranges):
    from .shared_tensor import _pointer_type, opcode
    from .source_pointer_recurrences import _identity

    source = loop.body.block
    pointer_slots = [i for i, v in enumerate(source.args[1:]) if _pointer_type(v.type)]
    step = _identity(loop.step)
    plans = {}
    for position in pointer_slots:
        value, argument, total = source.last_op.operands[position], source.args[position + 1], Form()
        for _ in range(32):
            if value is argument:
                break
            op = value.owner
            if (not hasattr(op, "name") or opcode(op) != "tt.addptr"
                    or value.type != argument.type or op.operands[0].type != argument.type):
                return None
            form = uniform_form(op.operands[1], loop, opcode, ranges, [256], allow_varying=True)
            if form is None:
                return None
            total = total.add(form)
            value = op.operands[0]
        else:
            return None
        coefficient = total.terms.pop(step, 0)
        factor = reduce(gcd, (abs(c) for c in (total.constant, *total.terms.values())), 0) or 1
        normalized = Form(total.constant // factor, {v: c // factor for v, c in total.terms.items()})
        low = high = normalized.constant
        # Validate the exact construction order, including each new multiply
        # and partial sum, rather than only its cancelled final expression.
        for atom, amount in normalized.terms.items():
            if atom not in ranges:
                return None
            a, b = sorted(value * amount for value in ranges[atom])
            low, high = low + a, high + b
            if not all(-(1 << 63) <= n < 1 << 63 for n in (amount, a, b, low, high)):
                return None
        try:
            body_bounds, final_bounds = recurrence_bounds(domain, (0, 0), (low, high), width=64)
        except ConversionError:
            return None
        from .loop_domains import exit_induction_bounds
        exit_low, exit_high = exit_induction_bounds(domain)
        diff = exit_low - domain.lower_interval[1], exit_high - domain.lower_interval[0]
        scaled = sorted(n * coefficient for n in diff)
        progress = min(0, *body_bounds, *final_bounds) * factor, max(0, *body_bounds, *final_bounds) * factor
        checks = [factor, coefficient, normalized.constant, *scaled, *progress,
                  scaled[0] + progress[0], scaled[1] + progress[1]]
        if coefficient:
            checks.extend(diff)
        if any(not -(1 << 63) <= n < 1 << 63 for n in checks):
            return None
        plans[position] = coefficient, factor, normalized
    if not any(form.terms for _a, _g, form in plans.values()):
        return None
    varying = any(not is_loop_invariant(atom, loop, opcode) for _a, _factor, form in plans.values() for atom in form.terms)

    kept = [i for i in range(len(loop.iter_args)) if i not in pointer_slots]
    body = Block(arg_types=[source.args[0].type, *[source.args[i+1].type for i in kept], loop.lb.type,
                           *([builtin.i64] * len(pointer_slots))])
    mapping = {source.args[0]: body.args[0]}
    mapping.update((source.args[i+1], body.args[j+1]) for j, i in enumerate(kept))
    progress_args = dict(zip(pointer_slots, body.args[len(kept)+2:], strict=True))

    def emit(operations, op):
        op.location = loop.location
        operations.append(op)
        return op.results[0]

    def constant(operations, value):
        return emit(operations, arith.ConstantOp.from_int_and_width(value, 64))

    def wide(operations, value):
        if isinstance(value.type, builtin.IndexType):
            return emit(operations, arith.IndexCastOp(value, builtin.i64))
        return value if value.type == builtin.i64 else emit(operations, arith.ExtSIOp(value, builtin.i64))

    def scale(operations, value, amount):
        return value if amount == 1 else emit(operations, arith.MuliOp(value, constant(operations, amount)))

    def pointer_at(operations, initial, current, progress, coefficient, factor):
        offset = scale(operations, progress, factor)
        if coefficient:
            difference = emit(operations, arith.SubiOp(wide(operations, current), wide(operations, loop.lb)))
            offset = emit(operations, arith.AddiOp(scale(operations, difference, coefficient), offset))
        if isinstance(initial.type, builtin.TensorType):
            offset = emit(operations, builtin.UnregisteredOp.with_name("tt.splat").create(
                operands=[offset], result_types=[builtin.TensorType(builtin.i64, initial.type.get_shape())]))
        return emit(operations, builtin.UnregisteredOp.with_name("tt.addptr").create(
            operands=[initial, offset], result_types=[initial.type]))

    prefix = []
    for position, (coefficient, factor, _form) in plans.items():
        mapping[source.args[position+1]] = pointer_at(prefix, loop.iter_args[position], body.args[0],
            progress_args[position], coefficient, factor)
    body.add_ops(prefix)
    for op in source.ops:
        if op is not source.last_op:
            body.add_op(op.clone(value_mapper=mapping))
    edge = []
    next_iv = emit(edge, arith.AddiOp(body.args[0], loop.step))
    next_progress = []
    for position, (_coefficient, _factor, form) in plans.items():
        increment = constant(edge, form.constant)
        for atom, amount in form.terms.items():
            term = scale(edge, wide(edge, mapping.get(atom, atom)), amount)
            increment = emit(edge, arith.AddiOp(increment, term))
        next_progress.append(emit(edge, arith.AddiOp(progress_args[position], increment)))
    body.add_ops(edge)
    end = scf.YieldOp(*[mapping.get(source.last_op.operands[i], source.last_op.operands[i]) for i in kept], next_iv, *next_progress)
    end.location = source.last_op.location
    body.add_op(end)
    before = []
    zeros = [constant(before, 0) for _ in pointer_slots]
    replacement = scf.ForOp(loop.lb, loop.ub, loop.step, [*[loop.iter_args[i] for i in kept], loop.lb, *zeros], Region(body))
    replacement.location = loop.location
    replacement.attributes.update(loop.attributes)
    results = {i: replacement.results[j] for j, i in enumerate(kept)}
    after = []
    for j, (position, (coefficient, factor, _form)) in enumerate(plans.items()):
        results[position] = pointer_at(after, loop.iter_args[position], replacement.results[len(kept)],
            replacement.results[len(kept)+1+j], coefficient, factor)
    Rewriter.replace_op(loop, [*before, replacement, *after], [results[i] for i in range(len(loop.results))])
    return {"kind": "source-bounded-runtime-pointer-recurrences" if varying else "source-invariant-runtime-pointer-recurrences", "source_ids": source_ids(loop.location),
        "streams": [{"carried_argument": i, "step_coefficient": a, "logical_unit_factor": factor,
                     "bounded_scalar_terms" if varying else "invariant_scalar_terms": len(form.terms)}
                    for i, (a, factor, form) in plans.items()],
        "control": "original-SSA; actual-zero-initialized-bounded-increment-progress",
        "evaluation": "original-body-definitions-retained; no-hoisting-or-zero-trip-speculation",
        "scope": "source-width-checked-logical-offsets; no-ownership-complete-write-or-effect-grant"}

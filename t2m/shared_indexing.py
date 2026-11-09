"""Small symbolic facts for logical-program ownership; no element enumeration."""

from dataclasses import dataclass

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, TensorType

from .diagnostics import ConversionError


@dataclass(frozen=True)
class Linear:
    constant: int = 0
    programs: tuple[int, ...] = (0, 0, 0)
    # (identity, lower, upper, coefficient) for bounded source loop IVs.
    loops: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def dynamic(self):
        return any(self.programs) or bool(self.loops)

    def __add__(self, other):
        loops = {identity: (low, high, coefficient) for identity, low, high, coefficient in self.loops}
        for identity, low, high, coefficient in other.loops:
            if identity in loops:
                old_low, old_high, old_coefficient = loops[identity]
                if (low, high) != (old_low, old_high):
                    raise ConversionError("inconsistent source loop range")
                coefficient += old_coefficient
            loops[identity] = low, high, coefficient
        return Linear(self.constant + other.constant,
                      tuple(a + b for a, b in zip(self.programs, other.programs, strict=True)),
                      tuple((i, lo, hi, c) for i, (lo, hi, c) in sorted(loops.items()) if c))

    def scale(self, amount):
        return Linear(self.constant * amount, tuple(a * amount for a in self.programs),
                      tuple((i, lo, hi, c * amount) for i, lo, hi, c in self.loops if c * amount))

    def divide(self, amount):
        if amount <= 0 or any(n % amount for n in (self.constant, *self.programs, *(c for _, _, _, c in self.loops))):
            raise ConversionError("element offsets do not align with declared logical axes")
        return Linear(self.constant // amount, tuple(n // amount for n in self.programs),
                      tuple((i, lo, hi, c // amount) for i, lo, hi, c in self.loops))

    def bounds(self, grid):
        grid = (*grid, *((1,) * (3 - len(grid))))
        spans = [coefficient * (size - 1) for coefficient, size in zip(self.programs, grid, strict=True)]
        return (self.constant + sum(min(0, n) for n in spans) + sum(min(lo * c, hi * c) for _, lo, hi, c in self.loops),
                self.constant + sum(max(0, n) for n in spans) + sum(max(lo * c, hi * c) for _, lo, hi, c in self.loops))


def linear(value, facts, grid):
    if value in facts:
        return facts[value]
    op = value.owner
    if not hasattr(op, "name"):
        raise ConversionError("memory index has an unproved loop/block recurrence")
    name = op.name
    if name == "arith.constant" and isinstance(op.value, IntegerAttr):
        result = Linear(op.value.value.data)
    elif name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
        result = linear(op.operands[0], facts, grid)
        if name == "arith.extui" and result.bounds(grid)[0] < 0:
            raise ConversionError("unsigned address extension requires a nonnegative source")
    elif name in ("arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui"):
        a, b = (linear(v, facts, grid) for v in op.operands)
        if name == "arith.addi":
            result = a + b
        elif name == "arith.subi":
            result = a + b.scale(-1)
        elif name == "arith.muli" and not b.dynamic:
            result = a.scale(b.constant)
        elif name == "arith.muli" and not a.dynamic:
            result = b.scale(a.constant)
        elif name in ("arith.divsi", "arith.divui") and not b.dynamic:
            result = a.divide(b.constant)
        else:
            raise ConversionError("nonlinear program indexing needs a separate ownership proof")
    else:
        raise ConversionError("memory index is not a qualified affine program expression", opcode=name)
    low, high = result.bounds(grid)
    type_ = value.type.get_element_type() if isinstance(value.type, TensorType) else value.type
    width = getattr(getattr(type_, "width", None), "data", 64)
    if not -(1 << (width - 1)) <= low <= high < 1 << (width - 1):
        raise ConversionError("logical address arithmetic may overflow")
    facts[value] = result
    return result


def independent_tile(offsets, sizes, grid):
    """Each nontrivial grid axis owns a distinct tensor axis with a tile stride."""
    if any(off.loops for off in offsets):
        raise ConversionError("parallel store coordinates depend on an unfunctionalized memory loop")
    used = set()
    for axis, programs in enumerate(grid):
        if programs == 1:
            continue
        candidates = [i for i, (off, size) in enumerate(zip(offsets, sizes, strict=True))
                      if size > 0 and abs(off.programs[axis]) >= size
                      and all(c == 0 or j == axis or j >= len(grid) or grid[j] == 1
                              for j, c in enumerate(off.programs))]
        match = next((i for i in candidates if i not in used), None)
        if match is None:
            raise ConversionError("parallel write ownership is not proved; refusing to serialize conflicting programs")
        used.add(match)


def seed_recurrences(operation, arguments, mapping, facts, grid, lower, upper, step, integer):
    """Reuse upstream's pointer-to-index recurrences, retaining source loop SSA.

    Only invariant affine increments are established. Other carried values stay
    ordinary computation; using an unproved recurrence as an address still fails.
    """
    iterations = (upper - lower + step - 1) // step
    ordinal = (facts[arguments[0]] + Linear(-lower)).divide(step)
    yielded = operation.body.block.last_op.operands
    finals, evidence = {}, []
    for position, (old, target, initial, result) in enumerate(zip(operation.body.block.args[1:], arguments[1:],
                                                               operation.operands[3:], yielded, strict=True)):
        if str(old.type) != "index" and not hasattr(old.type, "width"):
            continue
        delta = None
        if result is old:
            delta = 0
        else:
            producer = result.owner
            if hasattr(producer, "name") and producer.name in ("arith.addi", "arith.subi"):
                a, b = producer.operands
                try:
                    if a is old:
                        delta = integer(mapping.get(b, b)) * (-1 if producer.name == "arith.subi" else 1)
                    elif b is old and producer.name == "arith.addi":
                        delta = integer(mapping.get(a, a))
                except ConversionError:
                    pass
        if delta is None:
            continue
        try:
            start = linear(mapping.get(initial, initial), facts, grid)
        except ConversionError:
            continue
        current, final = start + ordinal.scale(delta), start + Linear(iterations * delta)
        width = getattr(getattr(old.type, "width", None), "data", 64)
        for form in (current, final):
            low, high = form.bounds(grid)
            if not -(1 << (width - 1)) <= low <= high < 1 << (width - 1):
                raise ConversionError("loop-carried address recurrence may overflow")
        facts[target] = current
        finals[position] = final
        evidence.append({"carried_argument": position, "increment": delta, "iterations": iterations})
    return finals, evidence


def logical_coordinates(total, strides, grid):
    """Split a flat affine base offset into bounded row-major coordinates.

    Source pointer increments can land in a different dimension than the offset
    slot chosen by pointer analysis. Exact quotient terms and bounded remainders
    recover that structure without enumerating programs or loop iterations.
    """
    low, high = total.bounds(grid)
    if not -(1 << 63) <= low <= high < 1 << 63:
        raise ConversionError("combined pointer offsets may overflow the logical index type")
    result, remainder = [], total
    for stride in strides:
        quotient = Linear(remainder.constant // stride,
                          tuple(c // stride if c % stride == 0 else 0 for c in remainder.programs),
                          tuple((i, lo, hi, c // stride) for i, lo, hi, c in remainder.loops if c % stride == 0))
        remainder = remainder + quotient.scale(-stride)
        low, high = remainder.bounds(grid)
        if low < 0 or high >= stride:
            raise ConversionError("flat pointer recurrence needs a non-affine coordinate/range adapter")
        result.append(quotient)
    return tuple(result)


def check_source_addresses(module, abi):
    """Check source integer widths before upstream re-raises them to index.

    Checking only recovered index arithmetic would miss an i32 overflow that
    the source performed before pointer analysis promoted the expression.
    """
    def name(op):
        return op.op_name.data if op.name == "builtin.unregistered" else op.name

    function = next(op for op in module.ops if name(op) == "tt.func")
    facts = {v: (n, n) for v, n in zip(function.regions[0].block.args[len(abi.buffers):], abi.scalar_constants, strict=True)}
    pending = set()

    def bounds(value):
        if value in facts:
            return facts[value]
        op = value.owner
        if not hasattr(op, "name"):
            parent = op.parent_op()
            if parent is None or parent.name != "scf.for":
                raise ConversionError("source memory index needs a loop/runtime range proof")
            limits = [bounds(v) for v in parent.operands[:3]]
            if any(lo != hi for lo, hi in limits):
                raise ConversionError("source memory loop bounds need static specialization")
            lb, ub, step = (lo for lo, _ in limits)
            if step <= 0 or ub <= lb:
                raise ConversionError("source memory loop needs a positive nonempty iteration domain")
            iterations = (ub - lb + step - 1) // step
            if value.index == 0:
                result = lb, lb + (iterations - 1) * step
            else:
                if value in pending:
                    raise ConversionError("source memory recurrence has cyclic/unproved increments")
                pending.add(value)
                initial = bounds(parent.operands[value.index + 2])
                yielded = parent.body.block.last_op.operands[value.index - 1]
                delta = (0, 0)
                if yielded is not value:
                    update = yielded.owner
                    if getattr(update, "name", "") == "arith.addi" and value in update.operands:
                        delta = bounds(update.operands[1] if update.operands[0] is value else update.operands[0])
                    elif getattr(update, "name", "") == "arith.subi" and update.operands[0] is value:
                        lo, hi = bounds(update.operands[1])
                        delta = -hi, -lo
                    else:
                        raise ConversionError("source memory recurrence needs an invariant affine increment")
                if delta[0] != delta[1]:
                    raise ConversionError("source memory recurrence increment is not invariant")
                span = (iterations - 1) * delta[0]
                result = initial[0] + min(0, span), initial[1] + max(0, span)
                pending.remove(value)
            width = getattr(getattr(value.type, "width", None), "data", 64)
            if not -(1 << (width - 1)) <= result[0] <= result[1] < 1 << (width - 1):
                raise ConversionError("source address recurrence may overflow before upstream index promotion")
            facts[value] = result
            return result
        kind = name(op)
        if kind in ("tt.get_program_id", "tt.get_num_programs"):
            axis = op.properties["axis"].value.data
            size = abi.grid[axis] if axis < len(abi.grid) else 1
            low, high = (0, size - 1) if kind == "tt.get_program_id" else (size, size)
        elif kind == "tt.make_range":
            low, high = op.properties["start"].value.data, op.properties["end"].value.data - 1
        elif kind == "tt.load":
            # An integer read has its complete source-width interval. This
            # proves no address-arithmetic overflow by itself; later arithmetic
            # must still fit, and the tensor gather adapter checks live indices
            # against the actual logical allocation before extracting a value.
            type_ = value.type.get_element_type() if isinstance(value.type, TensorType) else value.type
            width = getattr(getattr(type_, "width", None), "data", None)
            if width not in (32, 64):
                raise ConversionError("loaded source indices require signed i32/i64 values", opcode=kind)
            low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        elif kind == "arith.constant":
            attr = op.properties["value"]
            if isinstance(attr, IntegerAttr):
                low = high = attr.value.data
            elif isinstance(attr, DenseIntOrFPElementsAttr):
                values = attr.get_values()
                low, high = min(values), max(values)
            else:
                raise ConversionError("memory offset constant must be integer")
        elif kind in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans", "tt.reshape",
                       "arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
            low, high = bounds(op.operands[0])
            if kind == "arith.extui" and low < 0:
                raise ConversionError("source unsigned address extension needs a nonnegative range")
        elif kind == "arith.select":
            a, b = (bounds(v) for v in op.operands[1:])
            low, high = min(a[0], b[0]), max(a[1], b[1])
        elif kind in ("arith.addi", "arith.subi", "arith.muli", "arith.minsi", "arith.maxsi"):
            a, b = (bounds(v) for v in op.operands)
            if kind == "arith.addi":
                low, high = a[0] + b[0], a[1] + b[1]
            elif kind == "arith.subi":
                low, high = a[0] - b[1], a[1] - b[0]
            elif kind == "arith.minsi":
                low, high = min(a[0], b[0]), min(a[1], b[1])
            elif kind == "arith.maxsi":
                low, high = max(a[0], b[0]), max(a[1], b[1])
            else:
                products = [x * y for x in a for y in b]
                low, high = min(products), max(products)
        else:
            raise ConversionError("source memory arithmetic needs an integer range proof", opcode=kind)
        type_ = value.type.get_element_type() if isinstance(value.type, TensorType) else value.type
        width = getattr(getattr(type_, "width", None), "data", 64)
        if not -(1 << (width - 1)) <= low <= high < 1 << (width - 1):
            raise ConversionError("source address arithmetic may overflow before upstream index promotion", opcode=kind)
        facts[value] = low, high
        return low, high

    for op in module.walk():
        if name(op) == "tt.addptr":
            bounds(op.operands[1])

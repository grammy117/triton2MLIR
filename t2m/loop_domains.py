"""Conservative serial iteration domains from checked scalar intervals.

Upstream owns SCF conversion and pointer/index decomposition. This layer proves
the source-width and logical-memory premises needed at the functional boundary;
it does not substitute a worst-case trip count for the executable runtime bound.
"""

from dataclasses import dataclass

from .diagnostics import ConversionError


def require_signed_loop(operation):
    if "unsignedCmp" in operation.attributes or "unsignedCmp" in operation.properties:
        raise ConversionError("unsigned memory-loop comparison needs its own source-width domain proof", opcode="scf.for")


@dataclass(frozen=True)
class LoopDomain:
    lower: int
    upper: tuple[int, int]
    step: int
    minimum_iterations: int
    maximum_iterations: int

    @property
    def variable_header(self):
        return False

    @property
    def lower_interval(self):
        return self.lower, self.lower

    @property
    def step_interval(self):
        return self.step, self.step

    @property
    def exact(self):
        return self.upper[0] == self.upper[1]

    @property
    def induction_bounds(self):
        if self.maximum_iterations == 0:
            raise ConversionError("empty source loop has no reachable induction value")
        return self.lower, self.lower + (self.maximum_iterations - 1) * self.step


def bounded_loop(limits, *, width=64, allow_empty=False):
    lower, upper, step = limits
    if lower[0] != lower[1] or step[0] != step[1] or step[0] <= 0:
        raise ConversionError("memory loop requires constant lower/positive step and a proved upper interval")
    lb, stride = lower[0], step[0]
    counts = tuple(max(0, (ub - lb + stride - 1) // stride) for ub in upper)
    if counts[1] == 0 and not allow_empty:
        raise ConversionError("memory loop has no proved reachable iteration; needs empty-region elimination")
    # Check the actual IV update after the last reachable body execution as
    # well as its body interval, before upstream promotes source indices.
    final = lb + counts[1] * stride
    if not -(1 << (width - 1)) <= lb <= final < 1 << (width - 1):
        raise ConversionError("source memory loop induction update may overflow")
    return LoopDomain(lb, tuple(upper), stride, *counts)


@dataclass(frozen=True)
class IntervalLoopDomain:
    lower_interval: tuple[int, int]
    upper: tuple[int, int]
    step_interval: tuple[int, int]
    minimum_iterations: int
    maximum_iterations: int

    @property
    def variable_header(self):
        return True

    @property
    def exact(self):
        return False

    @property
    def induction_bounds(self):
        if self.maximum_iterations == 0:
            raise ConversionError("empty source loop has no reachable induction value")
        return self.lower_interval[0], min(self.upper[1] - 1,
            self.lower_interval[1] + (self.maximum_iterations - 1) * self.step_interval[1])


def enclosed_loop(limits, *, width=64, allow_empty=False):
    """Enclose actual positive-step SSA loops without inventing an ordinal.

    LLVM IntegerRangeAnalysis uses the incoming lower/upper enclosures for IVs.
    Also check the source-width update after the final body, and require actual
    step positivity rather than assuming an undiscoverable step is positive.
    Counts supply no stride, congruence, complete-write or replacement bound.
    """
    lower, upper, step = (tuple(interval) for interval in limits)
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    if any(not minimum <= low <= high <= maximum for low, high in (lower, upper, step)):
        raise ConversionError("source loop bounds exceed their signed integer width")
    if lower[0] == lower[1] and step[0] == step[1]:
        return bounded_loop((lower, upper, step), width=width, allow_empty=allow_empty)
    if step[0] <= 0:
        raise ConversionError("memory loop requires a proved positive runtime step")
    counts = (max(0, (upper[0] - lower[1] + step[1] - 1) // step[1]),
              max(0, (upper[1] - lower[0] + step[0] - 1) // step[0]))
    if counts[1] == 0 and not allow_empty:
        raise ConversionError("memory loop has no proved reachable iteration; needs empty-region elimination")
    final = lower[1] if counts[1] == 0 else max(lower[1], min(
        lower[1] + counts[1] * step[1], upper[1] - 1 + step[1]))
    if final > maximum:
        raise ConversionError("source memory loop induction update may overflow")
    return IntervalLoopDomain(lower, upper, step, *counts)


def exit_induction_bounds(domain):
    """Enclose the actual exit IV, including its unchanged zero-trip value."""
    if not domain.variable_header:
        return (domain.lower + domain.minimum_iterations * domain.step,
                domain.lower + domain.maximum_iterations * domain.step)
    lower, upper, step = domain.lower_interval, domain.upper, domain.step_interval
    high = lower[1] if domain.maximum_iterations == 0 else max(lower[1], min(
        lower[1] + domain.maximum_iterations * step[1], upper[1] - 1 + step[1]))
    return lower[0], high


def is_exit_induction(operation, position):
    """Recognize executable SCF edges, without trusting a source annotation."""
    if operation.iter_args[position] is not operation.lb:
        return False
    value = operation.body.block.last_op.operands[position]
    update = value.owner
    return (getattr(update, "name", None) == "arith.addi"
            and (tuple(update.operands) == (operation.body.block.args[0], operation.step)
                 or tuple(update.operands) == (operation.step, operation.body.block.args[0])))


def is_iteration_ordinal(operation, position):
    """Recognize an actual zero-initialized, unit-increment scalar carrier."""
    from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType

    initial = operation.iter_args[position]
    if not isinstance(initial.type, (IntegerType, IndexType)):
        return False

    def constant(value, number):
        op = value.owner
        attribute = getattr(op, "properties", {}).get("value")
        return (getattr(op, "name", None) == "arith.constant" and isinstance(attribute, IntegerAttr)
                and attribute.value.data == number)

    if not constant(initial, 0):
        return False
    value = operation.body.block.last_op.operands[position]
    update = value.owner
    if getattr(update, "name", None) != "arith.addi":
        return False
    argument = operation.body.block.args[position + 1]
    a, b = update.operands
    return (a is argument and constant(b, 1)) or (b is argument and constant(a, 1))

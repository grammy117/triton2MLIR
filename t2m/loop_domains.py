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
    def exact(self):
        return self.upper[0] == self.upper[1]

    @property
    def induction_bounds(self):
        return self.lower, self.lower + (self.maximum_iterations - 1) * self.step


def bounded_loop(limits, *, width=64):
    lower, upper, step = limits
    if lower[0] != lower[1] or step[0] != step[1] or step[0] <= 0:
        raise ConversionError("memory loop requires constant lower/positive step and a proved upper interval")
    lb, stride = lower[0], step[0]
    counts = tuple(max(0, (ub - lb + stride - 1) // stride) for ub in upper)
    if counts[1] == 0:
        raise ConversionError("memory loop has no proved reachable iteration; needs empty-region elimination")
    # Check the actual IV update after the last reachable body execution as
    # well as its body interval, before upstream promotes source indices.
    final = lb + counts[1] * stride
    if not -(1 << (width - 1)) <= lb <= final < 1 << (width - 1):
        raise ConversionError("source memory loop induction update may overflow")
    return LoopDomain(lb, tuple(upper), stride, *counts)

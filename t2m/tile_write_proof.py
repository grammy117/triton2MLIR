"""Actual source masks and lexical SCF choices for ordered tile observations.

LLVM a47bddcc SCF.cpp selects exactly one IfOp successor. Boolean facts stay
inside those actual choices; later choices cannot initialize an earlier read.
No range, ownership, effect or caller-provenance premise is introduced.
"""

from dataclasses import dataclass

from xdsl.dialects import builtin, scf
from xdsl.ir import Operation

from .diagnostics import ConversionError
from .source_boolean import BooleanCoverage


def lexical_choices(operation):
    choices, child = [], operation
    while (parent := child.parent_op()) is not None:
        if isinstance(parent, scf.IfOp):
            choices.append((parent.cond, child.parent_region() is parent.true_region))
        child = parent
    return tuple(choices)


def candidates(value, type_, result, depth=0):
    if depth >= 128 or len(result) >= 128 or value in result:
        return
    if value.type == type_:
        result.append(value)
    owner = value.owner
    if isinstance(owner, Operation) and owner.name in ("arith.andi", "arith.ori", "arith.xori", "arith.select"):
        for operand in owner.operands:
            candidates(operand, type_, result, depth + 1)


@dataclass
class TileWriteProof:
    booleans: BooleanCoverage
    total: int | None
    premise: int | None
    candidates: list

    def matches(self, value, *, global_scope=False):
        expected = self.booleans.condition(value) if value is not None else 1
        if expected is None or self.total is None:
            return False
        if self.total == expected:
            return True
        premise = 1 if global_scope else self.premise
        return (self.booleans.covers([self.total], premise=self.booleans.combine("and", premise, expected))
                and self.booleans.covers([expected], premise=self.booleans.combine("and", premise, self.total)))


def analyze_writes(events, value_type, *, choices=None):
    from .shared_tensor import _array, opcode

    if not events or len(events) > 128:
        return None
    proof, total, premise, retained = BooleanCoverage(pointwise_tensors=True), 0, 1, []
    for condition, selected in lexical_choices(events[-1]) if choices is None else choices:
        fact = proof.condition(condition)
        premise = proof.combine("and", premise, fact if selected else proof.negate(fact))
    mask_type = builtin.TensorType(builtin.i1, value_type.get_shape())
    for op in events:
        if opcode(op) == "tt.store":
            if len(op.operands) not in (2, 3) or op.operands[1].type != value_type:
                return None
            if len(op.operands) == 3:
                if op.operands[2].type != mask_type:
                    return None
                candidates(op.operands[2], mask_type, retained)
            guard = proof.condition(op.operands[2]) if len(op.operands) == 3 else 1
            total = proof.combine("or", total, guard)
        elif opcode(op) == "tt.load":
            try:
                segments = _array(op, "operandSegmentSizes")
            except ConversionError:
                return None
            valid = segments == (1, 0, 0) or (segments == (1, 1, 1) and op.operands[1].type == mask_type
                                                           and op.operands[2].type == value_type)
            if not valid or len(op.results) != 1 or op.results[0].type != value_type:
                return None
            active = proof.condition(op.operands[1]) if len(op.operands) == 3 else 1
            if not proof.covers([total], premise=proof.combine("and", premise, active)):
                return None
        else:
            return None
    return TileWriteProof(proof, total, premise, retained)

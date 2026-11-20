"""Bounded Boolean coverage of actual scalar SSA conditions.

Reduced ordered decision diagrams preserve correlations between the same
source observations. No arithmetic range, memory-effect or provenance premise
is invented. Unknown producers remain distinct SSA atoms, including loads.
"""

from xdsl.dialects import arith, builtin, tensor


class _Budget(Exception):
    pass


class BooleanCoverage:
    def __init__(self, *, pointwise_tensors=False):
        self.nodes, self.unique, self.atoms = [None, None], {}, {}
        self.cache, self.operations = {}, {}
        self.pointwise_tensors = pointwise_tensors

    @staticmethod
    def scalar(value):
        if isinstance(value.type, builtin.TensorType):
            owner = value.owner
            return owner.operands[0] if isinstance(owner, tensor.FromElementsOp) and len(owner.operands) == 1 else None
        return value

    @staticmethod
    def _value_key(value):
        owner = value.owner
        if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr):
            word = owner.value.value.data
            if isinstance(value.type, builtin.IntegerType):
                word %= 1 << value.type.bitwidth
            return "literal", value.type, word
        return "ssa", value

    @staticmethod
    def _splat_bit(attribute):
        # xDSL stores i1 elements as one packed byte each. Check the actual
        # representation without unpacking a Python tuple for every lane.
        type_, data = attribute.get_element_type(), attribute.data.data
        if type_ == builtin.i1 and type_.compile_time_size == 1 and data and data.count(data[0]) == len(data):
            bit = type_.unpack(data[:1], 1)[0]
            if bit in (-1, 0, 1):
                return int(bool(bit))
        return None

    def _node(self, variable, low, high):
        if low == high:
            return low
        key = variable, low, high
        if key not in self.unique:
            if len(self.nodes) >= 4096:
                raise _Budget
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def _atom(self, key):
        if key not in self.atoms:
            if len(self.atoms) >= 128:
                raise _Budget
            self.atoms[key] = len(self.atoms)
        return self._node(self.atoms[key], 0, 1)

    def _apply(self, kind, left, right):
        if left > right:
            left, right = right, left
        if kind == "and":
            if left == 0 or left == right:
                return left
            if left == 1:
                return right
        elif kind == "or":
            if left == 0 or left == right:
                return right
            if left == 1:
                return 1
        elif kind == "xor":
            if left == 0:
                return right
            if left == right:
                return 0
        key = kind, left, right
        if key in self.operations:
            return self.operations[key]
        if len(self.operations) >= 16384:
            raise _Budget
        variable = min(self.nodes[value][0] for value in (left, right) if value > 1)

        def children(value):
            node = self.nodes[value] if value > 1 else None
            return node[1:] if node is not None and node[0] == variable else (value, value)

        a, b = children(left), children(right)
        result = self._node(variable, self._apply(kind, a[0], b[0]), self._apply(kind, a[1], b[1]))
        self.operations[key] = result
        return result

    def combine(self, kind, left, right):
        if kind == "and" and (left == 0 or right == 0):
            return 0
        if kind == "or" and (left == 1 or right == 1):
            return 1
        if left is None or right is None:
            return None
        try:
            return self._apply(kind, left, right)
        except _Budget:
            return None

    def negate(self, value):
        return self.combine("xor", value, 1)

    def condition(self, value, depth=0):
        if value is None or depth >= 128:
            return None
        if not self.pointwise_tensors:
            value = self.scalar(value)
        type_ = value.type if value is not None else None
        if self.pointwise_tensors and isinstance(type_, builtin.TensorType):
            type_ = type_.element_type
        if type_ != builtin.i1:
            return None
        if value in self.cache:
            return self.cache[value]
        self.cache[value] = None
        owner = value.owner
        try:
            if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr):
                result = int(bool(owner.value.value.data))
            elif (self.pointwise_tensors and isinstance(owner, arith.ConstantOp)
                  and isinstance(owner.value, builtin.DenseIntOrFPElementsAttr)):
                result = self._splat_bit(owner.value)
                if result is None:
                    result = self._atom(("ssa", value))
            elif (self.pointwise_tensors and getattr(getattr(owner, "op_name", None), "data", None) == "tt.splat"
                  and len(owner.operands) == 1):
                result = self.condition(owner.operands[0], depth + 1)
            elif isinstance(owner, (arith.AndIOp, arith.OrIOp, arith.XOrIOp)):
                kind = "and" if isinstance(owner, arith.AndIOp) else "or" if isinstance(owner, arith.OrIOp) else "xor"
                result = self.combine(kind, *(self.condition(v, depth + 1) for v in owner.operands))
            elif isinstance(owner, arith.SelectOp):
                test, a, b = (self.condition(v, depth + 1) for v in owner.operands)
                result = self.combine("or", self.combine("and", test, a), self.combine("and", self.negate(test), b))
            elif isinstance(owner, arith.CmpiOp):
                predicate = arith.CMPI_COMPARISON_OPERATIONS[owner.predicate.value.data]
                left, right = (self._value_key(v) for v in owner.operands)
                if predicate in ("eq", "ne"):
                    result = self._atom(("eq", frozenset((left, right))))
                    result = self.negate(result) if predicate == "ne" else result
                else:
                    reverse = predicate in ("sgt", "sle", "ugt", "ule")
                    left, right = (right, left) if reverse else (left, right)
                    result = self._atom(("ult" if predicate.startswith("u") else "slt", left, right))
                    result = self.negate(result) if predicate in ("sle", "sge", "ule", "uge") else result
            else:
                result = self._atom(("ssa", value))
        except _Budget:
            result = None
        self.cache[value] = result
        return result

    def covers(self, guards, *, premise=1):
        total = 0
        for guard in guards:
            total = self.combine("or", total, guard)
        return self.combine("or", self.negate(premise), total) == 1

"""Check current byte-word payloads against their standard executable recipe.

Compare effect/control skeletons and typed integer expression DAGs. Only exact
literal folding and neutral integer identities are normalized. Provenance is
not an input to this check; the endian flag must come from a private literal
probe whose current allocation, stores, view, load and assertion are checked.
"""

from xdsl.dialects import arith, builtin, cf, memref, scf
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block

from .diagnostics import ConversionError
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp


def refuse():
    raise ConversionError("byte atomic payload differs from its complete typed word/byte recipe")


def literal(value):
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr):
        return owner.value.value.data
    return None


def checked_probe(flag, word, generic):
    compare = flag.owner
    if not isinstance(compare, arith.CmpiOp) or compare.predicate.value.data != 0:
        refuse()
    loaded, one = compare.operands
    if literal(one) != 1 or not isinstance(loaded.owner, memref.LoadOp) or loaded.type != word:
        refuse()
    load = loaded.owner
    if len(load.indices) != 1 or literal(load.indices[0]) != 0:
        refuse()
    view = load.memref.owner
    if (not isinstance(view, memref.ViewOp) or view.result.type != builtin.MemRefType(word, [1])
            or literal(view.byte_shift) != 0 or view.sizes):
        refuse()
    allocation = view.source.owner
    size = word.bitwidth // 8
    if (not isinstance(allocation, memref.AllocOp) or allocation.memref.type != builtin.MemRefType(builtin.i8, [size])
            or allocation.alignment is None or allocation.alignment.value.data < size):
        refuse()
    users = {use.operation for use in allocation.memref.uses}
    stores = [op for op in users if isinstance(op, memref.StoreOp)]
    if users != {view, *stores} or len(stores) != size or {use.operation for use in view.result.uses} != {load}:
        refuse()
    block = allocation.parent
    sequence = list(block.ops)
    if any(op.parent is not block for op in (*stores, view, load, compare)):
        refuse()
    observed = {}
    for store in stores:
        index = literal(store.indices[0]) if len(store.indices) == 1 else None
        if index in observed or index is None or sequence.index(store) >= sequence.index(load):
            refuse()
        observed[index] = literal(store.value)
    if observed != {index: int(index == 0) for index in range(size)}:
        refuse()
    big = [use.operation for use in loaded.uses if isinstance(use.operation, arith.CmpiOp)
           and use.operation.predicate.value.data == 0 and use.operation.operands[0] is loaded
           and literal(use.operation.operands[1]) == 1 << (8 * (size - 1))]
    if len(big) != 1 or big[0].parent is not block:
        refuse()
    valid = [use.operation for use in flag.uses if isinstance(use.operation, arith.OrIOp)
             and set(use.operation.operands) == {flag, big[0].result}]
    assertions = [use.operation for op in valid for use in op.result.uses
                  if isinstance(use.operation, cf.AssertOp) and use.operation.parent is block
                  and use.operation.msg.data == "Unsupported standard integer byte representation"]
    ancestor = generic
    while ancestor is not None and ancestor.parent is not block:
        ancestor = ancestor.parent_op()
    if ancestor is None or not any(sequence.index(op) < sequence.index(ancestor) for op in assertions):
        refuse()


COMMUTATIVE = {"arith.addi", "arith.muli", "arith.andi", "arith.ori", "arith.xori",
               "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui"}
PURE = COMMUTATIVE | {"arith.constant", "arith.subi", "arith.shli", "arith.shrui", "arith.shrsi",
                     "arith.divsi", "arith.divui", "arith.remsi", "arith.cmpi", "arith.select",
                     "arith.index_cast", "arith.extui", "arith.extsi", "arith.trunci"}


class Expressions:
    def __init__(self, stops):
        self.values = dict(stops)
        self.seen = set()

    def value(self, value):
        if value in self.values:
            return self.values[value]
        op = value.owner
        if not hasattr(op, "name") or len(self.seen) > 1024:
            refuse()
        self.seen.add(op)
        name = op.name
        if name not in PURE | {"memref.view", "memref.dim", "memref.cast"}:
            refuse()
        args = tuple(self.value(v) for v in op.operands)
        type_ = value.type
        fields = {k: str(v) for k, v in {**op.properties, **op.attributes}.items() if not k.startswith("prov.")}
        if name == "arith.constant":
            n = literal(value)
            if n is None:
                refuse()
            result = self.constant(n, type_)
        elif name in PURE:
            if not isinstance(type_, (builtin.IntegerType, builtin.IndexType)):
                refuse()
            result = self.normalize(op, args, fields)
        elif isinstance(op, memref.CastOp):
            source, target = op.source.type, op.dest.type
            if (not isinstance(source, builtin.MemRefType) or not isinstance(target, builtin.MemRefType)
                    or source.element_type != target.element_type or len(source.get_shape()) != 1
                    or len(target.get_shape()) != 1 or source.get_offset() != 0 or target.get_offset() != 0
                    or source.get_strides() != (1,) or target.get_strides() != (1,)
                    or source.memory_space != target.memory_space
                    or any(a >= 0 and b >= 0 and a != b for a, b in zip(source.get_shape(), target.get_shape(), strict=True))):
                refuse()
            result = args[0]
        elif isinstance(op, memref.DimOp) and literal(op.index) is not None and op.source.type.get_shape()[literal(op.index)] >= 0:
            result = self.constant(op.source.type.get_shape()[literal(op.index)], type_)
        elif isinstance(op, memref.ViewOp):
            if len(type_.get_shape()) != 1 or type_.get_offset() != 0 or type_.get_strides() != (1,):
                refuse()
            dimensions = iter(args[2:])
            shape = tuple(next(dimensions) if n < 0 else self.constant(n, builtin.IndexType()) for n in type_.get_shape())
            result = ("byte-word-view", str(type_.element_type), str(type_.memory_space), args[:2], shape)
        else:
            result = (name, str(type_), tuple(sorted(fields.items())), args)
        self.values[value] = result
        return result

    @staticmethod
    def constant(n, type_):
        return ("constant", str(type_), n % (1 << type_.bitwidth) if isinstance(type_, builtin.IntegerType) else n)

    def normalize(self, op, args, fields):
        name, type_ = op.name, op.results[0].type
        for key in ("overflowFlags", "overflow_flags"):
            if key in fields:
                if fields[key] not in ("#arith.overflow<none>", "#arith.overflow<>"):
                    refuse()
                fields = {k: v for k, v in fields.items() if k != key}
        zero = self.constant(0, type_)
        if name in ("arith.ori", "arith.xori", "arith.addi"):
            if args[0] == zero:
                return args[1]
            if args[1] == zero:
                return args[0]
        if name in ("arith.shli", "arith.shrui", "arith.shrsi", "arith.subi") and args[1] == zero:
            return args[0]
        if name == "arith.select" and args[1] == args[2]:
            return args[1]
        if name in COMMUTATIVE:
            args = tuple(sorted(args, key=repr))
        if name in ("arith.andi", "arith.ori") and args[0] == args[1]:
            return args[0]
        if name in ("arith.index_cast", "arith.extui", "arith.trunci") and args[0][0] == "constant":
            n = args[0][2]
            source = op.operands[0].type
            if name == "arith.index_cast" and isinstance(source, builtin.IntegerType) and n >= 1 << (source.bitwidth - 1):
                n -= 1 << source.bitwidth
            return self.constant(n, type_)
        if all(arg[0] == "constant" for arg in args) and name in ("arith.shli", "arith.shrui", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi", "arith.muli", "arith.divui"):
            a, b = (arg[2] for arg in args)
            if name == "arith.divui" and b == 0:
                refuse()
            if name in ("arith.shli", "arith.shrui") and b >= type_.bitwidth:
                refuse()
            result = {"arith.shli": lambda: a << b, "arith.shrui": lambda: a >> b,
                      "arith.andi": lambda: a & b, "arith.ori": lambda: a | b,
                      "arith.xori": lambda: a ^ b, "arith.addi": lambda: a + b,
                      "arith.subi": lambda: a - b, "arith.muli": lambda: a * b,
                      "arith.divui": lambda: a // b}[name]()
            return self.constant(result, type_)
        return (name, str(type_), tuple(sorted(fields.items())), args)

    def block(self, block, path=()):
        effects = []
        for op in block.ops:
            for arg in op.operands:
                if arg.type == builtin.i1 or isinstance(arg.type, (builtin.IntegerType, builtin.IndexType)):
                    self.value(arg)
            if op.name in PURE:
                continue
            if isinstance(op, GenericAtomicRMWOp):
                token = (*path, "atomic", len(effects))
                self.values[op.results[0]] = ("atomic-old", token, str(op.results[0].type))
                self.values[op.body.block.args[0]] = ("atomic-current", token, str(op.results[0].type))
                effects.append(("atomic", tuple(self.value(v) for v in op.operands), self.block(op.body.block, token)))
            elif isinstance(op, scf.IfOp):
                token = (*path, "if", len(effects))
                for index, value in enumerate(op.results):
                    self.values[value] = ("if-result", token, index, str(value.type))
                effects.append(("if", self.value(op.cond), tuple(self.block(region.block, (*token, r)) if region.blocks else ()
                    for r, region in enumerate(op.regions))))
            elif isinstance(op, cf.AssertOp):
                effects.append(("assert", self.value(op.arg), op.msg.data))
            elif isinstance(op, (YieldOp, scf.YieldOp, AtomicYieldOp)):
                effects.append((op.name, tuple(self.value(v) for v in op.operands)))
            else:
                refuse()
        # Dead extra arithmetic cannot disappear from payload accounting.
        for op in block.ops:
            if op.name in PURE - {"arith.constant"} and op not in self.seen:
                refuse()
        return tuple(effects)


def match(generic, info, *, masked):
    from .byte_atomic_words import event

    if not isinstance(generic, GenericOp) or len(generic.results) != 1:
        refuse()
    cas = info["operation"] == "cas"
    atomic = info["byte_view"]
    inputs = list(generic.inputs)
    count = 2 + int(cas) + int(masked)
    if len(inputs) != count or not isinstance(inputs[1].type, builtin.TensorType):
        refuse()
    word = inputs[1].type.element_type
    shape = inputs[1].type.get_shape()
    if (str(word) != atomic["value_dtype"] or inputs[0].type != builtin.TensorType(builtin.i64, shape)
            or generic.results[0].type != inputs[1].type
            or cas and inputs[2].type != inputs[1].type
            or masked and inputs[-1].type != builtin.TensorType(builtin.i1, shape)):
        refuse()
    regions = [op for op in generic.walk() if isinstance(op, GenericAtomicRMWOp)]
    if len(regions) != 1:
        refuse()
    region = regions[0]
    viewed = region.operands[0]
    while isinstance(viewed.owner, memref.CastOp):
        viewed = viewed.owner.source
    view = viewed.owner
    if not isinstance(view, memref.ViewOp) or view.source.type.element_type != builtin.i8:
        refuse()
    flags = {op.operands[0] for op in generic.walk() if isinstance(op, arith.SelectOp)
             and isinstance(op.operands[0].owner, arith.CmpiOp)
             and isinstance(op.operands[0].owner.operands[0].owner, memref.LoadOp)}
    if len(flags) != 1:
        refuse()
    flag, = flags
    checked_probe(flag, word, generic)
    temporary = Block()
    expected_value = event(view.source, inputs[0], inputs[1], flag, temporary.add_op,
                           operation=info["operation"], byte_order=atomic["byte_order"],
                           mask=inputs[-1] if masked else None,
                           desired=inputs[2] if cas else None)
    expected = expected_value.owner
    if (generic.indexing_maps != expected.indexing_maps or generic.iterator_types != expected.iterator_types
            or [v.type for v in generic.body.block.args] != [v.type for v in expected.body.block.args]):
        refuse()
    stops = {view.source: ("byte-storage", str(view.source.type)), flag: ("literal-probe", str(word))}
    actual_expr, expected_expr = Expressions(stops), Expressions(stops)
    for index, (actual, wanted) in enumerate(zip(generic.body.block.args, expected.body.block.args, strict=True)):
        token = ("tile-argument", index, str(actual.type))
        actual_expr.values[actual] = expected_expr.values[wanted] = token
    if actual_expr.block(generic.body.block) != expected_expr.block(expected.body.block):
        refuse()

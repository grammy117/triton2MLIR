"""Recover logical Boolean memory from Triton's explicit byte-slot ABI.

Triton f797708c semantic.py _load_legacy/_store_legacy expands i1 pointers
through i8 storage; test_core.py::test_masked_load includes bool and fills.
Only byte loads observed exclusively by !=0 and stores of zero-extended i1
values are raised. General byte observations of Boolean storage stay refused.
"""

from xdsl.dialects import arith, builtin
from xdsl.ir import Operation

from .diagnostics import ConversionError


def normalize_boolean_memory(module, source_ids):
    from .ir import context
    from .shared_tensor import SharedParser, _array, _pointer_type, opcode
    def scalar(type_):
        return type_.element_type if isinstance(type_, builtin.TensorType) else type_
    def pointee(type_):
        pointer = scalar(type_)
        if not _pointer_type(pointer):
            return None
        result = SharedParser(context(), pointer.value.data).parse_type()
        return result if pointer.value.data == str(result) else None
    def boolean(type_):
        return builtin.TensorType(builtin.i1, type_.get_shape()) if isinstance(type_, builtin.TensorType) else builtin.i1
    def zero(value):
        op = value.owner
        if getattr(op, "name", None) != "arith.constant":
            return False
        attr = op.properties.get("value")
        return (isinstance(attr, builtin.IntegerAttr) and attr.value.data == 0
                or isinstance(attr, builtin.DenseIntOrFPElementsAttr) and attr.is_splat() and all(v == 0 for v in attr.get_values()))
    def constant(value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name == "arith.constant":
            attr = owner.properties.get("value")
            if isinstance(attr, builtin.IntegerAttr):
                return attr.value.data
            if isinstance(attr, builtin.DenseIntOrFPElementsAttr) and attr.is_splat() and isinstance(scalar(value.type), builtin.IntegerType):
                return next(iter(attr.get_values()))
        elif name in ("arith.trunci", "arith.extui", "arith.extsi", "tensor.splat") or isinstance(owner, Operation) and opcode(owner) == "tt.splat":
            number = constant(owner.operands[0])
            if number is not None:
                return number & ((1 << scalar(value.type).bitwidth) - 1)
        return None
    plans = []
    for cast in tuple(module.walk()):
        if (opcode(cast) != "tt.bitcast" or len(cast.operands) != 1 or len(cast.results) != 1
                or pointee(cast.operands[0].type) != builtin.i1 or pointee(cast.results[0].type) != builtin.i8):
            continue
        for use in tuple(cast.results[0].uses):
            op = use.operation
            def refuse(message):
                raise ConversionError(message, opcode=opcode(op), source_id=next(iter(source_ids(op.location)), None))
            if use.index != 0:
                refuse("Boolean byte pointers require direct load/store observations")
            if opcode(op) == "tt.load":
                comparisons = tuple(v.operation for v in op.results[0].uses)
                if not comparisons or any(opcode(c) != "arith.cmpi" or c.properties["predicate"].value.data != 1
                    or c.operands[0] != op.results[0] or not zero(c.operands[1]) for c in comparisons):
                    refuse("Boolean byte loads require exclusive nonzero tests; raw byte observations need another ABI")
                segments = _array(op, "operandSegmentSizes")
                if len(segments) != 3 or segments[0] != 1 or sum(segments) != len(op.operands):
                    refuse("Boolean load needs its complete optional mask/fill schema")
                plans.append((cast, op, comparisons, bool(segments[2])))
            elif opcode(op) == "tt.store":
                extended = op.operands[1].owner
                if not isinstance(extended, Operation) or opcode(extended) != "arith.extui" or scalar(extended.operands[0].type) != builtin.i1:
                    refuse("Boolean byte stores require an exact zero-extension from i1")
                plans.append((cast, op, (extended,), False))
            else:
                refuse("Boolean byte pointers require direct ordinary memory events")
    reports = []
    for cast, op, related, filled in plans:
        location = builtin.FusedLoc(builtin.ArrayAttr((cast.location, op.location, *(item.location for item in related))), builtin.NoneAttr())
        operands = [cast.operands[0], *op.operands[1:]]
        if opcode(op) == "tt.load":
            if filled:
                fill = operands[-1]
                number = constant(fill)
                if number is not None:
                    type_ = boolean(fill.type)
                    attr = (builtin.DenseIntOrFPElementsAttr.from_list(type_, [int(number != 0)]) if isinstance(type_, builtin.TensorType)
                            else builtin.IntegerAttr(int(number != 0), builtin.i1))
                    converted = arith.ConstantOp(attr)
                    converted.location = location
                    op.parent.insert_op_before(converted, op)
                    operands[-1] = converted.result
                else:
                    attr = (builtin.DenseIntOrFPElementsAttr.from_list(fill.type, [0]) if isinstance(fill.type, builtin.TensorType)
                            else builtin.IntegerAttr(0, fill.type))
                    empty = arith.ConstantOp(attr)
                    converted = arith.CmpiOp.create(operands=[fill, empty.result], result_types=[boolean(fill.type)],
                        properties={"predicate": builtin.IntegerAttr(1, builtin.i64)})
                    empty.location = converted.location = location
                    op.parent.insert_ops_before([empty, converted], op)
                    operands[-1] = converted.result
            loaded = type(op).create(operands=operands, result_types=[boolean(op.results[0].type)],
                properties=op.properties, attributes=op.attributes, location=location)
            op.parent.insert_op_before(loaded, op)
            for comparison in related:
                comparison.results[0].replace_all_uses_with(loaded.results[0])
                comparison.parent.erase_op(comparison)
            op.parent.erase_op(op)
        else:
            operands[1] = related[0].operands[0]
            stored = type(op).create(operands=operands, properties=op.properties, attributes=op.attributes, location=location)
            op.parent.insert_op_before(stored, op)
            op.parent.erase_op(op)
        reports.append({"kind": "logical-boolean-byte-memory", "source_ids": source_ids(location),
                        "rule": "exclusive-nonzero-load-or-zero-extended-store", "storage": "logical-i1"})
    for cast, *_ in plans:
        if cast.parent is not None and not cast.results[0].uses:
            cast.parent.erase_op(cast)
    return reports

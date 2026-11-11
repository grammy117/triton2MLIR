"""Reify source register groups as parallel logical tensor computation."""

import math

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.dialects.builtin import AffineMapAttr, DenseArrayBase, IndexType, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter
from xdsl.traits import is_side_effect_free

from .map_layout import validate
from .standard_ops import ForallOp, InParallelOp, ParallelInsertSliceOp


def _constant(value):
    return arith.ConstantOp.from_int_and_width(value, builtin.i64)


def _linear(plan, values, add):
    zero, one = add(_constant(0)), add(_constant(1))
    result = [zero] * len(plan["out_dims"])
    for name, vectors in plan["bases"]:
        for bit, vector in enumerate(vectors):
            selected = add(arith.AndIOp(add(arith.ShRUIOp(values[name], add(_constant(bit)))), one))
            for i, value in enumerate(vector):
                if value:
                    result[i] = add(arith.XOrIOp(result[i], add(arith.MuliOp(selected, add(_constant(value))))))
    return dict(zip((name for name, _ in plan["out_dims"]), result, strict=True))


def expand(op, source, end, pack, row, refuse, *, memory_events=False):
    from .shared_tensor import opcode
    sizes = validate(row)
    shape = list(op.operands[0].type.get_shape())
    if row["shape"] != shape or row["pack"] != pack:
        refuse("source map grouping differs from the current operation")
    has_memory = any(opcode(child) == "tt.load" for child in source.walk())
    for child in source.walk():
        if child is end:
            continue
        control = isinstance(child, (scf.IfOp, scf.ForOp, scf.WhileOp))
        terminal = isinstance(child, (scf.YieldOp, scf.ConditionOp))
        if memory_events and opcode(child) in ("tt.load", "tt.addptr", "tt.bitcast"):
            continue
        if not control and (child.regions or not (is_side_effect_free(child) or isinstance(child, arith.ExtSIOp))
                            or opcode(child).split(".")[0] not in ("arith", "math") and not terminal):
            refuse("source-grouped maps require pure numeric scalar computation; mapped effects need their own contract", child)
    if has_memory:
        from .grouped_reads import expand_reads
        return expand_reads(op, source, end, pack, row)
    groups = math.prod(shape) // pack
    grouped_types = [TensorType(value.type.element_type, [groups, pack]) for value in op.results]
    empty = [tensor.EmptyOp([], type_) for type_ in grouped_types]
    body = Block(arg_types=[IndexType(), *grouped_types])

    def append(block):
        def add(new):
            new.location = op.location
            block.add_op(new)
            return new.results[0] if new.results else None
        return add

    add = append(body)
    group = add(arith.IndexCastOp(body.args[0], builtin.i64))
    registers = sizes["register"] // pack
    high = add(arith.AndIOp(group, add(_constant(registers - 1))))
    base_register = add(arith.MuliOp(high, add(_constant(pack))))
    physical, bits = {}, registers.bit_length() - 1
    for name in ("lane", "warp", "block"):
        physical[name] = add(arith.AndIOp(add(arith.ShRUIOp(group, add(_constant(bits)))), add(_constant(sizes[name] - 1))))
        bits += sizes[name].bit_length() - 1
    mapping = {}
    for slot in range(pack):
        coordinates = _linear(row["forward"], {**physical, "register": add(arith.AddiOp(base_register, add(_constant(slot))))}, add)
        indices = [add(arith.IndexCastOp(coordinates[f"dim{i}"], IndexType())) for i in range(len(shape))]
        for operand, value in enumerate(op.operands):
            mapping[source.args[operand * pack + slot]] = add(tensor.ExtractOp(value, indices, value.type.element_type))
    for child in list(source.ops)[:-1]:
        body.add_op(child.clone(value_mapper=mapping))
    updates = Block()
    for i, target in enumerate(body.args[1:]):
        values = [mapping.get(value, value) for value in end.operands[i * pack:(i + 1) * pack]]
        packed = add(tensor.FromElementsOp(*values))
        insertion = ParallelInsertSliceOp.build(operands=[packed, target, [body.args[0]], [], []], properties={
            "static_offsets": DenseArrayBase.from_list(builtin.i64, [builtin.DYNAMIC_INDEX, 0]),
            "static_sizes": DenseArrayBase.from_list(builtin.i64, [1, pack]),
            "static_strides": DenseArrayBase.from_list(builtin.i64, [1, 1])})
        insertion.location = op.location
        updates.add_op(insertion)
    body.add_op(InParallelOp.build(regions=[Region(updates)]))
    loop = ForallOp.build(operands=[[], [], [], [value.tensor for value in empty]], result_types=[grouped_types],
        properties={"staticLowerBound": DenseArrayBase.from_list(builtin.i64, [0]),
                    "staticUpperBound": DenseArrayBase.from_list(builtin.i64, [groups]),
                    "staticStep": DenseArrayBase.from_list(builtin.i64, [1])}, regions=[Region(body)])
    loop.attributes.update({"prov.map_pack": IntegerAttr(pack, builtin.i32),
        "prov.map_grouping": StringAttr("source-linear-layout-v1"), "prov.logical_group_count": IntegerAttr(groups, builtin.i64)})
    generated, outputs = [*empty, loop], []
    for i, result in enumerate(op.results):
        block = Block(arg_types=[result.type.element_type])
        add = append(block)
        logical = {f"dim{d}": add(arith.IndexCastOp(add(IndexOp(d)), builtin.i64)) for d in range(len(shape))}
        physical = _linear(row["inverse"], logical, add)
        group = add(arith.ShRUIOp(physical["register"], add(_constant(pack.bit_length() - 1))))
        bits = registers.bit_length() - 1
        for name in ("lane", "warp", "block"):
            group = add(arith.AddiOp(group, add(arith.ShLIOp(physical[name], add(_constant(bits))))))
            bits += sizes[name].bit_length() - 1
        slot = add(arith.AndIOp(physical["register"], add(_constant(pack - 1))))
        value = add(tensor.ExtractOp(loop.results[i], [add(arith.IndexCastOp(group, IndexType())),
            add(arith.IndexCastOp(slot, IndexType()))], result.type.element_type))
        add(YieldOp(value))
        destination = tensor.EmptyOp([], result.type)
        restored = GenericOp([], [destination.tensor], Region(block), [AffineMapAttr(AffineMap.identity(len(shape)))],
            [IteratorTypeAttr.parallel()] * len(shape), [result.type])
        restored.attributes["prov.map_grouping"] = StringAttr("source-linear-layout-v1")
        generated.extend((destination, restored))
        outputs.append(restored.results[0])
    for value in generated:
        value.location = op.location
    Rewriter.replace_op(op, generated, outputs)
    return {"method": "source-linear-layout-v1", "source_map": row["id"], "layout": row,
        "logical_groups": groups, "group_compute": "one-original-scalar-region-per-logical-group",
        "physical_mapping": "source-grouping-reified; consumer-placement-undecided"}

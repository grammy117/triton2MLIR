"""One parallel scalar invocation per source pack, retaining its read positions.

Reuse the existing source-word event conversion and guarded-memory analysis.
Separate tensor<G> outputs retain every returned scalar slot without cloning
the original read/control body for each restored logical tensor element.
"""

import math

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .grouped_map import _constant, _linear
from .map_layout import validate


def expand_reads(op, source, end, pack, row):
    sizes = validate(row)
    shape = row["shape"]
    groups, registers = math.prod(shape) // pack, sizes["register"] // pack
    types = [TensorType(value.type, [groups]) for value in end.operands]
    empty = [tensor.EmptyOp([], type_) for type_ in types]
    body = Block(arg_types=[type_.element_type for type_ in types])

    def append(block):
        def add(new):
            new.location = op.location
            block.add_op(new)
            return new.results[0] if new.results else None
        return add

    add = append(body)
    group = add(arith.IndexCastOp(add(IndexOp(0)), builtin.i64))
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
    terminal = YieldOp(*(mapping.get(value, value) for value in end.operands))
    terminal.location = end.location
    body.add_op(terminal)
    grouped = GenericOp([], [value.tensor for value in empty], Region(body),
        [AffineMapAttr(AffineMap.identity(1))] * len(types), [IteratorTypeAttr.parallel()], types)
    grouped.attributes.update({"prov.map_pack": IntegerAttr(pack, builtin.i32),
        "prov.map_grouping": StringAttr("source-linear-layout-v1"), "prov.logical_group_count": IntegerAttr(groups, builtin.i64)})
    generated, outputs = [*empty, grouped], []
    for result_index, result in enumerate(op.results):
        block = Block(arg_types=[result.type.element_type])
        add = append(block)
        logical = {f"dim{d}": add(arith.IndexCastOp(add(IndexOp(d)), builtin.i64)) for d in range(len(shape))}
        physical = _linear(row["inverse"], logical, add)
        group = add(arith.ShRUIOp(physical["register"], add(_constant(pack.bit_length() - 1))))
        bits = registers.bit_length() - 1
        for name in ("lane", "warp", "block"):
            group = add(arith.AddiOp(group, add(arith.ShLIOp(physical[name], add(_constant(bits))))))
            bits += sizes[name].bit_length() - 1
        index = add(arith.IndexCastOp(group, IndexType()))
        slot = add(arith.AndIOp(physical["register"], add(_constant(pack - 1))))
        value = add(tensor.ExtractOp(grouped.results[result_index * pack], [index], result.type.element_type))
        for position in range(1, pack):
            incoming = add(tensor.ExtractOp(grouped.results[result_index * pack + position], [index], result.type.element_type))
            selected = add(arith.CmpiOp(slot, add(_constant(position)), "eq"))
            value = add(arith.SelectOp(selected, incoming, value))
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
        "group_storage": "one-vector-per-result-slot", "physical_mapping": "source-grouping-reified; consumer-placement-undecided"}

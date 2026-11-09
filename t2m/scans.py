"""Portable inclusive associative scans using parallel tensor doubling stages.

The shared CumSumConverter (Meta 17f0c234) accepts only a single add combiner
and emits ttx.cumsum, outside the functional tensor ABI. This expansion instead
preserves the general source combiner, tuple results, axis and traversal order.
No identity is invented and no physical lane/warp mapping is selected.
"""

from xdsl.dialects import builtin, tensor
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from .diagnostics import ConversionError


def expand_scans(module, source_ids):
    reports = []

    def name(op):
        return op.op_name.data if op.name == "builtin.unregistered" else op.name

    for scan in list(module.walk()):
        if name(scan) != "tt.scan":
            continue
        ids = source_ids(scan.location)

        def refuse(message, opcode="tt.scan"):
            raise ConversionError(message, source_id=next(iter(ids), None), opcode=opcode)

        types = [v.type for v in scan.operands]
        if (not types or len(types) != len(scan.results) or
                any(not isinstance(t, TensorType) for t in types) or
                any(t != v.type for t, v in zip(types, scan.results, strict=True))):
            refuse("scan operands/results need matching ranked tensor types")
        shape = types[0].get_shape()
        if any(t.get_shape() != shape for t in types) or any(n <= 0 for n in shape):
            refuse("scan needs equal positive static tensor shapes")
        axis, reverse = scan.properties.get("axis"), scan.properties.get("reverse")
        if (not isinstance(axis, IntegerAttr) or not 0 <= axis.value.data < len(shape) or
                not isinstance(reverse, IntegerAttr) or reverse.value.data not in (0, -1, 1)):
            refuse("scan axis/reverse attributes are not qualified")
        axis, reverse = axis.value.data, bool(reverse.value.data)
        if len(scan.regions) != 1 or len(scan.regions[0].blocks) != 1:
            refuse("scan needs one scalar combiner block")
        combiner = scan.regions[0].block
        element_types = [t.get_element_type() for t in types]
        if [v.type for v in combiner.args] != element_types * 2:
            refuse("scan combiner argument types differ from the tuple tensors")
        terminator = combiner.last_op
        if (terminator is None or name(terminator) != "tt.scan.return" or
                [v.type for v in terminator.operands] != element_types):
            refuse("scan combiner must return every tuple component")
        for op in list(combiner.ops)[:-1]:
            if op.regions or op.name.split(".")[0] not in ("arith", "math"):
                refuse("scan combiner needs inlined pure scalar arithmetic/math", opcode=name(op))
            if any(isinstance(v.type, TensorType) for v in (*op.operands, *op.results)):
                refuse("scan combiner cannot contain tensor-valued arithmetic", opcode=name(op))
        emitted = []

        def append(op):
            op.location = scan.location
            op.attributes.update({"prov.family": StringAttr("scan"), "prov.op": StringAttr("associative_scan"),
                                  "prov.composite_family": StringAttr("scan"),
                                  "prov.composite_id": StringAttr("scan:" + (ids[0] if ids else str(scan.location))),
                                  "prov.scan_algorithm": StringAttr("inclusive-parallel-doubling"),
                                  "prov.scan_axis": IntegerAttr(axis, builtin.i64),
                                  "prov.scan_reverse": IntegerAttr(int(reverse), builtin.i1)})
            emitted.append(op)
            return op

        values, distance, stages = list(scan.operands), 1, 0
        while distance < shape[axis]:
            sizes = list(shape)
            sizes[axis] -= distance
            current_offsets, prior_offsets = [0] * len(shape), [0] * len(shape)
            (prior_offsets if reverse else current_offsets)[axis] = distance
            prior = [append(tensor.ExtractSliceOp.from_static_parameters(v, prior_offsets, sizes)).result for v in values]
            current = [append(tensor.ExtractSliceOp.from_static_parameters(v, current_offsets, sizes)).result for v in values]
            sliced_types = [TensorType(t, sizes) for t in element_types]
            outputs = [append(tensor.EmptyOp([], t)).tensor for t in sliced_types]
            body = Block(arg_types=element_types * 3)
            mapper = dict(zip(combiner.args, body.args[:2 * len(types)], strict=True))
            for op in list(combiner.ops)[:-1]:
                body.add_op(op.clone(value_mapper=mapper))
            returned = YieldOp(*(mapper[v] for v in terminator.operands))
            returned.location = terminator.location
            body.add_op(returned)
            combined = append(GenericOp([*prior, *current], outputs, Region(body),
                [AffineMapAttr(AffineMap.identity(len(shape)))] * (3 * len(types)),
                [IteratorTypeAttr.parallel()] * len(shape), sliced_types))
            combined.attributes["prov.scan_distance"] = IntegerAttr(distance, builtin.i64)
            values = [append(tensor.InsertSliceOp.from_static_parameters(part, full, current_offsets, sizes)).result
                      for part, full in zip(combined.results, values, strict=True)]
            distance *= 2
            stages += 1
        parent = scan.parent
        parent.insert_ops_before(emitted, scan)
        for old, new in zip(scan.results, values, strict=True):
            old.replace_all_uses_with(new)
        reports.append({"kind": "associative-scan-network", "source_ids": ids, "location": str(scan.location),
                        "axis": axis, "reverse": reverse, "tuple_arity": len(types), "stages": stages,
                        "algorithm": "inclusive-parallel-doubling", "numeric_contract": "source-associative-combiner",
                        "identity": "none-invented", "physical_mapping": "undecided"})
        parent.erase_op(scan)
    return reports

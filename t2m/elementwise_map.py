"""Turn scalar TTIR maps into parallel tensor computation, retaining the body.

Triton 3.7.1 defines pack=1 as one scalar subregion invocation per element.
The shared Meta/Microsoft arithmetic converters do not handle this region op.
Use standard linalg generalization: identity maps, parallel iterators and the
original inlined scalar region, including its structured control flow.
"""

from xdsl.dialects import builtin, tensor
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def expand_elementwise_maps(module, source_ids, *, memory_events=False):
    from .shared_tensor import _pointer_type, opcode
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.map_elementwise":
            continue
        ids = source_ids(op.location)

        def refuse(message, source=op):
            raise ConversionError(message, opcode=opcode(source), source_id=next(iter(source_ids(source.location)), None))

        pack = op.properties.get("pack")
        if not isinstance(pack, IntegerAttr) or pack.value.data != 1:
            refuse("packed elementwise maps need an explicit logical grouping contract")
        inputs, outputs = [v.type for v in op.operands], [v.type for v in op.results]
        if not inputs or not outputs or not all(isinstance(t, TensorType) for t in inputs + outputs):
            refuse("elementwise maps require ranked numeric input/result tensors")
        shape = inputs[0].get_shape()
        if (any(n <= 0 for n in shape) or any(t.get_shape() != shape for t in inputs + outputs)
                or not all(isinstance(t.element_type, (builtin.IntegerType, builtin.AnyFloat)) for t in inputs + outputs)):
            refuse("elementwise map tensor shapes must match and elements must be numeric; source memory needs its own effect adapter")
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            refuse("elementwise maps need one inlined scalar region")
        source = op.regions[0].block
        has_memory = any(opcode(child) == "tt.load" for child in source.walk())
        end = source.last_op
        input_elements, output_elements = [t.element_type for t in inputs], [t.element_type for t in outputs]
        if ([v.type for v in source.args] != input_elements or end is None
                or opcode(end) != "tt.map_elementwise.return" or [v.type for v in end.operands] != output_elements):
            refuse("elementwise map scalar signature differs from its tensor elements")
        controls = {"scf.if", "scf.for", "scf.while", "scf.yield", "scf.condition"}
        for child in source.walk():
            name = opcode(child)
            if child is end:
                continue
            memory = memory_events and name in ("tt.load", "tt.addptr", "tt.bitcast")
            if name.split(".")[0] not in ("arith", "math") and name not in controls and not memory:
                refuse("elementwise map body requires inlined scalar arithmetic/math and structured value control", child)
            values = [*child.operands, *child.results,
                      *(arg for region in child.regions for block in region.blocks for arg in block.args)]
            if not all(isinstance(v.type, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))
                       or memory_events and _pointer_type(v.type) and not isinstance(v.type, TensorType) for v in values):
                refuse("elementwise map scalar body cannot access tensors, addresses or effects", child)
        body = Block(arg_types=input_elements + output_elements)
        mapping = dict(zip(source.args, body.args[:len(inputs)], strict=True))
        for child in list(source.ops)[:-1]:
            body.add_op(child.clone(value_mapper=mapping))
        terminal = YieldOp(*(mapping.get(v, v) for v in end.operands))
        terminal.location = end.location
        body.add_op(terminal)
        empty = [tensor.EmptyOp([], t) for t in outputs]
        for value in empty:
            value.location = op.location
        generic = GenericOp(list(op.operands), [v.tensor for v in empty], Region(body),
            [AffineMapAttr(AffineMap.identity(len(shape)))] * (len(inputs) + len(outputs)),
            [IteratorTypeAttr.parallel()] * len(shape), outputs)
        generic.location = op.location
        generic.attributes.update({"prov.composite_family": StringAttr("map_elementwise"),
            "prov.composite_id": StringAttr("map_elementwise:" + (ids[0] if ids else str(op.location))),
            "prov.map_pack": IntegerAttr(1, builtin.i32)})
        Rewriter.replace_op(op, [*empty, generic])
        reports.append({"kind": "ttir-elementwise-map", "source_ids": ids, "pack": 1,
            "shape": list(shape), "inputs": len(inputs), "results": len(outputs),
            "expansion": "parallel-linalg-generic-with-original-scalar-region", "physical_mapping": "undecided",
            **({"memory": "isolated-source-word-events-in-original-scalar-region"} if has_memory else {})})
    return reports

"""Preserve mutable gather observations before tensor optimizations can run.

Meta 17f0c234 GatherConverter creates to_tensor/restrict plus tensor.extract.
Source-base extractions remain actual reads, without competing restrict views.
Structured loads already copy into a distinct allocation and retain that copy.
"""

from xdsl.dialects import memref, tensor

from .diagnostics import ConversionError


def materialize(module, abi, source_ids):
    function = next(iter(module.ops))
    buffers = {arg: value for arg, (kind, value) in zip(function.body.block.args, abi.source_bindings) if kind == "buffer"}
    reports = []
    views = {"memref.cast", "memref.subview", "memref.reinterpret_cast", "memref.collapse_shape",
             "memref.expand_shape", "memref.memory_space_cast", "memref.view"}
    def base(value):
        while value not in buffers:
            op = value.owner
            if not hasattr(op, "name") or op.name not in views:
                return None
            value = op.operands[0]
        return buffers[value]
    for op in list(function.walk()):
        if op.name != "bufferization.to_tensor" or (buffer := base(op.operands[0])) is None:
            continue
        for use in list(op.results[0].uses):
            extract = use.operation
            if not isinstance(extract, tensor.ExtractOp) or use.index != 0:
                raise ConversionError("upstream source-buffer tensor view requires an effect-preserving observation adapter",
                    source_id=next(iter(source_ids(extract.location)), None), opcode=extract.name)
            load = memref.LoadOp.build(operands=[op.operands[0], list(extract.operands[1:])],
                result_types=[extract.results[0].type], attributes=extract.attributes)
            load.location = extract.location
            extract.parent.insert_op_before(load, extract)
            extract.results[0].replace_all_uses_with(load.results[0])
            extract.parent.erase_op(extract)
            reports.append({"buffer": buffer.name, "source_ids": source_ids(load.location),
                            "adapter": "source-buffer-tensor-extract-to-memory-load", "effect": buffer.effect})
        op.parent.erase_op(op)
    return reports

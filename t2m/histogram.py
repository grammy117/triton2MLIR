"""Tile-local integer histogram with a standard indexed state expansion.

Ascend 8f0ade66 keeps histogram as hfusion.histogram or a target builtin;
neither is a portable Merlin payload. Preserve that composite intent while
expanding to zero-filled tensor state and one indexed update per input element.
This uses O(elements + bins) work without choosing a hardware implementation.
"""

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.dialects.builtin import IndexType, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def expand_histograms(module, source_ids):
    from .shared_tensor import opcode
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.histogram":
            continue
        ids = source_ids(op.location)

        def refuse(message):
            raise ConversionError(message, opcode="tt.histogram", source_id=next(iter(ids), None))

        if len(op.operands) not in (1, 2) or len(op.results) != 1:
            refuse("histogram requires one source, an optional mask and one result")
        source, result_type = op.operands[0], op.results[0].type
        if (not isinstance(source.type, TensorType) or not isinstance(result_type, TensorType)
                or len(source.type.get_shape()) != 1 or len(result_type.get_shape()) != 1
                or source.type.get_shape()[0] <= 0 or result_type.get_shape()[0] <= 0
                or source.type.element_type not in (builtin.i32, builtin.i64)
                or result_type.element_type not in (builtin.i32, builtin.i64)):
            refuse("histogram requires positive rank-one i32/i64 source and counter tensors")
        mask = op.operands[1] if len(op.operands) == 2 else None
        if mask is not None and mask.type != TensorType(builtin.i1, source.type.get_shape()):
            refuse("histogram mask must match the source shape")
        elements, bins = source.type.get_shape()[0], result_type.get_shape()[0]
        emitted = []

        def add(block, new):
            new.location = op.location
            (emitted.append(new) if block is None else block.add_op(new))
            return new.results[0] if new.results else None

        zero = add(None, arith.ConstantOp.from_int_and_width(0, result_type.element_type))
        one = add(None, arith.ConstantOp.from_int_and_width(1, result_type.element_type))
        empty = add(None, tensor.EmptyOp([], result_type))
        initial = add(None, FillOp([zero], [empty], res=[result_type]))
        lower = add(None, arith.ConstantOp.from_int_and_width(0, IndexType()))
        upper = add(None, arith.ConstantOp.from_int_and_width(elements, IndexType()))
        step = add(None, arith.ConstantOp.from_int_and_width(1, IndexType()))
        body = Block(arg_types=[IndexType(), result_type])
        iv, state = body.args

        def update(block):
            value = add(block, tensor.ExtractOp(source, [iv], source.type.element_type))
            if value.type != builtin.i64:
                value = add(block, arith.ExtSIOp(value, builtin.i64))
            low = add(block, arith.ConstantOp.from_int_and_width(0, builtin.i64))
            high = add(block, arith.ConstantOp.from_int_and_width(bins, builtin.i64))
            above = add(block, arith.CmpiOp(value, low, "sge"))
            below = add(block, arith.CmpiOp(value, high, "slt"))
            valid = add(block, arith.AndIOp(above, below))
            inside, outside = Block(), Block()
            index = add(inside, arith.IndexCastOp(value, IndexType()))
            old = add(inside, tensor.ExtractOp(state, [index], result_type.element_type))
            count = add(inside, arith.AddiOp(old, one))
            changed = add(inside, tensor.InsertOp(count, state, [index]))
            add(inside, scf.YieldOp(changed))
            add(outside, scf.YieldOp(state))
            return add(block, scf.IfOp(valid, [result_type], Region(inside), Region(outside)))

        if mask is None:
            changed = update(body)
        else:
            selected = add(body, tensor.ExtractOp(mask, [iv], builtin.i1))
            active, inactive = Block(), Block()
            add(active, scf.YieldOp(update(active)))
            add(inactive, scf.YieldOp(state))
            changed = add(body, scf.IfOp(selected, [result_type], Region(active), Region(inactive)))
        add(body, scf.YieldOp(changed))
        loop = scf.ForOp(lower, upper, step, [initial], Region(body))
        loop.location = op.location
        loop.attributes.update({"prov.composite_family": StringAttr("histogram"),
            "prov.composite_id": StringAttr("histogram:" + (ids[0] if ids else str(op.location))),
            "prov.histogram_bins": IntegerAttr(bins, builtin.i64),
            "prov.histogram_algorithm": StringAttr("indexed-integer-accumulation")})
        emitted.append(loop)
        Rewriter.replace_op(op, emitted, list(loop.results))
        reports.append({"kind": "ttir-histogram", "source_ids": ids, "elements": elements, "bins": bins,
            "masked": mask is not None, "source_bits": source.type.element_type.width.data,
            "counter_bits": result_type.element_type.width.data, "expansion": "indexed-tensor-state-loop",
            "work": "elements-plus-bins", "out_of_range": "no-contribution", "physical_mapping": "undecided"})
    return reports

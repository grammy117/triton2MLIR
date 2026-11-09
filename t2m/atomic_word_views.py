"""Parallel integer atomic updates on the actual floating storage word.

Meta 17f0c234's unstructured RMW converter supplies the parallel event pattern.
The remaining adapter retains the logical floating memref and integer operation
inside a standard generic atomic region, rather than creating physical aliases
or copying a whole buffer into integer storage.
"""

from xdsl.dialects import arith, builtin, memref, scf, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp

UPDATES = {"and": arith.AndIOp, "or": arith.OrIOp, "xor": arith.XOrIOp, "add": arith.AddiOp,
    "max": arith.MaxSIOp, "min": arith.MinSIOp, "umax": arith.MaxUIOp, "umin": arith.MinUIOp}
PAYLOAD = "standard-memref-generic-integer-word-update-with-original-returned-old-value"


def view_info(storage, value):
    if storage not in (builtin.f32, builtin.f64) or value not in (builtin.i32, builtin.i64) or storage.bitwidth != value.bitwidth:
        raise ValueError("atomic word views require equal-width f32/i32 or f64/i64 elements")
    return {"storage_dtype": str(storage), "value_dtype": str(value), "element_bits": storage.bitwidth,
            "address_units": "equal-width-elements"}


def bind_source(records, reports):
    inventory = {record["id"]: record for record in records}
    for report in reports:
        if report.get("kind") != "logical-atomic-word-view":
            continue
        info = {key: report[key] for key in ("storage_dtype", "value_dtype", "element_bits", "address_units")}
        for source in report["source_ids"]:
            record = inventory.get(source)
            if record is None or record["opcode"] != "tt.atomic_rmw" or record.get("atomic", {}).get("word_bits") != info["element_bits"]:
                raise ConversionError("atomic word view differs from its typed source RMW inventory", source_id=source)
            record["atomic"]["word_view"] = info


def match(op, value):
    """Check the exact scalar region against its source outside integer value."""
    if not isinstance(op, GenericAtomicRMWOp):
        raise ConversionError("integer word updates require a standard generic atomic region")
    block = op.body.block
    ops = list(block.ops)
    storage = op.result.type
    info = view_info(storage, value.type)
    if len(ops) == 4:
        viewed, updated, stored, terminal = ops
        kind = next((name for name, cls in UPDATES.items() if isinstance(updated, cls)), None)
        if (not isinstance(viewed, arith.BitcastOp) or tuple(viewed.operands) != (block.args[0],)
                or viewed.result.type != value.type or kind is None
                or tuple(updated.operands) != (viewed.result, value)
                or not isinstance(stored, arith.BitcastOp) or tuple(stored.operands) != tuple(updated.results)):
            raise ConversionError("atomic word region differs from the original integer update")
    elif len(ops) == 2:
        stored, terminal = ops
        kind = "exch"
        if not isinstance(stored, arith.BitcastOp) or tuple(stored.operands) != (value,):
            raise ConversionError("atomic word exchange differs from its original integer value")
    else:
        raise ConversionError("atomic word region has unaccounted operations")
    if (stored.result.type != storage or not isinstance(terminal, AtomicYieldOp)
            or tuple(terminal.operands) != tuple(stored.results)):
        raise ConversionError("atomic word region does not yield the updated storage word")
    return kind, info


def expand(module, admission, source_ids):
    from .shared_tensor import opcode
    from .pointer_views import element
    accesses = [a for a in admission["accesses"] if a.get("atomic", {}).get("word_view")]
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tts.unstructured_atomic_rmw":
            continue
        cast = op.operands[0].owner
        if not hasattr(cast, "name") or opcode(cast) != "tt.bitcast":
            continue
        base, = cast.operands
        storage, value_type = element(base.type), op.operands[2].type.element_type
        view = view_info(storage, value_type)
        ids = source_ids(op.location)
        candidates = [a for a in accesses if set(ids) & set(a["source_ids"]) and a["atomic"]["word_view"] == view]
        if not candidates or any(a["atomic"] != candidates[0]["atomic"] for a in candidates):
            raise ConversionError("atomic word event lacks matching logical storage admission", source_id=next(iter(ids), None))
        info = candidates[0]["atomic"]
        offset, value = op.operands[1:3]
        mask = op.operands[3] if len(op.operands) == 4 else None
        type_ = op.results[0].type
        imported = builtin.UnrealizedConversionCastOp.get([base], [builtin.UnrankedMemRefType.from_type(storage)])
        ranked = memref.CastOp.get(imported.results[0], builtin.MemRefType(storage, [builtin.DYNAMIC_INDEX]))
        empty = tensor.EmptyOp([], type_)
        inputs = [offset, value, *([mask] if mask is not None else [])]
        block = Block(arg_types=[v.type.element_type for v in inputs] + [value_type])
        body = Block() if mask is not None else block
        index = arith.IndexCastOp(block.args[0], builtin.IndexType())
        atomic_body = Block(arg_types=[storage])
        if info["operation"] == "exch":
            word = block.args[1]
        else:
            viewed = arith.BitcastOp(atomic_body.args[0], value_type)
            updated = UPDATES[info["operation"]](viewed.result, block.args[1])
            atomic_body.add_ops([viewed, updated])
            word = updated.result
        stored = arith.BitcastOp(word, storage)
        atomic_body.add_ops([stored, AtomicYieldOp.create(operands=[stored.result])])
        atomic = GenericAtomicRMWOp.create(operands=[ranked.dest, index.result], result_types=[storage], regions=[Region(atomic_body)])
        old = arith.BitcastOp(atomic.result, value_type)
        body.add_ops([index, atomic, old])
        if mask is not None:
            body.add_op(scf.YieldOp(old.result))
            inactive = Block([scf.YieldOp(block.args[-1])])
            control = scf.IfOp(block.args[2], [value_type], Region(body), Region(inactive))
            block.add_op(control)
            returned = control.results[0]
        else:
            returned = old.result
        block.add_op(YieldOp(returned))
        identity = builtin.AffineMapAttr(AffineMap.identity(len(type_.get_shape())))
        generic = GenericOp(inputs, [empty.tensor], Region(block), [identity] * (len(inputs) + 1),
            [IteratorTypeAttr.parallel()] * len(type_.get_shape()), [type_])
        generated = [imported, ranked, empty, generic]
        for outer in generated:
            for nested in outer.walk():
                nested.location = op.location
        reports.append({"kind": "standard-integer-word-atomic-region", "source_ids": ids, "word_view": view,
            "operation": info["operation"], "result": "original-atomic-old-word", "iteration": "parallel-logical-tile"})
        Rewriter.replace_op(op, generated, generic.results)
    for op in reversed(list(module.walk())):
        if opcode(op) == "tt.bitcast" and op.results and not op.results[0].uses:
            from .shared_tensor import _pointer_type
            if _pointer_type(op.results[0].type):
                op.parent.erase_op(op)
    return reports

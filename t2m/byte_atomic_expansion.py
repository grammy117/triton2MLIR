"""Expand admitted byte-backed words without copying or splitting effects."""

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def expand(module, admission, source_ids):
    from .atomic_memory import contract
    from .byte_atomic_events import view_info
    from .byte_atomic_words import event, probe
    from .pointer_views import element
    from .shared_tensor import TensorABI, opcode
    from .standard_memref import ReinterpretCastOp

    accesses = [access for access in admission["accesses"] if "byte_view" in access.get("atomic", {})]
    entry = next(iter(module.ops)).regions[0].block
    abi = TensorABI.from_dict(admission["spec"])
    names = {arg: value.name for arg, (kind, value) in zip(entry.args, abi.source_bindings)
             if kind == "buffer"}
    probes, reports = {}, []
    for op in list(module.walk()):
        name = opcode(op)
        if name not in ("tts.unstructured_atomic_rmw", "tt.atomic_cas"):
            continue
        descriptor = op.operands[0].owner if name == "tt.atomic_cas" else None
        pointer = descriptor.operands[0] if descriptor is not None else op.operands[0]
        cast = pointer.owner
        if not hasattr(cast, "operands") or opcode(cast) != "tt.bitcast":
            continue
        base, = cast.operands
        if element(base.type) != builtin.i8:
            continue
        ids = source_ids(op.location)
        primitive = contract(op, source_ids, typed=True)
        candidates = [access for access in accesses if access["buffer"] == names.get(base) and set(ids) & set(access["source_ids"])
                      and {k: v for k, v in access["atomic"].items() if k != "byte_view"} == primitive]
        if not candidates:
            raise ConversionError("byte atomic expansion lacks its complete word/byte admission", source_id=next(iter(ids), None))
        infos = [access["atomic"]["byte_view"] for access in candidates]
        word = op.results[0].type.element_type
        info = view_info(word, infos[0]["byte_order"])
        if any(value != info for value in infos):
            raise ConversionError("byte atomic expansion changed its original storage/word boundary")
        if word not in probes:
            temporary = Block()
            little = probe(word, temporary.add_op)
            generated = list(temporary.ops)
            for generated_op in generated:
                temporary.detach_op(generated_op)
                for nested in generated_op.walk():
                    nested.location = op.location
            entry.insert_ops_before(generated, entry.first_op)
            probes[word] = little
        emitted = []

        def append(item, emitted=emitted):
            emitted.append(item)
            return item.results[0] if item.results else None

        imported = append(builtin.UnrealizedConversionCastOp.get([base], [builtin.UnrankedMemRefType.from_type(builtin.i8)]))
        extent_arg = next(i for i, argument in enumerate(admission["internal_arguments"])
                          if argument == {"kind": "extent", "buffer": names[base]})
        extent = append(arith.IndexCastOp(entry.args[len(abi.source_bindings) + extent_arg], builtin.IndexType()))
        # Native ViewOpMemrefCastFolder unsafely folds a ranked cast from an
        # unranked source into memref.view. A standard explicit zero-offset,
        # unit-stride reinterpretation retains our checked owning-root ABI and
        # its exact runtime extent without passing through that invalid fold.
        storage = append(ReinterpretCastOp.build(operands=[imported, [], [extent], []],
            result_types=[builtin.MemRefType(builtin.i8, [builtin.DYNAMIC_INDEX])],
            properties={"static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, [0]),
                        "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, [ReinterpretCastOp.DYNAMIC_INDEX]),
                        "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1])}))
        offsets = descriptor.operands[1] if descriptor is not None else op.operands[1]
        type_ = offsets.type
        factor = append(arith.ConstantOp.from_int_and_width(info["alignment_bytes"], builtin.i64))
        empty = append(tensor.EmptyOp([], type_))
        scalar = Block(arg_types=[builtin.i64, builtin.i64])
        scaled = arith.MuliOp(scalar.args[0], factor)
        scalar.add_ops([scaled, YieldOp(scaled.result)])
        byte_offsets = append(GenericOp([offsets], [empty], Region(scalar),
            [builtin.AffineMapAttr(AffineMap.identity(len(type_.get_shape())))] * 2,
            [IteratorTypeAttr.parallel()] * len(type_.get_shape()), [type_]))
        values = op.operands[1] if descriptor is not None else op.operands[2]
        mask = op.operands[3] if descriptor is None and len(op.operands) == 4 else None
        result = event(storage, byte_offsets, values, probes[word], append,
                       operation=primitive["operation"], byte_order=info["byte_order"], mask=mask,
                       desired=op.operands[2] if descriptor is not None else None)
        for generated_op in emitted:
            for nested in generated_op.walk():
                nested.location = op.location
        reports.append({"kind": "standard-byte-backed-word-atomic", "source_ids": ids,
                        "byte_view": info, "operation": primitive["operation"],
                        "atomicity": "one-word-region-per-active-original-lane"})
        Rewriter.replace_op(op, emitted, [result])
    for op in reversed(list(module.walk())):
        if opcode(op) in ("tts.make_tptr", "tts.make_gather_scatter_tptr", "tt.bitcast") and op.results and not any(v.uses for v in op.results):
            op.parent.erase_op(op)
    return reports

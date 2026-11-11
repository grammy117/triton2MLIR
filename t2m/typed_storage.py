"""Adapt width-changing logical roots without changing the public tensor ABI."""

import json
from dataclasses import replace

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .typed_storage_contract import SCHEMA, WIDTHS, result_guard_costs


def prepare(module, abi, source_ids, byte_order):
    """Reuse finite-root/control analysis; adapt only connected width-changing roots."""
    from .pointer_control import normalize_pointer_control
    from .pointer_views import element

    state = normalize_pointer_control(
        module, source_ids, isolate_memory=True, _inspect_roots=True
    )
    selected = set()

    def pointee(value):
        try:
            return element(value.type)
        except ValueError as error:
            from xdsl.ir import BlockArgument

            from .shared_tensor import opcode

            owner = (
                value.block.parent_op()
                if isinstance(value, BlockArgument)
                else value.owner
            )
            raise ConversionError(
                "typed storage requires supported scalar types and the same logical address space",
                opcode=opcode(owner),
                source_id=next(iter(source_ids(owner.location)), None),
            ) from error

    for value in state.values:
        roots = state.bases[value]
        scalar = (
            value.type.element_type
            if isinstance(value.type, builtin.TensorType)
            else value.type
        )
        if all(scalar == root.type for root in roots):
            continue
        view = pointee(value)
        if any(view.bitwidth != pointee(root).bitwidth for root in roots):
            selected.update(roots)
    # All alternatives of a selected pointer need the same offset/storage units.
    pending = True
    while pending:
        pending = False
        for roots in state.bases.values():
            if selected.intersection(roots) and not roots.issubset(selected):
                selected.update(roots)
                pending = True
    changes, records = {}, []
    for arg, (kind, buffer) in zip(state.entry.args, abi.source_bindings, strict=True):
        if kind != "buffer" or arg not in selected or buffer.dtype == "i8":
            continue
        size = WIDTHS.get(buffer.dtype)
        if size is None:
            raise ConversionError(
                "typed storage requires supported byte-aligned root words",
                opcode="tt.func",
            )
        storage = replace(buffer, dtype="i8", shape=(*buffer.shape, size))
        changes[buffer.name] = storage
        records.append(
            {
                "name": buffer.name,
                "logical_dtype": buffer.dtype,
                "logical_shape": list(buffer.shape),
                "storage_dtype": "i8",
                "storage_shape": list(storage.shape),
                "element_bytes": size,
                "effect": buffer.effect,
            }
        )
        original = arg.type
        from .ir import context
        from .shared_tensor import SharedParser

        ctx = context()
        ctx.allow_unregistered = True
        pointer = SharedParser(ctx, "!tt.ptr<i8>").parse_type()
        root = Rewriter.replace_value_with_new_type(arg, pointer)
        cast = builtin.UnregisteredOp.with_name("tt.bitcast").create(
            operands=[root], result_types=[original]
        )
        cast.location = state.function.location
        root.replace_uses_with_if(
            cast.results[0], lambda use, cast=cast: use.operation is not cast
        )
        state.entry.insert_op_before(cast, state.entry.first_op)
    if not changes:
        return abi, None
    state.function.properties["function_type"] = builtin.FunctionType.from_lists(
        [arg.type for arg in state.entry.args], []
    )
    profile = {
        "schema": SCHEMA,
        "byte_order": byte_order,
        "roots": records,
        "grants_semantics": False,
    }
    return replace(
        abi, buffers=tuple(changes.get(b.name, b) for b in abi.buffers)
    ), profile


def restore(module, public, storage, profile, base_cost):
    from .word_codec import transport

    old = next(op for op in module.ops if isinstance(op, func.FuncOp))
    block = Block(arg_types=[value.type for value in public.inputs])
    selected = {root["name"] for root in profile["roots"]}

    def append(op):
        op.location = old.location
        op.attributes.update(
            {
                "prov.family": builtin.StringAttr("abi"),
                "prov.op": builtin.StringAttr("typed-storage-transport"),
            }
        )
        block.add_op(op)

    mapper = {}
    for arg, value, source, target in zip(
        block.args, old.body.block.args, public.inputs, storage.inputs, strict=True
    ):
        mapper[value] = (
            transport(arg, target.type, profile["byte_order"], append)
            if source.name in selected
            else arg
        )
    for op in old.body.block.ops:
        if not isinstance(op, func.ReturnOp):
            block.add_op(op.clone(value_mapper=mapper))
    returned = [mapper.get(value, value) for value in old.body.block.last_op.operands]
    outputs = [
        transport(value, target.type, profile["byte_order"], append)
        if target.name in selected
        else value
        for value, target in zip(returned, public.outputs, strict=True)
    ]
    block.add_op(func.ReturnOp(*outputs))
    new = func.FuncOp(
        public.entry,
        ([v.type for v in public.inputs], [v.type for v in public.outputs]),
        Region(block),
    )
    new.attributes.update(old.attributes)
    new.attributes["prov.typed_storage_transport"] = builtin.StringAttr(
        json.dumps(profile, sort_keys=True)
    )
    costs = result_guard_costs(
        {
            "spec": public.as_dict(),
            "profile": "shared-source-word-buffer",
            "typed_storage_transport": profile,
        },
        base_cost,
    )
    new.attributes["prov.memory_guard_bytes_per_result_element"] = builtin.IntegerAttr(
        max(costs, default=base_cost), builtin.i64
    )
    new.attributes["prov.memory_guard_bytes_by_result_element"] = (
        builtin.DenseArrayBase.from_list(builtin.i64, costs)
    )
    Rewriter.replace_op(old, new)
    module.verify()
    return module

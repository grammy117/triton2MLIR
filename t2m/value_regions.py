"""Retain local numeric tensor computation inside scalar TTIR regions.

Meta/shared 17f0c234 and archived Microsoft 050b93a already supply shape,
splat/range and elementwise-to-Linalg conversions. Keep those operations in
place for that pipeline. Expand nested region containers from the leaves so
their complete bodies, source initialization and tuple order remain visible.
No pointer/effect admission or physical lane mapping follows from tensor types.
"""

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, ReduceOp, YieldOp

from .scalar_tile_reads import numeric_tile


def numeric_value(type_):
    return numeric_tile(type_) or isinstance(
        type_, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)
    )


def tensor_computation(operation):
    """Exact functional operations; all descendants are checked separately.

    The source native verifier establishes TTIR geometry/prototypes. This is
    only admission to the existing shared conversion, never a new definition
    of a TTIR shape operation or permission for unknown source properties.
    """
    from .shared_tensor import opcode

    standard = isinstance(
        operation,
        (
            tensor.CastOp,
            tensor.CollapseShapeOp,
            tensor.DimOp,
            tensor.EmptyOp,
            tensor.ExpandShapeOp,
            tensor.ExtractOp,
            tensor.ExtractSliceOp,
            tensor.FromElementsOp,
            tensor.InsertOp,
            tensor.InsertSliceOp,
            tensor.PadOp,
            tensor.ReshapeOp,
            tensor.SplatOp,
            tensor.YieldOp,
            GenericOp,
            ReduceOp,
            YieldOp,
        ),
    )
    if standard:
        return True
    fields = {
        "tt.make_range": {"start", "end"},
        "tt.splat": set(),
        "tt.broadcast": set(),
        "tt.expand_dims": {"axis"},
        "tt.trans": {"order"},
        "tt.reshape": {"allow_reorder", "efficient_layout"},
        "tt.join": set(),
        "tt.split": set(),
        "tt.cat": {"can_reorder"},
    }
    allowed = fields.get(opcode(operation))
    return (
        allowed is not None
        and not operation.regions
        and not operation.successors
        and all(
            key in {"op_name", "op_name__", *allowed} or key.startswith("prov.")
            for key in (*operation.properties, *operation.attributes)
        )
    )


def clone_body_operation(operation, mapping, generated_guards):
    """Transport guard authority only along the actual compiler clone edge."""
    cloned = operation.clone(value_mapper=mapping)
    if isinstance(generated_guards, set):
        for original, copy in zip(operation.walk(), cloned.walk(), strict=True):
            if original in generated_guards:
                generated_guards.add(copy)
    return cloned


def expand_value_regions(
    module, source_ids, *, generated_guards=(), memory_events=False, layouts=None
):
    """Validate/expand a detached copy; publish only the complete expansion.

    Mixed map/reduction/scan nesting needs one postorder, rather than three
    passes that each encounter an unconverted inner container. A failure in
    any candidate leaves every original operation and source guard unchanged.
    """
    from .elementwise_map import expand_elementwise_maps
    from .reductions import expand_reductions
    from .region_integer_casts import expand_region_integer_casts
    from .scans import expand_scans
    from .shared_tensor import opcode

    prepared = module.clone()
    guards = {
        copy
        for original, copy in zip(module.walk(), prepared.walk(), strict=True)
        if original in generated_guards
    }
    expand_region_integer_casts(prepared, source_ids)
    reductions, scans, maps, tensors = [], [], [], []
    containers = [operation for operation in prepared.walk()
                  if opcode(operation) in ("tt.reduce", "tt.scan", "tt.map_elementwise")]
    for operation in containers:
        local = [child for region in operation.regions for child in region.walk()
                 if any(numeric_tile(value.type) for value in child.results)]
        if local:
            tensors.append({
                "kind": "source-region-tensor-dataflow", "source_ids": source_ids(operation.location),
                "container": opcode(operation), "tensor_operations": len(local),
                "tensor_source_ids": list(dict.fromkeys(
                    identity for child in local for identity in source_ids(child.location))),
                "expansion": "functional-local-tensor-values-and-postorder-standard-regions",
                "physical_mapping": "undecided",
            })
    for operation in reversed(containers):
        name = opcode(operation)
        if name == "tt.reduce":
            reductions.extend(
                expand_reductions(
                    prepared, source_ids, generated_guards=guards,
                    operations=(operation,),
                )
            )
        elif name == "tt.scan":
            scans.extend(
                expand_scans(
                    prepared, source_ids, generated_guards=guards,
                    operations=(operation,),
                )
            )
        elif name == "tt.map_elementwise":
            maps.extend(
                expand_elementwise_maps(
                    prepared, source_ids, generated_guards=guards,
                    operations=(operation,), memory_events=memory_events,
                    layouts=layouts,
                )
            )
    for block in list(module.body.blocks):
        module.body.erase_block(block, safe_erase=False)
    prepared.body.move_blocks(module.body)
    return reductions, scans, maps, tensors

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
    from .shared_tensor import _source_ids, opcode

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
        # Pinned TritonOps.td declares these numeric/domain operations Pure.
        # Original preflight and the existing converters retain their complete
        # numeric/property contracts. Admission here supplies no new expansion,
        # memory effect, physical layout or source precision policy.
        "tt.dot": {"inputPrecision", "maxNumImpreciseAcc"},
        "tt.dot_scaled": {"a_elem_type", "b_elem_type", "fastMath", "lhs_k_pack",
                          "rhs_k_pack", "operandSegmentSizes"},
        "tt.histogram": set(),
        "tt.gather": {"axis", "efficient_layout"},
        "tt.bitcast": set(),
        "tt.fp_to_fp": {"rounding"},
        "tt.precise_divf": set(),
        "tt.precise_sqrt": set(),
        "tt.clampf": {"propagateNan"},
        "tt.mulhiui": set(),
        "tt.get_program_id": {"axis"},
        "tt.get_num_programs": {"axis"},
        "tt.extern_elementwise": {"pure", "symbol", "libname", "libpath"},
    }
    allowed = fields.get(opcode(operation))
    if allowed is not None and opcode(operation) == "tt.extern_elementwise":
        # Externals have conditional effects rather than a Pure trait. Retain
        # the established library/type/prototype gate, including unused calls.
        from .external_math import contract
        contract(operation, next(iter(_source_ids(operation.location)), None))
    return (
        allowed is not None
        and not operation.regions
        and not operation.successors
        and all(
            key in {"op_name", "op_name__", *allowed} or key.startswith("prov.")
            for key in (*operation.properties, *operation.attributes)
        )
    )


def pure_value_region(operation, *, maximum_operations=1024):
    """Recognize numeric regions without granting their conversion contracts.

    Inspect every descendant, including unused instructions and captures.
    Parent Pure traits and provenance never hide an effect. Source containers
    remain intact for the existing postorder expansion, which separately checks
    geometry, initialization, tuple order, packing and numeric semantics.
    """
    from xdsl.dialects import arith, scf
    from xdsl.traits import is_side_effect_free

    from .shared_tensor import opcode

    if not operation.regions:
        return False
    containers = {"tt.reduce": {"axis"}, "tt.scan": {"axis", "reverse"},
                  "tt.map_elementwise": {"pack"}}
    terminals = {"tt.reduce.return", "tt.scan.return", "tt.map_elementwise.return"}
    pending, count = [(operation, 0)], 0
    while pending:
        child, depth = pending.pop()
        count += 1
        if count > maximum_operations or depth > 32 or child.successors:
            return False
        values = [*child.operands, *child.results,
            *(arg for region in child.regions for block in region.blocks for arg in block.args)]
        if not all(numeric_value(value.type) for value in values):
            return False
        name = opcode(child)
        if name in containers:
            if len(child.regions) != 1 or any(
                    key not in {"op_name", "op_name__", *containers[name]} and not key.startswith("prov.")
                    for key in (*child.properties, *child.attributes)):
                return False
        elif isinstance(child, (scf.IfOp, scf.ForOp, scf.WhileOp)):
            pass
        elif child.regions:
            if not tensor_computation(child):
                return False
        elif name in terminals or isinstance(child, (scf.YieldOp, scf.ConditionOp)):
            pass
        elif not (tensor_computation(child) or child.name.split(".", 1)[0] in {"arith", "math"}
                and (is_side_effect_free(child) or isinstance(child, (arith.ExtSIOp, arith.RemUIOp)))):
            return False
        for region in child.regions:
            if len(region.blocks) != 1:
                # Verified SCF permits an absent false region on a void if.
                if isinstance(child, scf.IfOp) and region is child.false_region and not region.blocks and not child.results:
                    continue
                return False
            pending.extend((nested, depth + 1) for nested in region.block.ops)
    return True


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
    from .region_launch_queries import normalize_region_launch_queries
    from .scans import expand_scans
    from .shared_tensor import opcode

    prepared = module.clone()
    guards = {
        copy
        for original, copy in zip(module.walk(), prepared.walk(), strict=True)
        if original in generated_guards
    }
    launch_reports = normalize_region_launch_queries(prepared, source_ids)
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
    return reductions, scans, maps, [*launch_reports, *tensors]

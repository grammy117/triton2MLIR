"""Portable inclusive associative scans using parallel tensor doubling stages.

The shared CumSumConverter (Meta 17f0c234) accepts only a single add combiner
and emits ttx.cumsum, outside the functional tensor ABI. This expansion instead
preserves the general source combiner, tuple results, axis and traversal order.
No identity is invented and no physical lane/warp mapping is selected.
"""

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.traits import is_side_effect_free
from xdsl.utils.exceptions import VerifyException

from .diagnostics import ConversionError


def audit_source_combiners(module, source_ids):
    """Reject unimplemented combiner effects before inlining or folding.

    Follow complete original helper bodies, including unused effects. Native
    Pure on a parent reduction is not permission to discard its nested reads
    or diagnostics. This source audit grants no scalar-operation admission;
    the complete inlined body still passes the later typed combiner checker.
    """
    from .shared_tensor import opcode

    functions = {
        operation.properties["sym_name"].data: operation
        for operation in module.ops
        if opcode(operation) == "tt.func"
    }
    checked = set()

    def visit(operation, active):
        name = opcode(operation)
        if (
            name
            in {
                "tt.load",
                "tt.store",
                "tt.atomic_rmw",
                "tt.atomic_cas",
                "tt.print",
                "tt.assert",
                "cf.assert",
            }
            or name.startswith(("tt.descriptor_", "memref."))
            or name == "func.call"
        ):
            raise ConversionError(
                "source reduction/scan combiner effects require an executable region contract before normalization",
                opcode=name,
                source_id=next(iter(source_ids(operation.location)), None),
            )
        if name == "tt.call":
            callee = operation.properties.get("callee")
            symbol = (
                callee.root_reference.data
                if isinstance(callee, builtin.SymbolRefAttr)
                and not callee.nested_references
                else None
            )
            if symbol not in functions or symbol in active:
                raise ConversionError(
                    "source reduction/scan combiner requires a complete nonrecursive helper body",
                    opcode=name,
                    source_id=next(iter(source_ids(operation.location)), None),
                )
            if symbol not in checked:
                for child in functions[symbol].regions[0].walk():
                    visit(child, active | {symbol})
                checked.add(symbol)

    for operation in module.walk():
        if opcode(operation) in ("tt.reduce", "tt.scan"):
            for region in operation.regions:
                for child in region.walk():
                    visit(child, set())


def _scalar_combiner(block, terminal, refuse, source_ids, *, family="scan"):
    """Check complete scalar execution, including nested controls and captures.

    Only exact SCF value regions have recursive-effect permission. Inspect every
    nested operation, including unused effects. No loop is unrolled, reordered
    or discarded here; source termination and per-call state remain executable.
    """
    controls, ids, control_ops = {}, [], []
    for child in block.walk():
        ids.extend(i for i in source_ids(child.location) if i not in ids)
        if child is terminal:
            continue
        control = isinstance(child, (scf.IfOp, scf.ForOp, scf.WhileOp))
        end = isinstance(child, (scf.YieldOp, scf.ConditionOp))
        arithmetic = (
            child.name.split(".")[0] in ("arith", "math")
            and not child.regions
            # xDSL 0.68 omits the standard no-memory-effect trait on these
            # exact arithmetic classes. Their original control position is
            # retained; this grants no speculation or divisor validity.
            and (
                is_side_effect_free(child)
                or isinstance(child, (arith.ExtSIOp, arith.RemUIOp))
            )
        )
        if not (control or end or arithmetic):
            name = (
                child.op_name.data
                if child.name == "builtin.unregistered"
                else child.name
            )
            refuse(
                f"{family} combiner requires pure scalar arithmetic and structured value control",
                opcode=name,
            )
        values = [
            *child.operands,
            *child.results,
            *(
                arg
                for region in child.regions
                for region_block in region.blocks
                for arg in region_block.args
            ),
        ]
        if not all(
            isinstance(
                value.type, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)
            )
            for value in values
        ):
            refuse(
                f"{family} combiner cannot capture tensors, addresses or memory state",
                opcode=child.name,
            )
        if control:
            if any(len(region.blocks) != 1 for region in child.regions):
                refuse(
                    f"{family} combiner controls require single-block structured regions",
                    opcode=child.name,
                )
            controls[child.name] = controls.get(child.name, 0) + 1
            control_ops.append(child)
    for child in control_ops:
        try:
            child.verify()
        except VerifyException as error:
            refuse(
                f"{family} combiner structured control fails verification: "
                + str(error),
                opcode=child.name,
            )
    return controls, ids


def expand_scans(module, source_ids):
    reports, plans = [], []

    def name(op):
        return op.op_name.data if op.name == "builtin.unregistered" else op.name

    for scan in list(module.walk()):
        if name(scan) != "tt.scan":
            continue
        ids = source_ids(scan.location)

        def refuse(message, opcode="tt.scan"):
            raise ConversionError(
                message, source_id=next(iter(ids), None), opcode=opcode
            )

        types = [v.type for v in scan.operands]
        if (
            not types
            or len(types) != len(scan.results)
            or any(not isinstance(t, TensorType) for t in types)
            or any(t != v.type for t, v in zip(types, scan.results, strict=True))
        ):
            refuse("scan operands/results need matching ranked tensor types")
        shape = types[0].get_shape()
        if any(t.get_shape() != shape for t in types) or any(n <= 0 for n in shape):
            refuse("scan needs equal positive static tensor shapes")
        axis, reverse = scan.properties.get("axis"), scan.properties.get("reverse")
        if (
            not isinstance(axis, IntegerAttr)
            or not 0 <= axis.value.data < len(shape)
            or not isinstance(reverse, IntegerAttr)
            or reverse.value.data not in (0, -1, 1)
        ):
            refuse("scan axis/reverse attributes are not qualified")
        axis, reverse = axis.value.data, bool(reverse.value.data)
        if len(scan.regions) != 1 or len(scan.regions[0].blocks) != 1:
            refuse("scan needs one scalar combiner block")
        combiner = scan.regions[0].block
        element_types = [t.get_element_type() for t in types]
        if [v.type for v in combiner.args] != element_types * 2:
            refuse("scan combiner argument types differ from the tuple tensors")
        terminator = combiner.last_op
        if (
            terminator is None
            or name(terminator) != "tt.scan.return"
            or [v.type for v in terminator.operands] != element_types
        ):
            refuse("scan combiner must return every tuple component")
        if not all(
            isinstance(t, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))
            for t in element_types
        ):
            refuse("scan tuple elements must be scalar numeric values")
        controls, combiner_ids = _scalar_combiner(
            combiner, terminator, refuse, source_ids
        )
        if not all(
            isinstance(
                value.type, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)
            )
            for value in terminator.operands
        ):
            refuse("scan combiner returns must be scalar numeric values")
        plans.append(
            (
                scan,
                ids,
                types,
                shape,
                axis,
                reverse,
                combiner,
                terminator,
                element_types,
                controls,
                combiner_ids,
            )
        )

    # Every source candidate is validated before attaching any expansion.
    for (
        scan,
        ids,
        types,
        shape,
        axis,
        reverse,
        combiner,
        terminator,
        element_types,
        controls,
        combiner_ids,
    ) in plans:
        emitted = []

        def append(op):
            op.location = scan.location
            op.attributes.update(
                {
                    "prov.family": StringAttr("scan"),
                    "prov.op": StringAttr("associative_scan"),
                    "prov.composite_family": StringAttr("scan"),
                    "prov.composite_id": StringAttr(
                        "scan:" + (ids[0] if ids else str(scan.location))
                    ),
                    "prov.scan_algorithm": StringAttr("inclusive-parallel-doubling"),
                    "prov.scan_axis": IntegerAttr(axis, builtin.i64),
                    "prov.scan_reverse": IntegerAttr(int(reverse), builtin.i1),
                }
            )
            emitted.append(op)
            return op

        values, distance, stages = list(scan.operands), 1, 0
        while distance < shape[axis]:
            sizes = list(shape)
            sizes[axis] -= distance
            current_offsets, prior_offsets = [0] * len(shape), [0] * len(shape)
            (prior_offsets if reverse else current_offsets)[axis] = distance
            prior = [
                append(
                    tensor.ExtractSliceOp.from_static_parameters(
                        v, prior_offsets, sizes
                    )
                ).result
                for v in values
            ]
            current = [
                append(
                    tensor.ExtractSliceOp.from_static_parameters(
                        v, current_offsets, sizes
                    )
                ).result
                for v in values
            ]
            sliced_types = [TensorType(t, sizes) for t in element_types]
            outputs = [append(tensor.EmptyOp([], t)).tensor for t in sliced_types]
            body = Block(arg_types=element_types * 3)
            mapper = dict(zip(combiner.args, body.args[: 2 * len(types)], strict=True))
            for op in list(combiner.ops)[:-1]:
                body.add_op(op.clone(value_mapper=mapper))
            returned = YieldOp(*(mapper.get(v, v) for v in terminator.operands))
            returned.location = terminator.location
            body.add_op(returned)
            combined = append(
                GenericOp(
                    [*prior, *current],
                    outputs,
                    Region(body),
                    [AffineMapAttr(AffineMap.identity(len(shape)))] * (3 * len(types)),
                    [IteratorTypeAttr.parallel()] * len(shape),
                    sliced_types,
                )
            )
            combined.attributes["prov.scan_distance"] = IntegerAttr(
                distance, builtin.i64
            )
            if controls:
                combined.attributes["prov.scan_combiner_control"] = (
                    builtin.DictionaryAttr(
                        {
                            kind: IntegerAttr(count, builtin.i64)
                            for kind, count in controls.items()
                        }
                    )
                )
            values = [
                append(
                    tensor.InsertSliceOp.from_static_parameters(
                        part, full, current_offsets, sizes
                    )
                ).result
                for part, full in zip(combined.results, values, strict=True)
            ]
            distance *= 2
            stages += 1
        parent = scan.parent
        parent.insert_ops_before(emitted, scan)
        for old, new in zip(scan.results, values, strict=True):
            old.replace_all_uses_with(new)
        reports.append(
            {
                "kind": "associative-scan-network",
                "source_ids": ids,
                "location": str(scan.location),
                "axis": axis,
                "reverse": reverse,
                "tuple_arity": len(types),
                "stages": stages,
                "algorithm": "inclusive-parallel-doubling",
                "numeric_contract": "source-associative-combiner",
                "identity": "none-invented",
                "physical_mapping": "undecided",
                **(
                    {
                        "combiner_control_flow": controls,
                        "combiner_source_ids": combiner_ids,
                    }
                    if controls
                    else {}
                ),
            }
        )
        parent.erase_op(scan)
    return reports

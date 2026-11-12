"""Re-raise closed pointer-integer expressions without choosing an address.

The pinned TT casts use i64. A value here denotes P + D modulo 2**64,
where P is an actual source pointer and D is executable integer arithmetic.
Only conversion back to a pointer may consume that absolute integer. Physical
observations and unproved byte-to-element conversions remain located refusals.
Shared's base/offset and memory conversions still own the resulting pointers.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin
from xdsl.utils.exceptions import ParseError

from .diagnostics import ConversionError

SHAPES = {
    "tt.splat",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.trans",
    "tt.reshape",
    "tt.unsplat",
}


def _scalar(type_):
    return type_.element_type if isinstance(type_, builtin.TensorType) else type_


def _like(type_, element):
    return (
        builtin.TensorType(element, type_.get_shape(), type_.encoding)
        if isinstance(type_, builtin.TensorType)
        else element
    )


def _constant(type_, number):
    if isinstance(type_, builtin.TensorType):
        return arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(type_, [number])
        )
    return arith.ConstantOp.from_int_and_width(number, type_)


def _low_zeros(value, cache, depth=0):
    if value in cache:
        return cache[value]
    if depth >= 128 or len(cache) >= 8192:
        raise ValueError("pointer divisibility proof exceeds its bounded SSA budget")
    cache[value] = 0
    result = _factor(value, lambda v: _low_zeros(v, cache, depth + 1))
    cache[value] = result
    return result


def _factor(value, known):
    """A conservative power-of-two divisor of every lane, capped at 3."""
    from .shared_tensor import opcode

    op = value.owner
    if not hasattr(op, "name"):
        return 0
    name = opcode(op)
    if name == "arith.constant":
        attr = op.properties.get("value")
        if isinstance(attr, builtin.IntegerAttr):
            numbers = [attr.value.data]
        elif isinstance(attr, builtin.DenseIntOrFPElementsAttr) and isinstance(
            _scalar(value.type), builtin.IntegerType
        ):
            numbers = attr.get_values()
        else:
            return 0
        return min(
            (3 if n == 0 else min(3, (n & -n).bit_length() - 1) for n in numbers),
            default=0,
        )
    if (
        name in SHAPES | {"arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast"}
        and len(op.operands) == 1
    ):
        return known(op.operands[0])
    if name in {"arith.addi", "arith.subi", "arith.ori", "arith.xori"}:
        return min(known(v) for v in op.operands)
    if name in {"arith.muli", "arith.andi"}:
        factors = [known(v) for v in op.operands]
        return min(3, sum(factors)) if name == "arith.muli" else max(factors)
    if name == "arith.select":
        return min(known(v) for v in op.operands[1:])
    if name == "arith.shli":
        shift = op.operands[1].owner
        attr = shift.properties.get("value") if hasattr(shift, "properties") else None
        numbers = (
            [attr.value.data]
            if isinstance(attr, builtin.IntegerAttr)
            else list(attr.get_values())
            if isinstance(attr, builtin.DenseIntOrFPElementsAttr)
            else []
        )
        if numbers and all(0 <= n < _scalar(value.type).bitwidth for n in numbers):
            return min(3, known(op.operands[0]) + min(numbers))
    return 0


@dataclass(frozen=True)
class Address:
    pointer: object
    displacement: object
    sources: frozenset
    origin: object


def normalize_pointer_addresses(module, source_ids):
    """Validate the whole address-use closure before committing any mutation."""
    from .pointer_views import element
    from .shared_tensor import opcode

    states, plans, reports = {}, [], []
    divisibility = {}
    emitted = []
    committed = False

    def refuse(op, message):
        raise ConversionError(
            message,
            opcode=opcode(op),
            source_id=next(iter(source_ids(op.location)), None),
        )

    def neutral(op):
        flags = op.properties.get("overflowFlags", op.attributes.get("overflowFlags"))
        if flags is not None and (
            not isinstance(flags, arith.IntegerOverflowAttr) or flags.data
        ):
            refuse(
                op,
                "pointer integer arithmetic requires modular, non-poisoning overflow semantics",
            )

    try:
        for op in list(module.walk()):
            name = opcode(op)
            affected = [v for v in op.operands if v in states]
            if name not in {"tt.ptr_to_int", "tt.int_to_ptr"} and not affected:
                continue
            emitted = []
            sources = frozenset(source_ids(op.location)).union(
                *(states[v].sources for v in affected)
            )
            if len(sources) > 4096:
                refuse(op, "pointer integer expression exceeds its 4096-source budget")
            if name == "tt.ptr_to_int":
                if (
                    len(op.operands) != 1
                    or len(op.results) != 1
                    or _scalar(op.results[0].type) != builtin.i64
                ):
                    refuse(
                        op,
                        "pointer integers require the pinned scalar/tensor i64 interface",
                    )
                try:
                    element(op.operands[0].type)
                except (ValueError, ParseError):
                    refuse(
                        op,
                        "pointer integers require a logical byte-aligned pointer type",
                    )
                zero = _constant(op.results[0].type, 0)
                emitted.append(zero)
                states[op.results[0]] = Address(
                    op.operands[0], zero.result, sources, op
                )
            elif name == "tt.int_to_ptr":
                if (
                    len(op.operands) != 1
                    or op.operands[0] not in states
                    or len(op.results) != 1
                ):
                    refuse(
                        op,
                        "integer-derived pointers require a closed original pointer base",
                    )
                address = states[op.operands[0]]
                try:
                    width = element(op.results[0].type).bitwidth // 8
                except (ValueError, ParseError):
                    refuse(
                        op,
                        "integer-derived pointers require a logical byte-aligned pointer type",
                    )
                try:
                    low_zeros = _low_zeros(address.displacement, divisibility)
                except ValueError as error:
                    refuse(op, str(error))
                if width not in (1, 2, 4, 8) or low_zeros < width.bit_length() - 1:
                    refuse(
                        op,
                        "pointer byte displacement is not proved divisible by its element width",
                    )
                base = address.pointer
                if _like(base.type, _scalar(op.results[0].type)) != op.results[0].type:
                    refuse(
                        op,
                        "integer-derived pointer shapes or encodings differ from their source base",
                    )
                if base.type != op.results[0].type:
                    cast = builtin.UnregisteredOp.with_name("tt.bitcast").create(
                        operands=[base], result_types=[op.results[0].type]
                    )
                    emitted.append(cast)
                    base = cast.results[0]
                offset = address.displacement
                if width != 1:
                    divisor = _constant(offset.type, width)
                    quotient = arith.DivSIOp(offset, divisor.result)
                    emitted.extend((divisor, quotient))
                    offset = quotient.result
                pointer = builtin.UnregisteredOp.with_name("tt.addptr").create(
                    operands=[base, offset], result_types=[op.results[0].type]
                )
                emitted.append(pointer)
                reports.append(
                    {
                        "kind": "closed-pointer-integer-displacement",
                        "source_ids": sorted(sources),
                        "address_bits": 64,
                        "element_bytes": width,
                        "expansion": "original-base-plus-exact-signed-byte-displacement-in-element-units",
                        "arithmetic": "modulo-2^64; source-offset-computation-retained",
                        "physical_address": "unobserved; no base integer materialized",
                    }
                )
            elif name in {"arith.addi", "arith.subi"}:
                neutral(op)
                if (
                    len(affected) != 1
                    or len(op.operands) != 2
                    or len(op.results) != 1
                    or name == "arith.subi"
                    and op.operands[0] not in states
                ):
                    refuse(
                        op,
                        "pointer integer arithmetic requires one base with an independent displacement",
                    )
                address = states[affected[0]]
                operands = [
                    states[v].displacement if v in states else v for v in op.operands
                ]
                result = (arith.AddiOp if name == "arith.addi" else arith.SubiOp)(
                    *operands
                )
                emitted.append(result)
                states[op.results[0]] = Address(
                    address.pointer, result.result, sources, address.origin
                )
            elif name == "arith.select":
                if op.operands[0] in states or any(
                    v not in states for v in op.operands[1:]
                ):
                    refuse(
                        op,
                        "pointer integer selection requires two closed bases and an independent condition",
                    )
                left, right = (states[v] for v in op.operands[1:])
                if left.pointer.type != right.pointer.type:
                    refuse(
                        op,
                        "pointer integer selection requires matching logical pointer types",
                    )
                pointer = arith.SelectOp(op.operands[0], left.pointer, right.pointer)
                delta = arith.SelectOp(
                    op.operands[0], left.displacement, right.displacement
                )
                emitted.extend((pointer, delta))
                states[op.results[0]] = Address(
                    pointer.result, delta.result, sources, left.origin
                )
            elif (
                name in SHAPES
                or name == "arith.bitcast"
                and op.operands[0].type == op.results[0].type
            ):
                if len(op.operands) != 1 or len(op.results) != 1:
                    refuse(
                        op,
                        "pointer integer shape conversion requires one value and result",
                    )
                address = states[op.operands[0]]
                if name == "arith.bitcast":
                    states[op.results[0]] = Address(
                        address.pointer, address.displacement, sources, address.origin
                    )
                else:
                    if name == "tt.reshape" and (
                        "allow_reorder" in op.properties
                        or "allow_reorder" in op.attributes
                    ):
                        refuse(
                            op,
                            "pointer integer reshape needs established paired lane ordering",
                        )
                    pointer = builtin.UnregisteredOp.with_name(name).create(
                        operands=[address.pointer],
                        result_types=[
                            _like(op.results[0].type, _scalar(address.pointer.type))
                        ],
                        properties=dict(op.properties),
                    )
                    delta = op.clone(
                        value_mapper={op.operands[0]: address.displacement}
                    )
                    delta.attributes = {
                        k: v
                        for k, v in delta.attributes.items()
                        if k in ("op_name", "op_name__") or k.startswith("prov.")
                    }
                    emitted.extend((pointer, delta))
                    states[op.results[0]] = Address(
                        pointer.results[0], delta.results[0], sources, address.origin
                    )
            else:
                refuse(
                    states[affected[0]].origin,
                    "physical pointer integers observe or escape their closed displacement expression"
                    f"; escape at {next(iter(source_ids(op.location)), 'unknown source')} ({name})",
                )
            for created in emitted:
                created.location = builtin.FusedLoc(
                    builtin.ArrayAttr(
                        [
                            op.location,
                            *(
                                builtin.NameLoc(
                                    builtin.StringAttr(s), builtin.UnknownLoc()
                                )
                                for s in sorted(sources)
                            ),
                        ]
                    ),
                    builtin.NoneAttr(),
                )
            plans.append((op, emitted, name == "tt.int_to_ptr"))
        # Absolute integers disappear only after all of their users are checked.
        for op, emitted, terminal in plans:
            op.parent.insert_ops_before(emitted, op)
            if terminal:
                op.results[0].replace_all_uses_with(emitted[-1].results[0])
        for op, _, _ in reversed(plans):
            op.parent.erase_op(op)
        committed = True
        return reports
    finally:
        if not committed:
            planned = [item for _, recipe, _ in plans for item in recipe]
            for created in reversed(
                [*planned, *(item for item in emitted if item not in planned)]
            ):
                if created.parent is None:
                    created.drop_all_references()

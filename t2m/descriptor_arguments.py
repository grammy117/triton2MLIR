"""Describe the actual upstream one-to-many logical descriptor interface.

Only native-verified type signatures enter this adapter. It supplies no memory,
alias or numeric rewrite permission; the official pass emits those operations.
"""

from xdsl.dialects.builtin import IntegerType, TensorType, UnregisteredAttr
from xdsl.parser import Parser

from .diagnostics import ConversionError
from .ir import context

SCHEMA = "t2m.logical-descriptor-arguments.v1"


def has_parameters(functions):
    from xdsl.utils.exceptions import ParseError

    ctx = context()
    ctx.allow_unregistered = True
    for arguments in functions.values():
        for spelling in arguments:
            try:
                type_ = Parser(ctx, spelling).parse_type()
            except ParseError:
                continue  # Other native types are checked by their own intake adapters.
            if isinstance(type_, UnregisteredAttr) and type_.attr_name.data == "tt.tensordesc":
                return True
    return False


def expansions(original, normalized):
    ctx = context()
    ctx.allow_unregistered = True
    functions = []
    for name, arguments in original.items():
        records, position, found = [], 0, False
        for index, spelling in enumerate(arguments):
            type_ = Parser(ctx, spelling).parse_type()
            descriptor = (
                isinstance(type_, UnregisteredAttr)
                and type_.attr_name.data == "tt.tensordesc"
            )
            if descriptor:
                found = True
                block = Parser(ctx, type_.value.data).parse_type()
                if not isinstance(block, TensorType):
                    raise ConversionError(
                        "logical descriptor argument requires a ranked block type"
                    )
                element = block.element_type
                if isinstance(element, IntegerType):
                    element = IntegerType(element.width.data)
                rank = len(block.get_shape())
                fields = [("base", f"!tt.ptr<{element}>")]
                fields += [(f"shape.{i}", "i64") for i in range(rank)]
                fields += [(f"stride.{i}", "i64") for i in range(rank)]
                fields += [("padding_nan", "i1"), ("round_f32_to_tf32", "i1")]
            else:
                fields = [("value", spelling)]
            record = {
                "source_argument": index,
                "source_type": spelling,
                "descriptor": descriptor,
                "fields": [
                    {"role": role, "type": dtype, "expanded_argument": position + i}
                    for i, (role, dtype) in enumerate(fields)
                ],
            }
            if descriptor:
                record["block_shape"] = list(block.get_shape())
            records.append(record)
            position += len(fields)
        if not found:
            continue
        actual = normalized.get(name)
        expected = [
            field["type"] for argument in records for field in argument["fields"]
        ]
        if actual != expected:
            raise ConversionError(
                "native descriptor argument expansion differs from the pinned logical field layout"
            )
        functions.append(
            {"entry": name, "expanded_argument_count": position, "arguments": records}
        )
    return (
        {"schema": SCHEMA, "functions": functions, "grants_semantics": False}
        if functions
        else None
    )


def bind_entry(expansion, abi):
    if expansion is None:
        return None
    if expansion.get("schema") != SCHEMA:
        raise ConversionError("unknown logical descriptor argument interface")
    matches = [
        function
        for function in expansion["functions"]
        if function["entry"] == abi.entry
    ]
    if not matches:
        return None
    (function,) = matches
    bindings = abi.source_bindings
    if len(bindings) != function["expanded_argument_count"]:
        raise ConversionError(
            "external descriptor ABI must bind every expanded base, shape, stride, padding and rounding field"
        )
    records = []
    for argument in function["arguments"]:
        fields = []
        for field in argument["fields"]:
            kind, value = bindings[field["expanded_argument"]]
            if kind == "buffer":
                type_ = f"!tt.ptr<{value.dtype}>"
                bound = {"kind": "allocation", "name": value.name}
            elif kind == "runtime":
                type_ = value.dtype
                bound = {"kind": "scalar", "name": value.name, "bounds": value.bounds}
                if field["role"].startswith("shape.") and (
                    value.bounds is None or value.bounds[0] < 0
                ):
                    raise ConversionError(
                        "descriptor shape fields require declared nonnegative runtime bounds"
                    )
            else:
                type_ = field["type"]
                if type_ not in ("i1", "i32", "i64") or not (
                    value in (0, 1)
                    if type_ == "i1"
                    else -(2 ** (int(type_[1:]) - 1))
                    <= value
                    < 2 ** (int(type_[1:]) - 1)
                ):
                    raise ConversionError(
                        "descriptor field specialization must fit its exact integer type"
                    )
                if field["role"].startswith("shape.") and value < 0:
                    raise ConversionError("descriptor shape fields must be nonnegative")
                bound = {"kind": "constant", "value": value}
            if type_ != field["type"]:
                raise ConversionError(
                    "descriptor field binding differs from its actual expanded native type"
                )
            fields.append({**field, "binding": bound})
        records.append({**argument, "fields": fields})
    return {
        "schema": SCHEMA,
        "entry": abi.entry,
        "arguments": records,
        "representation": "official-pre-layout-one-to-many-logical-fields",
        "grants_semantics": False,
    }

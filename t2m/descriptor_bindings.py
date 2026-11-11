"""Bind original descriptor arguments to roots before native helper inlining."""

from dataclasses import asdict, dataclass

from .diagnostics import ConversionError

ARGUMENTS_SCHEMA = "t2m.source-allocation-arguments.v2"


def _integer(value):
    return (
        type(value) is int
        and -(2**63) <= value < 2**63
        or type(value) is str
        and value.isidentifier()
    )


@dataclass(frozen=True)
class DescriptorArgument:
    allocation: str
    shape: tuple[int | str, ...]
    strides: tuple[int | str, ...]
    offset: int | str = 0
    padding_nan: bool | str = False
    round_f32_to_tf32: bool | str = False

    def __post_init__(self):
        if (
            type(self.allocation) is not str
            or not self.allocation.isidentifier()
            or not _integer(self.offset)
        ):
            raise ValueError(
                "descriptor allocation/origin requires an identifier and signed integer binding"
            )
        if (
            not isinstance(self.shape, (tuple, list))
            or not isinstance(self.strides, (tuple, list))
            or not self.shape
            or len(self.shape) != len(self.strides)
            or any(not _integer(v) or type(v) is int and v < 0 for v in self.shape)
            or any(not _integer(v) for v in self.strides)
        ):
            raise ValueError(
                "descriptor requires matching nonnegative shapes and signed element strides"
            )
        for flag in (self.padding_nan, self.round_f32_to_tf32):
            if type(flag) is not bool and (
                type(flag) is not str or not flag.isidentifier()
            ):
                raise ValueError(
                    "descriptor flags require Boolean constants or declared scalar names"
                )
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "strides", tuple(self.strides))


@dataclass(frozen=True)
class ExtendScalar:
    """An executable signed i32-to-i64 lift for a logical descriptor field."""

    name: str


def expand(bindings, expansion, source_entry, abi):
    from .allocation_bindings import PointerArgument, ScalarArgument

    functions = expansion.get("functions", []) if expansion else []
    matches = [function for function in functions if function["entry"] == source_entry]
    if not matches:
        if any(isinstance(binding, DescriptorArgument) for binding in bindings):
            raise ConversionError(
                "descriptor binding needs an actual native descriptor parameter expansion"
            )
        return tuple(bindings), None
    (function,) = matches
    if len(bindings) != len(function["arguments"]):
        raise ConversionError(
            "descriptor allocation bindings require the complete original argument roster"
        )
    scalars = {scalar.name: scalar for scalar in abi.runtime_scalars}
    flat, descriptors = [], []
    for original, binding in zip(function["arguments"], bindings, strict=True):
        if not original["descriptor"]:
            if isinstance(binding, DescriptorArgument):
                raise ConversionError(
                    "descriptor binding supplied for an ordinary source argument"
                )
            flat.append(binding)
            continue
        if not isinstance(binding, DescriptorArgument) or len(binding.shape) != len(
            original["block_shape"]
        ):
            raise ConversionError(
                "native descriptor argument needs a matching typed logical descriptor binding"
            )
        flat.append(PointerArgument(binding.allocation, binding.offset))
        values = [
            *binding.shape,
            *binding.strides,
            binding.padding_nan,
            binding.round_f32_to_tf32,
        ]
        for field, value in zip(original["fields"][1:], values, strict=True):
            if type(value) is str:
                scalar = scalars.get(value)
                if scalar is None:
                    raise ConversionError(
                        "descriptor field needs a declared runtime scalar"
                    )
                shape = field["role"].startswith("shape.")
                if shape and (scalar.bounds is None or scalar.bounds[0] < 0):
                    raise ConversionError(
                        "descriptor shapes require nonnegative declared runtime bounds"
                    )
                if scalar.dtype == "i32" and field["type"] == "i64":
                    flat.append(ExtendScalar(value))
                elif scalar.dtype == field["type"]:
                    flat.append(ScalarArgument(value))
                else:
                    raise ConversionError(
                        "descriptor field type requires its actual native width or signed i32 extension"
                    )
            else:
                flat.append(int(value))
        descriptors.append(
            {
                "source_argument": original["source_argument"],
                "source_type": original["source_type"],
                "block_shape": original["block_shape"],
                "fields": original["fields"],
                **asdict(binding),
            }
        )
    if len(flat) != function["expanded_argument_count"]:
        raise ConversionError(
            "descriptor binding expansion differs from the actual native interface"
        )
    return tuple(flat), {
        "schema": ARGUMENTS_SCHEMA,
        "source_entry": source_entry,
        "allocation_entry": abi.entry,
        "arguments": descriptors,
        "grants_semantics": False,
    }

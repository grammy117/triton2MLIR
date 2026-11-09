"""Runtime scalar arguments keep source position, type and by-value semantics."""

from dataclasses import dataclass

from xdsl.dialects import builtin


@dataclass(frozen=True)
class RuntimeScalar:
    name: str
    dtype: str
    source_arg: int
    bounds: tuple[int, int] | None = None

    def __post_init__(self):
        if (not self.name.isidentifier() or self.dtype not in ("f32", "f64", "i1", "i8", "i16", "i32", "i64")
                or type(self.source_arg) is not int or self.source_arg < 0):
            raise ValueError("runtime scalar requires a name, supported scalar dtype and source argument position")
        if self.bounds is not None:
            width = getattr(getattr(self.type, "width", None), "data", 0)
            if (self.dtype not in ("i8", "i16", "i32", "i64") or len(self.bounds) != 2
                    or any(type(n) is not int for n in self.bounds)
                    or not -(1 << (width - 1)) <= self.bounds[0] <= self.bounds[1] < 1 << (width - 1)):
                raise ValueError("runtime scalar bounds require a nonempty inclusive range within the signed integer type")
            object.__setattr__(self, "bounds", tuple(self.bounds))

    def as_dict(self):
        fields = {"name": self.name, "dtype": self.dtype, "source_arg": self.source_arg}
        if self.bounds is not None:
            fields["bounds"] = self.bounds
        return fields

    @property
    def type(self):
        return getattr(builtin, self.dtype)

    @property
    def shape(self):
        # The array archive stores one scalar value; the MLIR ABI remains plain
        # scalar, distinct from a tensor with rank zero.
        return ()


def guard_scalar(scalar, value, block, location):
    """Materialize an ABI premise before using its range as an analysis fact."""
    from xdsl.dialects import arith, cf
    if scalar.bounds is None:
        return
    low, high = (arith.ConstantOp.from_int_and_width(n, value.type) for n in scalar.bounds)
    lower, upper = arith.CmpiOp(value, low, "sge"), arith.CmpiOp(value, high, "sle")
    within = arith.AndIOp(lower, upper)
    check = cf.AssertOp(within, "Triton runtime scalar outside declared bounds: " + scalar.name)
    operations = [low, high, lower, upper, within, check]
    for op in operations:
        op.location = location
        op.attributes.update({"prov.family": builtin.StringAttr("abi"), "prov.op": builtin.StringAttr("runtime_scalar_bounds")})
    block.add_ops(operations)

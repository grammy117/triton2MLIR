"""Runtime scalar arguments keep source position, type and by-value semantics."""

from dataclasses import dataclass

from xdsl.dialects import builtin


@dataclass(frozen=True)
class RuntimeScalar:
    name: str
    dtype: str
    source_arg: int

    def __post_init__(self):
        if (not self.name.isidentifier() or self.dtype not in ("f32", "f64", "i1", "i8", "i16", "i32", "i64")
                or type(self.source_arg) is not int or self.source_arg < 0):
            raise ValueError("runtime scalar requires a name, supported scalar dtype and source argument position")

    @property
    def type(self):
        return getattr(builtin, self.dtype)

    @property
    def shape(self):
        # The array archive stores one scalar value; the MLIR ABI remains plain
        # scalar, distinct from a tensor with rank zero.
        return ()

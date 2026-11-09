"""Named runtime dimensions and logical grid expressions, without target facts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridExtent:
    dimension: str
    divisor: int

    def __post_init__(self):
        if not isinstance(self.dimension, str) or not self.dimension.isidentifier() or type(self.divisor) is not int or not 0 < self.divisor < 2**63:
            raise ValueError("runtime grid needs a named dimension and positive ceil-divisor")

    def as_dict(self):
        return {"dimension": self.dimension, "divisor": self.divisor}


def validate_dimensions(abi):
    scalars = {s.name: s for s in abi.runtime_scalars}
    names = {d for b in abi.buffers for d in b.shape if isinstance(d, str)}
    names.update(g.dimension for g in abi.grid if isinstance(g, GridExtent))
    for name in names:
        scalar = scalars.get(name)
        if scalar is None or scalar.bounds is None or scalar.bounds[0] < 0:
            raise ValueError("runtime dimensions require a nonnegative bounded integer scalar: " + name)
    return names


def shape_contract(abi):
    if not abi.dynamic:
        return None
    return {"schema": "t2m.runtime-shapes.v1",
            "inputs": [{"shape": list(b.shape), "dtype": b.dtype,
                        **({"scalar": b.name, **({"bounds": list(b.bounds)} if b.bounds is not None else {})}
                           if b in abi.runtime_scalars else {})} for b in abi.inputs],
            "outputs": [{"shape": list(b.shape), "dtype": b.dtype} for b in abi.outputs]}

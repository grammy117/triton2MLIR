"""Explicit launch ABI. Source intent is descriptive, never a semantic override."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    effect: str
    dtype: str = "f32"
    strides: tuple[int, ...] = (1,)
    storage_offset: int = 0
    storage_extent: int | None = None

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"invalid buffer name: {self.name!r}")
        if len(self.shape) != 1 or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("MVP requires a positive static one-dimensional shape")
        if self.dtype != "f32" or self.effect not in ("read", "write", "readwrite"):
            raise ValueError("supported effects are f32 read, write and readwrite")
        if len(self.strides) != 1 or type(self.strides[0]) is not int or self.strides[0] <= 0:
            raise ValueError("rank-one views require a positive static element stride")
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise ValueError("storage_offset must be a nonnegative element offset")
        required = self.storage_offset + (self.shape[0] - 1) * self.strides[0] + 1
        if self.storage_extent is not None and (type(self.storage_extent) is not int or self.storage_extent < required):
            raise ValueError("storage_extent does not contain the declared logical view")


@dataclass(frozen=True)
class KernelSpec:
    entry: str
    buffers: tuple[BufferSpec, ...]
    grid: tuple[int, ...]
    constexprs: tuple[tuple[str, int], ...] = ()
    intent: str = ""
    independent_allocations: bool = True

    def __post_init__(self):
        if not self.entry.isidentifier():
            raise ValueError("entry must be an identifier")
        if len(self.grid) != 1 or type(self.grid[0]) is not int or self.grid[0] <= 0:
            raise ValueError("MVP requires a positive static one-dimensional grid")
        names = [b.name for b in self.buffers]
        if not names or len(names) != len(set(names)):
            raise ValueError("buffer names must be nonempty and unique")
        if sum(b.effect != "read" for b in self.buffers) != 1:
            raise ValueError("the current frontend requires exactly one write or readwrite output")
        if self.independent_allocations is not True:
            raise ValueError("aliasing is unsupported; independent allocations are an ABI precondition")
        keys = [k for k, _ in self.constexprs]
        if len(keys) != len(set(keys)) or set(keys) & set(names):
            raise ValueError("constexpr names must be unique and distinct from buffers")
        if any(not k.isidentifier() or type(v) is not int for k, v in self.constexprs):
            raise ValueError("MVP constexprs must have identifier names and integer values")

    def as_dict(self):
        result = asdict(self)
        # Preserve R0's canonical spec hashes for its implicit contiguous views.
        for buffer in result["buffers"]:
            if buffer["strides"] == (1,):
                del buffer["strides"]
            if buffer["storage_offset"] == 0:
                del buffer["storage_offset"]
            if buffer["storage_extent"] is None:
                del buffer["storage_extent"]
        return result

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        fields["buffers"] = tuple(BufferSpec(**{**b, "shape": tuple(b["shape"]),
                                               "strides": tuple(b.get("strides", (1,)))}) for b in fields["buffers"])
        fields["grid"] = tuple(fields["grid"])
        fields["constexprs"] = tuple(tuple(item) for item in fields.get("constexprs", ()))
        return cls(**fields)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


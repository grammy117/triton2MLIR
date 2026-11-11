"""Source-captured byte atomics with selected roots and original recurrence."""

from pathlib import Path

from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer


def convert(work, *, width=32, cas=False, operation="xor", while_=False,
            byte_order="little", optimizer=None, root_dtype="i8", carry=False):
    from .examples.byte_atomic_kernels import byte_atomic_state
    from .frontend import capture_ttir
    from .shared_tensor import convert_shared

    if width not in ((16, 32, 64) if cas else (32, 64)) or operation not in ("add", "xor"):
        raise ValueError("byte state demo requires integer CAS or add/xor RMW")
    root = Path(work)
    root.mkdir(parents=True, exist_ok=False)
    constants = {"B": 4, "WIDTH": width, "CAS": cas, "OP": operation, "WHILE": while_, "NARROW": True, "CARRY": carry}
    pointer = "*" + {"f16": "fp16", "f32": "fp32", "f64": "fp64"}.get(root_dtype, root_dtype)
    source = root / "source.ttir"
    source.write_text(capture_ttir(byte_atomic_state, {
        "a": pointer, "b": pointer, "out": f"*i{width}", "n": "i32", "steps": "i32",
        "capacity": "i32", "choice": "i1", **{name: "constexpr" for name in constants},
    }, constants))
    abi = TensorABI("byte_atomic_state", (
        TensorBuffer("a", ("capacity",), "readwrite", root_dtype),
        TensorBuffer("b", ("capacity",), "readwrite", root_dtype),
        TensorBuffer("out", ("steps", "n"), "write", f"i{width}"),
    ), grid=(1,), runtime_scalars=(
        RuntimeScalar("n", "i32", 3, bounds=(0, 4)), RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
        RuntimeScalar("capacity", "i32", 5, bounds=(0, 1024)), RuntimeScalar("choice", "i1", 6),
    ))
    return convert_shared(source, abi, root / "translation", optimizer=optimizer,
                          memory_profile="guarded-buffer", address_policy="source-words",
                          effect_profile="typed-atomics", byte_order=byte_order)

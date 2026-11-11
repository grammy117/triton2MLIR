"""Executable typed aliases share one root and preserve every uncomputed word."""

from pathlib import Path

from .allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer


def convert(
    work,
    *,
    width=32,
    storage="integer",
    bfloat=False,
    while_=True,
    optimizer=None,
    byte_storage=False,
    byte_order=None,
):
    from .examples.allocation_word_kernels import allocation_word_updates

    if (
        width not in (16, 32, 64)
        or storage not in ("integer", "float")
        or bfloat
        and width != 16
    ):
        raise ValueError(
            "word aliases require 16/32/64-bit integer/float words or BF16"
        )
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    float_type = "bf16" if bfloat else f"fp{width}"
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            allocation_word_updates,
            {
                "floating": "*" + float_type,
                "integer": f"*i{width}",
                **{name: "i32" for name in ("n", "steps", "capacity")},
                **{name: "constexpr" for name in ("B", "WIDTH", "WHILE")},
            },
            {"B": 8, "WIDTH": width, "WHILE": while_},
        )
    )
    dtype = f"i{width}" if storage == "integer" else "bf16" if bfloat else f"f{width}"
    if byte_storage:
        if byte_order not in ("little", "big"):
            raise ValueError("byte-root word demo requires explicit byte order")
        dtype = "i8"
    abi = TensorABI(
        "allocation_word_state",
        (TensorBuffer("state", ("capacity",), "readwrite", dtype),),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 1, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 2, bounds=(0, 4)),
            RuntimeScalar("capacity", "i32", 3, bounds=(0, 256)),
            RuntimeScalar("read_origin", "i64", 4, bounds=(-512, 512)),
            RuntimeScalar("write_origin", "i64", 5, bounds=(-512, 512)),
        ),
        intent="Integer/floating source arguments share one allocation; preserve word payloads, tile snapshots and read-after-write recurrence",
    )
    return convert_allocations(
        source,
        abi,
        (
            PointerArgument("state", "read_origin"),
            PointerArgument("state", "write_origin"),
            *(ScalarArgument(name) for name in ("n", "steps", "capacity")),
        ),
        work / "binding",
        source_entry="allocation_word_updates",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=byte_order,
    )


def demo(
    out,
    *,
    width=32,
    storage="integer",
    bfloat=False,
    optimizer=None,
    python=None,
    memory_profile="guarded-buffer",
    address_policy="source-words",
):
    from .effect_bundle import write_effect_bundle
    from .examples.allocation_word_reference import arguments
    from .examples.descriptor_reference import policy

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError(
            "allocation word demo requires guarded-buffer memory and source-words addresses"
        )
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(
        work, width=width, storage=storage, bfloat=bfloat, optimizer=optimizer
    )
    values, expected = arguments(width, storage, bfloat)
    write_effect_bundle(
        result,
        out,
        values,
        policy(),
        references=[expected],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": result.spec.entry,
        "word_bits": width,
        "native_execution": "passed",
        "hardware_mapping": "undecided",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }

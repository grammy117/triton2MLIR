"""Retain external descriptor origins and writable sharing as real TTIR state."""

from pathlib import Path

from .allocation_bindings import DescriptorArgument, ScalarArgument, convert_allocations
from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer


def convert(work, *, dtype="i64", optimizer=None):
    from .examples.descriptor_state_kernels import descriptor_state_updates

    if dtype not in ("i64", "f32", "bf16"):
        raise ValueError("descriptor state demo requires i64, f32 or BF16 storage")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    ptr = {"i64": "i64", "f32": "fp32", "bf16": "bf16"}[dtype]
    descriptor = f"tensordesc<{ptr}[2,16]>"
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            descriptor_state_updates,
            {
                "first": descriptor,
                "second": descriptor,
                **{name: "i32" for name in ("m", "n", "steps", "capacity")},
            },
            {},
        )
    )
    scalar_fields = [
        ("m", "i32", (0, 5)),
        ("n", "i32", (0, 19)),
        ("steps", "i32", (0, 3)),
        ("capacity", "i32", (0, 1024)),
        ("read_origin", "i64", (-8192, 8192)),
        ("write_origin", "i64", (-8192, 8192)),
        *(
            (name, "i64", (-128, 128))
            for name in ("read_row", "read_col", "write_row", "write_col")
        ),
    ]
    abi = TensorABI(
        "descriptor_allocation_state",
        (TensorBuffer("state", ("capacity",), "readwrite", dtype),),
        grid=(GridExtent("m", 2), GridExtent("n", 16)),
        runtime_scalars=tuple(
            RuntimeScalar(name, type_, i, bounds=bounds)
            for i, (name, type_, bounds) in enumerate(scalar_fields, 1)
        ),
        intent="External descriptor views share one allocation with runtime origins/strides; preserve whole-tile reads, recurrence and untouched state",
    )

    def view(prefix):
        return DescriptorArgument(
            "state",
            ("m", "n"),
            (prefix + "_row", prefix + "_col"),
            offset=prefix + "_origin",
        )

    def metadata(prefix):
        return tuple(
            ScalarArgument(name)
            for name in ("m", "n", prefix + "_row", prefix + "_col")
        )

    bindings = (
        view("read"),
        *metadata("read"),
        view("write"),
        *metadata("write"),
        *(ScalarArgument(name) for name in ("m", "n", "steps", "capacity")),
    )
    return convert_allocations(
        source,
        abi,
        bindings,
        work / "binding",
        source_entry=descriptor_state_updates.__name__,
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
    )


def demo(
    out,
    *,
    dtype="i64",
    optimizer=None,
    python=None,
    memory_profile="guarded-buffer",
    address_policy="source-words",
):
    from .effect_bundle import write_effect_bundle
    from .examples.descriptor_reference import policy
    from .examples.descriptor_state_reference import arguments

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError(
            "descriptor state demo requires guarded-buffer memory and source-words addresses"
        )
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work, dtype=dtype, optimizer=optimizer)
    values, expected = arguments(dtype)
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
        "argument_profile": "t2m.source-allocation-arguments.v2",
        "native_execution": "passed",
        "hardware_mapping": "undecided",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }

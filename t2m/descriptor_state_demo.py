"""Retain external descriptor origins and writable sharing as real TTIR state."""

from pathlib import Path

from .allocation_bindings import DescriptorArgument, ScalarArgument, convert_allocations
from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer


def convert(
    work,
    *,
    dtype="i64",
    optimizer=None,
    word_views=False,
    byte_storage=False,
    byte_order=None,
    root_dtype=None,
):
    from .examples.descriptor_state_kernels import descriptor_state_updates

    if word_views:
        from .examples.allocation_word_kernels import descriptor_word_updates

        if dtype != "i32":
            raise ValueError("descriptor word demo requires i32 owning storage")
        descriptor_state_updates = descriptor_word_updates
    elif dtype not in ("i64", "f32", "bf16"):
        raise ValueError("descriptor state demo requires i64, f32 or BF16 storage")
    if byte_storage and (not word_views or byte_order not in ("little", "big")):
        raise ValueError(
            "byte descriptor word demo requires explicit byte order and word views"
        )
    if root_dtype is not None and (byte_storage or not word_views or byte_order not in ("little", "big")):
        raise ValueError("typed-root descriptor binding requires explicit byte order and word views")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    ptr = {"i64": "i64", "i32": "i32", "f32": "fp32", "bf16": "bf16"}[dtype]
    descriptor = f"tensordesc<{ptr}[2,16]>"
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            descriptor_state_updates,
            {
                "first": "tensordesc<fp32[2,16]>" if word_views else descriptor,
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
        (
            TensorBuffer(
                "state", ("capacity",), "readwrite", root_dtype or ("i8" if byte_storage else dtype)
            ),
        ),
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
        byte_order=byte_order,
    )


def demo(
    out,
    *,
    dtype="i64",
    root_dtype=None,
    byte_order=None,
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
    result = convert(work, dtype="i32" if root_dtype else dtype, optimizer=optimizer,
        word_views=root_dtype is not None, root_dtype=root_dtype, byte_order=byte_order)
    if root_dtype is None:
        values, expected = arguments(dtype)
    else:
        from .examples.typed_parameter_reference import arguments as typed_arguments
        values, expected = typed_arguments(root_dtype, byte_order, descriptor=True)
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

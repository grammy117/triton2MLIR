"""Capture real indexed descriptors and reuse the official pointer expansion."""

from pathlib import Path

from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared

KINDS = ("gather", "scatter", "add", "min", "max", "and", "or", "xor")


def convert(
    work,
    *,
    kind="gather",
    dtype="f32",
    padding="zero",
    optimizer=None,
    unsigned=False,
    f32_atomic_add=None,
):
    from .examples.descriptor_kernels import descriptor_updates, indexed_descriptor_copy

    if kind not in KINDS or dtype not in ("i16", "i32", "i64", "f16", "bf16", "f32"):
        raise ValueError("unknown descriptor demo operation or storage dtype")
    if kind in ("gather", "scatter") and dtype == "i64":
        raise ValueError(
            "pinned TTIR descriptor gather/scatter verifier limits element widths to 32 bits"
        )
    if padding not in ("zero", "nan") or (
        padding == "nan" and (kind != "gather" or dtype.startswith("i"))
    ):
        raise ValueError("NaN descriptor padding requires a floating gather")
    if unsigned and not dtype.startswith("i"):
        raise ValueError("unsigned descriptor semantics require integer storage")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    transfer = kind in ("gather", "scatter")
    kernel = indexed_descriptor_copy if transfer else descriptor_updates
    ptr = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}.get(dtype, dtype)
    if unsigned:
        ptr = "u" + dtype[1:]
    if transfer:
        signature = {"x": "*" + ptr, "indices": "*i32", "out": "*" + ptr}
        scalar_bounds = (
            ("rows", (0, 9)),
            ("cols", (0, 19)),
            ("count", (0, 17)),
            ("y", (-32, 32)),
            ("stride", (16, 32)),
        )
        constants = {"B": 8, "C": 16, "SCATTER": kind == "scatter", "PADDING": padding}
        buffers = (
            TensorBuffer(
                "x",
                ("rows", "stride") if kind == "gather" else ("count", 16),
                dtype=dtype,
            ),
            TensorBuffer("indices", ("count",), dtype="i32"),
            TensorBuffer(
                "out",
                ("count", 16) if kind == "gather" else ("rows", "stride"),
                "write" if kind == "gather" else "readwrite",
                dtype=dtype,
            ),
        )
        grid = (GridExtent("count", 8),)
    else:
        signature = {"x": "*" + ptr, "out": "*" + ptr}
        scalar_bounds = (
            ("rows", (0, 9)),
            ("cols", (0, 19)),
            ("tiles", (0, 3)),
            ("steps", (0, 3)),
            ("row", (-16, 16)),
            ("y", (-32, 32)),
            ("stride", (16, 32)),
        )
        constants = {"KIND": kind}
        buffers = (
            TensorBuffer("x", ("tiles", 2, 16), dtype=dtype),
            TensorBuffer("out", ("rows", "stride"), "readwrite", dtype=dtype),
        )
        grid = (GridExtent("tiles", 1),)
    signature.update({name: "i32" for name, _ in scalar_bounds})
    signature.update({name: "constexpr" for name in constants})
    source = work / "source.ttir"
    source.write_text(capture_ttir(kernel, signature, constants))
    scalars = tuple(
        RuntimeScalar(name, "i32", position, bounds=bounds)
        for position, (name, bounds) in enumerate(scalar_bounds, len(buffers))
    )
    abi = TensorABI(
        kernel.__name__,
        buffers,
        grid=grid,
        runtime_scalars=scalars,
        intent="Indexed descriptor tiles; exact logical bounds and padding; hardware mapping undecided",
    )
    return convert_shared(
        source,
        abi,
        work / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        effect_profile="ordinary" if transfer else "typed-atomics",
        f32_atomic_add=f32_atomic_add,
    )


def demo(
    out,
    *,
    kind="gather",
    optimizer=None,
    python=None,
    memory_profile="guarded-buffer",
    address_policy="source-words",
):
    from .effect_bundle import write_effect_bundle
    from .examples.descriptor_reference import (
        policy,
        reduction_arguments,
        transfer_arguments,
    )

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError(
            "indexed descriptor demo requires guarded-buffer memory and source-words addresses"
        )
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    dtype = "f32" if kind in ("gather", "scatter") else "i32"
    result = convert(work, kind=kind, dtype=dtype, optimizer=optimizer)
    values, expected = (
        transfer_arguments(kind, dtype)
        if kind in ("gather", "scatter")
        else reduction_arguments(kind, dtype)
    )
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
        "descriptor_operation": kind,
        "numeric_storage": dtype,
        "normalization": "official pre-layout descriptor-to-pointer pass",
        "hardware_mapping": "undecided",
        "native_execution": "passed",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }

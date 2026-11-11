"""A real byte-cache kernel with explicit logical byte order and word intent."""

from pathlib import Path

from .dynamic_abi import GridExtent
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer


def convert(
    work, *, dtype="i32", byte_order="little", scalar=False, narrow=True, optimizer=None
):
    from .examples.byte_buffer_kernels import byte_buffer_updates
    from .frontend import capture_ttir
    from .shared_tensor import convert_shared

    root = Path(work)
    root.mkdir(parents=True, exist_ok=False)
    if dtype not in ("i16", "i32", "i64", "f16", "bf16", "f32", "f64"):
        raise ValueError("byte-buffer demo requires a supported 16/32/64-bit word type")
    width = (
        16 if dtype in ("i16", "f16", "bf16") else 32 if dtype in ("i32", "f32") else 64
    )
    source = root / "source.ttir"
    constants = {
        "B": 4,
        "TYPE": dtype,
        "WIDTH": width,
        "SCALAR": scalar,
        "NARROW": narrow,
    }
    source.write_text(
        capture_ttir(
            byte_buffer_updates,
            {
                "raw": "*i8",
                "out": f"*i{width}",
                **{
                    name: "i32"
                    for name in (
                        "n",
                        "steps",
                        "capacity",
                        "origin",
                        "stride",
                        "advance",
                    )
                },
                **{name: "constexpr" for name in constants},
            },
            constants,
        )
    )
    abi = TensorABI(
        "byte_buffer_updates",
        (
            TensorBuffer("raw", ("capacity",), "readwrite", "i8"),
            TensorBuffer("out", ("steps", "n"), "write", f"i{width}"),
        ),
        grid=(GridExtent("n", 1 if scalar else 4),),
        runtime_scalars=tuple(
            RuntimeScalar(name, "i32", index, bounds=bounds)
            for index, (name, bounds) in enumerate(
                (
                    ("n", (0, 13)),
                    ("steps", (0, 3)),
                    ("capacity", (0, 1024)),
                    ("origin", (-8192, 8192)),
                    ("stride", (-128, 128)),
                    ("advance", (-64, 64)),
                ),
                2,
            )
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=byte_order,
    )


def convert_fills(work, *, dtype="f32", byte_order="little", optimizer=None):
    from .examples.byte_buffer_kernels import byte_buffer_fills
    from .frontend import capture_ttir
    from .shared_tensor import convert_shared

    if dtype not in ("f16", "bf16", "f32", "f64"):
        raise ValueError("byte fill demo requires a supported floating word")
    width = 16 if dtype in ("f16", "bf16") else 32 if dtype == "f32" else 64
    root = Path(work)
    root.mkdir(parents=True, exist_ok=False)
    constants = {"B": 4, "TYPE": dtype, "WIDTH": width}
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            byte_buffer_fills,
            {
                "raw": "*i8",
                "out": f"*i{width}",
                **{
                    name: "i32"
                    for name in ("n", "padded", "capacity", "origin", "stride")
                },
                "fill": f"i{width}",
                **{name: "constexpr" for name in constants},
            },
            constants,
        )
    )
    fields = [
        ("n", "i32", (0, 13)),
        ("padded", "i32", (0, 16)),
        ("capacity", "i32", (0, 1024)),
        ("origin", "i32", (-8192, 8192)),
        ("stride", "i32", (-128, 128)),
        ("fill", f"i{width}", None),
    ]
    abi = TensorABI(
        byte_buffer_fills.__name__,
        (
            TensorBuffer("raw", ("capacity",), "read", "i8"),
            TensorBuffer("out", ("padded",), "write", f"i{width}"),
        ),
        grid=(GridExtent("n", 4),),
        runtime_scalars=tuple(
            RuntimeScalar(name, type_, index, bounds=bounds)
            for index, (name, type_, bounds) in enumerate(fields, 2)
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=byte_order,
    )


def convert_selected(work, *, byte_order="big", optimizer=None):
    from .examples.byte_buffer_kernels import byte_buffer_selected
    from .frontend import capture_ttir
    from .shared_tensor import convert_shared

    root = Path(work)
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source.ttir"
    fields = [
        ("n", "i32", (0, 13)),
        ("steps", "i32", (0, 3)),
        ("capacity", "i32", (0, 1024)),
        ("origin", "i32", (-8192, 8192)),
        ("stride", "i32", (-128, 128)),
        ("advance", "i32", (-64, 64)),
        ("choice", "i1", None),
    ]
    source.write_text(
        capture_ttir(
            byte_buffer_selected,
            {
                "a": "*i8",
                "b": "*i8",
                "out": "*i32",
                **{name: type_ for name, type_, _ in fields},
                "B": "constexpr",
            },
            {"B": 4},
        )
    )
    abi = TensorABI(
        byte_buffer_selected.__name__,
        (
            TensorBuffer("a", ("capacity",), "readwrite", "i8"),
            TensorBuffer("b", ("capacity",), "readwrite", "i8"),
            TensorBuffer("out", ("steps", "n"), "write", "i32"),
        ),
        grid=(GridExtent("n", 4),),
        runtime_scalars=tuple(
            RuntimeScalar(name, type_, index, bounds=bounds)
            for index, (name, type_, bounds) in enumerate(fields, 3)
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=byte_order,
    )


def demo(
    out,
    *,
    byte_order="little",
    optimizer=None,
    python=None,
    memory_profile="guarded-buffer",
    address_policy="source-words",
):
    from .effect_bundle import write_effect_bundle
    from .examples.byte_buffer_reference import arguments, policy

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError("byte-buffer demo requires guarded source-word memory")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work, byte_order=byte_order, optimizer=optimizer)
    values, expected = arguments(order=byte_order)
    write_effect_bundle(
        result,
        out,
        values,
        policy(),
        references=expected,
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": result.spec.entry,
        "byte_order": byte_order,
        "native_execution": "passed",
        "hardware_mapping": "undecided",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }

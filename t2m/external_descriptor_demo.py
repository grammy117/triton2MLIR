"""Capture kernels with actual descriptor parameters and explicit logical fields."""

import math
from pathlib import Path

from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def convert(work, *, kind="gather", dtype="f32", shape=(2, 16), optimizer=None):
    from .examples.external_descriptor_kernels import (
        external_descriptor_block,
        external_descriptor_reduce,
        external_descriptor_transfer,
    )

    if kind not in (
        "gather",
        "scatter",
        "load",
        "store",
        "add",
        "min",
        "max",
        "and",
        "or",
        "xor",
    ):
        raise ValueError("unknown external descriptor demonstration")
    if dtype not in ("f16", "bf16", "f32", "i32", "i64"):
        raise ValueError("unknown external descriptor storage type")
    if kind in ("gather", "scatter") and dtype == "i64":
        raise ValueError("pinned TTIR gather/scatter limits element widths to 32 bits")
    transfer, block = kind in ("gather", "scatter"), kind in ("load", "store")
    shape = (1, 16) if transfer else tuple(shape) if block else (2, 16)
    rank = len(shape)
    if not 1 <= rank <= 3 or any(
        type(n) is not int or n <= 0 or n & (n - 1) for n in shape
    ):
        raise ValueError(
            "descriptor demo block shape requires one to three positive power-of-two dimensions"
        )
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    kernel = (
        external_descriptor_transfer
        if transfer
        else external_descriptor_block
        if block
        else external_descriptor_reduce
    )
    ptr = {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}.get(dtype, dtype)
    signature = {
        "desc": f"tensordesc<{ptr}[" + ",".join(map(str, shape)) + "]>",
        "values": "*" + ptr,
    }
    # Triton's real AST descriptor parameter also carries ordinary shape/stride
    # metadata. These remain separate source inputs after the descriptor itself
    # expands, and are deliberately retained even when this kernel ignores them.
    fields = [(f"shape{i}", "i64", (0, 32)) for i in range(rank)]
    fields += [(f"stride{i}", "i64", (-256, 256)) for i in range(rank)]
    fields += [("padding_nan", "i1", None), ("round_tf32", "i1", None)]
    fields += [(f"source_shape{i}", "i32", (0, 32)) for i in range(rank)]
    fields += [(f"source_stride{i}", "i64", (-256, 256)) for i in range(rank)]
    scalars = [
        RuntimeScalar(name, type_, i, bounds=bounds)
        for i, (name, type_, bounds) in enumerate(fields, 1)
    ]
    next_position = 3 + 4 * rank
    parameters = (
        [("count", (0, 17)), ("y", (-32, 32)), ("capacity", (0, 1024))]
        if transfer
        else [
            ("first", (-32, 32)),
            ("second", (-32, 32)),
            ("third", (-32, 32)),
            ("capacity", (0, 1024)),
        ]
        if block
        else [
            ("tiles", (0, 3)),
            ("steps", (0, 3)),
            ("row", (-32, 32)),
            ("y", (-32, 32)),
            ("capacity", (0, 1024)),
        ]
    )
    if transfer:
        signature["indices"] = "*i32"
        next_position += 1
    for i, (name, bounds) in enumerate(parameters, next_position + 1):
        signature[name] = "i32"
        scalars.append(RuntimeScalar(name, "i32", i, bounds=bounds))
    constants = (
        {"SCATTER": kind == "scatter"}
        if transfer
        else {"SHAPE": shape, "ELEMENTS": math.prod(shape), "STORE": kind == "store"}
        if block
        else {"KIND": kind}
    )
    signature.update({name: "constexpr" for name in constants})
    source = work / "source.ttir"
    source.write_text(capture_ttir(kernel, signature, constants))
    writable = kind not in ("gather", "load")
    buffers = (
        TensorBuffer(
            "allocation", ("capacity",), "readwrite" if writable else "read", dtype
        ),
        TensorBuffer(
            "values",
            ("count", 16)
            if transfer
            else (math.prod(shape),)
            if block
            else ("tiles", 2, 16),
            "read" if writable else "write",
            dtype,
        ),
    )
    if transfer:
        buffers += (TensorBuffer("indices", ("count",), dtype="i32"),)
    abi = TensorABI(
        kernel.__name__,
        buffers,
        runtime_scalars=tuple(scalars),
        grid=(GridExtent("count", 8),)
        if transfer
        else (1,)
        if block
        else (GridExtent("tiles", 1),),
        intent="External logical descriptor fields; runtime strides, bounds and padding; hardware mapping undecided",
    )
    return convert_shared(
        source,
        abi,
        work / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        effect_profile="ordinary" if transfer or block else "typed-atomics",
        f32_atomic_add="ieee" if kind == "add" and dtype == "f32" else None,
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
    from .examples.descriptor_reference import policy
    from .examples.external_descriptor_reference import arguments, matmul_arguments

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError(
            "external descriptor demo requires guarded-buffer memory and source-words addresses"
        )
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    dtype = "i32" if kind == "add" else "f32"
    result = (
        convert_matmul(work, dtype=dtype, optimizer=optimizer)
        if kind == "matmul"
        else convert(work, kind=kind, dtype=dtype, optimizer=optimizer)
    )
    values, expected = (
        matmul_arguments(17, 19, 37, dtype=dtype)
        if kind == "matmul"
        else arguments(kind, dtype)
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
        "argument_profile": "t2m.logical-descriptor-arguments.v1",
        "native_execution": "passed",
        "hardware_mapping": "undecided",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }


def convert_matmul(work, *, dtype="bf16", optimizer=None):
    from .examples.external_descriptor_kernels import external_descriptor_matmul

    if dtype not in ("f32", "bf16"):
        raise ValueError("external descriptor matmul demo requires f32 or BF16 inputs")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    ptr = "bf16" if dtype == "bf16" else "fp32"
    signature = {
        "x": f"tensordesc<{ptr}[16,32]>",
        "y": f"tensordesc<{ptr}[32,16]>",
        "initial": "*fp32",
        "out": "*fp32",
        **{name: "i32" for name in ("m", "n", "k", "a_count", "b_count")},
    }
    source = work / "source.ttir"
    source.write_text(capture_ttir(external_descriptor_matmul, signature, {}))
    scalars = []
    for prefix, start, bounds in (
        ("x", 0, ((0, 35), (0, 65))),
        ("y", 11, ((0, 65), (0, 21))),
    ):
        fields = [(prefix + f"_shape{i}", "i64", b) for i, b in enumerate(bounds)]
        fields += [(prefix + f"_stride{i}", "i64", (-512, 512)) for i in range(2)]
        fields += [(prefix + "_padding", "i1", None), (prefix + "_round", "i1", None)]
        fields += [
            (prefix + f"_source_shape{i}", "i32", b) for i, b in enumerate(bounds)
        ]
        fields += [
            (prefix + f"_source_stride{i}", "i64", (-512, 512)) for i in range(2)
        ]
        scalars.extend(
            RuntimeScalar(name, type_, start + i, bounds=b)
            for i, (name, type_, b) in enumerate(fields, 1)
        )
    scalars.extend(
        RuntimeScalar(name, "i32", i, bounds=b)
        for i, (name, b) in enumerate(
            (
                ("m", (0, 35)),
                ("n", (0, 21)),
                ("k", (0, 65)),
                ("a_count", (0, 8192)),
                ("b_count", (0, 8192)),
            ),
            24,
        )
    )
    abi = TensorABI(
        external_descriptor_matmul.__name__,
        (
            TensorBuffer("x", ("a_count",), dtype=dtype),
            TensorBuffer("y", ("b_count",), dtype=dtype),
            TensorBuffer("initial", ("m", "n")),
            TensorBuffer("out", ("m", "n"), "write"),
        ),
        runtime_scalars=tuple(scalars),
        grid=(GridExtent("m", 16), GridExtent("n", 16)),
        intent="Two external descriptors; original 16x16x32 tiled contraction and C-carrying K reuse; hardware mapping undecided",
    )
    return convert_shared(
        source,
        abi,
        work / "translation",
        optimizer=optimizer,
        memory_profile="guarded-buffer",
        address_policy="source-words",
    )

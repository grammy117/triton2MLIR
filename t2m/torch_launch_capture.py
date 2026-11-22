"""Capture actual static Torch Triton launches without a GPU or tensor values.

This module can be loaded directly in model2MLIR's separately pinned environment.
Keep it independent of t2m/xDSL imports. Torch's private tracing adapter is
version-probed here; Triton compiler internals remain in frontend.py.
"""

import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "t2m.discovered-launches.v1"


def discover(model, inputs):
    """Find real triton_op nodes and trace their launch wrappers on fake tensors.

    Initial automatic profile: static dense tensors, pure custom-op schemas,
    one launch, fresh returned allocations, no wrapper arithmetic or views.
    The existing explicit binder remains available for more complex wrappers.
    No CPU reference body supplies implementation or pointer bindings.
    """
    import torch
    from m2m.ir.kernel_registry import is_triton_op
    from torch._higher_order_ops.triton_kernel_wrap import (
        kernel_side_table,
        triton_kernel_wrapper_mutation,
    )
    from torch._library.custom_ops import _maybe_get_opdef
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.node import map_arg
    from torch.fx.operator_schemas import normalize_function

    if not torch.__version__.startswith("2.10."):
        raise ValueError("automatic launch discovery requires the qualified Torch 2.10 adapter")
    exported = torch.export.export(model, tuple(inputs))
    records = []
    for call in exported.graph_module.graph.nodes:
        if call.op != "call_function" or not is_triton_op(call.target):
            continue
        schema = call.target._schema
        if any(arg.alias_info is not None for arg in schema.arguments):
            raise ValueError("automatic launches require a pure nonaliasing schema; use explicit storage bindings")
        arguments = normalize_function(call.target, call.args, call.kwargs, normalize_to_only_use_kwargs=True)
        if arguments is None:
            raise ValueError("cannot normalize the actual Triton-op schema")
        values = [map_arg(arguments.kwargs[arg.name], lambda node: node.meta["val"]) for arg in schema.arguments]
        definition = _maybe_get_opdef(schema.name)
        if definition is None or not callable(definition._abstract_fn):
            raise ValueError("Triton op has no traceable launch wrapper")
        graph = make_fx(definition._abstract_fn, tracing_mode="fake")(*values).graph
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(schema.arguments):
            raise ValueError("automatic wrapper must retain the complete flat schema")
        names = {node: arg.name for node, arg in zip(placeholders, schema.arguments, strict=True)}
        launches = [node for node in graph.nodes if node.op == "call_function"
                    and node.target is triton_kernel_wrapper_mutation]
        if len(launches) != 1:
            raise ValueError("automatic profile requires one Triton launch per wrapper")
        launch = launches[0]
        allocations = {torch.ops.aten.empty.memory_format, torch.ops.aten.empty_like.default}
        for node in graph.nodes:
            if node.op in ("placeholder", "output") or node is launch:
                continue
            if node.op != "call_function" or node.target not in allocations:
                raise ValueError("automatic wrapper contains computation/views outside the launch; use explicit binding")
        returned = next(node for node in graph.nodes if node.op == "output").args[0]
        returned = list(returned) if isinstance(returned, (tuple, list)) else [returned]
        if len(set(returned)) != len(returned) or any(node in names or node.op != "call_function"
                or node.target not in allocations for node in returned):
            raise ValueError("automatic outputs must be distinct fresh tensor allocations")
        metadata = launch.kwargs
        if metadata.get("tma_descriptor_metadata"):
            raise ValueError("automatic physical descriptor metadata needs explicit logical field bindings")
        grids = metadata["grid"]
        if len(grids) != 1 or any(type(n) is not int or n <= 0 for n in grids[0]):
            raise ValueError("automatic launch grid must be a single positive static logical grid")
        kernel = kernel_side_table.get_kernel(metadata["kernel_idx"])
        if not hasattr(kernel, "fn") or not hasattr(kernel, "arg_names"):
            raise ValueError("automatic autotuning needs an explicit selected specialization")
        source = Path(inspect.getfile(kernel.fn)).resolve()
        if "<locals>" in kernel.fn.__qualname__ or kernel.fn.__name__ != kernel.fn.__qualname__:
            raise ValueError("automatic capture requires a file-backed top-level Triton function")
        actual = dict(kernel_side_table.get_constant_args(metadata["constant_args_idx"]))
        if set(actual) & set(metadata["kwargs"]):
            raise ValueError("ambiguous constant and graph launch arguments")
        actual.update(metadata["kwargs"])
        if set(actual) != set(kernel.arg_names):
            raise ValueError("automatic launch does not expose every source argument")
        dtype_names = {torch.float32: ("f32", "fp32"), torch.float64: ("f64", "fp64"),
                       torch.float16: ("f16", "fp16"), torch.bfloat16: ("bf16", "bf16"),
                       torch.int8: ("i8", "i8"), torch.int16: ("i16", "i16"),
                       torch.int32: ("i32", "i32"), torch.int64: ("i64", "i64")}
        signature, constexprs, buffers, operands, outputs = {}, {}, [], {}, {}
        used_inputs = set()
        for name in kernel.arg_names:
            value = actual[name]
            if not isinstance(value, torch.fx.Node):
                if type(value) not in (bool, int, float):
                    raise ValueError("automatic specialization needs literal bool/int/float launch arguments")
                signature[name] = "constexpr"
                constexprs[name] = value
                continue
            tensor = value.meta.get("val")
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype not in dtype_names
                    or not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or any(type(n) is not int for n in tensor.shape)):
                raise ValueError("automatic pointers require static dense zero-origin tensors")
            if value in names:
                if value in used_inputs:
                    raise ValueError("repeated source pointers require explicit alias bindings")
                used_inputs.add(value)
                effect = "read"
                operands[name] = names[value]
            elif value in returned:
                effect = "write"
                outputs[returned.index(value)] = name
            else:
                raise ValueError("automatic launch uses hidden scratch or unsupported pointer arithmetic")
            logical, native = dtype_names[tensor.dtype]
            signature[name] = "*" + native
            buffers.append({"name": name, "shape": list(tensor.shape), "effect": effect, "dtype": logical})
        if set(outputs) != set(range(len(returned))) or len(set(outputs.values())) != len(returned):
            raise ValueError("every actual wrapper result needs one source output pointer")
        grid = list(grids[0])
        while len(grid) > 1 and grid[-1] == 1:
            grid.pop()
        scalars = {arg.name: value for arg, value in zip(schema.arguments, values, strict=True)
                   if not isinstance(value, torch.Tensor)}
        if any(type(value) not in (bool, int, float) for value in scalars.values()):
            raise ValueError("automatic custom-op scalar arguments must be literal specializations")
        records.append({"qualname": schema.name, "call": call.name,
            "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "function": kernel.fn.__name__},
            "signature": signature, "constexprs": constexprs, "buffers": buffers, "grid": grid,
            "operands": operands, "results": [outputs[index] for index in range(len(returned))],
            "scalar_arguments": scalars})
    return {"schema": SCHEMA, "torch_version": torch.__version__, "calls": records,
            "discovery": "actual-exported-operator-and-fake-launch-graph", "tensor_values_used": False}


@contextmanager
def register_discovered(record):
    """Expose discovered identities during ordinary model2MLIR conversion only."""
    from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec

    if record.get("schema") != SCHEMA:
        raise ValueError("unknown automatic launch record")
    names = {call["qualname"] for call in record["calls"]}
    previous = {name: KERNEL_REGISTRY.get(name) for name in names}
    try:
        for name in names:
            KERNEL_REGISTRY.setdefault(name, KernelSpec(name))
        yield
    finally:
        for name, original in previous.items():
            if original is None:
                KERNEL_REGISTRY.pop(name, None)
            else:
                KERNEL_REGISTRY[name] = original

"""Bind logical model views to explicit source allocation roots.

Coordinate relationships come from executable parent SSA. Physical aliasing in
the kernel comes from its existing allocation-bound TTIR, never these records.
"""

import math

from xdsl.dialects import builtin, func
from xdsl.ir import Operation

PROFILE = "t2m.captured-source-tensor-views.v1"
SCHEMA = "t2m.model-kernel-binding.v3"


def _shape(value):
    type_ = value.type
    if (not isinstance(type_, builtin.TensorType) or not isinstance(type_.encoding, builtin.NoneAttr)
            or len(type_.get_shape()) > 8 or any(n < 0 for n in type_.get_shape())
            or math.prod(type_.get_shape()) >= 1 << 63):
        raise ValueError("source model views require static unencoded tensor values")
    return list(type_.get_shape())


def recover(view, root, depth=0):
    shape = _shape(view)
    _shape(root)
    if view.type.element_type != root.type.element_type or depth > 32:
        raise ValueError("source model view changes element type or exceeds its bounded SSA path")
    if view is root:
        stride, strides = 1, []
        for size in reversed(shape): strides.insert(0, stride); stride *= size
        return {"shape": shape, "strides": strides, "offset": 0, "steps": []}
    op = view.owner
    if not isinstance(op, Operation) or len(op.results) != 1:
        raise ValueError("source model view has no proved path to its declared allocation root")
    attrs = {**op.attributes, **op.properties}

    def integers(name):
        value = attrs.get(name)
        if not isinstance(value, builtin.DenseArrayBase):
            raise ValueError("source model view needs complete static coordinate fields")
        return list(value.iter_values())

    if op.name == "linalg.transpose":
        permutation = integers("permutation")
        if (len(op.operands) != 2 or sorted(permutation) != list(range(len(shape))) or len(op.regions) != 1
                or len(op.regions[0].blocks) != 1):
            raise ValueError("source model view requires a complete permutation")
        body = op.regions[0].block
        if (len(body.args) not in (1, 2) or len(tuple(body.ops)) != 1 or body.last_op.name != "linalg.yield"
                or tuple(body.last_op.operands) != (body.args[0],)):
            raise ValueError("source model transpose must preserve every element word")
        parent = recover(op.operands[0], root, depth + 1)
        if shape != [parent["shape"][axis] for axis in permutation]:
            raise ValueError("source model transpose dimensions disagree with its permutation")
        return {"shape": shape, "strides": [parent["strides"][axis] for axis in permutation],
                "offset": parent["offset"], "steps": [*parent["steps"], {"op": op.name, "permutation": permutation}]}
    if op.name == "tensor.extract_slice":
        if len(op.operands) != 1:
            raise ValueError("source model view needs static slice coordinates")
        offsets, sizes, strides = (integers(name) for name in ("static_offsets", "static_sizes", "static_strides"))
        parent = recover(op.operands[0], root, depth + 1)
        if (sizes != shape or len(shape) != len(parent["shape"]) or len(offsets) != len(shape) or len(strides) != len(shape)
                or any(start < 0 or step <= 0 or start > bound or size and start + (size - 1) * step >= bound
                       for start, size, step, bound in zip(offsets, sizes, strides, parent["shape"], strict=True))):
            raise ValueError("source model slice changes rank or exceeds its parent domain")
        return {"shape": shape, "strides": [a * b for a, b in zip(parent["strides"], strides, strict=True)],
                "offset": parent["offset"] + sum(a * b for a, b in zip(parent["strides"], offsets, strict=True)),
                "steps": [*parent["steps"], {"op": op.name, "offsets": offsets, "sizes": sizes, "strides": strides}]}
    if op.name == "tensor.cast" and len(op.operands) == 1 and op.operands[0].type == view.type:
        return recover(op.operands[0], root, depth + 1)
    raise ValueError("source model view requires a proved transpose/slice path to its root")


def bind(configurations, mapping, by_name, values, provenance):
    if not isinstance(configurations, dict) or not configurations or len(configurations) > 64:
        raise ValueError("source model views require a bounded explicit operand binding")
    pointers = provenance.get("source_argument_binding", {}).get("bindings", [])
    result = {}
    for name, config in configurations.items():
        if (not isinstance(name, str) or name not in by_name or name in mapping.values()
                or not isinstance(config, dict) or set(config) != {"root", "source_arg", "shape", "strides", "offset"}
                or not isinstance(config["root"],str) or config["root"] not in mapping or type(config["source_arg"]) is not int
                or not 0 <= config["source_arg"] < len(pointers)
                or not isinstance(config["shape"], list) or not isinstance(config["strides"], list)
                or len(config["shape"]) != len(config["strides"])):
            raise ValueError("source model view binding is incomplete or ambiguous")
        fields = [*config["shape"], *config["strides"], config["offset"]]
        if any(not isinstance(field, str) or field not in values or type(values[field]) is not int for field in fields):
            raise ValueError("source model view fields need captured integer kernel arguments")
        pointer = pointers[config["source_arg"]]
        if pointer != {"kind": "pointer", "allocation": config["root"], "offset": config["offset"]}:
            raise ValueError("source model view differs from its actual typed allocation pointer binding")
        root_name = mapping[config["root"]]
        root, view = by_name[root_name], by_name[name]
        layout = recover(view, root)
        if (layout["shape"] != [values[field] for field in config["shape"]]
                or layout["strides"] != [values[field] for field in config["strides"]]
                or layout["offset"] != values[config["offset"]]):
            raise ValueError("captured kernel view coordinates disagree with actual parent tensor dataflow")
        result[name] = {"binding": config, "root_schema_operand": root_name, "root_type": str(root.type),
                        "view_type": str(view.type), "layout": layout}
    return result


def validate_model(trace, module):
    records = trace.get("calls", [])
    present = any("source_tensor_views" in call for call in records)
    if not present:
        if "source_tensor_view_binding_profile" in trace:
            raise ValueError("source model view profile lacks its original call bindings")
        return
    if trace.get("schema") != SCHEMA or trace.get("source_tensor_view_binding_profile") != PROFILE:
        raise ValueError("source model views require the explicit v3 binding profile")
    calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    from .model_arguments import runtime_constants
    from .shared_tensor import TensorABI
    for record in records:
        views = record.get("source_tensor_views")
        ordinal = record.get("parent_call_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(calls):
            raise ValueError("source model view lacks its original typed parent call")
        call = calls[ordinal]
        if record.get("model_call_attributes") != {k: str(v) for k, v in call.attributes.items()}:
            raise ValueError("source model view differs from its captured call attributes")
        operands = call.attributes.get("prov.kernel_operands")
        mapping = record.get("source_operand_bindings")
        if (not isinstance(operands, builtin.ArrayAttr) or any(not isinstance(name, builtin.StringAttr) for name in operands)
                or len(operands) != len(call.operands) or len({name.data for name in operands}) != len(operands)
                or not isinstance(mapping, dict) or any(not isinstance(name, str) for name in mapping.values())
                or views is not None and not isinstance(views, dict)
                or set(mapping.values()) | set(views or {}) != {name.data for name in operands}):
            raise ValueError("source model view bindings must account for every original tensor operand")
        if views is None:
            continue
        kernel = trace["kernels"][record["implementation_sha256"]]
        spec = TensorABI.from_dict(kernel["spec"])
        mapping = {name: value["schema_argument"] for name, value in record["runtime_scalar_bindings"].items()}
        _, values = runtime_constants(spec, mapping, call.attributes["prov.kernel_args"])
        by_name = dict(zip((name.data for name in operands), call.operands, strict=True))
        configs = {name: view["binding"] for name, view in views.items()}
        if bind(configs, record["source_operand_bindings"], by_name, values, kernel) != views:
            raise ValueError("source model view records differ from the original executable dataflow")

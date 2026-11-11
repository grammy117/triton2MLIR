"""Bind semantic source-layout selections to original captured call arguments."""

from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, StringAttr, i64
from xdsl.dialects.func import CallOp

from .map_layout import options, validate_trace

PROFILE = "t2m.captured-source-layout.v1"


def bind_arguments(trace, mapping, attributes):
    witness = trace.get("source_layout_specialization")
    if witness is None:
        if mapping:
            raise ValueError("source layout arguments require a source specialization")
        return {}
    validate_trace(trace)
    selection = options(witness["source_options"])
    if (not isinstance(mapping, dict) or set(mapping) != set(selection)
            or any(not isinstance(name, str) for name in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("bind every source layout option to a distinct captured scalar argument")
    records = {}
    for key, value in selection.items():
        name = mapping[key]
        attribute = attributes.data.get(name)
        expected = StringAttr(value) if isinstance(value, str) else IntegerAttr(value, i64)
        if attribute != expected:
            raise ValueError("source layout selection disagrees with captured scalar argument: " + name)
        records[key] = {"schema_argument": name, "attribute": str(attribute)}
    return records


def validate_model(trace, parent_module):
    kernels = trace.get("kernels", {})
    selected = {digest for digest, kernel in kernels.items() if "source_layout_specialization" in kernel}
    claimed = "source_layout_binding_profile" in trace or any("source_layout_bindings" in c for c in trace["calls"])
    if not selected and not claimed:
        return
    if not selected or trace.get("source_layout_binding_profile") != PROFILE:
        raise ValueError("source-specialized models require the explicit captured source-layout profile")
    calls = [op for op in parent_module.walk() if isinstance(op, CallOp)]
    for digest in selected:
        validate_trace(kernels[digest])
    for record in trace["calls"]:
        digest = record["implementation_sha256"]
        bindings = record.get("source_layout_bindings", {})
        if digest not in selected:
            if bindings:
                raise ValueError("source layout bindings lack a source specialization")
            continue
        ordinal = record.get("parent_call_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(calls):
            raise ValueError("source layout binding lacks its original call")
        call = calls[ordinal]
        attributes = call.attributes.get("prov.kernel_args")
        if (call.attributes.get("prov.kernel") != StringAttr(record["kernel"])
                or call.callee.root_reference.data != record["callee"]
                or not isinstance(attributes, DictionaryAttr)
                or record["model_call_attributes"] != {k: str(v) for k, v in call.attributes.items()}
                or not isinstance(bindings, dict)
                or any(not isinstance(item, dict) or set(item) != {"schema_argument", "attribute"}
                       for item in bindings.values())):
            raise ValueError("source layout binding differs from its original typed call")
        mapping = {key: item["schema_argument"] for key, item in bindings.items()}
        if bind_arguments(kernels[digest], mapping, attributes) != bindings:
            raise ValueError("source layout binding attributes changed")

"""Bind mutable-view pointers to one actual source allocation per cloned base."""

from xdsl.dialects import builtin

SOURCE_PROFILE = "m2m.functional-kernel-storage.v1"
PROFILE = "t2m.functional-kernel-binding.v2"
FIELDS = {"prov.kernel_storage_bases", "prov.kernel_storage_views"}


def check(operation, spec, mapping, outputs, provenance, pointer_arguments, runtime_operands):
    attrs = operation.attributes
    bases, views = (attrs.get(key) for key in ("prov.kernel_storage_bases", "prov.kernel_storage_views"))
    mutated, ties, slots = (attrs[key] for key in ("prov.kernel_mutated", "prov.kernel_tied", "prov.kernel_result_slots"))
    names = attrs["prov.kernel_operands"]
    if (any(not isinstance(value, builtin.ArrayAttr) for value in (bases, views, mutated, ties, slots, names))
            or not bases or not views or len(bases) > 64 or len(views) != len(mutated)
            or any(not isinstance(value, builtin.StringAttr) for value in (*mutated, *names))
            or len(set(mutated)) != len(mutated) or len(ties) != len(bases)
            or any(not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in slots)):
        raise ValueError("mutable storage lacks its complete typed base/view/result roster")
    raw = [value.value.data for value in slots]
    if (len(outputs) < len(bases) or raw not in (list(range(len(outputs))), [-1, *range(len(outputs))])
            or raw[0] == -1 and len(outputs) != len(bases)):
        raise ValueError("mutable storage changes its original tuple/base slots")
    by_name = dict(zip((value.data for value in names), operation.operands, strict=True))
    buffers = {buffer["name"]: buffer for buffer in spec["buffers"]}
    reverse = {argument: source for source, argument in mapping.items()}
    scalars = {scalar["name"]: scalar for scalar in spec.get("runtime_scalars", ())}
    source = provenance.get("source_argument_binding", {})
    pointers = source.get("bindings")
    if (source.get("schema") != "t2m.source-allocation-bindings.v1" or not isinstance(pointers, list)
            or not isinstance(pointer_arguments, dict) or set(pointer_arguments) != {value.data for value in mutated}):
        raise ValueError("mutable storage requires complete original pointer argument bindings")
    fields, roots = set(), []

    def geometry(values):
        if not isinstance(values, builtin.ArrayAttr) or any(not isinstance(value, builtin.StringAttr) for value in values):
            raise ValueError("mutable storage coordinates require named scalar SSA")
        for value in values:
            if value.data in fields or value.data not in by_name or by_name[value.data].type != builtin.i64:
                raise ValueError("mutable storage coordinate lacks a distinct logical i64 operand")
            fields.add(value.data)
            selected = [name for name, argument in runtime_operands.items() if argument == value.data]
            if not selected or any(scalars[name]["dtype"] != "i64" for name in selected):
                raise ValueError("mutable storage coordinates must reach actual source scalar formals")

    for index, (base, tie) in enumerate(zip(bases, ties, strict=True)):
        if (not isinstance(base, builtin.DictionaryAttr) or set(base.data) != {"operand", "shape"}
                or not isinstance(base.data["operand"], builtin.StringAttr)
                or not isinstance(tie, builtin.DictionaryAttr) or set(tie.data) != {"base_index", "operand", "result"}
                or any(not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in tie.data.values())):
            raise ValueError("mutable storage base needs an exact typed operand/result tie")
        name = base.data["operand"].data
        tensor, buffer = by_name.get(name), reverse.get(name)
        operand, result = tie.data["operand"].value.data, tie.data["result"].value.data
        if (tensor is None or not isinstance(tensor.type, builtin.TensorType) or buffer not in buffers
                or buffer in roots or buffers[buffer]["effect"] != "readwrite"
                or tie.data["base_index"].value.data != index or not 0 <= operand < len(operation.operands)
                or names.data[operand] != base.data["operand"] or result != len(outputs) - len(bases) + index
                or not 0 <= result < len(outputs) or outputs[result] != buffer or tensor.type != operation.results[result].type):
            raise ValueError("mutable storage tie changes its complete source readwrite allocation")
        geometry(base.data["shape"])
        if len(base.data["shape"]) != len(tensor.type.get_shape()):
            raise ValueError("mutable storage base shape does not cover every source axis")
        roots.append(buffer)
    if set(roots) != {name for name, buffer in buffers.items() if buffer["effect"] == "readwrite"}:
        raise ValueError("mutable storage loses a source readwrite allocation")
    used = set()
    for view, argument in zip(views, mutated, strict=True):
        if (not isinstance(view, builtin.DictionaryAttr)
                or set(view.data) != {"argument", "base_index", "kind", "shape", "strides", "offset"}
                or view.data["argument"] != argument or not isinstance(view.data["kind"], builtin.StringAttr)
                or view.data["kind"].data not in ("base", "alias", "slice", "as_strided")
                or not isinstance(view.data["base_index"], builtin.IntegerAttr) or view.data["base_index"].type != builtin.i64
                or not 0 <= view.data["base_index"].value.data < len(roots)
                or not isinstance(view.data["offset"], builtin.StringAttr)):
            raise ValueError("mutable storage has an unknown or incomplete source view")
        geometry(view.data["shape"])
        geometry(view.data["strides"])
        geometry(builtin.ArrayAttr([view.data["offset"]]))
        if len(view.data["shape"]) != len(view.data["strides"]):
            raise ValueError("mutable storage view changes its coordinate rank")
        position = pointer_arguments[argument.data]
        if type(position) is not int or not 0 <= position < len(pointers) or position in used:
            raise ValueError("mutable storage lacks a distinct original source pointer position")
        used.add(position)
        pointer = pointers[position]
        root = roots[view.data["base_index"].value.data]
        if (not isinstance(pointer, dict) or set(pointer) != {"kind", "allocation", "offset"}
                or pointer["kind"] != "pointer" or pointer["allocation"] != root
                or runtime_operands.get(pointer["offset"]) != view.data["offset"].data):
            raise ValueError("mutable storage view pointer does not use its actual cloned base and offset")
    return True

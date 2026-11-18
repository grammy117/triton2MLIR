"""Check functional mutation ties against the actual translated memory ABI."""

import json

from xdsl.dialects import builtin

from .diagnostics import ConversionError

PROFILE = "t2m.functional-kernel-binding.v1"
SOURCE_PROFILE = "m2m.functional-kernel-bases.v1"
STATE_PROFILE = "m2m.functional-state-results.v1"
FIELDS = {"prov.kernel_functionalization", "prov.kernel_source_schema", "prov.kernel_mutated",
          "prov.kernel_tied", "prov.kernel_result_slots"}


def entry(function):
    value = function.attributes.get("prov.functional_state")
    if value is None:
        return False
    if not isinstance(value, builtin.StringAttr):
        raise ConversionError("functional state entry requires its explicit typed contract")
    contract = json.loads(value.data)
    if (not isinstance(contract, dict) or set(contract) != {"schema", "state_outputs", "user_outputs", "output_count", "invocation"}
            or contract["schema"] != STATE_PROFILE or contract["invocation"] != "explicit-state-results"
            or type(contract["output_count"]) is not int or contract["output_count"] != len(function.function_type.outputs)
            or not isinstance(contract["state_outputs"], list) or not contract["state_outputs"]
            or len(contract["state_outputs"]) > contract["output_count"]
            or not isinstance(contract["user_outputs"], list) or any(type(value) is not int for value in contract["user_outputs"])
            or contract["user_outputs"] != list(range(len(contract["state_outputs"]), contract["output_count"]))):
        raise ValueError("functional state entry changes its complete exported result contract")
    used = set()
    for index, state in enumerate(contract["state_outputs"]):
        if (not isinstance(state, dict) or set(state) != {"result", "input", "kind", "target"}
                or type(state["result"]) is not int or state["result"] != index or type(state["input"]) is not int
                or not 0 <= state["input"] < len(function.function_type.inputs) or state["input"] in used
                or state["kind"] not in ("USER_INPUT_MUTATION", "BUFFER_MUTATION")
                or not isinstance(state["target"], str) or not state["target"]
                or function.function_type.inputs.data[state["input"]] != function.function_type.outputs.data[index]):
            raise ValueError("functional state entry has an invalid original input/result tie")
        used.add(state["input"])
    return True


def call(operation, spec, mapping, outputs):
    present = FIELDS.intersection(operation.attributes)
    if not present:
        return False
    if present != FIELDS or operation.attributes["prov.kernel_functionalization"] != builtin.StringAttr(SOURCE_PROFILE):
        raise ValueError("functional kernel needs its complete known mutation profile")
    schema = operation.attributes["prov.kernel_source_schema"]
    names = operation.attributes["prov.kernel_operands"]
    mutated, ties, slots = (operation.attributes[key] for key in
                          ("prov.kernel_mutated", "prov.kernel_tied", "prov.kernel_result_slots"))
    if (not isinstance(schema, builtin.StringAttr) or not schema.data
            or any(not isinstance(value, builtin.ArrayAttr) for value in (names, mutated, ties, slots))
            or any(not isinstance(value, builtin.StringAttr) for value in (*names, *mutated))
            or not mutated or len(set(mutated)) != len(mutated) or len(ties) != len(mutated)
            or any(not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in slots)):
        raise ValueError("functional kernel has incomplete schema/base/result records")
    raw = [value.value.data for value in slots]
    if (len(outputs) < len(ties) or raw not in (list(range(len(outputs))), [-1, *range(len(outputs))])
            or raw[0] == -1 and len(outputs) != len(ties)):
        raise ValueError("functional kernel changes its original tuple result slots")
    buffers = {value["name"]: value for value in spec["buffers"]}
    reverse = {argument: buffer for buffer, argument in mapping.items()}
    checked = set()
    for base, tie in enumerate(ties):
        if not isinstance(tie, builtin.DictionaryAttr) or set(tie.data) != {"argument", "base_index", "operand", "result"}:
            raise ValueError("functional kernel has an incomplete updated-base tie")
        fields = tie.data
        argument = fields["argument"]
        if (not isinstance(argument, builtin.StringAttr) or argument not in mutated or argument.data in checked
                or any(not isinstance(fields[key], builtin.IntegerAttr) or fields[key].type != builtin.i64
                       for key in ("base_index", "operand", "result"))):
            raise ValueError("functional kernel has an invalid typed updated-base tie")
        operand, result = fields["operand"].value.data, fields["result"].value.data
        buffer = reverse.get(argument.data)
        if (fields["base_index"].value.data != base or not 0 <= operand < len(names) or names.data[operand] != argument
                or result != len(outputs) - len(ties) + base or not 0 <= result < len(outputs)
                or buffer not in buffers or buffers[buffer]["effect"] != "readwrite" or outputs[result] != buffer
                or operation.operands[operand].type != operation.results[result].type):
            raise ValueError("functional kernel mutation tie differs from its actual readwrite source buffer")
        checked.add(argument.data)
    if checked != {mapping[name] for name, buffer in buffers.items() if buffer["effect"] == "readwrite"}:
        raise ValueError("functional kernel loses a complete source mutation base")
    return True

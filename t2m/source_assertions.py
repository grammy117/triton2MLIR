"""Retain TTIR fail-stop diagnostics using the actual upstream assertion pass.

Meta triton-shared 17f0c234 AssertConverter and its triton_assert.mlir tests
lower scalar i1 to cf.assert and tensor i1 to a parallel generic of cf.assert.
The archived Microsoft converter registers the same conversion. This adapter
adds source accounting and the functional package contract, not a new lowering.
"""

from xdsl.dialects.builtin import TensorType, StringAttr, i1

from .diagnostics import ConversionError


def contract(op, source_id):
    name = "cf.assert" if op.name == "cf.assert" else "tt.assert"
    condition = op.operands[0].type if len(op.operands) == 1 else None
    scalar = condition.element_type if isinstance(condition, TensorType) else condition
    key = "msg" if name == "cf.assert" else "message"
    message = op.properties.get(key, op.attributes.get(key))
    if scalar != i1 or name == "cf.assert" and isinstance(condition, TensorType) or not isinstance(message, StringAttr) or op.results or op.regions:
        raise ConversionError("source assertion requires one logical i1 condition and a literal diagnostic message",
                              source_id=source_id, opcode=name)
    if "\x00" in message.data:
        raise ConversionError("embedded NUL diagnostics require a length-aware runtime ABI",
                              source_id=source_id, opcode=name)
    return {"source_ids":[source_id], "condition_type":str(condition), "message":message.data,
        "diagnostic_text": ("Assertion `"+message.data+"` failed" if name == "tt.assert" else message.data) + "\n",
        "semantics":"abort-if-any-source-condition-is-false",
        "expansion":"source-standard-cf.assert" if name == "cf.assert" else "upstream-assert-to-cf; scalar-or-lane-wise-standard-assertions"}

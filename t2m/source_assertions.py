"""Retain TTIR fail-stop diagnostics using the actual upstream assertion pass.

Meta triton-shared 17f0c234 AssertConverter and its triton_assert.mlir tests
lower scalar i1 to cf.assert and tensor i1 to a parallel generic of cf.assert.
The archived Microsoft converter registers the same conversion. This adapter
adds source accounting and the functional package contract, not a new lowering.
"""

from xdsl.dialects.builtin import TensorType, StringAttr, i1

from .diagnostics import ConversionError


def contract(op, source_id):
    condition = op.operands[0].type if len(op.operands) == 1 else None
    scalar = condition.element_type if isinstance(condition, TensorType) else condition
    message = op.properties.get("message", op.attributes.get("message"))
    if scalar != i1 or not isinstance(message, StringAttr) or op.results or op.regions:
        raise ConversionError("source assertion requires one logical i1 condition and a literal diagnostic message",
                              source_id=source_id, opcode="tt.assert")
    if "\x00" in message.data:
        raise ConversionError("embedded NUL diagnostics require a length-aware runtime ABI",
                              source_id=source_id, opcode="tt.assert")
    return {"source_ids":[source_id], "condition_type":str(condition), "message":message.data,
        "semantics":"abort-if-any-source-condition-is-false",
        "expansion":"upstream-assert-to-cf; scalar-or-lane-wise-standard-assertions"}

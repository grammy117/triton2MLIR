"""Explicit inlining with executable contracts and execution-owned results.

The original SSA graph computes sizes once. No symbolic result-size program
is required, and no opaque calls or GPU launch ABI survive binding. Source
effects, tiles, loops and every ordinary model operation remain executable.
Package/native/consumer qualification is separate from this transformation.
"""

import hashlib
from io import StringIO

from xdsl.dialects import builtin, func
from xdsl.printer import Printer

from .diagnostics import ConversionError
from .executed_model_arguments import accepts, arguments, cast_tensor, tensor_contract
from .shared_tensor import parse_shared

MODE = "execution-owned"
SCHEMA = "t2m.model-kernel-binding.v4"
PROFILE = "t2m.execution-checked-calls.v1"


def serialize(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(
        module
    )
    return stream.getvalue() + "\n"


def bind(model_mlir, bindings):
    from .model_binding import BoundModel

    prepared = [(binding, *binding.validate()) for binding in bindings]
    module = parse_shared(model_mlir)
    module.verify()
    functions = {
        op.sym_name.data: op for op in module.ops if isinstance(op, func.FuncOp)
    }
    calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    entries = [
        op
        for op in functions.values()
        if op.sym_name.data == "forward" and op.body.blocks
    ]
    if len(entries) != 1:
        raise ConversionError(
            "executed model binding requires one defined forward entry"
        )
    records, used = [], set()
    for ordinal, call in enumerate(calls):
        symbol = call.callee.root_reference.data
        declaration = functions.get(symbol)
        identity = call.attributes.get("prov.kernel")
        if (
            not isinstance(identity, builtin.StringAttr)
            or declaration is None
            or declaration.attributes.get("prov.kernel") != identity
            or declaration.body.blocks
        ):
            raise ConversionError(
                "executed model call needs a matching original kernel declaration",
                opcode="func.call",
            )
        schema = call.attributes.get("prov.kernel_operands")
        if (
            not isinstance(schema, builtin.ArrayAttr)
            or any(not isinstance(name, builtin.StringAttr) for name in schema)
            or len(schema) != len(call.operands)
            or len({name.data for name in schema}) != len(schema)
        ):
            raise ConversionError(
                "kernel operands lack complete unique schema names", opcode="func.call"
            )
        by_name = dict(zip((name.data for name in schema), call.operands, strict=True))
        attrs = call.attributes.get("prov.kernel_args", builtin.DictionaryAttr({}))
        if not isinstance(attrs, builtin.DictionaryAttr):
            raise ConversionError(
                "kernel scalar arguments must be a typed dictionary", opcode="func.call"
            )
        candidates = [
            item
            for item in prepared
            if item[0].qualname == identity.data
            and item[4] == attrs
            and set(item[2].values()) | set(item[0].runtime_operands.values())
            == set(by_name)
            and accepts(
                item[0], item[2], item[3], by_name, [v.type for v in call.results]
            )
        ]
        if len(candidates) != 1:
            raise ConversionError(
                f"{identity.data}: expected one matching typed/scalar specialization, found {len(candidates)}",
                opcode="func.call",
            )
        binding, function, names, outputs, _ = candidates[0]
        callsite = f"kernel_call.{len(records):04d}"
        emitted = []

        def append(
            op,
            role,
            *,
            call=call,
            identity=identity,
            callsite=callsite,
            emitted=emitted,
            binding=binding,
        ):
            op.location = call.location
            op.attributes.update(
                {
                    "prov.family": builtin.StringAttr("binding"),
                    "prov.op": builtin.StringAttr(role),
                    "prov.kernel_origin": identity,
                    "prov.kernel_callsite": builtin.StringAttr(callsite),
                    "prov.region_id": builtin.StringAttr(callsite + "/abi"),
                    "prov.emitted_id": builtin.StringAttr(
                        f"{callsite}/abi.{len(emitted):04d}"
                    ),
                    "prov.kernel_source_sha256": builtin.StringAttr(
                        binding.implementation.trace["source_sha256"]
                    ),
                }
            )
            emitted.append(op)
            return op.results[0] if op.results else None

        values, dimensions = arguments(binding, function, names, by_name, attrs, append)
        mapper = dict(zip(function.body.block.args, values, strict=True))
        for source in function.body.block.ops:
            if isinstance(source, func.ReturnOp):
                returned = [mapper[value] for value in source.arguments]
                break
            cloned = source.clone(value_mapper=mapper)
            for op in cloned.walk():
                op.attributes.update(
                    {
                        "prov.kernel_origin": identity,
                        "prov.kernel_callsite": builtin.StringAttr(callsite),
                        "prov.kernel_source_sha256": builtin.StringAttr(
                            binding.implementation.trace["source_sha256"]
                        ),
                    }
                )
                for attribute in ("prov.region_id", "prov.emitted_id"):
                    prior = op.attributes.get(attribute)
                    if isinstance(prior, builtin.StringAttr):
                        op.attributes[attribute] = builtin.StringAttr(
                            callsite + "/" + prior.data
                        )
            emitted.append(cloned)
        results = {}
        for buffer, value in zip(
            binding.implementation.spec.outputs, returned, strict=True
        ):
            tensor_contract(append, value, buffer, dimensions)
            results[buffer.name] = value
        replacements = [
            cast_tensor(append, results[name], old.type)
            for name, old in zip(outputs, call.results, strict=True)
        ]
        call.parent.insert_ops_before(emitted, call)
        for old, value in zip(call.results, replacements, strict=True):
            old.replace_all_uses_with(value)
        call.parent.erase_op(call)
        records.append(
            {
                "id": callsite,
                "parent_call_ordinal": ordinal,
                "callee": symbol,
                "kernel": identity.data,
                "schema_operands": [name.data for name in schema],
                "model_call_attributes": {
                    name: str(value) for name, value in call.attributes.items()
                },
                "source_operand_bindings": names,
                "source_result_order": list(outputs),
                "captured_scalar_arguments": binding.scalar_arguments,
                "runtime_scalar_arguments": binding.runtime_arguments,
                "runtime_operand_arguments": binding.runtime_operands,
                "source_layout_arguments": binding.source_layout_arguments,
                "implementation_sha256": binding.implementation.trace["mlir_sha256"],
                "source_sha256": binding.implementation.trace["source_sha256"],
            }
        )
        used.add(symbol)
    for symbol, function in functions.items():
        if "prov.kernel" in function.attributes:
            if symbol not in used or function.body.blocks:
                raise ConversionError(
                    "unresolved executed model kernel declaration: " + symbol
                )
            module.body.block.erase_op(function)
    if any(
        isinstance(op, func.CallOp) or "prov.kernel" in op.attributes
        for op in module.walk()
    ):
        raise ConversionError("executed model retains an unresolved source dependency")
    module.attributes["prov.kernel_binding"] = builtin.StringAttr(SCHEMA)
    module.verify()
    parse_shared(serialize(module)).verify()
    selected = {record["implementation_sha256"] for record in records}
    return BoundModel(
        module,
        {
            "schema": SCHEMA,
            "binding_profile": PROFILE,
            "result_mode": MODE,
            "resource_scope": "execution-allocator-accounted",
            "calls": records,
            "kernels": {
                binding.implementation.trace[
                    "mlir_sha256"
                ]: binding.implementation.trace
                for binding in bindings
                if binding.implementation.trace["mlir_sha256"] in selected
            },
            "located_implementations": {
                binding.implementation.trace["mlir_sha256"]: {
                    "sha256": hashlib.sha256(
                        serialize(binding.implementation.module).encode()
                    ).hexdigest(),
                    "bytes": len(serialize(binding.implementation.module).encode()),
                }
                for binding in bindings
                if binding.implementation.trace["mlir_sha256"] in selected
            },
            "input_model_sha256": hashlib.sha256(model_mlir.encode()).hexdigest(),
            "mlir_sha256": hashlib.sha256(serialize(module).encode()).hexdigest(),
            "model_abi": "unchanged",
            "structural_verification": "passed",
            "native_execution": "not-performed",
            "merlin_execution": "not-performed",
        },
    )

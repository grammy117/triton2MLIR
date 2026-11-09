"""Bind translated tensor bodies to model2MLIR's actual custom-kernel calls.

The stable identity is prov.kernel, not a generated callee name. Binding is
per call and checks schema names, typed SSA operands/results and scalar values.
No Torch or target backend is imported by this adapter.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math

from xdsl.dialects import func
from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr,
                                  StringAttr, UnitAttr, f64, i1, i64)

from .diagnostics import ConversionError
from .ir import text
from .shared_tensor import SharedTensorResult, parse_shared
from .memory_contract import GUARDED_PROFILES, validate_memory_evidence


def scalar_attribute(value):
    """The scalar schema used by m2m.ir.kernel_registry._scalar_attr."""
    if value is None:
        return UnitAttr()
    if type(value) is bool:
        return IntegerAttr(int(value), i1)
    if type(value) is int:
        return IntegerAttr(value, i64)
    if type(value) is float and math.isfinite(value):
        return FloatAttr(value, f64)
    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, (tuple, list)):
        return ArrayAttr(scalar_attribute(v) for v in value)
    raise ValueError("kernel scalar binding requires finite JSON-compatible values")


@dataclass
class KernelBinding:
    qualname: str
    implementation: SharedTensorResult
    # Source input-buffer name -> model2MLIR schema operand name.
    operands: dict[str, str] = field(default_factory=dict)
    # Source output-buffer names in model-call result order; empty = source order.
    results: tuple[str, ...] = ()
    scalar_arguments: dict = field(default_factory=dict)

    def validate(self):
        if len(self.qualname.split("::")) != 2 or not all(self.qualname.split("::")):
            raise ValueError("kernel identity must be namespace::name")
        result = self.implementation
        validate_memory_evidence(result.trace, result.auxiliary_artifacts)
        if hashlib.sha256(result.mlir_text.encode()).hexdigest() != result.trace["mlir_sha256"]:
            raise ConversionError("kernel implementation changed after conversion")
        if json.loads(json.dumps(result.spec.as_dict())) != json.loads(json.dumps(result.trace["spec"])):
            raise ConversionError("kernel ABI changed after conversion")
        functions = list(result.module.ops)
        if len(functions) != 1 or not isinstance(functions[0], func.FuncOp):
            raise ConversionError("binding requires one translated tensor function")
        function = functions[0]
        if len(function.body.blocks) != 1 or not isinstance(function.body.block.last_op, func.ReturnOp):
            raise ConversionError("kernel body requires one entry block and a tensor return")
        inputs = [b.name for b in result.spec.inputs]
        names = self.operands or {n: n for n in inputs}
        if set(names) != set(inputs) or len(set(names.values())) != len(names):
            raise ConversionError("bind every source tensor input to a distinct schema operand name")
        output_names = [b.name for b in result.spec.outputs]
        outputs = self.results or tuple(output_names)
        if len(outputs) != len(output_names) or set(outputs) != set(output_names):
            raise ConversionError("bind every source result exactly once")
        scalar = DictionaryAttr({n: scalar_attribute(v) for n, v in self.scalar_arguments.items()})
        return function, names, outputs, scalar


@dataclass
class BoundModel:
    module: object
    trace: dict

    @property
    def mlir_text(self):
        return text(self.module)


def bind_model(model_mlir, bindings):
    """Inline all registered kernel calls, preserving the enclosing model ABI.

    No bundle receipt or native/consumer execution is claimed by binding alone.
    The caller can qualify the resulting serialized model with its original
    argument manifest and independently captured model outputs.
    """
    bindings = list(bindings)
    prepared = [(b, *b.validate()) for b in bindings]
    module = parse_shared(model_mlir)
    module.verify()
    functions = {op.sym_name.data: op for op in module.ops if isinstance(op, func.FuncOp)}
    calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    used, records, guard_storage = set(), [], 0
    for call in calls:
        symbol = call.callee.root_reference.data
        declaration = functions.get(symbol)
        identity = call.attributes.get("prov.kernel")
        declared = declaration.attributes.get("prov.kernel") if declaration is not None else None
        if identity is None and declared is None:
            continue
        if not isinstance(identity, StringAttr) or identity != declared:
            raise ConversionError("kernel call and declaration identities disagree", opcode="func.call")
        if declaration.body.blocks:
            raise ConversionError("registered kernel declaration already has a body", opcode="func.call")
        names = call.attributes.get("prov.kernel_operands")
        if (not isinstance(names, ArrayAttr) or any(not isinstance(n, StringAttr) for n in names)
            or len(names) != len(call.operands) or len(set(n.data for n in names)) != len(names)):
            raise ConversionError("kernel operands lack complete unique schema names", opcode="func.call")
        by_name = dict(zip((n.data for n in names), call.operands, strict=True))
        arguments = call.attributes.get("prov.kernel_args", DictionaryAttr({}))
        if not isinstance(arguments, DictionaryAttr):
            raise ConversionError("kernel scalar arguments must be a typed dictionary", opcode="func.call")
        candidates = []
        for binding, function, operand_names, output_names, scalars in prepared:
            if binding.qualname != identity.data or scalars != arguments or set(operand_names.values()) != set(by_name):
                continue
            operands = [by_name[operand_names[b.name]] for b in binding.implementation.spec.inputs]
            outputs = dict(zip((b.name for b in binding.implementation.spec.outputs),
                               function.function_type.outputs, strict=True))
            if ([v.type for v in operands] == list(function.function_type.inputs)
                and [outputs[n] for n in output_names] == [v.type for v in call.results]):
                candidates.append((binding, function, operands, output_names))
        if len(candidates) != 1:
            raise ConversionError(f"{identity.data}: expected one matching typed/scalar specialization, found {len(candidates)}",
                                  opcode="func.call")
        binding, function, operands, output_names = candidates[0]
        if binding.implementation.trace.get("profile") in GUARDED_PROFILES:
            outputs = binding.implementation.spec.outputs
            if any(type(n) is not int for b in outputs for n in b.shape):
                raise ConversionError("guarded model binding needs static kernel scratch dimensions", opcode="func.call")
            contract = function.attributes.get("prov.memory_guard_bytes_per_result_element")
            cost = json.loads(binding.implementation.auxiliary_artifacts["memory-admission.json"])["guard_storage_bytes_per_output_element"]
            if not isinstance(contract, IntegerAttr) or contract.value.data != cost:
                raise ConversionError("guarded kernel lacks its scratch storage contract", opcode="func.call")
            guard_storage += cost * sum(math.prod(b.shape) for b in outputs)
        callsite = f"kernel_call.{len(records):04d}"
        mapper = dict(zip(function.body.block.args, operands, strict=True))
        emitted = []
        for source in function.body.block.ops:
            if isinstance(source, func.ReturnOp):
                returned = [mapper[v] for v in source.arguments]
                break
            cloned = source.clone(value_mapper=mapper)
            for op in cloned.walk():
                op.attributes["prov.kernel_origin"] = identity
                op.attributes["prov.kernel_callsite"] = StringAttr(callsite)
                op.attributes["prov.kernel_source_sha256"] = StringAttr(binding.implementation.trace["source_sha256"])
                for attribute in ("prov.region_id", "prov.emitted_id"):
                    if isinstance(value := op.attributes.get(attribute), StringAttr):
                        op.attributes[attribute] = StringAttr(f"{callsite}/{value.data}")
            emitted.append(cloned)
        result_map = dict(zip((b.name for b in binding.implementation.spec.outputs), returned, strict=True))
        block = call.parent
        block.insert_ops_before(emitted, call)
        for old, name in zip(call.results, output_names, strict=True):
            old.replace_all_uses_with(result_map[name])
        block.erase_op(call)
        used.add(symbol)
        records.append({"id": callsite, "kernel": identity.data, "callee": symbol,
                        "model_call_attributes": {n: str(v) for n, v in call.attributes.items()},
                        "schema_operands": [n.data for n in names], "source_result_order": list(output_names),
                        "implementation_sha256": binding.implementation.trace["mlir_sha256"],
                        "source_sha256": binding.implementation.trace["source_sha256"]})
    remaining_calls = {op.callee.root_reference.data for op in module.walk() if isinstance(op, func.CallOp)}
    for symbol, function in functions.items():
        if "prov.kernel" in function.attributes:
            if symbol not in used or symbol in remaining_calls:
                raise ConversionError(f"unresolved kernel declaration @{symbol}")
            module.body.block.erase_op(function)
    module.attributes["prov.kernel_binding"] = StringAttr("t2m.model-kernel-binding.v1")
    if guard_storage:
        entries = [f for f in functions.values() if "prov.kernel" not in f.attributes and f.body.blocks]
        if len(entries) != 1:
            raise ConversionError("guarded model scratch accounting requires one defined model entry")
        entries[0].attributes["prov.memory_guard_scratch_bytes"] = IntegerAttr(guard_storage, i64)
    module.verify()
    selected = {r["implementation_sha256"] for r in records}
    kernels = {b.implementation.trace["mlir_sha256"]: b.implementation.trace for b in bindings
               if b.implementation.trace["mlir_sha256"] in selected}
    report = {"schema": "t2m.model-kernel-binding.v1", "calls": records, "kernels": kernels,
              "input_model_sha256": hashlib.sha256(model_mlir.encode()).hexdigest(),
              "mlir_sha256": hashlib.sha256(text(module).encode()).hexdigest(),
              "model_abi": "unchanged", "structural_verification": "passed",
              "native_execution": "not-performed", "merlin_execution": "not-performed"}
    if guard_storage:
        report["guard_storage_bytes"] = guard_storage
    return BoundModel(module, report)

"""Bind translated tensor bodies to model2MLIR's actual custom-kernel calls.

The stable identity is prov.kernel, not a generated callee name. Binding is
per call and checks schema names, typed SSA operands/results and scalar values.
No Torch or target backend is imported by this adapter.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr,
                                  StringAttr, UnitAttr, f64, i1, i64)

from .diagnostics import ConversionError
from .ir import text
from .shared_tensor import SharedTensorResult, parse_shared
from .memory_contract import GUARDED_PROFILES, validate_memory_evidence
from .model_arguments import materialize_arguments, plan_arguments, runtime_constants
from .scalar_abi import RuntimeScalar
from .model_tensor_views import BINDING_SCHEMA, POLICY as VIEW_POLICY, PROFILE as VIEW_PROFILE, reshape


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
    # Source runtime-scalar name -> captured prov.kernel_args schema name.
    runtime_arguments: dict[str, str] = field(default_factory=dict)
    # Explicit logical row-major reshapes, never physical view/alias inference.
    tensor_views: str | None = None
    # Semantic source-layout option -> original captured scalar argument.
    source_layout_arguments: dict[str, str] = field(default_factory=dict)

    def validate(self):
        if self.tensor_views not in (None,VIEW_POLICY):
            raise ValueError("unknown model tensor view policy")
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
        if (list(function.function_type.inputs) != [value.type for value in result.spec.inputs]
                or list(function.function_type.outputs) != [value.type for value in result.spec.outputs]):
            raise ConversionError("kernel function signature differs from its declared tensor/scalar ABI")
        inputs = [b.name for b in result.spec.inputs if not isinstance(b, RuntimeScalar)]
        names = self.operands or {n: n for n in inputs}
        if set(names) != set(inputs) or len(set(names.values())) != len(names):
            raise ConversionError("bind every source tensor input to a distinct schema operand name")
        output_names = [b.name for b in result.spec.outputs]
        outputs = self.results or tuple(output_names)
        if len(outputs) != len(output_names) or set(outputs) != set(output_names):
            raise ConversionError("bind every source result exactly once")
        scalar = DictionaryAttr({n: scalar_attribute(v) for n, v in self.scalar_arguments.items()})
        runtime_constants(result.spec, self.runtime_arguments, scalar)
        from .model_source_layout import bind_arguments
        bind_arguments(result.trace, self.source_layout_arguments, scalar)
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
    for parent_call_ordinal, call in enumerate(calls):
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
            plan = plan_arguments(binding, function, operand_names, output_names, by_name, arguments,
                                  [value.type for value in call.results])
            if plan is not None:
                candidates.append((binding, function, plan, output_names))
        if len(candidates) != 1:
            raise ConversionError(f"{identity.data}: expected one matching typed/scalar specialization, found {len(candidates)}",
                                  opcode="func.call")
        binding, function, plan, output_names = candidates[0]
        if binding.implementation.trace.get("profile") in GUARDED_PROFILES:
            outputs = binding.implementation.spec.outputs
            cost = json.loads(binding.implementation.auxiliary_artifacts["memory-admission.json"])["guard_storage_bytes_per_output_element"]
            from .typed_storage_contract import checked_guard_costs
            costs = checked_guard_costs(function, binding.implementation.trace, cost)
            guard_storage += sum(cost * math.prod(plan.outputs[b.name].get_shape()) for cost, b in zip(costs, outputs, strict=True))
            if not 0 <= guard_storage < 2**63:
                raise ConversionError("guarded model scratch exceeds its i64 domain", opcode="func.call")
        callsite = f"kernel_call.{len(records):04d}"
        emitted = []
        adaptations = list(plan.adaptations)

        def append_adapter(op, kind):
            op.location = call.location
            op.attributes.update({"prov.family": StringAttr("binding"), "prov.op": StringAttr(kind),
                "prov.kernel_origin": identity, "prov.kernel_callsite": StringAttr(callsite),
                "prov.region_id": StringAttr(callsite + "/abi"),
                "prov.emitted_id": StringAttr(f"{callsite}/abi.{len(emitted):04d}"),
                "prov.kernel_source_sha256": StringAttr(binding.implementation.trace["source_sha256"])})
            emitted.append(op)
            return op.results[0]

        operands = materialize_arguments(plan, append_adapter)
        mapper = dict(zip(function.body.block.args, operands, strict=True))
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
        for old, name in zip(call.results, output_names, strict=True):
            if name in plan.result_views:
                if result_map[name].type != plan.outputs[name]:
                    result_map[name] = append_adapter(tensor.CastOp(result_map[name],plan.outputs[name]),"kernel-concrete-result-shape")
                result_map[name] = reshape(result_map[name],old.type,append_adapter,role="kernel-to-caller-view")
            if result_map[name].type != old.type:
                result_map[name] = append_adapter(tensor.CastOp(result_map[name], old.type), "kernel-to-caller-shape")
                adaptations.append({"kind": "result-shape-cast", "name": name, "caller_type": str(old.type)})
        block = call.parent
        block.insert_ops_before(emitted, call)
        for old, name in zip(call.results, output_names, strict=True):
            old.replace_all_uses_with(result_map[name])
        block.erase_op(call)
        used.add(symbol)
        records.append({"id": callsite, "kernel": identity.data, "callee": symbol,
                        "parent_call_ordinal": parent_call_ordinal, "source_operand_bindings": operand_names,
                        "model_call_attributes": {n: str(v) for n, v in call.attributes.items()},
                        "schema_operands": [n.data for n in names], "source_result_order": list(output_names),
                        "argument_adaptations": adaptations, "runtime_scalar_bindings": plan.scalar_bindings,
                        "implementation_sha256": binding.implementation.trace["mlir_sha256"],
                        "source_sha256": binding.implementation.trace["source_sha256"]})
        if binding.tensor_views is not None:
            records[-1]["tensor_views"] = binding.tensor_views
        if "source_layout_specialization" in binding.implementation.trace:
            from .model_source_layout import bind_arguments
            records[-1]["source_layout_bindings"] = bind_arguments(
                binding.implementation.trace, binding.source_layout_arguments, arguments)
    remaining_calls = {op.callee.root_reference.data for op in module.walk() if isinstance(op, func.CallOp)}
    for symbol, function in functions.items():
        if "prov.kernel" in function.attributes:
            if symbol not in used or symbol in remaining_calls:
                raise ConversionError(f"unresolved kernel declaration @{symbol}")
            module.body.block.erase_op(function)
    view_binding = any(record.get("tensor_views") for record in records)
    binding_schema = BINDING_SCHEMA if view_binding else "t2m.model-kernel-binding.v1"
    module.attributes["prov.kernel_binding"] = StringAttr(binding_schema)
    if guard_storage:
        entries = [f for f in functions.values() if "prov.kernel" not in f.attributes and f.body.blocks]
        if len(entries) != 1:
            raise ConversionError("guarded model scratch accounting requires one defined model entry")
        entries[0].attributes["prov.memory_guard_scratch_bytes"] = IntegerAttr(guard_storage, i64)
    module.verify()
    selected = {r["implementation_sha256"] for r in records}
    kernels = {b.implementation.trace["mlir_sha256"]: b.implementation.trace for b in bindings
               if b.implementation.trace["mlir_sha256"] in selected}
    report = {"schema": binding_schema, "calls": records, "kernels": kernels,
              "input_model_sha256": hashlib.sha256(model_mlir.encode()).hexdigest(),
              "mlir_sha256": hashlib.sha256(text(module).encode()).hexdigest(),
              "model_abi": "unchanged", "structural_verification": "passed",
              "native_execution": "not-performed", "merlin_execution": "not-performed"}
    if guard_storage:
        report["guard_storage_bytes"] = guard_storage
    if any(record["runtime_scalar_bindings"] for record in records):
        report["argument_binding_profile"] = "t2m.captured-runtime-arguments.v1"
    if view_binding:
        report["tensor_view_binding_profile"] = VIEW_PROFILE
    if any("source_layout_bindings" in record for record in records):
        from .model_source_layout import PROFILE
        report["source_layout_binding_profile"] = PROFILE
    return BoundModel(module, report)

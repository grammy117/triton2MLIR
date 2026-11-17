"""Bind actual scalar SSA edges without turning evaluator values into facts.

Use standard Tensor dim/extract and Arith index casts. Singleton tensors carry
one logical value regardless of rank. Static dimension facts come only from
actual parent SSA/type semantics, never a sampled input or provenance claim.
Checked unsigned index ranges additionally require dominating source assertions;
they do not become concrete shape facts.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin, tensor

from .diagnostics import ConversionError

PROFILE = "t2m.live-runtime-arguments.v1"
RANGE_PROFILE = "t2m.checked-index-runtime-arguments.v1"


def binding_profile(calls):
    live = []
    for call in calls:
        bindings = call.get("runtime_operand_bindings", {})
        if not isinstance(bindings, dict) or any(not isinstance(binding, dict) for binding in bindings.values()):
            raise ValueError("live runtime operand bindings must be dictionaries")
        live.extend(bindings.values())
    if any(binding.get("adapter") == "checked-index-castui" for binding in live):
        return RANGE_PROFILE
    return PROFILE if live else None


def validate_mapping(spec, mapping):
    names = {scalar.name for scalar in spec.runtime_scalars}
    if (not isinstance(mapping, dict) or not set(mapping) <= names
            or any(not isinstance(value, str) or not value for value in mapping.values())):
        raise ConversionError("live runtime operands need schema names for declared scalars", opcode="func.call")
    return set(mapping)


def known_integer(value, depth=0):
    if depth >= 16 or not hasattr(value.owner, "name"):
        return None
    operation = value.owner
    if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
        return operation.value.value.data
    if isinstance(operation, tensor.DimOp):
        source, axis = operation.operands
        number = known_integer(axis, depth + 1)
        if (isinstance(source.type, builtin.TensorType) and type(number) is int
                and 0 <= number < len(source.type.get_shape())):
            extent = source.type.get_shape()[number]
            return extent if extent >= 0 else None
    if isinstance(operation, tensor.FromElementsOp) and len(operation.operands) == 1:
        return known_integer(operation.operands[0], depth + 1)
    return None


@dataclass(frozen=True)
class LiveOperand:
    value: object
    target: object
    adapter: str
    record: dict

    def materialize(self, append):
        value = self.value
        if self.adapter == "singleton-tensor-extract":
            indices = [append(arith.ConstantOp.from_int_and_width(0, builtin.IndexType()), "live-scalar-index")
                       for _ in value.type.get_shape()]
            value = append(tensor.ExtractOp.build(operands=[value, indices], result_types=[self.target]),
                           "live-runtime-scalar")
        elif self.adapter == "proved-index-cast":
            value = append(arith.IndexCastOp(value, self.target), "live-runtime-scalar")
        elif self.adapter == "checked-index-castui":
            from .standard_arith import IndexCastUIOp

            value = append(IndexCastUIOp(value, self.target), "live-runtime-scalar")
        return value


def plan_operands(spec, mapping, by_name, *, source_facts=None, source_ranges=None, source_point=None,
                  symbolic_dimensions=None):
    validate_mapping(spec, mapping)
    plans, facts = {}, {}
    for scalar in spec.runtime_scalars:
        if scalar.name not in mapping:
            continue
        name = mapping[scalar.name]
        if name not in by_name:
            raise ConversionError("live runtime scalar has no original schema operand: " + name, opcode="func.call")
        value, target = by_name[name], scalar.type
        fact = known_integer(value) if source_facts is None else source_facts[name]
        enclosure = None
        if isinstance(value.type, builtin.TensorType):
            if (not isinstance(value.type.encoding, builtin.NoneAttr) or value.type.element_type != target
                    or any(extent != 1 for extent in value.type.get_shape())):
                raise ConversionError("live runtime scalar tensor needs one logical element of its exact declared type",
                                      opcode="func.call")
            adapter = "singleton-tensor-extract"
        elif value.type == target:
            adapter = "exact-scalar"
        elif isinstance(value.type, builtin.IndexType) and scalar.dtype in ("i8", "i16", "i32", "i64"):
            width = target.bitwidth
            if type(fact) is int and -(1 << (width - 1)) <= fact < 1 << (width - 1):
                adapter = "proved-index-cast"
            elif fact is None:
                from .source_index_ranges import unsigned_index_range

                enclosure = (source_ranges[name] if source_ranges is not None else
                             unsigned_index_range(value, source_point) if source_point is not None else None)
                if enclosure is None or not 0 <= enclosure[0] <= enclosure[1] < 1 << (width - 1):
                    raise ConversionError("live index conversion needs a representable source SSA shape/integer fact",
                                          opcode="func.call")
                adapter = "checked-index-castui"
            else:
                raise ConversionError("live index conversion needs a representable source SSA shape/integer fact",
                                      opcode="func.call")
        else:
            raise ConversionError("live runtime scalar must retain its exact numeric/Boolean type", opcode="func.call")
        if scalar.bounds is not None and fact is not None and not scalar.bounds[0] <= fact <= scalar.bounds[1]:
            raise ConversionError("live runtime scalar's proved value is outside its declared bounds", opcode="func.call")
        record = {"schema_operand": name, "caller_type": str(value.type), "kernel_type": str(target), "adapter": adapter}
        if fact is not None:
            record["constant_integer"] = fact
            facts[scalar.name] = fact
        if enclosure is not None:
            record["unsigned_integer_range"] = list(enclosure)
        plans[scalar.name] = LiveOperand(value, target, adapter, record)
    dimensions = {dimension for buffer in spec.buffers for dimension in buffer.shape if isinstance(dimension, str)}
    if dimensions & set(mapping) - set(facts) - set(symbolic_dimensions or {}):
        raise ConversionError("live runtime shape dimension needs an actual parent SSA/type fact", opcode="func.call")
    return plans, facts


def validate_model(trace, parent, *, implementations=None, bound_module=None):
    from xdsl.dialects import func

    from .shared_tensor import TensorABI

    calls = trace.get("calls", [])
    from .model_shape_facts import PROFILE as SHAPE_PROFILE
    from .model_shape_facts import ShapeFacts

    shape_facts = None
    entries = [op for op in parent.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward" and op.body.blocks]
    shape_profile = trace.get("model_shape_profile")
    dynamic = len(entries) == 1 and any(isinstance(type_, builtin.TensorType) and any(n < 0 for n in type_.get_shape())
                                      for type_ in (*entries[0].function_type.inputs, *entries[0].function_type.outputs))
    if (shape_profile is not None or dynamic or any("symbolic_tensor_shapes" in record for record in calls)
            or "runtime_shape_contract" in trace):
        if shape_profile != SHAPE_PROFILE or len(entries) != 1 or bound_module is None or implementations is None:
            raise ValueError("symbolic model shapes require their explicit profile and executable implementations")
        ordinals = [record.get("parent_call_ordinal") for record in calls]
        if any(type(number) is not int for number in ordinals) or ordinals != sorted(set(ordinals)):
            raise ValueError("symbolic model call shapes require original source order")
        shape_facts = ShapeFacts.entry(entries[0])
    expected_profile = binding_profile(calls)
    if (expected_profile is not None and trace.get("argument_binding_profile") != expected_profile
            or expected_profile is None and trace.get("argument_binding_profile") in (PROFILE, RANGE_PROFILE)):
        raise ValueError("live runtime operands require their explicit binding profile")
    parent_calls = [op for op in parent.walk() if isinstance(op, func.CallOp)]
    for record in calls:
        live = record.get("runtime_operand_bindings", {})
        if not isinstance(live, dict) or any(not isinstance(name, str) for name in live):
            raise ValueError("live runtime operand bindings must be complete dictionaries")
        if not live:
            if "symbolic_tensor_shapes" in record:
                raise ValueError("symbolic tensor dimensions require actual live scalar bindings")
            continue
        ordinal = record.get("parent_call_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(parent_calls):
            raise ValueError("live runtime operand lacks its original typed call")
        call = parent_calls[ordinal]
        names = call.attributes.get("prov.kernel_operands")
        if (not isinstance(names, builtin.ArrayAttr) or len(names) != len(call.operands)
                or any(not isinstance(name, builtin.StringAttr) for name in names)
                or call.attributes.get("prov.kernel") != builtin.StringAttr(record["kernel"])
                or record.get("model_call_attributes") != {key: str(value) for key, value in call.attributes.items()}):
            raise ValueError("live runtime operands disagree with original typed call")
        from types import SimpleNamespace

        from .model_arguments import plan_arguments

        mapping = record.get("source_operand_bindings", {})
        outputs = record.get("source_result_order", [])
        captures = record.get("runtime_scalar_bindings", {})
        names = [name.data for name in names]
        if (len(set(names)) != len(names) or set(mapping.values()) | {item["schema_operand"] for item in live.values()}
                | set(record.get("source_tensor_views", {})) != set(names)):
            raise ValueError("live runtime operands lack complete distinct source mappings")
        try:
            spec = TensorABI.from_dict(trace["kernels"][record["implementation_sha256"]]["spec"])
            binding = SimpleNamespace(implementation=SimpleNamespace(spec=spec),
                runtime_arguments={name: item["schema_argument"] for name, item in captures.items()},
                runtime_operands={name: item["schema_operand"] for name, item in live.items()},
                tensor_views=record.get("tensor_views"))
            function = func.FuncOp.external("checked_interface", [item.type for item in spec.inputs],
                                             [item.type for item in spec.outputs])
            if shape_facts is not None:
                function = implementations[record["implementation_sha256"]]
                if (list(function.function_type.inputs) != [item.type for item in spec.inputs]
                        or list(function.function_type.outputs) != [item.type for item in spec.outputs]):
                    raise ValueError("symbolic kernel body types differ from its checked ABI")
            plan = plan_arguments(binding, function, mapping, outputs, dict(zip(names, call.operands, strict=True)),
                                  call.attributes.get("prov.kernel_args", builtin.DictionaryAttr({})),
                                  [value.type for value in call.results], source_point=call, shape_facts=shape_facts)
        except (KeyError, TypeError, ConversionError) as error:
            raise ValueError("live runtime operands do not reconstruct the parent call interface") from error
        if plan is None or plan.scalar_bindings != captures or plan.live_bindings != live:
            raise ValueError("live runtime operands disagree with parent tensor/result shapes")
        expected_shapes = {name: list(shape) for name, shape in plan.symbolic_shapes.items()}
        if record.get("symbolic_tensor_shapes") != (expected_shapes or None):
            raise ValueError("symbolic model tensor relationships differ from original SSA and executable kernel shapes")
        for value, name in zip(call.results, outputs, strict=True):
            if plan.symbolic_shapes:
                shape_facts.tensors[value] = plan.symbolic_shapes[name]
    if shape_facts is not None:
        if (("guard_storage_bytes" in trace) != ("guard_storage_scope" in trace)
                or "guard_storage_scope" in trace and trace["guard_storage_scope"] != "sum-of-source-shape-upper-bounds"):
            raise ValueError("symbolic guarded scratch requires its explicit shape-bound scope")
        bound_entries = [op for op in bound_module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"]
        contract = shape_facts.contract(entries[0])
        if (len(bound_entries) != 1 or bound_entries[0].function_type != entries[0].function_type
                or trace.get("runtime_shape_contract") != contract
                or ShapeFacts.entry(bound_entries[0]).contract(bound_entries[0]) != contract):
            raise ValueError("symbolic model runtime shape contract differs from its complete executable source")

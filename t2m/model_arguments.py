"""Checked model call attributes specialize a kernel's runtime ABI.

LLVM a47bddcc TensorOps.cpp CastOp::areCastCompatible and Tensor canonicalize
tests define the standard shape-cast boundary. We additionally prove every
concrete extent from the typed caller and captured scalar attributes. Neither
reference archives nor evaluator input values participate in this adapter.
"""

import math
from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, tensor

from .diagnostics import ConversionError
from .model_tensor_views import POLICY, compatible, reshape
from .scalar_abi import RuntimeScalar


def runtime_constants(spec, mapping, attributes, *, live=()):
    scalars = {scalar.name: scalar for scalar in spec.runtime_scalars if scalar.name not in live}
    if set(mapping) != set(scalars) or any(
        not isinstance(name, str) or name not in attributes.data
        for name in mapping.values()
    ):
        raise ConversionError(
            "bind every runtime scalar to a captured schema scalar argument",
            opcode="func.call",
        )
    constants, values = {}, {}
    for name, scalar in scalars.items():
        value = attributes.data[mapping[name]]
        if scalar.dtype == "i1":
            if not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i1:
                raise ConversionError(
                    "runtime Boolean requires a captured Boolean attribute",
                    opcode="func.call",
                )
            number = bool(value.value.data)
            constant = builtin.IntegerAttr(int(number), builtin.i1)
        elif scalar.dtype.startswith("i"):
            if not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64:
                raise ConversionError(
                    "runtime integer requires a captured integer attribute",
                    opcode="func.call",
                )
            number = value.value.data
            width = scalar.type.width.data
            if (
                not -(1 << (width - 1)) <= number < 1 << (width - 1)
                or scalar.bounds is not None
                and not scalar.bounds[0] <= number <= scalar.bounds[1]
            ):
                raise ConversionError(
                    "captured runtime scalar is outside its declared type/bounds: "
                    + name,
                    opcode="func.call",
                )
            constant = builtin.IntegerAttr(number, scalar.type)
        else:
            if (
                not isinstance(value, builtin.FloatAttr)
                or value.type != builtin.f64
                or not math.isfinite(value.value.data)
            ):
                raise ConversionError(
                    "runtime float requires a finite captured floating attribute",
                    opcode="func.call",
                )
            number, constant = value.value.data, value
        constants[name] = constant
        values[name] = number
    return constants, values


def concrete_type(buffer, runtime_values):
    return builtin.TensorType(
        buffer.type.element_type,
        [runtime_values[dim] if isinstance(dim, str) else dim for dim in buffer.shape],
    )


@dataclass
class ArgumentPlan:
    arguments: list
    outputs: dict
    adaptations: list
    scalar_bindings: dict
    input_views: dict = field(default_factory=dict)
    result_views: dict = field(default_factory=dict)
    live_arguments: dict = field(default_factory=dict)
    live_bindings: dict = field(default_factory=dict)
    runtime_values: dict = field(default_factory=dict)
    symbolic_shapes: dict = field(default_factory=dict)
    output_max_shapes: dict = field(default_factory=dict)


def plan_arguments(
    binding, function, operand_names, output_names, by_name, attributes, result_types, *, source_facts=None,
    source_ranges=None, source_point=None, shape_facts=None, source_shape_values=None
):
    spec = binding.implementation.spec
    from .model_live_arguments import known_integer, plan_operands

    if set(binding.runtime_operands.values()) & set(attributes.data):
        raise ConversionError("schema arguments cannot be both live SSA operands and captured attributes", opcode="func.call")
    dimensions = {dim for buffer in spec.buffers for dim in buffer.shape if isinstance(dim, str)}
    symbolic = {}
    original = by_name if source_shape_values is None else source_shape_values
    if shape_facts is not None and shape_facts.expressions and source_point is not None:
        source_ranges = dict(source_ranges or {})
        for name, schema in binding.runtime_operands.items():
            if source_ranges.get(schema) is None and (isinstance(original[schema].type, builtin.IndexType)
                    or shape_facts.data_words and name in dimensions and isinstance(original[schema].type, builtin.IntegerType)):
                source_ranges[schema] = shape_facts.range_at(original[schema], source_point)
    if shape_facts is not None:
        for name, schema in binding.runtime_operands.items():
            fact = source_facts[schema] if source_facts is not None else known_integer(original[schema])
            if name in dimensions and fact is None:
                expression = shape_facts.size(original[schema], source_point)
                if expression is not None:
                    symbolic[name] = expression
    live, facts = plan_operands(spec, binding.runtime_operands, by_name, source_facts=source_facts,
                               source_ranges=source_ranges, source_point=source_point, symbolic_dimensions=symbolic,
                               word_sizes=shape_facts is not None and shape_facts.data_words)
    constants, values = runtime_constants(spec, binding.runtime_arguments, attributes, live=live)
    values.update(facts)
    symbolic = {name: expression for name, expression in symbolic.items() if name not in values}
    for name in symbolic:
        enclosure = live[name].record.get("unsigned_integer_range", live[name].record.get("integer_range"))
        scalar = next(item for item in spec.runtime_scalars if item.name == name)
        if (enclosure is None or scalar.bounds is None
                or not scalar.bounds[0] <= enclosure[0] <= enclosure[1] <= scalar.bounds[1]):
            raise ConversionError("symbolic tensor dimension needs checked source bounds inside the kernel domain",
                                  opcode="func.call")
    logical_shapes = {buffer.name: tuple(values[dim] if dim in values else symbolic[dim] if isinstance(dim, str) else dim
                                        for dim in buffer.shape) for buffer in spec.buffers}
    type_values = {**values, **{name: builtin.DYNAMIC_INDEX for name in symbolic}}
    arguments, adaptations, input_views = [], [], {}
    live_arguments = {}
    for index, (source, target) in enumerate(
        zip(spec.inputs, function.function_type.inputs, strict=True)
    ):
        if isinstance(source, RuntimeScalar):
            if source.name in live:
                plan = live[source.name]
                arguments.append((plan.value, target))
                live_arguments[index] = plan
                adaptations.append({"kind": "live-runtime-scalar", "argument": index,
                                    "name": source.name, **plan.record})
                continue
            arguments.append((constants[source.name], target))
            adaptations.append(
                {
                    "kind": "captured-runtime-scalar",
                    "argument": index,
                    "name": source.name,
                    "schema_argument": binding.runtime_arguments[source.name],
                    "kernel_type": str(target),
                    "captured_attribute": str(
                        attributes.data[binding.runtime_arguments[source.name]]
                    ),
                    "conversion": "rtne-f64-to-f32"
                    if target == builtin.f32
                    else "exact-typed-constant",
                }
            )
        else:
            value = by_name[operand_names[source.name]]
            concrete = concrete_type(source, type_values)
            if symbolic and not shape_facts.same_shape(shape_facts.shape(original[operand_names[source.name]]),
                                                       logical_shapes[source.name], source_point):
                return None
            if value.type != concrete:
                if binding.tensor_views != POLICY or not compatible(value.type,concrete):
                    return None
                input_views[index] = concrete
                adaptations.append({"kind":"dense-tensor-input-view","argument":index,"name":source.name,
                    "caller_type":str(value.type),"concrete_kernel_type":str(concrete),"policy":POLICY})
            arguments.append((value, target))
            if concrete != target:
                adaptations.append(
                    {
                        "kind": "tensor-shape-cast",
                        "argument": index,
                        "name": source.name,
                        "caller_type": str(value.type),
                        "kernel_type": str(target),
                    }
                )
    outputs = {buffer.name: concrete_type(buffer, type_values) for buffer in spec.outputs}
    if len(output_names) != len(result_types):
        return None
    result_views = {}
    for name,target in zip(output_names,result_types,strict=True):
        if outputs[name] != target:
            if binding.tensor_views != POLICY or not compatible(outputs[name],target):
                return None
            result_views[name] = target
            adaptations.append({"kind":"dense-tensor-result-view","name":name,"caller_type":str(target),
                "concrete_kernel_type":str(outputs[name]),"policy":POLICY})
    maximum = {}
    if symbolic:
        from .model_shape_facts import ShapeFacts

        kernel_facts = ShapeFacts(expressions=shape_facts.expressions)
        shape_facts.copy_context(kernel_facts, source_point)
        for source, argument in zip(spec.inputs, function.body.block.args, strict=True):
            if isinstance(source, RuntimeScalar):
                expression = values.get(source.name, symbolic.get(source.name))
                if expression is not None and (isinstance(expression, str) or type(expression) is int and expression >= 0):
                    kernel_facts.integers[argument] = expression
            else:
                kernel_facts.tensors[argument] = logical_shapes[source.name]
        terminal = function.body.block.last_op
        from xdsl.dialects import func

        if (not isinstance(terminal, func.ReturnOp)
                or any(not kernel_facts.same_shape(kernel_facts.shape(value), logical_shapes[buffer.name], terminal)
                       for value, buffer in zip(terminal.operands, spec.outputs, strict=True))):
            raise ConversionError("symbolic kernel result shape needs an executable standard-MLIR proof", opcode="func.call")
        maximum = {buffer.name: tuple(values[dim] if dim in values else
                                     live[dim].record.get("unsigned_integer_range", live[dim].record.get("integer_range"))[1]
                                     if isinstance(dim, str) else dim for dim in buffer.shape)
                   for buffer in spec.outputs}
    return ArgumentPlan(
        arguments,
        outputs,
        adaptations,
        {
            name: {
                "schema_argument": binding.runtime_arguments[name],
                "attribute": str(attributes.data[binding.runtime_arguments[name]]),
            }
            for name in constants
        },input_views,result_views,live_arguments,{name: plan.record for name, plan in live.items()},values,
        symbolic_shapes=logical_shapes if symbolic else {}, output_max_shapes=maximum,
    )


def materialize_arguments(plan, append):
    arguments = []
    for index,(value, target) in enumerate(plan.arguments):
        if index in plan.live_arguments:
            value = plan.live_arguments[index].materialize(append)
        elif isinstance(value, (builtin.IntegerAttr, builtin.FloatAttr)):
            value = append(arith.ConstantOp(value), "captured-runtime-scalar")
            if value.type != target:
                value = append(
                    arith.TruncFOp(value, target), "captured-runtime-scalar-rounding"
                )
        else:
            if index in plan.input_views:
                value = reshape(value,plan.input_views[index],append,role="caller-to-kernel-view")
            if value.type != target:
                value = append(tensor.CastOp(value, target), "caller-to-kernel-shape")
        arguments.append(value)
    return arguments

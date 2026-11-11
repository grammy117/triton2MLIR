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


def runtime_constants(spec, mapping, attributes):
    scalars = {scalar.name: scalar for scalar in spec.runtime_scalars}
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


def plan_arguments(
    binding, function, operand_names, output_names, by_name, attributes, result_types
):
    spec = binding.implementation.spec
    constants, values = runtime_constants(spec, binding.runtime_arguments, attributes)
    arguments, adaptations, input_views = [], [], {}
    for index, (source, target) in enumerate(
        zip(spec.inputs, function.function_type.inputs, strict=True)
    ):
        if isinstance(source, RuntimeScalar):
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
            concrete = concrete_type(source,values)
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
    outputs = {buffer.name: concrete_type(buffer, values) for buffer in spec.outputs}
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
        },input_views,result_views,
    )


def materialize_arguments(plan, append):
    arguments = []
    for index,(value, target) in enumerate(plan.arguments):
        if isinstance(value, (builtin.IntegerAttr, builtin.FloatAttr)):
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

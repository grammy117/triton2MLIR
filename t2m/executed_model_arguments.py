"""Executable call contracts for sizes produced inside the model.

LLVM a47bddccec30255619bb8c37fa59700e661d4e66 TensorOps.cpp,
ArithOps.cpp and Tensor invalid/Arith canonicalize tests define compatible
tensor casts and signed index extension/truncation. A cast is not a guard:
check dimensions and integer round trips before narrowing or allocating.
No public-input size prediction or evaluator value participates here.
"""

from xdsl.dialects import arith, builtin, cf, tensor

from .diagnostics import ConversionError
from .model_arguments import runtime_constants
from .model_tensor_views import RUNTIME_POLICY, reshape
from .model_tensor_views import compatible as dense_compatible
from .scalar_abi import RuntimeScalar


def compatible(source, target):
    return (
        isinstance(source, builtin.TensorType)
        and isinstance(target, builtin.TensorType)
        and isinstance(source.encoding, builtin.NoneAttr)
        and isinstance(target.encoding, builtin.NoneAttr)
        and source.element_type == target.element_type
        and len(source.get_shape()) == len(target.get_shape())
        and all(
            a < 0 or b < 0 or a == b
            for a, b in zip(source.get_shape(), target.get_shape(), strict=True)
        )
    )


def accepts(binding, names, outputs, by_name, result_types, attributes):
    spec = binding.implementation.spec
    _, numbers = runtime_constants(
        spec, binding.runtime_arguments, attributes, live=binding.runtime_operands
    )

    def matches(caller, buffer, *, input_view):
        concrete = concrete_view_type(buffer, numbers)
        if compatible(caller, buffer.type):
            return True
        if binding.tensor_views == RUNTIME_POLICY:
            from .executed_runtime_views import admissible

            return (
                admissible(caller, buffer.type, input_view=True)
                if input_view
                else admissible(buffer.type, caller, input_view=False)
            )
        return (
            binding.tensor_views is not None
            and concrete is not None
            and dense_compatible(caller, concrete)
        )

    if set(binding.runtime_operands.values()) & set(binding.scalar_arguments):
        raise ConversionError(
            "schema arguments cannot be both live and captured", opcode="func.call"
        )
    if len(outputs) != len(result_types):
        return False
    if any(
        not matches(by_name[names[b.name]].type, b, input_view=True)
        for b in spec.inputs
        if not isinstance(b, RuntimeScalar)
    ):
        return False
    buffers = {b.name: b for b in spec.outputs}
    if any(
        not matches(target, buffers[name], input_view=False)
        for name, target in zip(outputs, result_types, strict=True)
    ):
        return False
    for scalar in spec.runtime_scalars:
        if scalar.name not in binding.runtime_operands:
            continue
        typ = by_name[binding.runtime_operands[scalar.name]].type
        if isinstance(typ, builtin.TensorType):
            if (
                not isinstance(typ.encoding, builtin.NoneAttr)
                or typ.element_type != scalar.type
                or any(size != 1 for size in typ.get_shape())
            ):
                return False
        elif typ != scalar.type and not (
            isinstance(typ, builtin.IndexType)
            and scalar.dtype in ("i8", "i16", "i32", "i64")
        ):
            return False
    return True


def concrete_view_type(buffer, values):
    shape = [
        values.get(size) if isinstance(size, str) else size for size in buffer.shape
    ]
    return (
        builtin.TensorType(buffer.type.element_type, shape)
        if all(type(size) is int and size >= 0 for size in shape)
        else None
    )


def adapt_tensor(
    append, value, target, buffer, numbers, policy, *, role, dimensions=None
):
    if policy == RUNTIME_POLICY and (
        role == "caller-to-kernel-view" or not compatible(value.type, target)
    ):
        from .executed_runtime_views import RuntimeViews

        return RuntimeViews(append).reshape(
            value,
            target,
            buffer.shape,
            dimensions or {},
            input_view=role == "caller-to-kernel-view",
        )
    if compatible(value.type, target):
        return cast_tensor(append, value, target)
    concrete = concrete_view_type(buffer, numbers)
    if policy is None or concrete is None:
        raise ConversionError(
            "executed tensor view needs source-bound concrete dimensions",
            opcode="func.call",
        )
    if role == "caller-to-kernel-view":
        value = reshape(value, concrete, append, role=role)
        return cast_tensor(append, value, target)
    value = cast_tensor(append, value, concrete)
    return reshape(value, target, append, role=role)


def check(append, condition, message):
    append(cf.AssertOp(condition, "Triton executed call: " + message), "call-contract")


def constant(append, number, typ):
    return append(
        arith.ConstantOp.from_int_and_width(number, typ), "call-contract-constant"
    )


def cast_index(append, value, target):
    """Reject lossy signed integer/index transport at the actual invocation."""
    converted = append(arith.IndexCastOp(value, target), "checked-call-index-cast")
    restored = append(
        arith.IndexCastOp(converted, value.type), "checked-call-index-roundtrip"
    )
    equal = append(arith.CmpiOp(value, restored, "eq"), "checked-call-index-equal")
    check(append, equal, "integer/index conversion is not representable")
    return converted


def cast_tensor(append, value, target):
    if not compatible(value.type, target):
        raise ConversionError(
            "executed call tensor rank/storage/static dimensions disagree",
            opcode="func.call",
        )
    for axis, (actual, expected) in enumerate(
        zip(value.type.get_shape(), target.get_shape(), strict=True)
    ):
        if actual < 0 <= expected:
            index = constant(append, axis, builtin.IndexType())
            extent = append(tensor.DimOp(value, index), "call-actual-dimension")
            expected_value = constant(append, expected, builtin.IndexType())
            equal = append(
                arith.CmpiOp(extent, expected_value, "eq"), "call-static-dimension"
            )
            check(append, equal, "tensor narrowing changes a static dimension")
    return (
        value
        if value.type == target
        else append(tensor.CastOp(value, target), "checked-call-tensor-cast")
    )


def tensor_contract(append, value, buffer, dimensions):
    for axis, expected in enumerate(buffer.shape):
        if not isinstance(expected, str):
            continue
        index = constant(append, axis, builtin.IndexType())
        extent = append(tensor.DimOp(value, index), "call-actual-dimension")
        equal = append(
            arith.CmpiOp(extent, dimensions[expected], "eq"), "call-dimension-binding"
        )
        check(append, equal, "tensor dimension differs from scalar: " + buffer.name)


def arguments(binding, function, names, by_name, attributes, append):
    spec = binding.implementation.spec
    captured, numbers = runtime_constants(
        spec, binding.runtime_arguments, attributes, live=binding.runtime_operands
    )
    scalars = {}
    for scalar in spec.runtime_scalars:
        if scalar.name in captured:
            value = append(
                arith.ConstantOp(captured[scalar.name]), "captured-call-scalar"
            )
            if value.type != scalar.type:
                value = append(
                    arith.TruncFOp(value, scalar.type), "captured-call-scalar-rounding"
                )
        else:
            value = by_name[binding.runtime_operands[scalar.name]]
            if isinstance(value.type, builtin.TensorType):
                indices = [
                    constant(append, 0, builtin.IndexType())
                    for _ in value.type.get_shape()
                ]
                value = append(
                    tensor.ExtractOp.build(
                        operands=[value, indices], result_types=[scalar.type]
                    ),
                    "call-scalar-read",
                )
            elif value.type != scalar.type:
                value = cast_index(append, value, scalar.type)
        if scalar.bounds is not None:
            low, high = (constant(append, bound, value.type) for bound in scalar.bounds)
            lower = append(arith.CmpiOp(value, low, "sge"), "call-scalar-lower")
            upper = append(arith.CmpiOp(value, high, "sle"), "call-scalar-upper")
            valid = append(arith.AndIOp(lower, upper), "call-scalar-range")
            check(
                append, valid, "runtime scalar outside declared bounds: " + scalar.name
            )
        scalars[scalar.name] = value
    dimensions = {}
    for name in sorted(
        {dim for b in spec.buffers for dim in b.shape if isinstance(dim, str)}
    ):
        value = scalars[name]
        nonnegative = append(
            arith.CmpiOp(value, constant(append, 0, value.type), "sge"),
            "call-size-nonnegative",
        )
        check(append, nonnegative, "negative tensor dimension: " + name)
        dimensions[name] = cast_index(append, value, builtin.IndexType())
    values = []
    for source, target in zip(spec.inputs, function.function_type.inputs, strict=True):
        if isinstance(source, RuntimeScalar):
            values.append(scalars[source.name])
        else:
            value = by_name[names[source.name]]
            if getattr(
                binding, "tensor_views", None
            ) == RUNTIME_POLICY or not compatible(value.type, target):
                value = adapt_tensor(
                    append,
                    value,
                    target,
                    source,
                    numbers,
                    getattr(binding, "tensor_views", None),
                    role="caller-to-kernel-view",
                    dimensions=dimensions,
                )
            tensor_contract(append, value, source, dimensions)
            values.append(cast_tensor(append, value, target))
    return values, dimensions

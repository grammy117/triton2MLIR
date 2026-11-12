"""Lift a closed scalar recipe before native TTIR normalization.

The later shared arithmetic conversion supplies parallel Linalg. This adapter
changes only the computation carrier; source implementations still validate
their complete semantics, effects, shape and result interface separately.
"""

from xdsl.dialects import arith, builtin


def lift(operations, bindings, outputs, tile_type):
    """Map scalar SSA to same-shaped arithmetic, retaining types and fields."""
    shape, encoding = tile_type.get_shape(), tile_type.encoding
    values, emitted = {}, []

    def shaped(type_):
        return builtin.TensorType(type_, shape, encoding)

    for scalar, actual in bindings.items():
        if isinstance(actual.type, builtin.TensorType):
            if (
                actual.type.get_shape() != shape
                or actual.type.element_type != scalar.type
                or actual.type.encoding != encoding
            ):
                raise ValueError(
                    "scalar DAG inputs require the same logical tile and encoding"
                )
            values[scalar] = actual
        else:
            if actual.type != scalar.type:
                raise ValueError(
                    "scalar DAG broadcast must preserve its exact numeric type"
                )
            splat = builtin.UnregisteredOp.with_name("tt.splat").create(
                operands=[actual], result_types=[shaped(actual.type)]
            )
            values[scalar] = splat.results[0]
            emitted.append(splat)
    for operation in operations:
        if (
            operation.regions
            or operation.successors
            or operation.name.split(".")[0] not in ("arith", "math")
        ):
            raise ValueError(
                "scalar DAG carrier requires closed standard elementwise arithmetic"
            )
        if any(value not in values for value in operation.operands):
            raise ValueError("scalar DAG carrier found an unbound scalar capture")
        result_types = [shaped(value.type) for value in operation.results]
        properties = dict(operation.properties)
        if isinstance(operation, arith.ConstantOp):
            value = properties["value"]
            properties["value"] = builtin.DenseIntOrFPElementsAttr.from_list(
                result_types[0], [value.value.data]
            )
        converted = type(operation).create(
            operands=[values[value] for value in operation.operands],
            result_types=result_types,
            attributes=dict(operation.attributes),
            properties=properties,
            location=operation.location,
        )
        emitted.append(converted)
        values.update(zip(operation.results, converted.results, strict=True))
    if any(value not in values for value in outputs):
        raise ValueError("scalar DAG carrier returned an unbound scalar result")
    return emitted, [values[value] for value in outputs]

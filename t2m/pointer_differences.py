"""Cancel a shared logical pointer base in modular i64 address differences.

Reuse shared 17f0c234's base/offset boundary and Ascend 8f0ade66's per-addptr
byte-unit rule (CommonPtrRules, parseAddPtr/parseBitcast). Each source integer
increment keeps its width until signed extension; pointer bitcasts preserve
bytes. This analysis supplies no physical address or cross-allocation relation.
"""

from xdsl.dialects import arith, builtin

from .pointer_addresses import SHAPES, _constant, _like, _scalar
from .pointer_views import element


def difference(left, right, emit, source_ids):
    """Build a detached byte-offset recipe and the source facts it consumed.

    Roots are actual SSA values, never names, provenance or declared aliases.
    Scalar roots are uniform under shape operations. Nonuniform roots retain
    their exact shape-operation path before they can be compared.
    """
    from .shared_tensor import opcode

    cache, sources = {}, set()

    def checked(operation, fields=()):
        if (operation.regions or operation.successors
                or any(key not in {"op_name", "op_name__", *fields} and not key.startswith("prov.")
                       for key in (*operation.properties, *operation.attributes))):
            raise ValueError("pointer difference has unqualified source fields or effects")
        sources.update(source_ids(operation.location))

    def offsets(pointer, depth=0):
        if pointer in cache:
            if cache[pointer] is None:
                raise ValueError("pointer difference has a cyclic undecomposed SSA producer")
            return cache[pointer]
        if depth >= 128 or len(cache) >= 8192:
            raise ValueError("pointer difference exceeds its bounded SSA budget")
        cache[pointer] = None
        type_ = _like(pointer.type, builtin.i64)
        operation = pointer.owner
        name = opcode(operation) if hasattr(operation, "name") else ""
        if name == "tt.addptr":
            checked(operation)
            if len(operation.operands) != 2 or len(operation.results) != 1:
                raise ValueError("pointer difference needs an exact addptr interface")
            base, previous, uniform = offsets(operation.operands[0], depth + 1)
            increment = operation.operands[1]
            word_type = _scalar(increment.type)
            if (not isinstance(word_type, builtin.IntegerType) or not 1 <= word_type.bitwidth <= 64
                    or _like(increment.type, builtin.i64) != type_):
                raise ValueError("pointer difference needs same-shaped source integer increments up to i64")
            if increment.type != type_:
                increment = emit(arith.ExtSIOp(increment, type_))
            width = element(operation.operands[0].type).bitwidth // 8
            if width != 1:
                increment = emit(arith.MuliOp(increment, emit(_constant(type_, width))))
            result = base, emit(arith.AddiOp(previous, increment)), uniform
        elif name == "tt.bitcast":
            checked(operation)
            if (len(operation.operands) != 1 or len(operation.results) != 1
                    or _like(operation.operands[0].type, builtin.i64) != type_):
                raise ValueError("pointer difference bitcasts must preserve shape and logical address space")
            element(operation.operands[0].type)
            element(pointer.type)
            result = offsets(operation.operands[0], depth + 1)
        elif name in SHAPES:
            fields = {"tt.expand_dims": {"axis"}, "tt.trans": {"order"},
                      "tt.reshape": {"allow_reorder", "efficient_layout"}}
            checked(operation, fields.get(name, ()))
            if len(operation.operands) != 1 or len(operation.results) != 1:
                raise ValueError("pointer difference shape operations require one input and result")
            if name == "tt.reshape" and ("allow_reorder" in operation.properties or "allow_reorder" in operation.attributes):
                raise ValueError("pointer difference reshape needs established paired lane ordering")
            base, previous, uniform = offsets(operation.operands[0], depth + 1)
            shaped = builtin.UnregisteredOp.with_name(name).create(
                operands=[previous], result_types=[type_], properties=dict(operation.properties),
                attributes={key: value for key, value in operation.attributes.items() if key not in ("op_name", "op_name__")})
            if not uniform:
                def semantic_fields(fields):
                    return tuple(sorted((key, value) for key, value in fields.items()
                                        if key not in ("op_name", "op_name__") and not key.startswith("prov.")))
                base = (name, base, type_, semantic_fields(operation.properties), semantic_fields(operation.attributes))
            result = base, emit(shaped), uniform
        elif name == "arith.select":
            checked(operation)
            if len(operation.operands) != 3 or len(operation.results) != 1:
                raise ValueError("pointer difference selection needs its exact condition and two pointers")
            first, a, first_uniform = offsets(operation.operands[1], depth + 1)
            second, b, second_uniform = offsets(operation.operands[2], depth + 1)
            if first == second:
                result = first, emit(arith.SelectOp(operation.operands[0], a, b)), first_uniform and second_uniform
            else:
                result = pointer, emit(_constant(type_, 0)), not isinstance(pointer.type, builtin.TensorType)
        else:
            # An exact SSA pointer is a valid common anchor even when its
            # producer/region recurrence is not decomposed here. Its address
            # cancels only against this same value and matching lane path.
            result = pointer, emit(_constant(type_, 0)), not isinstance(pointer.type, builtin.TensorType)
        cache[pointer] = result
        return result

    if left.pointer is right.pointer:
        first, second = left.displacement, right.displacement
    else:
        first_root, first_offset, _ = offsets(left.pointer)
        second_root, second_offset, _ = offsets(right.pointer)
        if first_root != second_root:
            raise ValueError("pointer difference needs the same established logical base and lane correspondence")
        first = emit(arith.AddiOp(first_offset, left.displacement))
        second = emit(arith.AddiOp(second_offset, right.displacement))
    return emit(arith.SubiOp(first, second)), sources

"""Preserve Triton's narrow register packing before proving logical independence.

Packing order adapts ElementwiseInlineAsmOpConversion::packOperands and its
result unpacking, triton-lang/triton f797708c0626e5f9840ca5b0a98790e2c7cb09ad.
Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
See docs/licenses/triton-LICENSE.txt. Source register bits are a language ABI,
not an output hardware choice. The independent proof below contains no ISA.
"""

from xdsl.dialects import arith, builtin

from .source_assembly import AssemblyRefusal


def interface(input_types, output_types, pack):
    """Return complete source register interfaces and their typed packing record."""
    def registers(types):
        result, groups = [], []
        for typ in types:
            if not isinstance(typ, builtin.IntegerType) or typ.width.data not in (1, 2, 4, 8, 16, 32, 64):
                raise AssemblyRefusal("packed word transport needs integer bit representations; mixed float registers need their own bitcast contract")
            width = typ.width.data
            slots = max(32 // width, 1)
            if pack % slots:
                raise AssemblyRefusal("partially filled source registers need an explicit padding/constraint contract")
            groups.append(slots)
            result.extend([builtin.i32 if width < 32 else typ] * (pack // slots))
        return tuple(result), groups

    inputs, input_groups = registers(input_types)
    outputs, output_groups = registers(output_types)
    return inputs, outputs, {"schema": "triton.inline-asm-word-packing.v1", "minimum_register_bits": 32,
        "order": "operand/result-major; increasing slots occupy increasing register bit positions",
        "input_elements_per_register": input_groups, "output_elements_per_register": output_groups,
        "padding": "none; complete source register groups"}


def pack_inputs(arguments, types, pack):
    operations, results = [], []

    def emit(op):
        operations.append(op)
        return op.results[0]

    for index, typ in enumerate(types):
        width = typ.width.data
        slots = max(32 // width, 1)
        values = arguments[index * pack:(index + 1) * pack]
        for start in range(0, pack, slots):
            if slots == 1:
                results.append(values[start])
                continue
            word = emit(arith.ConstantOp.from_int_and_width(0, builtin.i32))
            for slot, value in enumerate(values[start:start + slots]):
                widened = emit(arith.ExtUIOp(value, builtin.i32))
                if slot:
                    shift = emit(arith.ConstantOp.from_int_and_width(slot * width, builtin.i32))
                    widened = emit(arith.ShLIOp(widened, shift))
                word = emit(arith.OrIOp(word, widened))
            results.append(word)
    return operations, results


def unpack_outputs(arguments, types, pack):
    operations, results, index = [], [], 0
    for typ in types:
        width = typ.width.data
        slots = max(32 // width, 1)
        for _ in range(pack // slots):
            word = arguments[index]
            index += 1
            if slots == 1:
                results.append(word)
                continue
            for slot in range(slots):
                selected = word
                if slot:
                    shift = arith.ConstantOp.from_int_and_width(slot * width, builtin.i32)
                    shifted = arith.ShRUIOp(word, shift.result)
                    operations.extend([shift, shifted])
                    selected = shifted.result
                truncated = arith.TruncIOp(selected, typ)
                operations.append(truncated)
                results.append(truncated.result)
    if index != len(arguments):
        raise AssemblyRefusal("packed word output register inventory differs from its logical interface")
    return operations, results


def project(block, terminal, pack, arguments):
    """Prove all words total and every slot equal to one logical scalar recipe.

    Quantify every original logical input and uniform capture. The candidate
    recipe specializes other slots to zero only after UNSAT proves every output
    independent of them, including carries and shifted cross-byte bits. Never
    substitute a register grouping inferred from adjacent tensor coordinates.
    """
    try:
        import z3
    except ImportError as exc:
        raise AssemblyRefusal("packed word independence needs the installed word-proofs extra") from exc
    from xdsl.utils.exceptions import VerifyException

    if len(block.args) > 2048 or len(tuple(block.ops)) > 16384:
        raise AssemblyRefusal("packed word proof exceeds its finite scalar interface budget")
    values, bad, captures = {}, [], {}

    def width(value):
        typ = value.type
        if not isinstance(typ, builtin.IntegerType) or not 1 <= typ.width.data <= 64:
            raise AssemblyRefusal("packed word proof requires integer scalars of at most 64 bits")
        return typ.width.data

    for index, value in enumerate(block.args):
        values[value] = z3.BitVec(f"slot_{index}", width(value))

    def get(value):
        if value not in values:
            captures.setdefault(value, z3.BitVec(f"uniform_{len(captures)}", width(value)))
            return captures[value]
        return values[value]

    arithmetic = {arith.AddiOp: lambda a, b: a + b, arith.SubiOp: lambda a, b: a - b,
        arith.MuliOp: lambda a, b: a * b, arith.AndIOp: lambda a, b: a & b,
        arith.OrIOp: lambda a, b: a | b, arith.XOrIOp: lambda a, b: a ^ b,
        arith.MinSIOp: lambda a, b: z3.If(a < b, a, b),
        arith.MaxSIOp: lambda a, b: z3.If(a > b, a, b),
        arith.MinUIOp: lambda a, b: z3.If(z3.ULT(a, b), a, b),
        arith.MaxUIOp: lambda a, b: z3.If(z3.UGT(a, b), a, b)}
    operations = list(block.ops)[:-1]
    for op in operations:
        try:
            op.verify()
        except VerifyException as exc:
            raise AssemblyRefusal("packed word proof received an invalid typed arithmetic operation") from exc
        allowed = {"value"} if type(op) is arith.ConstantOp else {"predicate"} if type(op) is arith.CmpiOp else set()
        overflow = op.properties.get("overflowFlags")
        if overflow is not None:
            if type(overflow) is not arith.IntegerOverflowAttr or overflow.data:
                raise AssemblyRefusal("packed word proof cannot discard poison-generating overflow flags")
            if type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.ShLIOp):
                allowed.add("overflowFlags")
        if (set(op.properties) - allowed or any(not name.startswith("prov.") for name in op.attributes)
                or len(op.results) != 1):
            raise AssemblyRefusal("packed word proof needs an explicit model for every semantic property and result")
        operands = [get(value) for value in op.operands]
        bits = width(op.results[0])
        if type(op) is arith.ConstantOp and isinstance(op.value, builtin.IntegerAttr):
            result = z3.BitVecVal(op.value.value.data, bits)
        elif type(op) in arithmetic:
            result = arithmetic[type(op)](*operands)
        elif type(op) in (arith.ExtUIOp, arith.ExtSIOp):
            extension = bits - width(op.operands[0])
            result = (z3.ZeroExt if type(op) is arith.ExtUIOp else z3.SignExt)(extension, operands[0])
        elif type(op) is arith.TruncIOp:
            result = z3.Extract(bits - 1, 0, operands[0])
        elif type(op) in (arith.ShLIOp, arith.ShRUIOp, arith.ShRSIOp):
            a, count = operands
            bad.append(z3.Not(z3.ULT(count, z3.BitVecVal(bits, count.size()))))
            result = a << count if type(op) is arith.ShLIOp else (
                z3.LShR(a, count) if type(op) is arith.ShRUIOp else a >> count)
        elif type(op) is arith.CmpiOp:
            a, b = operands
            conditions = (a == b, a != b, a < b, a <= b, a > b, a >= b,
                          z3.ULT(a, b), z3.ULE(a, b), z3.UGT(a, b), z3.UGE(a, b))
            predicate = op.predicate.value.data
            if not 0 <= predicate < len(conditions):
                raise AssemblyRefusal("packed word proof has an unknown comparison predicate")
            result = z3.If(conditions[predicate], z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
        elif type(op) is arith.SelectOp:
            result = z3.If(operands[0] == 1, operands[1], operands[2])
        else:
            raise AssemblyRefusal("packed word proof has no total integer semantics for " + op.name)
        values[op.results[0]] = result
    returned = list(terminal.operands)
    if not returned or len(returned) % pack:
        raise AssemblyRefusal("packed word proof lacks its complete logical result roster")
    selected = returned[::pack]
    for output, representative in enumerate(selected):
        for slot in range(pack):
            substitutions = [(values[value], values[block.args[index // pack * pack + slot]]
                              if index % pack == 0 else z3.BitVecVal(0, width(value)))
                             for index, value in enumerate(block.args)]
            candidate = z3.substitute(get(representative), *substitutions)
            bad.append(get(returned[output * pack + slot]) != candidate)
    solver = z3.Solver()
    solver.set(timeout=2000, rlimit=1_000_000)
    solver.add(z3.Or(*bad))
    verdict = solver.check()
    if verdict != z3.unsat:
        reason = "coupled/asymmetric slots or a possibly undefined arithmetic operation" if verdict == z3.sat else solver.reason_unknown()
        raise AssemblyRefusal("packed word independence is unproved: " + reason + "; retain an explicit source grouping contract")
    needed, selected_ops = set(selected), []
    for op in reversed(operations):
        if any(value in needed for value in op.results):
            needed.update(op.operands)
            selected_ops.append(op)
    selected_ops.reverse()
    mapping, emitted, zeros = {}, [], {}
    for index, value in enumerate(block.args):
        if index % pack == 0:
            mapping[value] = arguments[index // pack]
        elif value in needed:
            if value.type not in zeros:
                zero = arith.ConstantOp.from_int_and_width(0, value.type)
                zero.location = terminal.location
                emitted.append(zero)
                zeros[value.type] = zero.result
            mapping[value] = zeros[value.type]
    for op in selected_ops:
        emitted.append(op.clone(value_mapper=mapping))
    results = [mapping.get(value, value) for value in selected]
    proof = {"method": "typed-word-bitvector-equivalence-v1", "grouping": "independent-of-register-layout",
        "solver": "z3", "solver_version": z3.get_version_string(), "timeout_ms": 2000, "resource_limit": 1_000_000,
        "result": "unsat-counterexample", "arithmetic_definedness": "all decoded operations total for all source inputs",
        "source_scalar_operations": len(operations), "representative_scalar_operations": len(emitted),
        "logical_input_slots": len(block.args), "uniform_captures": len(captures),
        "logical_result_slots": len(returned), "scalar_slots": pack,
        "projection": "representative slot; other slots zero only after universal independence and congruence proof"}
    return emitted, results, proof

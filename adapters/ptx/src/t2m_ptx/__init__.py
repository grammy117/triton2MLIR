"""PTX source decoding is optional; emitted computation uses standard MLIR only.

Semantics reference: NVIDIA PTX ISA 8.7, integer arithmetic, comparison/selection
and logic/shift instruction sections. No driver, device compiler or execution
backend is imported. Unknown instructions/modifiers/effects are refused.
"""

from xdsl.dialects import arith, builtin

from t2m.source_assembly import AssemblyExpansion, AssemblyRefusal

SPECIFICATION = "https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html"


def refuse(message):
    raise AssemblyRefusal("PTX integer source semantics: " + message)


def statements(text):
    """Bounded lexical parsing of assembly, never printed MLIR parsing."""
    tokens, position = [], 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        if text.startswith("//", position):
            end = text.find("\n", position + 2)
            position = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                refuse("unterminated comment")
            position = end + 2
            continue
        if char in ",;<>!@{}":
            tokens.append(char)
            position += 1
        else:
            start = position
            while position < len(text) and text[position].isascii() and (text[position].isalnum() or text[position] in "_.$%+-"):
                position += 1
            if start == position:
                refuse(f"unsupported assembly token at byte {position}")
            tokens.append(text[start:position])
        if len(tokens) > 8192:
            refuse("assembly exceeds its token budget")
    if tokens and tokens[0] == "{":
        if tokens[-1] != "}":
            refuse("unclosed assembly scope")
        tokens = tokens[1:-1]
    if any(token in ("{", "}") for token in tokens):
        refuse("nested assembly scopes need explicit register-scope handling")
    result, current = [], []
    for token in tokens:
        if token == ";":
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    if current:
        refuse("every assembly declaration/instruction must terminate with a semicolon")
    if not result or len(result) > 512:
        refuse("assembly requires between one and 512 statements")
    return result


def word_type(name):
    if name in ("b32", "u32", "s32"):
        return builtin.i32
    if name in ("b64", "u64", "s64"):
        return builtin.i64
    if name == "pred":
        return builtin.i1
    refuse("unsupported register/instruction type " + name)


class Provider:
    api_version = 1

    def lower(self, request, arguments):
        if request.pack != 1:
            refuse("packed elements require register-group facts")
        types = [*request.output_types, *request.input_types]
        if any(t not in (builtin.i32, builtin.i64) for t in types):
            refuse("this profile requires i32/i64 scalar interfaces")
        constraints = request.constraints.split(",")
        expected = ["r" if t == builtin.i32 else "l" for t in types]
        outputs = len(request.output_types)
        if len(constraints) != len(types) or len(types) > 256:
            refuse("constraint count differs from the value interface")
        for index, (constraint, register) in enumerate(zip(constraints, expected, strict=True)):
            choices = ("=" + register, "=&" + register) if index < outputs else (register,)
            if constraint not in choices:
                refuse("this profile requires matching r/l outputs and readonly inputs; ties/clobbers need another contract")
        registers = {f"${i}": t for i, t in enumerate(types)}
        values = {f"${outputs + i}": v for i, v in enumerate(arguments)}
        operations, instructions = [], []

        def emit(op):
            operations.append(op)
            return op.results[0]

        def constant(number, type_):
            bits = type_.width.data
            number &= (1 << bits) - 1
            if number >= 1 << (bits - 1):
                number -= 1 << bits
            return emit(arith.ConstantOp.from_int_and_width(number, type_))

        def read(token, type_):
            if token in registers:
                if registers[token] != type_ or token not in values:
                    refuse("uninitialized or differently sized register " + token)
                return values[token]
            if len(token) > 22 or not token or token[0] not in "+-0123456789":
                refuse("unknown register, physical state or literal " + token)
            try:
                magnitude = token.lstrip("+-")
                base = 16 if magnitude.startswith(("0x", "0X")) else (8 if len(magnitude) > 1 and magnitude[0] == "0" else 10)
                number = int(token, base)
                if abs(number) >= 1 << 64:
                    refuse("integer literal exceeds a 64-bit source word")
            except ValueError:
                refuse("unsupported integer literal " + token)
            return constant(number, type_)

        def predicate(tokens):
            if len(tokens) == 1:
                return read(tokens[0], builtin.i1)
            if len(tokens) == 2 and tokens[0] == "!":
                return emit(arith.XOrIOp(read(tokens[1], builtin.i1), constant(1, builtin.i1)))
            refuse("predicate must name one initialized predicate register")

        for statement in statements(request.assembly):
            if statement[0] == ".reg":
                if len(statement) < 3 or not statement[1].startswith("."):
                    refuse("malformed register declaration")
                type_ = word_type(statement[1][1:])
                entries = statement[2:]
                while entries:
                    name = entries.pop(0)
                    stem = name[1:]
                    if (not name.startswith("%") or not stem or not (stem[0].isalpha() or stem[0] in "_$")
                            or any(not (c.isalnum() or c in "_$") for c in stem)
                            or name in registers or len(registers) >= 512):
                        refuse("register declarations require unique local names within budget")
                    count = None
                    if entries and entries[0] == "<":
                        if len(entries) < 3 or entries[2] != ">" or not entries[1].isdecimal() or len(entries[1]) > 3:
                            refuse("malformed register array declaration")
                        count = int(entries[1])
                        entries = entries[3:]
                        if not 1 <= count <= 256:
                            refuse("register array exceeds its size budget")
                    names = [name] if count is None else [name + str(i) for i in range(count)]
                    if len(registers) + len(names) > 512 or any(n in registers for n in names):
                        refuse("register declarations collide or exceed their budget")
                    registers.update((n, type_) for n in names)
                    if entries and (entries.pop(0) != "," or not entries):
                        refuse("malformed register declaration separator")
                continue
            guard = None
            if statement[0] == "@":
                size = 3 if len(statement) > 1 and statement[1] == "!" else 2
                guard = predicate(statement[1:size])
                statement = statement[size:]
            if len(statement) < 3:
                refuse("malformed instruction")
            mnemonic, tokens = statement[0], statement[1:]
            operands, current = [], []
            for token in tokens:
                if token == ",":
                    if not current:
                        refuse("missing instruction operand")
                    operands.append(current)
                    current = []
                else:
                    current.append(token)
            if not current:
                refuse("missing instruction operand")
            operands.append(current)
            if len(operands[0]) != 1:
                refuse("instruction needs one scalar destination")
            destination = operands[0][0]
            if destination not in registers or destination in {f"${outputs+i}" for i in range(len(arguments))}:
                refuse("instruction destination must be a declared local or output register")
            old = values.get(destination)
            if guard is not None and old is None:
                refuse("predicated writes require an initialized destination")
            parts = mnemonic.split(".")
            name, modifiers = parts[0], parts[1:]
            if not modifiers:
                refuse("instruction requires an explicit type")
            type_name = modifiers[-1]
            type_ = word_type(type_name)

            def sources(count, *, count_type=None, operands=operands, mnemonic=mnemonic, type_=type_):
                if len(operands) != count + 1 or any(len(v) != 1 for v in operands[1:]):
                    refuse("instruction operand count/form differs from " + mnemonic)
                return [read(v[0], count_type if count_type is not None and i == count - 1 else type_)
                        for i, v in enumerate(operands[1:])]

            integer = type_name in ("u32", "s32", "u64", "s64")
            if name in ("add", "sub", "min", "max") and len(modifiers) == 1 and integer:
                a, b = sources(2)
                cls = {"add": arith.AddiOp, "sub": arith.SubiOp,
                    "min": arith.MinSIOp if type_name.startswith("s") else arith.MinUIOp,
                    "max": arith.MaxSIOp if type_name.startswith("s") else arith.MaxUIOp}[name]
                value = emit(cls(a, b))
            elif name in ("mul", "mad") and modifiers[:-1] == ["lo"] and integer:
                args = sources(2 if name == "mul" else 3)
                value = emit(arith.MuliOp(args[0], args[1]))
                if name == "mad":
                    value = emit(arith.AddiOp(value, args[2]))
            elif name in ("neg", "abs") and modifiers == [type_name] and type_name in ("s32", "s64"):
                a, = sources(1)
                negative = emit(arith.SubiOp(constant(0, type_), a))
                value = negative if name == "neg" else emit(arith.SelectOp(emit(arith.CmpiOp(a, constant(0, type_), "slt")), negative, a))
            elif name in ("and", "or", "xor", "not") and modifiers == [type_name] and type_name in ("b32", "b64", "pred"):
                args = sources(1 if name == "not" else 2)
                if name == "not":
                    value = emit(arith.XOrIOp(args[0], constant(-1, type_)))
                else:
                    value = emit({"and": arith.AndIOp, "or": arith.OrIOp, "xor": arith.XOrIOp}[name](*args))
            elif name in ("shl", "shr") and modifiers == [type_name] and type_ != builtin.i1 and (name == "shr" or type_name.startswith("b")):
                a, count = sources(2, count_type=builtin.i32)
                width = type_.width.data
                within = emit(arith.CmpiOp(count, constant(width, builtin.i32), "ult"))
                safe = emit(arith.MinUIOp(count, constant(width - 1, builtin.i32)))
                if type_ == builtin.i64:
                    safe = emit(arith.ExtUIOp(safe, builtin.i64))
                cls = arith.ShLIOp if name == "shl" else (arith.ShRSIOp if type_name.startswith("s") else arith.ShRUIOp)
                shifted = emit(cls(a, safe))
                fallback = emit(arith.ShRSIOp(a, constant(width - 1, type_))) if name == "shr" and type_name.startswith("s") else constant(0, type_)
                value = emit(arith.SelectOp(within, shifted, fallback))
            elif name == "setp" and len(modifiers) == 2 and type_ != builtin.i1 and modifiers[0] in ("eq", "ne", "lt", "le", "gt", "ge"):
                comparison = modifiers[0]
                if comparison not in ("eq", "ne") and not integer:
                    refuse("ordered setp needs an explicitly signed/unsigned integer type")
                a, b = sources(2)
                predicate_name = comparison if comparison in ("eq", "ne") else ("s" if type_name.startswith("s") else "u") + comparison
                value = emit(arith.CmpiOp(a, b, predicate_name))
            elif name == "selp" and len(modifiers) == 1 and type_ != builtin.i1:
                if len(operands) != 4 or any(len(v) != 1 for v in operands[1:3]):
                    refuse("selp requires two words and one predicate")
                value = emit(arith.SelectOp(predicate(operands[3]), read(operands[1][0], type_), read(operands[2][0], type_)))
            elif name == "mov" and len(modifiers) == 1 and type_ != builtin.i1:
                value, = sources(1)
            else:
                refuse("unsupported instruction/modifiers " + mnemonic)
            if registers[destination] != value.type:
                refuse("instruction destination type differs from its result")
            values[destination] = emit(arith.SelectOp(guard, value, old)) if guard is not None else value
            instructions.append({"ordinal": len(instructions), "mnemonic": mnemonic, "predicated": guard is not None})
        names = [f"${i}" for i in range(outputs)]
        if not instructions or any(name not in values for name in names):
            refuse("every output must have a definite initialized value")
        return AssemblyExpansion(operations, [values[name] for name in names], {
            "profile": "ptx-integer-v1", "source_language": "PTX", "specification": SPECIFICATION,
            "specification_version": "8.7", "word_arithmetic": "modular-without-saturation",
            "shift_counts": "unsigned-i32-clamped-to-word-width", "instructions": instructions,
            "registers": len(registers), "effects": "none; all source instructions decoded",
            "register_constraints": "distinct logical operands under valid readonly/early-clobber source constraints"})

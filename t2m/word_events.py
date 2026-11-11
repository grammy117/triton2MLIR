"""Use the official pointer converter on isolated, typed memory events.

Meta 17f0c234's TritonToUnstructuredPass.cpp and gather_no_loop.mlir supply
the base/offset conversion. The pinned binary can leave region-local and
chained runtime-stride pointer preparation unresolved. Event templates avoid that traversal;
their converted bodies are substituted at the original event, never hoisted.
The parent receives arithmetic conversion only: the pointer pass is not
idempotent on already converted unstructured gathers/scatters.
"""

from io import StringIO

from xdsl.dialects import arith, builtin
from xdsl.ir import Block, BlockArgument, Region
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .upstream import lower_with_shared

MEMORY_EVENTS = {"tt.load": "tts.gather", "tt.store": "tts.scatter",
                 "tt.atomic_rmw": "tts.unstructured_atomic_rmw", "tt.atomic_cas": "tt.atomic_cas"}


def print_ir(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
    return stream.getvalue() + "\n"


def needs_isolation(module):
    from .shared_tensor import opcode
    from .shared_tensor import _pointer_type

    # The shared whole-function pointer pass can leave chained pointer offsets
    # unresolved when descriptor strides are runtime values. Normalize their
    # actual base/offset expressions before invoking the same native converter
    # on event templates, independently of source annotations or kernel names.
    def nested_offset(value):
        seen = set()
        while value not in seen:
            seen.add(value)
            owner = value.owner
            if not hasattr(owner, "operands"):
                return False
            name = opcode(owner)
            if name == "tt.addptr":
                return True
            if name not in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans") or not owner.operands:
                return False
            value = owner.operands[0]
        return False

    if any(opcode(op) == "tt.addptr" and nested_offset(op.operands[0]) for op in module.walk()):
        return True
    word_atomics = (any(opcode(op) == "tt.atomic_rmw" for op in module.walk())
        and any(opcode(op) == "tt.bitcast" and any(_pointer_type(v.type) for v in op.results) for op in module.walk()))
    return word_atomics or any(opcode(op) == "tt.atomic_cas" for op in module.walk()) or any(opcode(op) in ("scf.if", "scf.while")
               and any(opcode(nested) in MEMORY_EVENTS for nested in op.walk()) for op in module.walk())


def same_type(left, right):
    # xDSL creates a distinct Python class per parsed unregistered attribute.
    # Native verification plus exact attribute parameters checks TT pointer
    # types across those contexts without comparing their printed spelling.
    return left == right or (isinstance(left, builtin.UnregisteredAttr)
        and isinstance(right, builtin.UnregisteredAttr) and left.parameters == right.parameters)


def templates(module, source_ids):
    from .shared_tensor import _array, opcode
    function, = module.ops
    entry = function.regions[0].block
    events = [op for op in function.walk() if opcode(op) in MEMORY_EVENTS]
    functions, bindings, calls = [], [], []
    for index, op in enumerate(events):
        def refuse(message):
            raise ConversionError(message, opcode=opcode(op), source_id=next(iter(source_ids(op.location)), None))
        pointer = op.operands[0]
        address = pointer.owner
        if not hasattr(address, "operands") or opcode(address) != "tt.addptr":
            refuse("isolated memory event requires normalized base plus signed i64 offset")
        base, offset = address.operands
        if isinstance(pointer.type, builtin.TensorType):
            if not hasattr(base.owner, "operands") or opcode(base.owner) != "tt.splat":
                refuse("isolated tensor memory needs a scalar logical entry base")
            base, = base.owner.operands
        view_type, view = None, None
        if hasattr(base.owner, "operands") and opcode(base.owner) == "tt.bitcast":
            from .pointer_views import compatible, element
            cast = base.owner
            if len(cast.operands) != 1 or not compatible(cast.operands[0].type, base.type):
                refuse("isolated word view requires equal-width logical storage")
            view_type = base.type
            base, = cast.operands
            view = {"storage_dtype": str(element(base.type)), "value_dtype": str(element(view_type)),
                    "element_bits": element(view_type).bitwidth, "address_units": "equal-width-elements"}
        if not isinstance(base, BlockArgument) or base.block is not entry:
            refuse("isolated memory base must be a source-entry logical buffer")
        offset_element = offset.type.element_type if isinstance(offset.type, builtin.TensorType) else offset.type
        if offset_element != builtin.i64:
            refuse("isolated memory offsets must retain the signed i64 source-word accumulator")
        other, mask = None, None
        write_or_atomic = opcode(op) in ("tt.store", "tt.atomic_rmw")
        if opcode(op) == "tt.atomic_cas":
            if len(op.operands) != 3:
                refuse("isolated CAS requires exact comparison/value operands")
            captures = [base, offset, *op.operands[1:]]
        elif write_or_atomic:
            if len(op.operands) not in (2, 3):
                refuse("isolated store/RMW requires a value and optional mask")
            if len(op.operands) == 3:
                mask = op.operands[2]
            captures = [base, offset, op.operands[1], *([mask] if mask is not None else [])]
        else:
            segments = _array(op, "operandSegmentSizes")
            if (len(segments) != 3 or segments[0] != 1 or any(n not in (0, 1) for n in segments[1:])
                    or sum(segments) != len(op.operands)):
                refuse("isolated load requires explicit optional mask/fill operands")
            if segments[1]:
                mask = op.operands[1]
            if segments[2]:
                other = op.operands[-1]
                if mask is None:
                    refuse("isolated load fill requires its source mask")
            captures = [base, offset, *([mask] if mask is not None else [])]
        template_types = [v.type for v in captures]
        cas_view = view_type is not None and opcode(op) == "tt.atomic_cas"
        if cas_view:
            # CAS already has its official descriptor. Present the logical
            # word base to the pointer pass and restore the storage cast at
            # substitution, so that pass does not rewrite a descriptor base
            # into an integer offset.
            template_types[0] = view_type
        body = Block(arg_types=template_types)
        local_base, local_offset = body.args[:2]
        def emit(name, **kwargs):
            generated = builtin.UnregisteredOp.with_name(name).create(location=op.location, **kwargs)
            body.add_op(generated)
            return generated
        if view_type is not None and not cas_view:
            local_base = emit("tt.bitcast", operands=[local_base], result_types=[view_type]).results[0]
        if opcode(op) == "tt.atomic_cas":
            from .compare_exchange import template
            returned = [template(body, op, [local_base, *body.args[1:]])]
        else:
            if isinstance(pointer.type, builtin.TensorType):
                local_base = emit("tt.splat", operands=[local_base], result_types=[pointer.type]).results[0]
            ptr = emit("tt.addptr", operands=[local_base, local_offset], result_types=[pointer.type]).results[0]
            operands = [ptr, *body.args[2:]]
            if other is not None:
                type_ = op.results[0].type
                element = type_.element_type if isinstance(type_, builtin.TensorType) else type_
                zero = (arith.ConstantOp(builtin.FloatAttr(0.0, element)) if isinstance(element, builtin.AnyFloat)
                        else arith.ConstantOp.from_int_and_width(0, element))
                zero.location = op.location
                body.add_op(zero)
                fill = zero.result
                if isinstance(type_, builtin.TensorType):
                    fill = emit("tt.splat", operands=[fill], result_types=[type_]).results[0]
                operands.append(fill)
            memory = type(op).create(operands=operands, result_types=[v.type for v in op.results],
                                    properties=op.properties, attributes=op.attributes, location=op.location)
            body.add_op(memory)
            returned = list(memory.results)
        emit("tt.return", operands=returned)
        name = f"__t2m_memory_event_{index}"
        functions.append(builtin.UnregisteredOp.with_name("tt.func").create(
            properties={"function_type": builtin.FunctionType.from_lists(template_types, [v.type for v in op.results]),
                        "sym_name": builtin.StringAttr(name), "sym_visibility": builtin.StringAttr("public")},
            attributes={"noinline": builtin.IntegerAttr(0, builtin.i1)}, regions=[Region(body)], location=op.location))
        bindings.append({"symbol": name, "opcode": opcode(op), "source_ids": source_ids(op.location),
                         "input_types": [str(v.type) for v in captures], "result_types": [str(v.type) for v in op.results],
                         "fill": "source-mask-select-after-zero-filled-load" if other is not None else "unchanged",
                         **({"word_view": view} if view else {}),
                         **({"template_input_types": [str(t) for t in template_types]} if cas_view else {})})
        calls.append((op, captures, mask, other))
    if not events:
        raise ConversionError("isolated memory conversion requires source memory events")
    return builtin.ModuleOp(functions), bindings, calls


def substitute(module, converted, bindings, calls):
    from .shared_tensor import opcode
    functions = {op.properties.get("sym_name").data: op for op in converted.ops if opcode(op) == "tt.func"}
    if len(functions) != len(bindings) or len(list(converted.ops)) != len(bindings):
        raise ConversionError("isolated event conversion changed the function inventory")
    replacements = {}
    for record, (original, captures, mask, other) in zip(bindings, calls, strict=True):
        captures = [replacements.get(v, v) for v in captures]
        mask, other = replacements.get(mask, mask), replacements.get(other, other)
        function = functions.get(record["symbol"])
        if function is None or len(function.regions[0].blocks) != 1:
            raise ConversionError("isolated event conversion lost its typed template")
        body = function.regions[0].block
        prefix = []
        arguments = list(captures)
        if "template_input_types" in record:
            from .pointer_views import compatible, element
            from .atomic_word_views import view_info
            pointer_type = original.operands[0].type
            viewed_type = pointer_type.element_type if isinstance(pointer_type, builtin.TensorType) else pointer_type
            if (record["opcode"] != "tt.atomic_cas" or len(body.args) != len(captures)
                    or record["input_types"] != [str(v.type) for v in captures]
                    or record["template_input_types"] != [str(v.type) for v in body.args]
                    or not same_type(body.args[0].type, viewed_type)
                    or not compatible(captures[0].type, viewed_type)
                    or record.get("word_view") != view_info(element(captures[0].type), element(viewed_type))):
                raise ConversionError("isolated word CAS changed its typed storage/view boundary")
            cast = builtin.UnregisteredOp.with_name("tt.bitcast").create(operands=[captures[0]],
                result_types=[viewed_type], location=original.location)
            prefix.append(cast)
            arguments[0] = cast.results[0]
        if len(body.args) != len(arguments) or any(not same_type(a.type, b.type) for a, b in zip(body.args, arguments)):
            raise ConversionError("isolated event conversion changed its typed capture bindings")
        operations = list(body.ops)
        expected = MEMORY_EVENTS[record["opcode"]]
        memories = [op for op in body.walk() if opcode(op) in MEMORY_EVENTS.values()]
        if (len(memories) != 1 or opcode(memories[0]) != expected or opcode(operations[-1]) != "tt.return"
                or any(opcode(op) in {*MEMORY_EVENTS.keys() - {"tt.atomic_cas"}, "tt.addptr", "tt.call"} for op in body.walk())):
            raise ConversionError("isolated event conversion did not retain exactly one memory observation")
        if record["opcode"] in ("tt.atomic_rmw", "tt.atomic_cas"):
            def attribute(op, key):
                return op.properties.get(key, op.attributes.get(key))
            keys = ("sem", "scope") + (("atomic_rmw_op",) if record["opcode"] == "tt.atomic_rmw" else ())
            if any(attribute(memories[0], key) != attribute(original, key) for key in keys):
                raise ConversionError("isolated atomic event changed its source operation/order/scope")
        mapping = dict(zip(body.args, arguments, strict=True))
        emitted = [*prefix, *[op.clone(value_mapper=mapping) for op in operations[:-1]]]
        values = [mapping.get(v, v) for v in operations[-1].operands]
        if len(values) != len(original.results) or any(not same_type(a.type, b.type) for a, b in zip(values, original.results)):
            raise ConversionError("isolated event conversion changed its typed result bindings")
        if other is not None:
            selected = arith.SelectOp(mask, values[0], other)
            selected.location = original.location
            emitted.append(selected)
            values = [selected.result]
        replacements.update(zip(original.results, values, strict=True))
        Rewriter.replace_op(original, emitted, values)


def convert_events(module, work, optimizer, source_ids):
    from .shared_tensor import parse_shared
    work.mkdir(parents=True, exist_ok=False)
    isolated, bindings, calls = templates(module, source_ids)
    event_input = print_ir(isolated)
    source = work / "events.mlir"
    source.write_text(event_input)
    report = lower_with_shared(source, work / "native", optimizer=optimizer, stage="source-word-events")
    event_output = (work / "native/lowered.mlir").read_text()
    substitute(module, parse_shared(event_output), bindings, calls)
    return {"input": event_input, "output": event_output, "native_stage": report, "bindings": bindings}

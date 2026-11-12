"""Typed source RMW admission and standard native-payload accounting.

Reuse Meta 17f0c234's unstructured atomic converter. Triton enum values come
from v3.7.1 TritonAttrDefs.td; standard RMW values from LLVM ArithBase.td.
The public buffers are private to this invocation, so standard acq_rel atomics
cover the original relaxed/acquire/release/acq_rel and closed logical scopes.
"""

from xdsl.dialects import arith, builtin, memref, scf
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, Region

from .diagnostics import ConversionError
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp
from .atomic_float_add import CONTRACT as F32_ADD_CONTRACT, F64_CONTRACT, HALF_CONTRACTS, HALF_PAYLOADS

EFFECT_PROFILE = "private-invocation-integer-atomics-v1"
TYPED_EFFECT_PROFILE = "private-invocation-typed-atomics-v2"
ATOMIC_SCHEMAS = ("t2m.memory-admission.v3", "t2m.memory-admission.v4")
# Source enum -> source name, standard arith::AtomicRMWKind.
RMW = {1: ("and", 2), 2: ("or", 14), 3: ("xor", 15), 4: ("add", 1), 5: ("fadd", 0),
       6: ("max", 6), 7: ("min", 10), 8: ("umax", 7), 9: ("umin", 11), 10: ("exch", 3)}
ORDERS = {1: "relaxed", 2: "acquire", 3: "release", 4: "acq_rel"}
SCOPES = {1: "gpu", 2: "cta", 3: "sys"}
FLOAT_CONTRACTS = {("exch", "f32"): ("bitwise-exchange",), ("exch", "f64"): ("bitwise-exchange",),
                   ("fadd", "f64"): (F64_CONTRACT, "ieee-f64-rne-add"), ("fadd", "f32"): ("ieee-f32-rne-add", F32_ADD_CONTRACT)}
FLOAT_CONTRACTS.update({("fadd", dtype): (contract,) for dtype, contract in HALF_CONTRACTS.items()})
FLOAT_WIDTHS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def payload(info):
    if "byte_view" in info:
        return "standard-byte-backed-word-atomic-with-original-returned-old-value"
    if info.get("numeric_contract") in HALF_CONTRACTS.values():
        return HALF_PAYLOADS[info["value_dtype"]]
    if info.get("numeric_contract") == F64_CONTRACT:
        from .atomic_float_add import F64_PAYLOAD
        return F64_PAYLOAD
    if info.get("numeric_contract") == "ieee-f32-rne-add":
        from .atomic_float_add import IEEE_PAYLOAD
        return IEEE_PAYLOAD
    if info.get("numeric_contract") == F32_ADD_CONTRACT:
        from .atomic_float_add import PAYLOAD
        return PAYLOAD
    if "word_view" in info:
        from .atomic_word_views import PAYLOAD, CAS_PAYLOAD
        return CAS_PAYLOAD if info["operation"] == "cas" else PAYLOAD
    return ("standard-memref-generic-compare-exchange-with-original-returned-old-value" if info["operation"] == "cas"
            else "standard-memref-atomic-rmw-with-original-returned-old-value")


def validate_evidence(admission, memory, records, buffers):
    """Validate declared atomic custody; this does not prove source equivalence."""
    profile = TYPED_EFFECT_PROFILE if admission.get("schema") == ATOMIC_SCHEMAS[1] else EFFECT_PROFILE
    if (admission.get("effect_profile") != profile or memory.get("effect_profile") != profile
            or admission.get("f32_atomic_add") not in (None, "ieee", "ftz")
            or memory.get("f32_atomic_add") not in (None, "ieee", "ftz")
            or admission.get("address_policy") != "source-words-offset64"
            or not isinstance(admission.get("atomic_adaptations"), list)
            or admission["atomic_adaptations"] != memory.get("atomic_payload")):
        raise ValueError("atomic memory evidence lacks its explicit private-invocation effect contract")
    source = {r["id"]: r.get("atomic") for r in records if r.get("opcode") in ("tt.atomic_rmw", "tt.atomic_cas")}
    opcodes = {r["id"]: r.get("opcode") for r in records}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    if len(accesses) != len(admission["atomic_adaptations"]):
        raise ValueError("atomic memory evidence does not account for every payload event")
    for access, actual in zip(accesses, admission["atomic_adaptations"], strict=True):
        info = access.get("atomic")
        buffer = buffers[access["buffer"]]
        candidates = [source[s] for s in access["source_ids"] if s in source]
        floating = isinstance(info, dict) and info.get("value_dtype") in FLOAT_WIDTHS
        dtype = info.get("value_dtype") if floating else "i" + str(info.get("word_bits")) if isinstance(info, dict) else None
        word = isinstance(info, dict) and "word_view" in info
        # A callee's typed primitive is shared; storage views belong to uses.
        byte = isinstance(info, dict) and "byte_view" in info
        primitive = {key: value for key, value in info.items() if key not in ("word_view", "byte_view")} if word or byte else info
        expected_view = {"storage_dtype": "f" + str(info.get("word_bits")), "value_dtype": "i" + str(info.get("word_bits")),
            "element_bits": info.get("word_bits"), "address_units": "equal-width-elements"} if word else None
        if word:
            dtype = expected_view["storage_dtype"]
        if byte:
            from .byte_atomic_events import view_info
            if type(info.get("word_bits")) is not int:
                raise ValueError("byte atomic word widths must be exact integers")
            try:
                expected_byte = view_info(builtin.IntegerType(info["word_bits"]), admission.get("byte_order"))
            except (ValueError, KeyError):
                raise ValueError("byte atomic evidence lacks its explicit word/byte boundary") from None
            if (profile != TYPED_EFFECT_PROFILE or floating or word or info["byte_view"] != expected_byte
                    or not isinstance(info["byte_view"], dict)
                    or any(type(info["byte_view"].get(key)) is not int for key in ("word_bits", "alignment_bytes"))
                    or memory.get("byte_order") != admission.get("byte_order")
                    or access.get("mask") not in ("unmasked", "source-per-lane")
                    or not isinstance(access.get("guard_tile"), list)
                    or any(type(n) is not int for n in access.get("guard_tile", []))
                    or access.get("guard_tile") != [*access.get("tile", []), expected_byte["alignment_bytes"]]):
                raise ValueError("byte atomic evidence differs from its complete word/byte ownership contract")
            dtype = "i8"
        if (not isinstance(info, dict) or type(info.get("word_bits")) is not int
                or not candidates or any(c != info and c != primitive for c in candidates)
                or buffer.get("dtype") != dtype
                or word and (profile != TYPED_EFFECT_PROFILE or floating
                    or info.get("word_bits") not in ((16, 32, 64) if info.get("operation") == "cas" else (32, 64)) or info["word_view"] != expected_view)
                or floating and (profile != TYPED_EFFECT_PROFILE
                    or (info.get("operation"), dtype) not in FLOAT_CONTRACTS
                    or info.get("word_bits") != FLOAT_WIDTHS[dtype]
                    or info.get("numeric_contract") not in FLOAT_CONTRACTS.get((info.get("operation"), dtype), ()))
                or not floating and info.get("operation") == "fadd"
                or floating and info.get("operation") == "fadd" and dtype == "f32"
                    and (admission.get("f32_atomic_add") != memory.get("f32_atomic_add")
                         or info.get("numeric_contract") != {"ieee": "ieee-f32-rne-add", "ftz": F32_ADD_CONTRACT}.get(admission.get("f32_atomic_add")))
                or not floating and ("value_dtype" in info or "numeric_contract" in info)
                or info.get("word_bits") not in ((16, 32, 64) if floating or info.get("operation") == "cas" else (32, 64))
                or (info.get("operation"), info.get("standard_kind")) not in (*RMW.values(), ("cas", "compare-exchange"))
                or access.get("opcode") != ("tt.atomic_cas" if info["operation"] == "cas" else "tts.unstructured_atomic_rmw")
                or any(opcodes[s] != ("tt.atomic_cas" if info["operation"] == "cas" else "tt.atomic_rmw") for s in access["source_ids"] if s in source)
                or info["operation"] == "cas" and access.get("mask") != "unmasked"
                or info.get("source_order") not in ORDERS.values() or info.get("source_scope") not in SCOPES.values()
                or info.get("standard_order") != "acq_rel"
                or info.get("logical_scope") != ("one-logical-program" if info["source_scope"] == "cta"
                    else "all-logical-programs-in-private-invocation")
                or info.get("external_participants") != "excluded-by-private-buffer-ABI"
                or not isinstance(actual, dict) or any(actual.get(key) != value for key, value in info.items())
                or actual.get("buffer") != buffer["name"] or not actual.get("source_ids")
                or not set(actual["source_ids"]) & set(access["source_ids"])
                or actual.get("payload") != payload(info)
                or actual.get("effect_boundary") != "standard-execute-region-no-inline"
                or actual.get("inactive_old_value") != ("zero-refinement-of-source-undefined-old-value"
                    if access["mask"] == "source-per-lane" else "no-inactive-lanes")):
            raise ValueError("atomic memory payload differs from its typed source operation/order/scope inventory")


def contract(op, source_ids, *, typed=False, f32_atomic_add=None):
    from .shared_tensor import opcode
    ids = source_ids(op.location)
    def refuse(message):
        raise ConversionError(message, opcode=opcode(op), source_id=next(iter(ids), None))
    type_ = op.results[0].type if len(op.results) == 1 else None
    element = type_.element_type if isinstance(type_, builtin.TensorType) else type_
    cas = opcode(op) in ("tt.atomic_cas", "tts.atomic_cas")
    floating = element in (builtin.f16, builtin.bf16, builtin.f32, builtin.f64)
    if not (typed and floating and not cas) and element not in ((builtin.i16, builtin.i32, builtin.i64) if cas else (builtin.i32, builtin.i64)):
        if cas:
            refuse("atomic CAS needs a qualified integer i16/i32/i64 numeric profile")
        refuse("atomic RMW needs a qualified integer i32/i64 numeric profile")
    values = [None] if cas else []
    for name, allowed in ((("atomic_rmw_op", RMW),) if not cas else ()) + (("sem", ORDERS), ("scope", SCOPES)):
        value = op.properties.get(name, op.attributes.get(name))
        if not isinstance(value, builtin.IntegerAttr) or value.value.data not in allowed:
            refuse("atomic RMW has an unqualified operation/order/scope")
        values.append(value.value.data)
    kind, order, scope = values
    operation, standard_kind = ("cas", "compare-exchange") if cas else RMW[kind]
    if (not floating and operation == "fadd") or (floating and (operation, str(element)) not in FLOAT_CONTRACTS):
        refuse("floating arithmetic atomics need a qualified rounding/subnormal/NaN numeric contract")
    numeric = None
    if floating:
        numeric = FLOAT_CONTRACTS[operation, str(element)][0]
        if operation == "fadd" and element == builtin.f32:
            numeric = {"ieee": "ieee-f32-rne-add", "ftz": F32_ADD_CONTRACT}.get(f32_atomic_add)
            if numeric is None:
                refuse("f32 atomic addition needs an explicitly qualified ieee or ftz numeric policy")
    return {"operation": operation, "standard_kind": standard_kind, "word_bits": element.bitwidth,
        **({"value_dtype": str(element), "numeric_contract": numeric} if floating else {}),
        "source_order": ORDERS[order], "source_scope": SCOPES[scope],
        "logical_scope": "one-logical-program" if scope == 2 else "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel", "external_participants": "excluded-by-private-buffer-ABI"}


def adapt_payload(module, abi, admission, source_ids):
    """Account for actual source-buffer atomics, retaining old values/effects."""
    function = next(iter(module.ops))
    buffers = {arg: value for arg, (kind, value) in zip(function.body.block.args, abi.source_bindings) if kind == "buffer"}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    observed = [0] * len(accesses)
    views = {"memref.cast", "memref.subview", "memref.reinterpret_cast", "memref.collapse_shape", "memref.expand_shape", "memref.view"}
    def root(value):
        while value not in buffers:
            op = value.owner
            if not hasattr(op, "name") or op.name not in views:
                return None
            value = op.operands[0]
        return buffers[value]
    reports, generics = [], []
    for op in list(function.walk()):
        region = isinstance(op, GenericAtomicRMWOp)
        word = region and op.result.type in (builtin.f16, builtin.f32, builtin.f64)
        cas = region and not word
        if not isinstance(op, (memref.AtomicRMWOp, GenericAtomicRMWOp)) or (buffer := root(op.operands[0 if region else 1])) is None:
            continue  # Compiler-owned guard atomics use separate arguments.
        ids = source_ids(op.location)
        matches = [i for i, access in enumerate(accesses)
                   if access["buffer"] == buffer.name and set(ids) & set(access["source_ids"])]
        contexts = [i for i in matches if set(ids) == set(accesses[i]["source_ids"])]
        if contexts:
            matches = contexts
        generic = op.parent_op()
        while generic is not None and not isinstance(generic, GenericOp):
            generic = generic.parent_op()
        if generic is None:
            raise ConversionError("native atomic left its parallel tensor event")
        if len(matches) == 1 and "byte_view" in accesses[matches[0]]["atomic"]:
            access = accesses[matches[0]]
            info = access["atomic"]
            if not region or list(generic.results[0].type.get_shape()) != access["tile"]:
                raise ConversionError("native byte atomic changed its original word event tile")
            from .byte_atomic_match import match as match_byte
            match_byte(generic, info, masked=access["mask"] == "source-per-lane")
            observed[matches[0]] += 1
            ancestor = generic
            while ancestor is not None:
                if isinstance(ancestor, GenericOp) and ancestor not in generics:
                    generics.append(ancestor)
                ancestor = ancestor.parent_op()
            op.attributes.update({"prov.atomic_operation": builtin.StringAttr(info["operation"]),
                "prov.atomic_source_order": builtin.StringAttr(info["source_order"]),
                "prov.atomic_source_scope": builtin.StringAttr(info["source_scope"]),
                "prov.atomic_logical_scope": builtin.StringAttr(info["logical_scope"]),
                "prov.atomic_standard_order": builtin.StringAttr("acq_rel")})
            reports.append({"source_ids": ids, "buffer": buffer.name, **info,
                "inactive_old_value": "zero-refinement-of-source-undefined-old-value" if access["mask"] == "source-per-lane" else "no-inactive-lanes",
                "effect_boundary": "standard-execute-region-no-inline", "payload": payload(info)})
            continue
        kind = op.properties.get("kind")
        standard_kind = "compare-exchange" if cas else kind.value.data if isinstance(kind, builtin.IntegerAttr) else None
        word_info = None
        if word:
            from .atomic_word_views import match
            word_kind, word_info = match(op, generic.body.block.args[1],
                generic.body.block.args[2] if len(generic.body.block.args) >= 4 else None)
            standard_kind = "compare-exchange" if word_kind == "cas" else dict(RMW.values())[word_kind]
            uses = tuple(op.result.uses)
            if (len(uses) != 1 or not isinstance(uses[0].operation, arith.BitcastOp)
                    or uses[0].operation.parent is not op.parent
                    or uses[0].operation.result.type != generic.body.block.args[1].type
                    or tuple(op.parent.last_op.operands) != (uses[0].operation.result,)):
                raise ConversionError("native word atomic lost its original returned old word")
        if (len(matches) != 1 or standard_kind != accesses[matches[0]]["atomic"]["standard_kind"]
                or word_info != accesses[matches[0]]["atomic"].get("word_view")):
            raise ConversionError("native source-buffer atomic differs from its typed admission inventory",
                                  opcode=op.name, source_id=next(iter(ids), None))
        index = matches[0]
        observed[index] += 1
        access = accesses[index]
        info = access["atomic"]
        if info.get("numeric_contract") in (F32_ADD_CONTRACT, "ieee-f32-rne-add", F64_CONTRACT, *HALF_CONTRACTS.values()):
            from .atomic_float_add import materialize
            op = materialize(op, ftz=info["numeric_contract"] == F32_ADD_CONTRACT)
        ancestor = generic
        while ancestor is not None:
            if isinstance(ancestor, GenericOp) and ancestor not in generics:
                generics.append(ancestor)
            ancestor = ancestor.parent_op()
        if op.results[0].type != buffer.type.element_type:
            raise ConversionError("native atomic old-value type differs from its logical buffer")
        if cas:
            body = op.body.block
            operations = list(body.ops)
            if (len(operations) != 3 or not isinstance(operations[0], arith.CmpiOp)
                    or not isinstance(operations[1], arith.SelectOp) or not isinstance(operations[2], AtomicYieldOp)
                    or operations[0].predicate.value.data != 0
                    or tuple(operations[0].operands) != (body.args[0], generic.body.block.args[1])
                    or tuple(operations[1].operands) != (operations[0].result, generic.body.block.args[2], body.args[0])
                    or operations[2].value is not operations[1].result):
                raise ConversionError("native CAS region differs from exact compare/exchange and returned-old-value semantics")
        refinement = "no-inactive-lanes"
        if access["mask"] == "source-per-lane":
            control = op.parent_op()
            if not isinstance(control, scf.IfOp) or len(control.regions[1].blocks) != 1:
                raise ConversionError("masked native atomic lacks its exact inactive old-value branch")
            generic = control.parent_op()
            if not isinstance(generic, GenericOp):
                raise ConversionError("masked native atomic moved outside its parallel tensor event")
            inactive = control.regions[1].block
            end = inactive.last_op
            if (not isinstance(end, scf.YieldOp) or len(end.operands) != 1
                    or end.operands[0] is not generic.body.block.args[-1]):
                raise ConversionError("native atomic inactive branch differs from the undefined output iter-arg")
            element = end.operands[0].type
            zero = (arith.ConstantOp(builtin.FloatAttr(0.0, element)) if isinstance(element, builtin.AnyFloat)
                    else arith.ConstantOp.from_int_and_width(0, element))
            zero.location = op.location
            inactive.insert_op_before(zero, end)
            end.operands = [zero.result]
            refinement = "zero-refinement-of-source-undefined-old-value"
        op.attributes.update({"prov.atomic_operation": builtin.StringAttr(info["operation"]),
            "prov.atomic_source_order": builtin.StringAttr(info["source_order"]),
            "prov.atomic_source_scope": builtin.StringAttr(info["source_scope"]),
            "prov.atomic_logical_scope": builtin.StringAttr(info["logical_scope"]),
            "prov.atomic_standard_order": builtin.StringAttr("acq_rel")})
        if "numeric_contract" in info:
            op.attributes.update({"prov.atomic_numeric_contract": builtin.StringAttr(info["numeric_contract"]),
                                  "prov.atomic_value_dtype": builtin.StringAttr(info["value_dtype"])})
        reports.append({"source_ids": ids, "buffer": buffer.name, **info, "inactive_old_value": refinement,
                        "effect_boundary": "standard-execute-region-no-inline",
                        "payload": payload(info)})
    if any(count != 1 for count in observed):
        raise ConversionError("native atomic payload does not account for every admitted memory event exactly once")
    for generic in generics:
        # LLVM's elementwise fusion checks tensor operands/iterators, not all
        # captured scalar-region effects. Cloning this producer duplicates the
        # RMW even with one consumer. A standard exactly-once region preserves
        # the parallel tile while keeping its effects at the original event.
        body = Block()
        boundary = scf.ExecuteRegionOp.create(result_types=[v.type for v in generic.results], regions=[Region(body)],
            attributes={"no_inline": builtin.UnitAttr(), "prov.atomic_effect_boundary": builtin.StringAttr("exactly-once")},
            location=generic.location)
        generic.parent.insert_op_before(boundary, generic)
        generic.parent.detach_op(generic)
        terminal = scf.YieldOp(*generic.results)
        terminal.location = generic.location
        body.add_ops([generic, terminal])
        for old, new in zip(generic.results, boundary.results, strict=True):
            old.replace_uses_with_if(new, lambda use: use.operation is not terminal)
    return reports

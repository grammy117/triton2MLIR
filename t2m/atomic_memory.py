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

EFFECT_PROFILE = "private-invocation-integer-atomics-v1"
# Source enum -> source name, standard arith::AtomicRMWKind.
RMW = {1: ("and", 2), 2: ("or", 14), 3: ("xor", 15), 4: ("add", 1),
       6: ("max", 6), 7: ("min", 10), 8: ("umax", 7), 9: ("umin", 11), 10: ("exch", 3)}
ORDERS = {1: "relaxed", 2: "acquire", 3: "release", 4: "acq_rel"}
SCOPES = {1: "gpu", 2: "cta", 3: "sys"}


def validate_evidence(admission, memory, records, buffers):
    """Validate declared atomic custody; this does not prove source equivalence."""
    if (admission.get("effect_profile") != EFFECT_PROFILE or memory.get("effect_profile") != EFFECT_PROFILE
            or admission.get("address_policy") != "source-words-offset64"
            or not isinstance(admission.get("atomic_adaptations"), list)
            or admission["atomic_adaptations"] != memory.get("atomic_payload")):
        raise ValueError("atomic memory evidence lacks its explicit private-invocation effect contract")
    source = {r["id"]: r.get("atomic") for r in records if r.get("opcode") == "tt.atomic_rmw"}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    if len(accesses) != len(admission["atomic_adaptations"]):
        raise ValueError("atomic memory evidence does not account for every payload event")
    for access, actual in zip(accesses, admission["atomic_adaptations"], strict=True):
        info = access.get("atomic")
        buffer = buffers[access["buffer"]]
        candidates = [source[s] for s in access["source_ids"] if s in source]
        if (not isinstance(info, dict) or not candidates or any(c != info for c in candidates)
                or buffer.get("dtype") != "i" + str(info.get("word_bits"))
                or info.get("word_bits") not in (32, 64)
                or (info.get("operation"), info.get("standard_kind")) not in RMW.values()
                or info.get("source_order") not in ORDERS.values() or info.get("source_scope") not in SCOPES.values()
                or info.get("standard_order") != "acq_rel"
                or info.get("logical_scope") != ("one-logical-program" if info["source_scope"] == "cta"
                    else "all-logical-programs-in-private-invocation")
                or info.get("external_participants") != "excluded-by-private-buffer-ABI"
                or not isinstance(actual, dict) or any(actual.get(key) != value for key, value in info.items())
                or actual.get("buffer") != buffer["name"] or not actual.get("source_ids")
                or not set(actual["source_ids"]) & set(access["source_ids"])
                or actual.get("payload") != "standard-memref-atomic-rmw-with-original-returned-old-value"
                or actual.get("effect_boundary") != "standard-execute-region-no-inline"
                or actual.get("inactive_old_value") != ("zero-refinement-of-source-undefined-old-value"
                    if access["mask"] == "source-per-lane" else "no-inactive-lanes")):
            raise ValueError("atomic memory payload differs from its typed source operation/order/scope inventory")


def contract(op, source_ids):
    from .shared_tensor import opcode
    ids = source_ids(op.location)
    def refuse(message):
        raise ConversionError(message, opcode=opcode(op), source_id=next(iter(ids), None))
    type_ = op.results[0].type if len(op.results) == 1 else None
    element = type_.element_type if isinstance(type_, builtin.TensorType) else type_
    if element not in (builtin.i32, builtin.i64):
        refuse("atomic RMW needs a qualified integer i32/i64 numeric profile")
    values = []
    for name, allowed in (("atomic_rmw_op", RMW), ("sem", ORDERS), ("scope", SCOPES)):
        value = op.properties.get(name, op.attributes.get(name))
        if not isinstance(value, builtin.IntegerAttr) or value.value.data not in allowed:
            refuse("atomic RMW has an unqualified operation/order/scope")
        values.append(value.value.data)
    kind, order, scope = values
    return {"operation": RMW[kind][0], "standard_kind": RMW[kind][1], "word_bits": element.width.data,
        "source_order": ORDERS[order], "source_scope": SCOPES[scope],
        "logical_scope": "one-logical-program" if scope == 2 else "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel", "external_participants": "excluded-by-private-buffer-ABI"}


def adapt_payload(module, abi, admission, source_ids):
    """Account for actual source-buffer atomics, retaining old values/effects."""
    function = next(iter(module.ops))
    buffers = {arg: value for arg, (kind, value) in zip(function.body.block.args, abi.source_bindings) if kind == "buffer"}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    observed = [0] * len(accesses)
    views = {"memref.cast", "memref.subview", "memref.reinterpret_cast", "memref.collapse_shape", "memref.expand_shape"}
    def root(value):
        while value not in buffers:
            op = value.owner
            if not hasattr(op, "name") or op.name not in views:
                return None
            value = op.operands[0]
        return buffers[value]
    reports, generics = [], []
    for op in list(function.walk()):
        if not isinstance(op, memref.AtomicRMWOp) or (buffer := root(op.operands[1])) is None:
            continue  # Compiler-owned guard atomics use separate arguments.
        ids = source_ids(op.location)
        matches = [i for i, access in enumerate(accesses)
                   if access["buffer"] == buffer.name and set(ids) & set(access["source_ids"])]
        kind = op.properties.get("kind")
        if (len(matches) != 1 or not isinstance(kind, builtin.IntegerAttr)
                or kind.value.data != accesses[matches[0]]["atomic"]["standard_kind"]):
            raise ConversionError("native source-buffer atomic differs from its typed admission inventory",
                                  opcode=op.name, source_id=next(iter(ids), None))
        index = matches[0]
        observed[index] += 1
        access = accesses[index]
        info = access["atomic"]
        generic = op.parent_op()
        while generic is not None and not isinstance(generic, GenericOp):
            generic = generic.parent_op()
        if generic is None:
            raise ConversionError("native atomic left its parallel tensor event")
        if generic not in generics:
            generics.append(generic)
        if op.results[0].type != buffer.type.element_type:
            raise ConversionError("native atomic old-value type differs from its logical buffer")
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
            zero = arith.ConstantOp.from_int_and_width(0, op.results[0].type)
            zero.location = op.location
            inactive.insert_op_before(zero, end)
            end.operands = [zero.result]
            refinement = "zero-refinement-of-source-undefined-old-value"
        op.attributes.update({"prov.atomic_operation": builtin.StringAttr(info["operation"]),
            "prov.atomic_source_order": builtin.StringAttr(info["source_order"]),
            "prov.atomic_source_scope": builtin.StringAttr(info["source_scope"]),
            "prov.atomic_logical_scope": builtin.StringAttr(info["logical_scope"]),
            "prov.atomic_standard_order": builtin.StringAttr("acq_rel")})
        reports.append({"source_ids": ids, "buffer": buffer.name, **info, "inactive_old_value": refinement,
                        "effect_boundary": "standard-execute-region-no-inline",
                        "payload": "standard-memref-atomic-rmw-with-original-returned-old-value"})
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

"""Admit actual typed source effects separately from word-stage byte custody."""

from xdsl.dialects import builtin
from xdsl.utils.exceptions import ParseError

from .diagnostics import ConversionError


def validate_source_effects(source, stage, memory, records):
    """Check source primitives before any legacy stage can return successfully.

    This reconstructs selection and primitive fields, not source equivalence,
    address geometry, memory dependencies or final executable correctness.
    Unreachable declarations may remain in the original inventory; every
    actual effect must nevertheless match its retained source declaration.
    """
    from .atomic_memory import EFFECT_PROFILE, TYPED_EFFECT_PROFILE, contract as atomic_contract
    from .shared_tensor import _source_ids, opcode, parse_shared
    from .volatile_memory import EFFECT_PROFILE as VOLATILE_PROFILE, contract as volatile_contract

    if (not isinstance(source, (str, bytes)) or not isinstance(stage, dict) or not isinstance(memory, dict)
            or not isinstance(records, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                                                   or not row["id"] for row in records)):
        raise ValueError("source-word effects require typed source, stage, policy and inventory objects")
    try:
        module = parse_shared(source.decode() if isinstance(source, bytes) else source)
    except ParseError as error:
        raise ValueError("source-word effects require actual parseable typed IR") from error
    rows = {row.get("id"): row for row in records}
    if None in rows or len(rows) != len(records):
        raise ValueError("word source effects require distinct retained source identities")
    atomic_rows = {sid: row for sid, row in rows.items() if row.get("opcode") in ("tt.atomic_rmw", "tt.atomic_cas")}
    volatile_rows = {sid: row for sid, row in rows.items() if "volatile_read" in row}
    atomic_ids, volatile_ids = set(atomic_rows), set(volatile_rows)
    atomic_profile = memory.get("effect_profile")
    volatile_profile = memory.get("volatile_profile", atomic_profile)
    volatile_stage = stage.get("schema") in ("t2m.source-word-stages.v3", "t2m.source-word-stages.v4")
    legacy = stage.get("schema") in ("t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2")
    for op in module.walk():
        name = opcode(op)
        if name.startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl.", "llvm.")) or name in (
                "tt.inline_asm", "tt.elementwise_inline_asm"):
            raise ValueError("word source retains an unexpanded physical/effect operation: " + name)
        if name == "tt.load":
            flag = op.properties.get("isVolatile", op.attributes.get("isVolatile"))
            if flag is not None and (not isinstance(flag, builtin.IntegerAttr) or flag.type != builtin.i1):
                raise ValueError("word source volatile flag requires the actual pinned boolean")
            selected = set(_source_ids(op.location)) & volatile_ids
            if flag is None or not flag.value.data:
                if selected:
                    raise ValueError("word source lost an original volatile observation")
                continue
            if volatile_profile != VOLATILE_PROFILE or not volatile_stage:
                raise ValueError("actual volatile source requires its selected nondestructive word/effect profile at "
                                 + str(op.location))
            if not selected:
                raise ValueError("actual volatile source lacks its primitive declaration")
            for sid in selected:
                if (volatile_rows[sid].get("opcode") != name
                        or volatile_contract(op, sid) != volatile_rows[sid]["volatile_read"]):
                    raise ValueError("word source volatile primitive differs from its original declaration")
        elif name in ("tt.atomic_rmw", "tt.atomic_cas"):
            if atomic_profile not in (EFFECT_PROFILE, TYPED_EFFECT_PROFILE):
                raise ValueError("actual atomic source requires its selected private numeric/effect profile at "
                                 + str(op.location))
            selected = set(_source_ids(op.location)) & atomic_ids
            if not selected:
                raise ValueError("actual atomic source lacks its primitive declaration")
            try:
                actual = atomic_contract(op, _source_ids, typed=atomic_profile == TYPED_EFFECT_PROFILE,
                                         f32_atomic_add=memory.get("f32_atomic_add"))
            except ConversionError as error:
                raise ValueError("actual word source atomic is outside its selected primitive policy") from error
            for sid in selected:
                declared = atomic_rows[sid].get("atomic")
                if not isinstance(declared, dict):
                    raise ValueError("word source atomic lacks its original primitive fields")
                # View permissions remain checked by the separate address and
                # storage contracts. They cannot change the source primitive.
                primitive = {key: value for key, value in declared.items() if key not in ("word_view", "byte_view")}
                # Earlier v4 atomic evidence names the same f64 IEEE source
                # operation before the quiet-NaN payload acquired its new label.
                # Keep that exact legacy label in evidence; this is comparison
                # compatibility, not a new arithmetic or NaN permission.
                if (legacy and primitive.get("operation") == "fadd" and primitive.get("value_dtype") == "f64"
                        and primitive.get("numeric_contract") == "ieee-f64-rne-add"):
                    primitive["numeric_contract"] = "ieee-f64-rne-add-quiet-nans"
                if atomic_rows[sid].get("opcode") != name or primitive != actual:
                    raise ValueError("word source atomic primitive differs from its original declaration")

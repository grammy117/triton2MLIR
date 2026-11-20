"""Explicit representation planning; decisions grant no semantic permission."""

from .memory_contract import GUARDED_PROFILE, TENSOR_PROFILE, WORD_PROFILE

SCHEMA = "t2m.memory-selection.v1"
STRATEGY = "first-admitted-portable-representation"
SCOPE = "frontend-admission-only; numeric-effect-ownership-target-permissions-unchanged"
CANDIDATES = (
    ("tensor", "structured"),
    ("guarded-buffer", "structured"),
    ("guarded-buffer", "source-words"),
)
PROFILES = {
    CANDIDATES[0]: TENSOR_PROFILE,
    CANDIDATES[1]: GUARDED_PROFILE,
    CANDIDATES[2]: WORD_PROFILE,
}


def candidates(options):
    from .fp8_casts import SELECTIONS

    if options["address_policy"] not in ("structured", "source-words"):
        raise ValueError("automatic memory selection requires a known address policy")
    if options["effect_profile"] not in ("ordinary", "atomics", "typed-atomics", "volatile"):
        raise ValueError("automatic memory selection requires a known effect profile")
    if options["byte_order"] not in (None, "little", "big"):
        raise ValueError(
            "automatic memory selection requires a known logical byte order"
        )
    if options["fp8_overflow"] not in (None, *SELECTIONS):
        raise ValueError("automatic memory selection requires a known FP8 policy")
    if (
        options["f32_atomic_add"] not in (None, "ieee", "ftz")
        or options["f32_atomic_add"] is not None
        and options["effect_profile"] != "typed-atomics"
    ):
        raise ValueError(
            "automatic memory selection requires the explicit typed atomic policy"
        )
    if (
        options["address_policy"] == "source-words"
        or options["byte_order"] is not None
        or options["effect_profile"] != "ordinary"
    ):
        return CANDIDATES[-1:]
    return CANDIDATES


def validate_selection(trace):
    if "memory_selection" not in trace:
        return
    record = trace["memory_selection"]
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema",
            "strategy",
            "scope",
            "requested",
            "source_sha256",
            "attempts",
            "status",
        }
        or record["schema"] != SCHEMA
        or record["strategy"] != STRATEGY
        or record["scope"] != SCOPE
        or record["status"] != "admitted"
        or record["source_sha256"] != trace["source_sha256"]
    ):
        raise ValueError("automatic memory selection has an invalid admission record")
    options = record["requested"]
    if not isinstance(options, dict) or set(options) != {
        "address_policy",
        "effect_profile",
        "byte_order",
        "fp8_overflow",
        "f32_atomic_add",
    }:
        raise ValueError(
            "automatic memory selection changes its original policy roster"
        )
    allowed = candidates(options)
    attempts = record["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= len(allowed):
        raise ValueError(
            "automatic memory selection has an incomplete attempt inventory"
        )
    for index, row in enumerate(attempts):
        last = index == len(attempts) - 1
        expected_fields = {"memory_profile", "address_policy", "status"} | (
            set() if last else {"diagnostic", "source_id", "opcode"}
        )
        if (
            not isinstance(row, dict)
            or set(row) != expected_fields
            or (row["memory_profile"], row["address_policy"]) != allowed[index]
            or row["status"] != ("admitted" if last else "not-admitted")
        ):
            raise ValueError(
                "automatic memory selection changes its candidate order or disposition"
            )
        if not last and (
            not isinstance(row["diagnostic"], str)
            or not 1 <= len(row["diagnostic"]) <= 4096
            or any(
                row[key] is not None and not isinstance(row[key], str)
                for key in ("source_id", "opcode")
            )
        ):
            raise ValueError("automatic memory selection loses its refusal diagnostic")
    if trace.get("profile", TENSOR_PROFILE) != PROFILES[allowed[len(attempts) - 1]]:
        raise ValueError(
            "automatic memory selection disagrees with the actual representation"
        )

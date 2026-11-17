"""Explicit consumer allocation limits; no kernel or invocation shape facts."""

PROFILE = "merlin.owned-result-limits.v2"


def checked_policy(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "max_bytes", "max_allocations", "max_output_bytes", "index_bits"}
        or value.get("schema") != PROFILE
        or type(value["index_bits"]) is not int
        or value["index_bits"] != 64
        or any(
            type(value[key]) is not int or not 0 < value[key] < 1 << 63
            for key in ("max_bytes", "max_allocations", "max_output_bytes")
        )
    ):
        raise ValueError(
            "execution allocation policy requires the complete bounded v2 resource profile"
        )
    return dict(value)

"""Expose defined helper bodies while retaining source call-grouping intent.

Triton f797708c Dialect.cpp's inliner checks the Boolean noinline attribute.
Reuse that inliner after moving this performance hint into descriptive
provenance. Values, effects, types and the actual helper body are unchanged.
"""

from xdsl.dialects import builtin

from .diagnostics import ConversionError


def expose_defined_helpers(module, source_ids):
    from .shared_tensor import opcode

    reports, functions = [], []
    for function in module.ops:
        if opcode(function) != "tt.func" or "noinline" not in function.attributes:
            continue
        hint = function.attributes["noinline"]
        ids = source_ids(function.location)
        if (not isinstance(hint, builtin.IntegerAttr) or hint.type != builtin.i1
                or hint.value.data not in (-1, 0, 1)):
            raise ConversionError("source noinline hint requires its native Boolean type",
                opcode="tt.func", source_id=next(iter(ids), None))
        if hint.value.data == 0:
            continue
        if len(function.regions) != 1 or not function.regions[0].blocks:
            raise ConversionError("noinline declaration requires an actual portable function implementation",
                opcode="tt.func", source_id=next(iter(ids), None))
        reports.append({
            "kind": "source-call-grouping-intent",
            "function": function.properties["sym_name"].data,
            "source_ids": ids,
            "body_source_ids": list(dict.fromkeys(source for op in function.walk() for source in source_ids(op.location))),
            "source_noinline": True,
            "lowering": "defined-body-exposed-through-existing-upstream-inliner",
            "preserved": "complete-typed-body-values-effects-tiles-reuse-control-and-source-identity",
            "performance_boundary": "physical-call-boundary-code-size-and-register-pressure-not-preserved; downstream-may-outline",
            "grants_semantics": False,
        })
        functions.append(function)
    for function in functions:
        function.attributes.pop("noinline")
    return reports

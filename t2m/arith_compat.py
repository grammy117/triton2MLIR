"""Represent explicit neutral native defaults in the pinned xDSL schema.

LLVM a47bddccec30255619bb8c37fa59700e661d4e66 ArithOps.td defines
trunci overflowFlags with the default `none`. xDSL 0.68's TruncIOp has no
such property and implements ordinary modular truncation. Only the explicitly
empty flag set is equivalent to that schema; poison-producing flags refuse.
"""

from xdsl.dialects import arith, builtin

from .diagnostics import ConversionError


def normalize_truncation_defaults(module, source_ids):
    reports = []
    for op in module.walk():
        if op.name != "arith.trunci" or "overflowFlags" not in op.properties:
            continue
        flags = op.properties["overflowFlags"]
        ids = source_ids(op.location)
        if not isinstance(flags, arith.IntegerOverflowAttr) or flags.data:
            raise ConversionError("integer truncation with poison-producing overflow flags needs a qualified schema bridge",
                                  opcode=op.name, source_id=next(iter(ids), None))
        del op.properties["overflowFlags"]
        op.attributes["prov.arith_truncation_default"] = builtin.StringAttr("explicit-overflow-none-equivalent-to-absent-default")
        reports.append({"kind": "native-neutral-truncation-default", "source_ids": ids,
            "opcode": op.name, "source_default": "overflow<none>", "emitted_default": "ordinary-modular-truncation",
            "nonempty_flags": "refused-without-schema-and-numeric-qualification"})
    return reports
